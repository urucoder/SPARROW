import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import edge_inference as edge
import motion_capture as mc
from onboard import onboard
from test_motion_capture import FakeCamera, FakeServer, make_config


def output(*rows):
    result = np.zeros((1, 300, 6), np.float32)
    for index, row in enumerate(rows):
        result[0, index] = row
    return result


class PreprocessTest(unittest.TestCase):
    def test_rgb_unit_nchw_and_padding(self):
        frame = np.full((720, 1280, 3), [0, 127, 255], np.uint8)
        tensor, transform = edge.preprocess(frame)
        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)
        self.assertEqual((transform.left, transform.top), (0, 140))
        np.testing.assert_allclose(tensor[0, :, 140, 0], [1, 127 / 255, 0])
        np.testing.assert_allclose(tensor[0, :, 139, 0], [114 / 255] * 3)

    def test_exact_inverse_for_portrait_odd_landscape_and_clipping(self):
        for height, width in [(721, 1281), (1001, 333), (17, 29), (720, 1280)]:
            with self.subTest(shape=(height, width)):
                _, t = edge.preprocess(np.zeros((height, width, 3), np.uint8))
                x1, y1, x2, y2 = width // 4, height // 4, width * 3 // 4, height * 3 // 4
                row = [x1 * t.resized_width / width + t.left,
                       y1 * t.resized_height / height + t.top,
                       x2 * t.resized_width / width + t.left,
                       y2 * t.resized_height / height + t.top, .8, 0]
                detections, reason = edge.decode(output(row), t, .5, .2)
                self.assertEqual(reason, "animal")
                self.assertEqual(detections[0]["bbox"], [x1, y1, x2, y2])
                detections, _ = edge.decode(output([-10, -10, 650, 650, .8, 0]), t, .5, .2)
                self.assertEqual(detections[0]["bbox"], [0, 0, width, height])

    def test_invalid_frames(self):
        for frame in (np.zeros((1, 1)), np.zeros((0, 1, 3), np.uint8)):
            with self.assertRaises(ValueError):
                edge.preprocess(frame)


class DecodeTest(unittest.TestCase):
    def setUp(self):
        self.transform = edge.preprocess(np.zeros((720, 1280, 3), np.uint8))[1]

    def decode(self, *rows):
        return edge.decode(output(*rows), self.transform, .5, .25)

    def test_person_first_even_with_animal_or_offscreen_person(self):
        for person in ([5, 5, 30, 30, .25, 1], [0, 0, 0, 0, .25, 1]):
            self.assertEqual(self.decode([100, 200, 300, 350, .9, 0], person),
                             ([], "person"))

    def test_blank_vehicle_low_animal_and_degenerate_boxes(self):
        for rows in ([], [[0, 140, 10, 150, .49, 0]], [[0, 140, 10, 150, .9, 2]],
                     [[0, 0, 10, 10, .9, 0]], [[100, 200, 100, 300, .9, 0]]):
            self.assertEqual(self.decode(*rows), ([], "no_animal"))

    def test_confidence_boundary_and_metadata(self):
        detections, _ = self.decode([100, 200, 300, 350, .5, 0], [0, 0, 10, 10, .24, 1])
        self.assertEqual(detections, [{"label": "animal", "confidence": .5,
                                      "bbox": [200, 120, 600, 420], "class_id": 0}])

    def test_reject_invalid_tensors_and_rows(self):
        bad_outputs = [np.zeros((1, 6), np.float32), output().astype(np.float64)]
        for row in ([0, 0, 10, 10, float("nan"), 0], [0, 0, 10, 10, 1.1, 0],
                    [0, 0, 10, 10, .8, 3], [0, 0, 10, 10, .8, .5],
                    [10, 0, 1, 20, .8, 0], [0, 0, float("inf"), 1, .8, 0]):
            bad_outputs.append(output(row))
        for result in bad_outputs:
            with self.assertRaises(ValueError):
                edge.decode(result, self.transform, .5, .2)


class ConfigAndModelTest(unittest.TestCase):
    def test_settings_fail_closed(self):
        for kwargs in ({"edge_person_threshold": 0}, {"edge_animal_threshold": float("nan")},
                       {"edge_intra_threads": 0}, {"edge_inter_threads": 5},
                       {"edge_queue_dir": "/same", "queue_dir": "/same"}):
            with self.subTest(settings=kwargs), self.assertRaises(ValueError):
                mc.Config(edge_enabled=True, **kwargs).validate()
        with patch.dict("os.environ", {"EDGE_ENABLED": "maybe"}), self.assertRaises(ValueError):
            mc.Config()

    def test_bad_model_before_runtime_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.onnx"
            path.write_bytes(b"not the catalog artifact")
            with self.assertRaisesRegex(ValueError, "checksum"):
                edge.EdgeDetector(mc.Config(edge_model_path=str(path)))

    def test_missing_runtime_is_an_actionable_startup_error(self):
        with patch.object(edge, "verify_model", return_value=edge.MODEL_SHA256), \
                patch.dict(sys.modules, {"onnxruntime": None}):
            with self.assertRaisesRegex(RuntimeError, "needs ONNX Runtime"):
                edge.EdgeDetector(mc.Config())

    def test_runtime_contract_threads_and_single_session(self):
        session = Mock()
        session.get_inputs.return_value = [Mock(shape=edge.INPUT_SHAPE, type="tensor(float)", name="images")]
        session.get_outputs.return_value = [Mock(shape=edge.OUTPUT_SHAPE, type="tensor(float)", name="output")]
        session.get_modelmeta.return_value.custom_metadata_map = {
            "names": "{0: 'animal', 1: 'person', 2: 'vehicle'}"}
        session.run.return_value = [output([100, 200, 300, 350, .8, 0])]
        runtime = Mock()
        runtime.InferenceSession.return_value = session
        for name in ("Fail", "InvalidArgument", "RuntimeException", "EngineError", "EPFail"):
            setattr(runtime.capi.onnxruntime_pybind11_state, name, RuntimeError)
        with patch.dict(sys.modules, {"onnxruntime": runtime}), patch.object(edge, "verify_model", return_value=edge.MODEL_SHA256):
            detector = edge.EdgeDetector(mc.Config())
            for _ in range(2):
                self.assertEqual(detector.detect(np.zeros((720, 1280, 3), np.uint8))[1], "animal")
            runtime.InferenceSession.assert_called_once()
            options = runtime.SessionOptions.return_value
            self.assertEqual(options.intra_op_num_threads, 2)
            self.assertEqual(options.inter_op_num_threads, 1)
            self.assertEqual(runtime.InferenceSession.call_args.kwargs["providers"], ["CPUExecutionProvider"])
            session.run.side_effect = RuntimeError("test runtime failure")
            with self.assertRaisesRegex(RuntimeError, "ONNX Runtime execution failed"):
                detector.detect(np.zeros((720, 1280, 3), np.uint8))
            session.get_outputs.return_value[0].shape = [1, 84, 8400]
            with self.assertRaisesRegex(ValueError, "contract"):
                edge.EdgeDetector(mc.Config())
            session.get_outputs.return_value[0].shape = edge.OUTPUT_SHAPE
            session.get_modelmeta.return_value.custom_metadata_map["names"] = "{0: 'person'}"
            with self.assertRaisesRegex(ValueError, "label"):
                edge.EdgeDetector(mc.Config())


class EdgeCaptureTest(unittest.TestCase):
    def test_queue_cap_cannot_delete_an_inflight_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, "http://127.0.0.1:9")
            config.max_queue_mb = 100 / (1024 * 1024)
            session = Mock()
            entered, release = threading.Event(), threading.Event()

            def post(*args, **kwargs):
                entered.set()
                self.assertTrue(release.wait(3))
                return Mock()

            session.post.side_effect = post
            queue = mc.UploadQueue(config, "key", "unit", session)
            queue.enqueue(b"a" * 75, {"image_name": "a.jpg"})
            thread = threading.Thread(target=queue.drain)
            thread.start()
            self.assertTrue(entered.wait(3))
            try:
                queue.enqueue(b"b" * 75, {"image_name": "b.jpg"})
                self.assertTrue((queue.dir / "a.jpg").exists())
                self.assertFalse((queue.dir / "b.jpg").exists())
            finally:
                release.set()
                thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertFalse(queue.pending())

    def test_single_oversized_capture_is_dropped_visibly(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, "http://127.0.0.1:9")
            config.max_queue_mb = 10 / (1024 * 1024)
            queue = mc.UploadQueue(config, "key", "unit")
            with self.assertLogs(mc.logger, level="WARNING"):
                queue.enqueue(b"x" * 11, {"image_name": "large.jpg"})
            self.assertFalse(list(queue.dir.iterdir()))

    def test_disabled_mode_does_not_import_optional_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, "http://127.0.0.1:9")
            stop = threading.Event()
            stop.set()
            with patch.dict(sys.modules, {"onnxruntime": None}), \
                    patch.object(mc.UploadQueue, "start"), patch.object(mc.UploadQueue, "stop"):
                mc.run(config, stop)
            self.assertTrue(Path(config.queue_dir).exists())

    def test_shutdown_during_inference_does_not_enqueue(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, "http://127.0.0.1:9")
            config.edge_enabled = True
            config.edge_queue_dir = str(Path(tmp) / "edge")
            stop = threading.Event()
            detector = Mock()

            def detect(_):
                stop.set()
                return [{"label": "animal"}], "animal"

            detector.detect.side_effect = detect
            motion = Mock(last_ratio=.1)
            motion.update.return_value = True
            with patch.object(mc, "MotionDetector", return_value=motion), \
                    patch.object(mc.UploadQueue, "start"), patch.object(mc.UploadQueue, "stop"):
                mc.run(config, stop, lambda _: FakeCamera([np.zeros((20, 20, 3), np.uint8)]),
                       lambda _: detector)
            self.assertFalse(list(Path(config.edge_queue_dir).iterdir()))

    def test_policy_in_real_capture_loop_and_distinct_queues(self):
        for decision in ("animal", "person", "no_animal", "error"):
            with self.subTest(decision=decision), tempfile.TemporaryDirectory() as tmp:
                config = make_config(tmp, "http://127.0.0.1:9/v1")
                config.edge_enabled = True
                config.edge_queue_dir = str(Path(tmp) / "edge-queue")
                stop = threading.Event()
                frame = np.zeros((720, 1280, 3), np.uint8)
                detector = Mock()
                meta = {"model": edge.MODEL_ID, "sha256": edge.MODEL_SHA256,
                        "person_checked": True}
                detector.metadata.return_value = meta
                detection = {"label": "animal", "class_id": 0,
                             "confidence": .9, "bbox": [1, 2, 100, 200]}

                def detect(_):
                    if decision == "error":
                        stop.set()
                        raise ValueError("bad output")
                    return ([detection] if decision == "animal" else [], decision)

                detector.detect.side_effect = detect
                # Stop at the next read: exercises exactly one eligible event.
                class Camera(FakeCamera):
                    def read(self):
                        if not self.frames:
                            stop.set()
                        return super().read()

                camera = Camera([frame] * 5)
                motion = Mock()
                motion.last_ratio = .1
                motion.update.side_effect = [False, False, False, True, False, False]
                with patch.object(mc, "MotionDetector", return_value=motion), \
                        patch.object(mc.UploadQueue, "start"), patch.object(mc.UploadQueue, "stop"), \
                        patch.dict(sys.modules, {"onnxruntime": None}):
                    mc.run(config, stop, lambda _: camera, lambda _: detector)
                paths = list(Path(config.edge_queue_dir).glob("*.jpg"))
                self.assertEqual(len(paths), 1 if decision == "animal" else 0)
                self.assertFalse(Path(config.queue_dir).exists())
                if paths:
                    metadata = json.loads(paths[0].with_suffix(".json").read_text())
                    self.assertEqual(metadata["detections"], [detection])
                    self.assertEqual(metadata["edge_policy"], meta)

    def test_enabled_startup_failure_never_starts_uploading(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, "http://127.0.0.1:9")
            config.edge_enabled = True
            with patch.object(mc.UploadQueue, "start") as start, self.assertRaises(FileNotFoundError):
                mc.run(config, threading.Event())
            start.assert_not_called()

    def test_corrupt_or_unchecked_edge_queue_never_uploaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, "http://127.0.0.1:9")
            config.edge_enabled = True
            config.edge_queue_dir = str(Path(tmp) / "edge")
            session = Mock()
            queue = mc.UploadQueue(config, "key", "unit", session)
            jpg = queue.dir / "old.jpg"
            jpg.write_bytes(b"jpeg")
            for content in (None, "broken", '{"detections": []}', "[]",
                            '{"edge_policy": null}', '{"edge_policy": []}'):
                if content is not None:
                    jpg.with_suffix(".json").write_text(content)
                self.assertFalse(queue.drain())
            session.post.assert_not_called()
            self.assertTrue(jpg.exists())

    def test_nonempty_detection_upload_and_restart_retry(self):
        server = FakeServer(503)
        self.addCleanup(server.close)
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, server.url)
            config.edge_enabled = True
            config.edge_queue_dir = str(Path(tmp) / "edge")
            meta = {"image_name": "animal.jpg", "detections": [
                {"label": "animal", "class_id": 0, "confidence": .9, "bbox": [1, 2, 10, 20]}],
                "edge_policy": {"model": edge.MODEL_ID, "sha256": edge.MODEL_SHA256, "person_checked": True}}
            queue = mc.UploadQueue(config, "key", "unit")
            queue.enqueue(b"\xff\xd8jpeg", meta)
            self.assertFalse(queue.drain())
            server.status = 200
            restarted = mc.UploadQueue(config, "key", "unit")
            self.assertTrue(restarted.drain())
            self.assertEqual(server.received[0]["metadata"], meta)
            self.assertFalse(list(queue.dir.iterdir()))


class OnboardingTest(unittest.TestCase):
    def test_contract_and_idempotent_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, "http://localhost/v1")
            unit = mc.get_hardware_id(config.unique_id_path)
            session = Mock()
            session.post.return_value.json.return_value = {
                "ok": True, "unit_id": unit, "project_id": 1, "device_id": 1, "idempotent": True}
            self.assertTrue(onboard(config, "issuer@example.org", session)["idempotent"])
            kwargs = session.post.call_args.kwargs
            self.assertEqual(kwargs["json"], {"unit_id": unit})
            self.assertEqual(kwargs["headers"], {"X-API-Key": "secret-key", "X-Email": "issuer@example.org"})
            session.post.return_value.json.return_value["unit_id"] = "different"
            with self.assertRaises(ValueError):
                onboard(config, "issuer@example.org", session)


if __name__ == "__main__":
    unittest.main()
