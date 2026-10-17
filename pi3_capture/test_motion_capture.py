import hashlib
import json
import sys
import tempfile
import threading
import time
import unittest
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import motion_capture as mc  # noqa: E402


class FakeServer:
    """Minimal stand-in for POST /v1/uploads/image."""

    def __init__(self, status=200):
        self.status = status
        self.received = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                msg = BytesParser(policy=HTTP).parsebytes(
                    b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + body)
                parts = {p.get_param("name", header="content-disposition"): p for p in msg.iter_parts()}
                text = lambda k: parts[k].get_payload(decode=True).decode()
                if outer.status == 200:
                    outer.received.append({
                        "path": self.path,
                        "filename": parts["file"].get_filename(),
                        "image": parts["file"].get_payload(decode=True),
                        "auth_key": text("auth_key"),
                        "unique_id": text("unique_id"),
                        "metadata": json.loads(text("metadata")),
                    })
                self.send_response(outer.status)
                self.end_headers()

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def make_config(tmp, url):
    tmp = Path(tmp)
    (tmp / "key").write_text("secret-key\n")
    (tmp / "uuid").write_text("1234-uuid\n")
    return mc.Config(server_base_url=url, access_key_file=str(tmp / "key"),
                     unique_id_path=str(tmp / "uuid"), queue_dir=str(tmp / "queue"),
                     warmup_frames=3, cooldown_seconds=0.2)


class MotionDetectorTest(unittest.TestCase):
    def test_static_noise_and_lighting_do_not_trigger_but_object_does(self):
        rng = np.random.default_rng(0)
        base = np.full((90, 160), 100, np.uint8)
        det = mc.MotionDetector(warmup_frames=3)
        for _ in range(20):
            noisy = np.clip(base + rng.integers(-5, 6, base.shape), 0, 255).astype(np.uint8)
            self.assertFalse(det.update(noisy))
        self.assertFalse(det.update(np.full_like(base, 200)))  # lights on
        self.assertFalse(det.update(np.full_like(base, 200)))
        det = mc.MotionDetector(warmup_frames=3)
        for _ in range(5):
            det.update(base)
        moving = base.copy()
        moving[30:60, 50:90] = 220  # ~8% of the frame
        results = [det.update(moving) for _ in range(2)]
        self.assertEqual(results, [False, True])


class FakeCamera:
    def __init__(self, frames):
        self.frames = list(frames)

    def read(self):
        if not self.frames:
            time.sleep(0.05)
            return True, self.last
        self.last = self.frames.pop(0)
        return True, self.last

    def release(self):
        pass


class EndToEndTest(unittest.TestCase):
    def test_motion_triggers_capture_and_upload(self):
        server = FakeServer()
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, server.url)
            empty = np.full((720, 1280, 3), 90, np.uint8)
            animal = empty.copy()
            animal[200:500, 400:800] = (30, 160, 220)
            frames = [empty] * 10 + [animal] * 3 + [empty] * 10
            stop = threading.Event()
            t = threading.Thread(target=mc.run, args=(config, stop, lambda c: FakeCamera(frames)))
            t.start()
            deadline = time.time() + 30
            while not server.received and time.time() < deadline:
                time.sleep(0.1)
            time.sleep(0.5)
            stop.set()
            t.join(15)
            server.close()

            self.assertGreaterEqual(len(server.received), 1)
            up = server.received[0]
            expected_id = hashlib.sha256(b"1234-uuid").hexdigest()[:12]
            self.assertEqual(up["path"], "/v1/uploads/image")
            self.assertEqual(up["auth_key"], "secret-key")
            self.assertEqual(up["unique_id"], expected_id)
            self.assertTrue(up["image"].startswith(b"\xff\xd8"))  # JPEG
            self.assertEqual(up["metadata"]["image_name"], up["filename"])
            self.assertEqual(up["metadata"]["trigger"], "motion")
            self.assertEqual(up["metadata"]["detections"], [])
            self.assertEqual(list(Path(config.queue_dir).iterdir()), [])  # deleted after upload


class QueueTest(unittest.TestCase):
    def test_failed_upload_is_kept_and_retried(self):
        server = FakeServer(status=503)
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, server.url)
            q = mc.UploadQueue(config, "k", "id")
            q.enqueue(b"\xff\xd8jpeg", {"image_name": "a.jpg", "captured_at": "", "detections": []})
            self.assertFalse(q.drain())
            self.assertEqual([p.name for p in q.pending()], ["a.jpg"])
            server.status = 200
            self.assertTrue(q.drain())
            self.assertEqual(q.pending(), [])
            self.assertEqual(len(server.received), 1)
        server.close()

    def test_size_limit_drops_oldest(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, "http://127.0.0.1:9")
            config.max_queue_mb = 1.5 / 1024  # 1.5 KB
            q = mc.UploadQueue(config, "k", "id")
            for name in ("a.jpg", "b.jpg", "c.jpg"):
                q.enqueue(b"x" * 1024, {"image_name": name})
            self.assertEqual([p.name for p in q.pending()], ["c.jpg"])
            self.assertFalse((Path(config.queue_dir) / "a.json").exists())


if __name__ == "__main__":
    unittest.main()
