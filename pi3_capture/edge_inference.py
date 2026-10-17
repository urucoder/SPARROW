"""Adapter for the verified, end-to-end MDV6-yolov10-c ONNX export."""

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODEL_ID = "MDV6-yolov10-c"
MODEL_SHA256 = "d38a71b93dcf34147261c6af7665cf3ad2495c924d15503fd88304950cf8eb6a"
INPUT_SHAPE = [1, 3, 640, 640]
OUTPUT_SHAPE = [1, 300, 6]


@dataclass(frozen=True)
class Letterbox:
    width: int
    height: int
    resized_width: int
    resized_height: int
    left: int
    top: int


def preprocess(frame):
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected a nonempty uint8 BGR camera frame")
    height, width = frame.shape[:2]
    if not width or not height:
        raise ValueError("Empty camera frame")
    scale = min(640 / width, 640 / height)
    rw, rh = max(1, round(width * scale)), max(1, round(height * scale))
    left, top = (640 - rw) // 2, (640 - rh) // 2
    resized = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)
    padded = np.full((640, 640, 3), 114, dtype=np.uint8)
    padded[top:top + rh, left:left + rw] = resized
    tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None],
                                 dtype=np.float32) / 255.0
    return tensor, Letterbox(width, height, rw, rh, left, top)


def decode(output, transform, animal_threshold, person_threshold):
    """Validate every row before applying person-first policy. No second NMS."""
    if (not isinstance(output, np.ndarray) or output.shape != tuple(OUTPUT_SHAPE)
            or output.dtype != np.float32 or not np.isfinite(output).all()):
        raise ValueError("Expected finite float32 detector output [1,300,6]")
    rows = output[0]
    scores, classes = rows[:, 4], rows[:, 5]
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("Detector confidence outside [0,1]")
    if np.any((classes != np.floor(classes)) | (classes < 0) | (classes > 2)):
        raise ValueError("Unsupported detector class (expected animal=0, person=1, vehicle=2)")
    active = rows[scores > 0]
    if np.any((active[:, 2] < active[:, 0]) | (active[:, 3] < active[:, 1])):
        raise ValueError("Detector returned inverted coordinates")
    # Even a person whose box clips to zero area must suppress the whole frame.
    if np.any((classes == 1) & (scores >= person_threshold)):
        return [], "person"
    detections = []
    for x1, y1, x2, y2, score, class_id in rows:
        if class_id != 0 or score < animal_threshold:
            continue
        box = [
            (float(x1) - transform.left) * transform.width / transform.resized_width,
            (float(y1) - transform.top) * transform.height / transform.resized_height,
            (float(x2) - transform.left) * transform.width / transform.resized_width,
            (float(y2) - transform.top) * transform.height / transform.resized_height,
        ]
        box = [round(min(max(v, 0), bound)) for v, bound in
               zip(box, (transform.width, transform.height) * 2)]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        detections.append({"label": "animal", "confidence": float(score),
                           "bbox": box, "class_id": 0})
    return detections, "animal" if detections else "no_animal"


def verify_model(path):
    with Path(path).open("rb") as model:
        digest = hashlib.file_digest(model, "sha256").hexdigest()
    if digest != MODEL_SHA256:
        raise ValueError(f"Model checksum mismatch; expected {MODEL_ID} SHA-256 {MODEL_SHA256}")
    return digest


class EdgeDetector:
    def __init__(self, config):
        self.sha256 = verify_model(config.edge_model_path)
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Edge mode needs ONNX Runtime; rerun install.sh --edge-model PATH") from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = config.edge_intra_threads
        options.inter_op_num_threads = config.edge_inter_threads
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(config.edge_model_path, sess_options=options,
                                           providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or inputs[0].shape != INPUT_SHAPE or inputs[0].type != "tensor(float)"
                or len(outputs) != 1 or outputs[0].shape != OUTPUT_SHAPE
                or outputs[0].type != "tensor(float)"):
            raise ValueError("Model contract mismatch: expected float32 [1,3,640,640] -> [1,300,6]")
        names = ast.literal_eval(self.session.get_modelmeta().custom_metadata_map.get("names", "{}"))
        if names != {0: "animal", 1: "person", 2: "vehicle"}:
            raise ValueError("Model label mapping mismatch")
        self.input_name, self.output_name = inputs[0].name, outputs[0].name
        self.animal_threshold = config.edge_animal_threshold
        self.person_threshold = config.edge_person_threshold
        errors = ort.capi.onnxruntime_pybind11_state
        self.runtime_errors = (errors.Fail, errors.InvalidArgument, errors.RuntimeException,
                               errors.EngineError, errors.EPFail)

    def detect(self, frame):
        tensor, transform = preprocess(frame)
        try:
            output = self.session.run([self.output_name], {self.input_name: tensor})[0]
        except self.runtime_errors as exc:
            raise RuntimeError(f"ONNX Runtime execution failed: {exc}") from exc
        return decode(output, transform, self.animal_threshold, self.person_threshold)

    def metadata(self):
        return {"model": MODEL_ID, "sha256": self.sha256,
                "animal_threshold": self.animal_threshold,
                "person_threshold": self.person_threshold, "person_checked": True}
