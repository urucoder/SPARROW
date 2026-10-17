#!/usr/bin/env python3
"""
SPARROW capture-only mode for low-power boards (e.g. Raspberry Pi 3 B+).

Watches a USB webcam, takes a photo whenever motion is detected and uploads it
to the SPARROW server (POST {SERVER_BASE_URL}/uploads/image) for storage.
No server AI inference is triggered by this upload. Optional EDGE_ENABLED
mode performs local animal detection and best-effort person suppression
before encoding or queueing. No sensors, PyTorch or Docker are required.

Accepted photos are written to a bounded disk queue; a background thread
retries uploads. Oldest photos can be dropped at the configured size cap.
"""

import hashlib
import json
import logging
import math
import os
import signal
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

logger = logging.getLogger("sparrow_capture")


def _env(name, default, cast=str):
    def factory():
        value = os.getenv(name)
        return default if value in (None, "") else cast(value)
    return field(default_factory=factory)


def _bool(value):
    if value.lower() not in ("true", "false", "1", "0"):
        raise ValueError("Boolean settings must be true/false or 1/0")
    return value.lower() in ("true", "1")


@dataclass
class Config:
    server_base_url: str = _env("SERVER_BASE_URL", "https://server.sparrowstudio.azure.com/v1")
    access_key_file: str = _env("ACCESS_KEY_FILE", "/etc/sparrow-capture/access_key.txt")
    unique_id_path: str = _env("UNIQUE_ID_PATH", "/etc/unique_id")
    camera_device: str = _env("CAMERA_DEVICE", "/dev/video0")
    capture_width: int = _env("CAPTURE_WIDTH", 1280, int)
    capture_height: int = _env("CAPTURE_HEIGHT", 720, int)
    jpeg_quality: int = _env("JPEG_QUALITY", 90, int)
    # Motion detection
    detect_width: int = _env("DETECT_WIDTH", 160, int)
    pixel_threshold: int = _env("MOTION_PIXEL_THRESHOLD", 25, int)
    min_motion_ratio: float = _env("MOTION_MIN_RATIO", 0.01, float)
    max_motion_ratio: float = _env("MOTION_MAX_RATIO", 0.6, float)
    motion_frames: int = _env("MOTION_FRAMES", 2, int)
    background_alpha: float = _env("BACKGROUND_ALPHA", 0.05, float)
    cooldown_seconds: float = _env("COOLDOWN_SECONDS", 5, float)
    warmup_frames: int = _env("WARMUP_FRAMES", 30, int)
    # Upload queue
    queue_dir: str = _env("QUEUE_DIR", "/var/lib/sparrow-capture/queue")
    max_queue_mb: float = _env("MAX_QUEUE_MB", 1024, float)
    upload_timeout: float = _env("UPLOAD_TIMEOUT", 30, float)
    retry_max_seconds: float = _env("RETRY_MAX_SECONDS", 300, float)
    edge_enabled: bool = _env("EDGE_ENABLED", False, _bool)
    edge_model_path: str = _env("EDGE_MODEL_PATH", "/opt/sparrow-capture/models/mdv6-yolov10-c.onnx")
    edge_queue_dir: str = _env("EDGE_QUEUE_DIR", "/var/lib/sparrow-capture/edge-queue")
    edge_animal_threshold: float = _env("EDGE_ANIMAL_THRESHOLD", 0.5, float)
    edge_person_threshold: float = _env("EDGE_PERSON_THRESHOLD", 0.2, float)
    edge_intra_threads: int = _env("EDGE_INTRA_THREADS", 2, int)
    edge_inter_threads: int = _env("EDGE_INTER_THREADS", 1, int)

    def validate(self):
        for name in ("capture_width", "capture_height", "detect_width", "motion_frames"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("max_queue_mb", "upload_timeout", "retry_max_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 1 <= self.jpeg_quality <= 100 or not 0 <= self.pixel_threshold <= 255:
            raise ValueError("Invalid JPEG quality or motion pixel threshold")
        if not 0 < self.min_motion_ratio <= self.max_motion_ratio <= 1:
            raise ValueError("Motion ratios must satisfy 0 < min <= max <= 1")
        if not 0 < self.background_alpha <= 1:
            raise ValueError("BACKGROUND_ALPHA must be in (0,1]")
        if not math.isfinite(self.cooldown_seconds) or self.cooldown_seconds < 0 or self.warmup_frames < 0:
            raise ValueError("Cooldown and warmup must be nonnegative")
        if self.edge_enabled:
            for name in ("edge_animal_threshold", "edge_person_threshold"):
                if not 0 < getattr(self, name) <= 1:
                    raise ValueError(f"{name} must be in (0,1]")
            if not 1 <= self.edge_intra_threads <= 4 or not 1 <= self.edge_inter_threads <= 4:
                raise ValueError("Edge thread counts must be between 1 and 4")
            if Path(self.edge_queue_dir).resolve() == Path(self.queue_dir).resolve():
                raise ValueError("EDGE_QUEUE_DIR must differ from the unchecked capture-only queue")

    @property
    def upload_url(self):
        return f"{self.server_base_url.rstrip('/')}/uploads/image"


def get_hardware_id(uuid_path):
    """Same ID scheme as sparrow/utils/sparrow_id.py so the server sees a consistent device ID."""
    uuid_str = Path(uuid_path).read_text(encoding="utf-8").strip()
    if not uuid_str:
        raise ValueError(f"UUID file {uuid_path} is empty")
    return hashlib.sha256(uuid_str.encode("utf-8")).hexdigest()[:12]


class MotionDetector:
    """Running-average background subtraction on small grayscale frames.

    `update` takes a grayscale uint8 frame (already downscaled) and returns True
    once the share of changed pixels has stayed in [min_ratio, max_ratio] for
    `motion_frames` consecutive frames. Changes above max_ratio are treated as a
    global lighting/exposure change and only refresh the background.
    """

    def __init__(self, pixel_threshold=25, min_ratio=0.01, max_ratio=0.6,
                 motion_frames=2, alpha=0.05, warmup_frames=30):
        self.pixel_threshold = pixel_threshold
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.motion_frames = max(1, motion_frames)
        self.alpha = alpha
        self.warmup_frames = warmup_frames
        self.reset()

    def reset(self):
        self.background = None
        self.frames_seen = 0
        self.consecutive = 0
        self.last_ratio = 0.0

    def update(self, gray):
        frame = gray.astype(np.float32)
        self.frames_seen += 1
        if self.background is None or self.background.shape != frame.shape:
            self.background = frame
            self.consecutive = 0
            return False

        diff = np.abs(frame - self.background)
        ratio = float(np.count_nonzero(diff > self.pixel_threshold)) / diff.size
        self.last_ratio = ratio

        if ratio > self.max_ratio:
            # Lighting/exposure change: adopt the new scene immediately.
            self.background = frame
            self.consecutive = 0
            return False

        # Learn slowly so a moving subject isn't absorbed into the background.
        self.background = (1 - self.alpha) * self.background + self.alpha * frame

        if self.frames_seen <= self.warmup_frames:
            return False

        if ratio >= self.min_ratio:
            self.consecutive += 1
        else:
            self.consecutive = 0
        return self.consecutive >= self.motion_frames


class UploadQueue:
    """Disk-backed queue of JPEG + JSON sidecar pairs, uploaded by a background thread."""

    def __init__(self, config, auth_key, unique_id, session=None):
        self.config = config
        self.auth_key = auth_key
        self.unique_id = unique_id
        self.session = session or requests.Session()
        self.dir = Path(config.edge_queue_dir if config.edge_enabled else config.queue_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.RLock()
        self._uploading = None

    def enqueue(self, jpeg_bytes, metadata):
        with self._lock:
            self._enqueue(jpeg_bytes, metadata)

    def _enqueue(self, jpeg_bytes, metadata):
        self._validate_edge_metadata(metadata)
        stem = self.dir / Path(metadata["image_name"]).stem
        side_tmp = self.dir / f"{stem.name}.json.tmp"
        img_tmp = self.dir / f"{stem.name}.jpg.tmp"
        for path, data in ((side_tmp, json.dumps(metadata).encode()), (img_tmp, jpeg_bytes)):
            with path.open("wb") as target:
                target.write(data)
                target.flush()
                os.fsync(target.fileno())
        # Sidecar first: the uploader only picks up *.jpg, whose metadata must already exist.
        os.replace(side_tmp, stem.with_suffix(".json"))
        os.replace(img_tmp, stem.with_suffix(".jpg"))
        fd = os.open(self.dir, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        self.enforce_size_limit()
        self._wake.set()

    def pending(self):
        return sorted(self.dir.glob("*.jpg"))

    def enforce_size_limit(self):
        with self._lock:
            self._enforce_size_limit()

    def _enforce_size_limit(self):
        limit = self.config.max_queue_mb * 1024 * 1024
        files = self.pending()
        total = sum(p.stat().st_size for p in files)
        while files and total > limit:
            oldest = files.pop(0)
            if oldest == self._uploading:
                continue
            total -= oldest.stat().st_size
            logger.warning("Queue over %.0f MB, dropping oldest capture %s", self.config.max_queue_mb, oldest.name)
            self._remove(oldest)

    @staticmethod
    def _remove(jpg):
        jpg.unlink(missing_ok=True)
        jpg.with_suffix(".json").unlink(missing_ok=True)

    def upload_one(self, jpg):
        with self._lock:
            try:
                meta = json.loads(jpg.with_suffix(".json").read_text())
            except (OSError, ValueError):
                if self.config.edge_enabled:
                    raise ValueError(f"Missing or corrupt edge metadata: {jpg.name}")
                logger.warning("Missing/corrupt metadata for %s; using capture-only fallback", jpg.name)
                meta = {"image_name": jpg.name, "captured_at": "", "detections": []}
            self._validate_edge_metadata(meta)
            image = jpg.read_bytes()
            self._uploading = jpg
        started = time.monotonic()
        try:
            response = self.session.post(
                self.config.upload_url,
                files={"file": (jpg.name, image, "image/jpeg")},
                data={
                    "auth_key": self.auth_key,
                    "unique_id": self.unique_id,
                    "metadata": json.dumps(meta),
                },
                timeout=self.config.upload_timeout,
            )
            response.raise_for_status()
            with self._lock:
                self._remove(jpg)
            logger.info("Uploaded %s in %.3fs", jpg.name, time.monotonic() - started)
        finally:
            with self._lock:
                self._uploading = None

    def _validate_edge_metadata(self, meta):
        if not self.config.edge_enabled:
            return
        from edge_inference import MODEL_ID, MODEL_SHA256

        if not isinstance(meta, dict):
            raise ValueError("Invalid edge metadata object")
        policy = meta.get("edge_policy", {})
        detections = meta.get("detections", [])
        if (not isinstance(policy, dict) or policy.get("model") != MODEL_ID
                or policy.get("sha256") != MODEL_SHA256
                or policy.get("person_checked") is not True
                or not isinstance(detections, list) or not detections):
            raise ValueError("Refusing unchecked or invalid image in edge queue")
        for detection in detections:
            if not isinstance(detection, dict):
                raise ValueError("Invalid edge detection")
            score = detection.get("confidence")
            box = detection.get("bbox")
            if (detection.get("label") != "animal" or detection.get("class_id") != 0
                    or not isinstance(score, (float, int)) or not 0 < score <= 1
                    or not isinstance(box, list) or len(box) != 4
                    or any(type(v) is not int or v < 0 for v in box)
                    or box[2] <= box[0] or box[3] <= box[1]):
                raise ValueError("Invalid edge animal score or pixel box")

    def drain(self):
        """Upload everything currently queued, oldest first. Returns True if nothing failed."""
        for jpg in self.pending():
            if self._stop.is_set():
                return False
            try:
                self.upload_one(jpg)
            except FileNotFoundError:
                continue  # pruned by enforce_size_limit meanwhile
            except (requests.RequestException, OSError, ValueError) as e:
                logger.warning("Upload of %s failed: %s", jpg.name, e)
                return False
        return True

    def _run(self):
        backoff = min(5.0, self.config.retry_max_seconds)
        while not self._stop.is_set():
            self._wake.clear()
            if self.drain():
                backoff = min(5.0, self.config.retry_max_seconds)
                self._wake.wait(timeout=60)
            else:
                # New captures must not defeat retry backoff during an outage.
                self._stop.wait(timeout=backoff)
                backoff = min(backoff * 2, self.config.retry_max_seconds)

    def start(self):
        self._thread = threading.Thread(target=self._run, name="uploader", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join()
        self.session.close()


def open_camera(config):
    import cv2

    device = int(config.camera_device) if config.camera_device.isdigit() else config.camera_device
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    # MJPEG keeps USB 2.0 bandwidth and Pi 3 CPU usage low at 720p.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.capture_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.capture_height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open camera {config.camera_device}")
    return cap


def to_detect_frame(frame, detect_width):
    import cv2

    h, w = frame.shape[:2]
    small = cv2.resize(frame, (detect_width, max(1, int(h * detect_width / w))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def run(config, stop_event, camera_factory=open_camera, edge_factory=None):
    import cv2

    config.validate()
    edge = None
    if config.edge_enabled:
        from edge_inference import EdgeDetector

        edge = (edge_factory or EdgeDetector)(config)
        logger.info("Edge mode enabled: %s", edge.metadata())
    auth_key = Path(config.access_key_file).read_text(encoding="utf-8").strip()
    if not auth_key:
        raise ValueError(f"Access key file {config.access_key_file} is empty")
    unique_id = get_hardware_id(config.unique_id_path)
    logger.info("Device ID %s, uploading to %s", unique_id, config.upload_url)

    queue = UploadQueue(config, auth_key, unique_id)
    queue.start()
    detector = MotionDetector(
        config.pixel_threshold, config.min_motion_ratio, config.max_motion_ratio,
        config.motion_frames, config.background_alpha, config.warmup_frames,
    )
    hostname = socket.gethostname()
    last_capture = 0.0
    inference_backoff = 1.0
    cap = None
    try:
        while not stop_event.is_set():
            if cap is None:
                try:
                    cap = camera_factory(config)
                    detector.reset()
                    logger.info("Camera %s opened", config.camera_device)
                except (RuntimeError, OSError, cv2.error) as e:
                    logger.error("%s; retrying in 10 s", e)
                    stop_event.wait(10)
                    continue

            ok, frame = cap.read()
            if not ok or frame is None:
                logger.error("Camera read failed; reopening")
                cap.release()
                cap = None
                stop_event.wait(2)
                continue

            moved = detector.update(to_detect_frame(frame, config.detect_width))
            now = time.monotonic()
            if not moved or (last_capture and now - last_capture < config.cooldown_seconds):
                continue

            last_capture = now
            captured_at = datetime.now(timezone.utc)
            detections = []
            if edge is not None:
                started = time.monotonic()
                try:
                    detections, reason = edge.detect(frame)
                except (RuntimeError, ValueError, cv2.error) as exc:
                    logger.error("Edge inference failed; frame discarded; retry in %.1fs: %s",
                                 inference_backoff, exc)
                    stop_event.wait(inference_backoff)
                    inference_backoff = min(inference_backoff * 2, 60)
                    continue
                inference_backoff = 1.0
                logger.info("Edge decision=%s inference=%.3fs detections=%d",
                            reason, time.monotonic() - started, len(detections))
                if stop_event.is_set():
                    break
                if not detections:
                    continue
            name = f"{unique_id}_{captured_at.strftime('%Y%m%dT%H%M%S_%f')}.jpg"
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality])
            if not ok:
                logger.error("JPEG encoding failed")
                continue
            meta = {
                "image_name": name,
                "captured_at": captured_at.isoformat(),
                "detections": detections,
                "source": "pi3_capture",
                "trigger": "motion",
                "motion_ratio": round(detector.last_ratio, 4),
                "hostname": hostname,
            }
            if edge is not None:
                meta["edge_policy"] = edge.metadata()
                meta["image_width"], meta["image_height"] = frame.shape[1], frame.shape[0]
            queue.enqueue(buf.tobytes(), meta)
            logger.info("Motion (%.1f%% of frame) -> captured %s event_to_queue=%.3fs queue_depth=%d",
                        detector.last_ratio * 100, name, time.monotonic() - now, len(queue.pending()))
    finally:
        if cap is not None:
            cap.release()
        queue.stop()


def main():
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(threadName)s: %(message)s",
    )
    stop_event = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop_event.set())
    run(Config(), stop_event)


if __name__ == "__main__":
    main()
