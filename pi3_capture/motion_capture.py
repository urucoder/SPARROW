#!/usr/bin/env python3
"""
SPARROW capture-only mode for low-power boards (e.g. Raspberry Pi 3 B+).

Watches a USB webcam, takes a photo whenever motion is detected and uploads it
to the SPARROW server (POST {SERVER_BASE_URL}/uploads/image), where all AI
processing happens. No sensors, local models or Docker are required.

Photos are first written to an on-disk queue so captures are never lost while
the network is down; a background thread uploads and deletes them.
"""

import hashlib
import json
import logging
import os
import signal
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger("sparrow_capture")


def _env(name, default, cast=str):
    def factory():
        value = os.getenv(name)
        return default if value in (None, "") else cast(value)
    return field(default_factory=factory)


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
        import requests

        self.config = config
        self.auth_key = auth_key
        self.unique_id = unique_id
        self.session = session or requests.Session()
        self.dir = Path(config.queue_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None

    def enqueue(self, jpeg_bytes, metadata):
        stem = self.dir / Path(metadata["image_name"]).stem
        side_tmp = self.dir / f"{stem.name}.json.tmp"
        img_tmp = self.dir / f"{stem.name}.jpg.tmp"
        side_tmp.write_text(json.dumps(metadata))
        img_tmp.write_bytes(jpeg_bytes)
        # Sidecar first: the uploader only picks up *.jpg, whose metadata must already exist.
        os.replace(side_tmp, stem.with_suffix(".json"))
        os.replace(img_tmp, stem.with_suffix(".jpg"))
        self.enforce_size_limit()
        self._wake.set()

    def pending(self):
        return sorted(self.dir.glob("*.jpg"))

    def enforce_size_limit(self):
        limit = self.config.max_queue_mb * 1024 * 1024
        files = self.pending()
        total = sum(p.stat().st_size for p in files)
        while len(files) > 1 and total > limit:
            oldest = files.pop(0)
            total -= oldest.stat().st_size
            logger.warning("Queue over %.0f MB, dropping oldest capture %s", self.config.max_queue_mb, oldest.name)
            self._remove(oldest)

    @staticmethod
    def _remove(jpg):
        jpg.unlink(missing_ok=True)
        jpg.with_suffix(".json").unlink(missing_ok=True)

    def upload_one(self, jpg):
        try:
            meta = json.loads(jpg.with_suffix(".json").read_text())
        except (OSError, ValueError):
            meta = {"image_name": jpg.name, "captured_at": "", "detections": []}
        with open(jpg, "rb") as f:
            response = self.session.post(
                self.config.upload_url,
                files={"file": (jpg.name, f, "image/jpeg")},
                data={
                    "auth_key": self.auth_key,
                    "unique_id": self.unique_id,
                    "metadata": json.dumps(meta),
                },
                timeout=self.config.upload_timeout,
            )
        response.raise_for_status()
        self._remove(jpg)
        logger.info("Uploaded %s", jpg.name)

    def drain(self):
        """Upload everything currently queued, oldest first. Returns True if nothing failed."""
        for jpg in self.pending():
            if self._stop.is_set():
                return False
            try:
                self.upload_one(jpg)
            except FileNotFoundError:
                continue  # pruned by enforce_size_limit meanwhile
            except Exception as e:
                logger.warning("Upload of %s failed: %s", jpg.name, e)
                return False
        return True

    def _run(self):
        backoff = 5.0
        while not self._stop.is_set():
            if self.drain():
                backoff = 5.0
                self._wake.wait(timeout=60)
            else:
                self._wake.wait(timeout=backoff)
                backoff = min(backoff * 2, self.config.retry_max_seconds)
            self._wake.clear()

    def start(self):
        self._thread = threading.Thread(target=self._run, name="uploader", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=10)


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


def run(config, stop_event, camera_factory=open_camera):
    import cv2

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
    cap = None
    try:
        while not stop_event.is_set():
            if cap is None:
                try:
                    cap = camera_factory(config)
                    detector.reset()
                    logger.info("Camera %s opened", config.camera_device)
                except Exception as e:
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
            name = f"{unique_id}_{captured_at.strftime('%Y%m%dT%H%M%S_%f')}.jpg"
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality])
            if not ok:
                logger.error("JPEG encoding failed")
                continue
            queue.enqueue(buf.tobytes(), {
                "image_name": name,
                "captured_at": captured_at.isoformat(),
                "detections": [],
                "source": "pi3_capture",
                "trigger": "motion",
                "motion_ratio": round(detector.last_ratio, 4),
                "hostname": hostname,
            })
            logger.info("Motion (%.1f%% of frame) -> captured %s", detector.last_ratio * 100, name)
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
