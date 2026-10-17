# Raspberry Pi USB camera: capture-only or local animal detection

This standalone service runs on a Raspberry Pi 3/3 B+ without sensors,
Docker, PyTorch, audio or the other SPARROW hardware services.

- **Capture-only (default):** motion triggers a JPEG upload with empty
  detections. Sparrow Studio stores it; **empty detections do not trigger
  server-side inference**.
- **Edge mode (opt-in):** motion triggers local `MDV6-yolov10-c` CPU inference.
  A qualifying person detection rejects the **entire image**, including mixed
  animal/person scenes. Animal-free and vehicle-only images are discarded.
  Only accepted animal images are encoded, queued and uploaded, with generic
  `animal` labels, confidences and original-image pixel bounding boxes.
  Species classification and server-side inference are not included.

Person suppression is **best-effort, not a privacy guarantee**: the model can
miss people or falsely reject animal scenes. The initial person threshold (0.2) and animal threshold (0.5)
are starting values, not calibrated accuracy claims. Use non-private scenes.
Capture-only mode performs no person suppression.

## Requirements

- Pi 3 or newer, UVC USB webcam, stable power (Pi 3: 5 V / 2.5 A micro-USB).
  Use a powered hub if the webcam causes brownouts.
- Raspberry Pi OS Lite or Debian with apt OpenCV. Edge mode was tested on
  **Pi 3 Model B Rev 1.2, 64-bit Debian 13, Python 3.13.5**.
  Capture-only can use 32-bit OS; the pinned edge runtime needs a compatible
  binary wheel (use 64-bit on this Pi).
- Network access to Sparrow Studio, an existing project-issued onboarding key
  and the email of its issuer. No dashboard accounts are created by this tool.

## Install

Check out this fork branch, not upstream `main`:

```bash
git clone --branch pi3-usb-camera-motion-capture https://github.com/urucoder/SPARROW.git
cd SPARROW/pi3_capture
```

Capture-only:

```bash
sudo bash install.sh --server-base-url http://YOUR-SERVER:8080/v1 \
  --issuer-email ISSUER_EMAIL
```

The installer prompts for the key without echoing it. For HTTPS deployments,
use the server's HTTPS base URL. Plain HTTP should be confined to a trusted
local test network; it carries both images and credentials unencrypted.

Edge mode: obtain the exact `MDV6-yolov10-c` ONNX artifact from your installed
Sparrow Studio model catalog (`MDV6-yolov10-c/1/model.onnx`), copy it to the Pi,
then run:

```bash
sudo bash install.sh --edge-model /home/YOUR_USER/mdv6-yolov10-c.onnx \
  --server-base-url http://YOUR-SERVER:8080/v1 --issuer-email ISSUER_EMAIL
```

The supported artifact SHA-256 is:

```text
d38a71b93dcf34147261c6af7665cf3ad2495c924d15503fd88304950cf8eb6a
```

No weights are downloaded or committed. A different checksum, input/output
shape or label mapping is rejected, not automatically guessed. The catalog
specifies letterbox/unit normalization and end-to-end six-column output;
the ONNX metadata calls class 2 `vehicle` (some catalog labels use `car`).
Neither is retained by the animal-only policy.

**License:** ONNX Runtime is MIT ([upstream license](https://github.com/microsoft/onnxruntime/blob/main/LICENSE)).
The verified model's embedded metadata reports **AGPL-3.0** and links to the
[Ultralytics license](https://ultralytics.com/license). Review those model
terms for your use/distribution; optional installation does not change them.

### Preserve an already-onboarded device

The unit ID is the first 12 hex characters of SHA-256 of the stripped UUID
file, matching full SPARROW. **Keep the original UUID and key** when migrating:

```bash
sudo bash install.sh \
  --uuid-file /home/YOUR_USER/sparrow-pi-test/device_uuid \
  --access-key-file /home/YOUR_USER/sparrow-pi-test/access_key.txt \
  --edge-model /home/YOUR_USER/sparrow-pi-test/mdv6-yolov10-c.onnx \
  --server-base-url http://YOUR-SERVER:8080/v1 --issuer-email ISSUER_EMAIL \
  --no-start
```

Source paths are not secret values: never put the key itself on a command
line or in git. The installer refuses conflicting existing UUIDs/keys.
It preserves existing environment settings except explicitly selected
server/model settings. Rerunning onboarding with the same unit/key is
idempotent; a key bound to another unit fails visibly.

The installer uses apt NumPy/OpenCV and a `--system-site-packages` venv in
`/opt/sparrow-capture/venv`. Edge mode installs the pinned ONNX Runtime 1.31.0
wheel with `--only-binary=:all: --no-deps` and apt runtime dependencies.
It never attempts a source build or installs PyTorch. Missing wheel support
or an unreadable key/model/UUID fails installation rather than disabling edge
filtering. It validates imports/model access as the service user.

`--no-start` stages the unit without starting/restarting it. Otherwise the
installer enables and restarts it. A real service install requires local sudo;
do not paste sudo passwords into chat. Coordinate any reboot separately.

Onboarding can also be run explicitly:

```bash
sudo -u sparrow-capture /opt/sparrow-capture/venv/bin/python \
  /opt/sparrow-capture/onboard.py --server-base-url http://YOUR-SERVER:8080/v1 \
  --issuer-email ISSUER_EMAIL
```

It sends the existing key via `X-API-Key`, issuer email via `X-Email` and
`{"unit_id": "..."}` to `/onboarding`. It never provisions user accounts.

## Operation and configuration

Edit `/etc/sparrow-capture/sparrow-capture.env`, then restart:

```bash
sudo systemctl restart sparrow-capture
journalctl -u sparrow-capture -f
v4l2-ctl --list-devices
```

The environment file uses simple `KEY=value` assignments (optionally quoted,
no shell expansions). Preserve restrictive permissions on keys and queues.

| Setting | Default | Meaning |
|---|---|---|
| `SERVER_BASE_URL` | `https://server.sparrowstudio.azure.com/v1` | Set explicitly for local testing; never send fixtures to production |
| `CAMERA_DEVICE` | `/dev/video0` | UVC/MJPEG camera |
| `CAPTURE_WIDTH`, `CAPTURE_HEIGHT` | `1280`, `720` | Requested original JPEG resolution |
| `MOTION_MIN_RATIO` | `0.01` | Fraction changed to trigger motion |
| `MOTION_PIXEL_THRESHOLD` | `25` | Brightness delta per pixel |
| `MOTION_MAX_RATIO` | `0.6` | Larger changes treated as lighting changes |
| `MOTION_FRAMES` | `2` | Consecutive motion frames |
| `COOLDOWN_SECONDS` | `5` | Minimum interval between eligible events |
| `EDGE_ENABLED` | `false` | Explicit opt-in; no ONNX Runtime import when false |
| `EDGE_MODEL_PATH` | `/opt/sparrow-capture/models/mdv6-yolov10-c.onnx` | Checksum-validated artifact |
| `EDGE_ANIMAL_THRESHOLD` | `0.5` | Inclusive animal confidence threshold |
| `EDGE_PERSON_THRESHOLD` | `0.2` | Inclusive whole-frame rejection threshold |
| `EDGE_INTRA_THREADS`, `EDGE_INTER_THREADS` | `2`, `1` | CPU runtime threads, each bounded to 1–4 |
| `QUEUE_DIR` | `/var/lib/sparrow-capture/queue` | Unchecked capture-only items |
| `EDGE_QUEUE_DIR` | `/var/lib/sparrow-capture/edge-queue` | Separate, inference-approved items |
| `MAX_QUEUE_MB` | `1024` | JPEG queue cap, oldest dropped first with a warning |

To return to capture-only, explicitly set `EDGE_ENABLED=false`. Switching
modes does not migrate/delete the other queue. Returning to capture-only
will resume that unchecked queue; review its contents before doing so.
Never point both queue settings to the same directory.

## Processing and durability

Motion uses small grayscale frames. Edge inference runs synchronously, one
loaded CPU session, fixed float32 `[1,3,640,640]` RGB/NCHW input and
`[1,300,6]` end-to-end output. There is no second NMS or inference backlog.
Actual rounded resize dimensions and integer padding are inverted to produce
clipped, nondegenerate integer pixel boxes at the original image resolution.

Any qualifying person rejects the whole frame before JPEG encoding or disk
writes. Invalid inference output discards that frame with a logged error and
bounded retry delay; invalid startup configuration/model fails before the
uploader starts. Only generic animals are retained; species/classifier
fields are deliberately absent.

Accepted images and JSON metadata are flushed to disk and atomically
published to the queue. Multipart upload to `/uploads/image` uses the existing
`file`, `auth_key`, `unique_id`, `metadata` contract. Successful uploads remove
the JPEG/sidecar; failed uploads remain for retry, including after restart.
Missing/corrupt/unapproved edge metadata is retained with an error, never
uploaded using an empty-detections fallback.

Queue pruning and upload selection are synchronized; new captures do not
cancel network retry backoff. A single oversized JPEG is dropped with a
warning. An in-flight upload can temporarily exceed the JPEG cap; sidecar
and temporary-file overhead is not included in `MAX_QUEUE_MB`.
Use filesystem monitoring for disk-full conditions. Local queue failures are
visible and cause the service to restart, not claim a successful capture.

There is no exactly-once guarantee: a lost success response can cause a
duplicate on retry. The local backend tested here also returns HTTP 500 for
identical image bytes already stored under the same project (a database
SHA-256 uniqueness constraint). Such an item stays queued and blocks later
items until an operator verifies it was stored and moves it out of the queue;
the client does not discard a failed upload or alter the backend to mask this.
Shutdown waits for the in-flight upload (request timeout applies) and
discards an inference result if shutdown is observed when inference returns.

Synchronous inference pauses camera consumption and may miss short events or
briefly read a buffered frame afterward. `CAP_PROP_BUFFERSIZE=1` is requested,
not a guarantee from every camera driver. Vegetation and shadows still
trigger motion. Watch inference/upload latency, camera read errors, thermal
throttling, RSS and queue depth; no privacy/accuracy claims follow from a
short benchmark.

## Validation

```bash
python3 -m unittest discover -s pi3_capture -p 'test_*.py'
# From inside pi3_capture:
python3 -m unittest -v test_motion_capture test_edge_inference
bash -n install.sh
```

Tests use deterministic frames and fake ONNX outputs for preprocessing,
coordinate reversal, confidence boundaries, person-first filtering,
malformed tensors, optional runtime startup, accepted-image upload, outage
retry/restart and corrupt-edge-queue rejection. NumPy, OpenCV and requests
are needed; the unit tests do not need ONNX Runtime or a camera.

Deployment acceptance also needs a real nonempty detection upload, persisted
database boxes and an authenticated gallery overlay check. HTTP success alone
does not establish that. Reboot/outage service testing and live-person scenes
require explicit coordination; keep test traffic on a local backend.

### Integration evidence and remaining deployment checks

On the Pi 3 Model B / Debian 13 setup, all 24 tests passed locally and on
the Pi. A controlled animal fixture passed the real motion/camera-loop,
ONNX detector, queue and local Sparrow Studio API; the stored original
decoded at 1280x720, with a nonempty `animal` box/confidence verified in
`images`, `detections` and `detection_boxes`. Existing-device onboarding
returned idempotent success twice. A controlled unreachable endpoint retained
an accepted JPEG/sidecar, which a restarted uploader drained after recovery.
No private/live-person images were saved or uploaded for these checks.

A separate 900-second camera/runtime soak sampled 180 live frames without
saving/uploading them: median inference 1.53 s, maximum 3.13 s, peak process
RSS 263.8 MiB, peak temperature 65.0 C and zero failed camera reads. RSS
stabilized around 264 MiB. These measurements meet the provisional median
3 s / RSS 350 MiB budgets, **not** a full installed-service reliability SLA.
Other short validation processes ran concurrently for part of this soak.
USB JPEG decoder warnings recurred, and `vcgencmd get_throttled` intermittently
reported active undervoltage/throttling (`0x50005`, later historical flags
`0x50000`). Address power/cabling and investigate camera integrity before
unattended deployment; successful decodes do not establish frame freshness.

One small, composited animal scene was falsely rejected as `person` at the
starting threshold; the full animal scene was accepted. No thresholds were
weakened to force acceptance. This illustrates why both privacy misses and
false rejections need scene-specific evaluation.

The authenticated gallery overlay remains **unverified** because the browser
was at the login page. The root installer/service-user permissions, boot
startup and reboot recovery also remain **unverified** pending user-assisted
sudo and coordinated service installation. The test service was not installed
or enabled, and the existing dashboard/API/database were not recreated.
