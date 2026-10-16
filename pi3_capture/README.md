# SPARROW capture-only mode (Raspberry Pi 3 B+ + USB webcam)

A lightweight alternative to the full SPARROW stack for boards that are too
small to run the AI models locally. The Pi only:

1. watches a USB webcam,
2. takes a photo whenever motion is detected, and
3. uploads it to the SPARROW server, where all AI processing happens.

No sensors, Docker, PyTorch or local models are needed. It runs as a single
Python `systemd` service using the Raspberry Pi OS OpenCV package.

## What you need

- Raspberry Pi 3 B+ with **Raspberry Pi OS Lite** (64-bit recommended; 32-bit also works)
- A UVC (Linux plug-and-play) USB webcam; MJPEG support is recommended
- A 5 V / 2.5 A micro-USB power supply (use a powered USB hub if the webcam browns out the Pi)
- Network access (Wi-Fi, Ethernet or Starlink) to the SPARROW server
- A SPARROW server **access key**

## Install

```bash
git clone https://github.com/microsoft/SPARROW.git
cd SPARROW/pi3_capture
sudo ./install.sh
```

The installer:

- installs `python3-opencv`, `python3-numpy`, `python3-requests` and `v4l-utils`,
- creates `/etc/unique_id` if needed (same device-ID scheme as full SPARROW),
- asks for the access key and stores it in `/etc/sparrow-capture/access_key.txt`,
- installs and starts the `sparrow-capture` systemd service.

Check that it works:

```bash
journalctl -u sparrow-capture -f
# ... Motion (4.2% of frame) -> captured 1a2b3c4d5e6f_20260101T120000_000000.jpg
# ... Uploaded 1a2b3c4d5e6f_20260101T120000_000000.jpg
```

## Configuration

Edit `/etc/sparrow-capture/sparrow-capture.env`, then run
`sudo systemctl restart sparrow-capture`.

| Variable | Default | Meaning |
|---|---|---|
| `SERVER_BASE_URL` | `https://server.sparrowstudio.azure.com/v1` | SPARROW server |
| `CAMERA_DEVICE` | `/dev/video0` | Webcam (`v4l2-ctl --list-devices`) |
| `CAPTURE_WIDTH` / `CAPTURE_HEIGHT` | `1280` / `720` | Photo resolution |
| `MOTION_MIN_RATIO` | `0.01` | Fraction of the frame that must change. Raise it to reduce false triggers |
| `MOTION_PIXEL_THRESHOLD` | `25` | Per-pixel brightness change that counts as changed |
| `MOTION_MAX_RATIO` | `0.6` | Bigger changes are treated as lighting changes and ignored |
| `MOTION_FRAMES` | `2` | Consecutive motion frames before a photo is taken |
| `COOLDOWN_SECONDS` | `5` | Minimum time between photos |
| `MAX_QUEUE_MB` | `1024` | Offline storage cap; oldest photos are dropped first |

## How it works

- Frames are read in MJPEG, downscaled to 160 px grayscale and compared with a
  slowly updated background. This uses little CPU on a Pi 3.
- On motion, the full-resolution frame is saved as JPEG plus a JSON sidecar
  in `/var/lib/sparrow-capture/queue`.
- A background thread uploads each photo with the same request the full
  SPARROW stack uses (`POST {SERVER_BASE_URL}/uploads/image`, multipart
  `file`, `auth_key`, `unique_id`, `metadata`). Photos are deleted only after
  the server responds with success. While offline they stay queued and
  are retried with backoff.
- `metadata.detections` is empty, so the server runs all detection and
  classification itself.

Motion detection reacts to any visual change, such as wind-blown vegetation
or shadows, not just animals. Tune `MOTION_MIN_RATIO` and `COOLDOWN_SECONDS`
for your site.

## Tests

```bash
python3 -m unittest -v test_motion_capture   # needs numpy, requests, opencv
```
