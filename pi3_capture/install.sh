#!/usr/bin/env bash
# Installs USB capture and optional CPU edge inference on Raspberry Pi OS.
set -euo pipefail

[[ $EUID -eq 0 ]] || { echo "Run as root: sudo $0" >&2; exit 1; }

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR=/opt/sparrow-capture
CONF_DIR=/etc/sparrow-capture
DATA_DIR=/var/lib/sparrow-capture
UUID_FILE=/etc/unique_id
SVC_USER=sparrow-capture
EDGE_MODEL=""
UUID_SOURCE=""
KEY_SOURCE=""
ISSUER_EMAIL=""
SERVER_URL=""
START=true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --edge-model|--uuid-file|--access-key-file|--issuer-email|--server-base-url)
            [[ $# -ge 2 && -n "$2" ]] || { echo "Missing value for $1" >&2; exit 1; }
            case "$1" in
                --edge-model) EDGE_MODEL="$2";;
                --uuid-file) UUID_SOURCE="$2";;
                --access-key-file) KEY_SOURCE="$2";;
                --issuer-email) ISSUER_EMAIL="$2";;
                --server-base-url) SERVER_URL="$2";;
            esac
            shift 2;;
        --no-start) START=false; shift;;
        *) echo "Unknown option: $1" >&2; exit 1;;
    esac
done
if [[ -n "$EDGE_MODEL" ]]; then EDGE_MODEL="$(realpath -- "$EDGE_MODEL")"; fi
if [[ -n "$UUID_SOURCE" ]]; then UUID_SOURCE="$(realpath -- "$UUID_SOURCE")"; fi
if [[ -n "$KEY_SOURCE" ]]; then KEY_SOURCE="$(realpath -- "$KEY_SOURCE")"; fi

echo "==> Installing packages (OpenCV and NumPy from apt; no Docker/PyTorch)"
apt-get update
apt-get install -y python3 python3-venv python3-opencv python3-numpy python3-requests v4l-utils uuid-runtime

echo "==> Creating service user and folders"
id -u "$SVC_USER" >/dev/null 2>&1 || useradd --system --user-group --no-create-home --shell /usr/sbin/nologin "$SVC_USER"
usermod -aG video "$SVC_USER"
install -d -m 755 "$APP_DIR"
install -d -m 750 -o root -g "$SVC_USER" "$CONF_DIR"
install -d -m 750 -o "$SVC_USER" -g "$SVC_USER" "$DATA_DIR" "$DATA_DIR/queue" "$DATA_DIR/edge-queue"
for file in motion_capture.py edge_inference.py onboard.py; do
    install -m 644 "$SRC_DIR/$file" "$APP_DIR/$file"
done
cd "$APP_DIR"
python3 -m venv --system-site-packages "$APP_DIR/venv"

if [[ -n "$UUID_SOURCE" ]]; then
    [[ -s "$UUID_SOURCE" ]] || { echo "UUID source missing/empty" >&2; exit 1; }
    if [[ -s "$UUID_FILE" ]] && ! cmp -s "$UUID_SOURCE" "$UUID_FILE"; then
        echo "Existing UUID differs; refusing to change this device identity" >&2
        exit 1
    fi
    [[ "$UUID_SOURCE" -ef "$UUID_FILE" ]] || install -m 644 "$UUID_SOURCE" "$UUID_FILE"
fi
if [[ ! -s "$UUID_FILE" ]]; then
    uuidgen >"$UUID_FILE"
    chmod 644 "$UUID_FILE"
    echo "Generated device UUID in $UUID_FILE"
fi
echo "Device ID: $(tr -d '\r\n' <"$UUID_FILE" | sha256sum | cut -c1-12)"

if [[ ! -f "$CONF_DIR/sparrow-capture.env" ]]; then
    install -m 640 -o root -g "$SVC_USER" "$SRC_DIR/sparrow-capture.env" "$CONF_DIR/sparrow-capture.env"
fi

if [[ -n "$KEY_SOURCE" ]]; then
    [[ -s "$KEY_SOURCE" ]] || { echo "Key source missing/empty" >&2; exit 1; }
    if [[ -s "$CONF_DIR/access_key.txt" ]] && ! cmp -s "$KEY_SOURCE" "$CONF_DIR/access_key.txt"; then
        echo "Existing key differs; refusing to replace it" >&2
        exit 1
    fi
    [[ "$KEY_SOURCE" -ef "$CONF_DIR/access_key.txt" ]] ||
        install -m 640 -o root -g "$SVC_USER" "$KEY_SOURCE" "$CONF_DIR/access_key.txt"
fi
if [[ ! -s "$CONF_DIR/access_key.txt" ]]; then
    read -r -s -p "Enter the SPARROW server access key: " key; echo
    [[ -n "$key" ]] || { echo "Access key cannot be empty" >&2; exit 1; }
    printf '%s\n' "$key" >"$CONF_DIR/access_key.txt"
    chown root:"$SVC_USER" "$CONF_DIR/access_key.txt"
    chmod 640 "$CONF_DIR/access_key.txt"
fi

if [[ -n "$EDGE_MODEL" ]]; then
    # No source builds, no PyTorch, and no replacement of apt's NumPy.
    apt-get install -y python3-flatbuffers python3-packaging python3-protobuf
    "$APP_DIR/venv/bin/pip" install --only-binary=:all: --no-deps -r "$SRC_DIR/requirements-edge.txt"
    "$APP_DIR/venv/bin/python" -c \
        'import sys; from edge_inference import verify_model; verify_model(sys.argv[1])' "$EDGE_MODEL"
    install -d -m 755 "$APP_DIR/models"
    if [[ ! "$EDGE_MODEL" -ef "$APP_DIR/models/mdv6-yolov10-c.onnx" ]]; then
        install -m 644 "$EDGE_MODEL" "$APP_DIR/models/mdv6-yolov10-c.onnx"
    fi
fi

# Update only explicitly selected settings; preserve all site-specific tuning.
"$APP_DIR/venv/bin/python" - "$CONF_DIR/sparrow-capture.env" "$SERVER_URL" "$EDGE_MODEL" <<'PY'
import json
import sys
from pathlib import Path

path, url, model = sys.argv[1:]
p = Path(path)
lines = p.read_text().splitlines()
changes = {}
if url:
    changes["SERVER_BASE_URL"] = json.dumps(url.rstrip("/"))
if model:
    changes.update(EDGE_ENABLED="true",
                   EDGE_MODEL_PATH="/opt/sparrow-capture/models/mdv6-yolov10-c.onnx")
for key, value in changes.items():
    lines = [line for line in lines if not line.startswith(key + "=")]
    lines.append(f"{key}={value}")
p.write_text("\n".join(lines) + "\n")
PY

# Read the systemd-compatible template without executing shell content.
mapfile -t SETTINGS < <(runuser -u "$SVC_USER" -- "$APP_DIR/venv/bin/python" - "$CONF_DIR/sparrow-capture.env" <<'PY'
import shlex
import sys
from pathlib import Path
from motion_capture import Config, get_hardware_id
import os

for line in Path(sys.argv[1]).read_text().splitlines():
    if line.strip() and not line.lstrip().startswith("#"):
        key, value = line.split("=", 1)
        parts = shlex.split(value, comments=True)
        os.environ[key.strip()] = parts[0] if parts else ""
c = Config()
c.validate()
get_hardware_id(c.unique_id_path)
if not Path(c.access_key_file).read_text().strip():
    raise ValueError("Access key is empty")
if c.edge_enabled:
    from edge_inference import EdgeDetector
    EdgeDetector(c)
print(c.server_base_url)
print(c.access_key_file)
print(c.unique_id_path)
PY
)
[[ ${#SETTINGS[@]} -eq 3 ]] || { echo "Configuration/model/permissions validation failed" >&2; exit 1; }
if [[ -n "$ISSUER_EMAIL" ]]; then
    runuser -u "$SVC_USER" -- "$APP_DIR/venv/bin/python" "$APP_DIR/onboard.py" \
        --server-base-url "${SETTINGS[0]}" --issuer-email "$ISSUER_EMAIL" \
        --access-key-file "${SETTINGS[1]}" --unique-id-path "${SETTINGS[2]}"
fi

echo "==> Detected cameras"
v4l2-ctl --list-devices || echo "No V4L2 camera found - plug in the USB webcam and check CAMERA_DEVICE."

install -m 644 "$SRC_DIR/sparrow-capture.service" /etc/systemd/system/sparrow-capture.service
systemctl daemon-reload
if $START; then
    systemctl enable sparrow-capture.service
    systemctl restart sparrow-capture.service
    systemctl is-active --quiet sparrow-capture.service
else
    echo "Service installed but not started (--no-start)."
fi

echo "==> Done. Follow logs with: journalctl -u sparrow-capture -f"
