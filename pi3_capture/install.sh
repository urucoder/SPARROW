#!/usr/bin/env bash
# Installs SPARROW capture-only mode on Raspberry Pi OS (Pi 3 B+ or newer).
# Usage: sudo ./install.sh
set -euo pipefail

[[ $EUID -eq 0 ]] || { echo "Run as root: sudo $0" >&2; exit 1; }

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR=/opt/sparrow-capture
CONF_DIR=/etc/sparrow-capture
DATA_DIR=/var/lib/sparrow-capture
UUID_FILE=/etc/unique_id
SVC_USER=sparrow-capture

echo "==> Installing packages (OpenCV from Raspberry Pi OS, no pip/Docker needed)"
apt-get update
apt-get install -y python3 python3-opencv python3-numpy python3-requests v4l-utils uuid-runtime

echo "==> Creating service user and folders"
id -u "$SVC_USER" >/dev/null 2>&1 || useradd --system --no-create-home --shell /usr/sbin/nologin "$SVC_USER"
usermod -aG video "$SVC_USER"
install -d -m 755 "$APP_DIR"
install -d -m 750 -o root -g "$SVC_USER" "$CONF_DIR"
install -d -m 750 -o "$SVC_USER" -g "$SVC_USER" "$DATA_DIR" "$DATA_DIR/queue"
install -m 644 "$SRC_DIR/motion_capture.py" "$APP_DIR/motion_capture.py"

if [[ ! -s "$UUID_FILE" ]]; then
    uuidgen >"$UUID_FILE"
    chmod 644 "$UUID_FILE"
    echo "Generated device UUID in $UUID_FILE"
fi
echo "Device ID: $(tr -d '\r\n' <"$UUID_FILE" | sha256sum | cut -c1-12)"

if [[ ! -f "$CONF_DIR/sparrow-capture.env" ]]; then
    install -m 640 -o root -g "$SVC_USER" "$SRC_DIR/sparrow-capture.env" "$CONF_DIR/sparrow-capture.env"
fi

if [[ ! -s "$CONF_DIR/access_key.txt" ]]; then
    read -r -s -p "Enter the SPARROW server access key: " key; echo
    [[ -n "$key" ]] || { echo "Access key cannot be empty" >&2; exit 1; }
    printf '%s\n' "$key" >"$CONF_DIR/access_key.txt"
    chown root:"$SVC_USER" "$CONF_DIR/access_key.txt"
    chmod 640 "$CONF_DIR/access_key.txt"
fi

echo "==> Detected cameras"
v4l2-ctl --list-devices || echo "No V4L2 camera found - plug in the USB webcam and check CAMERA_DEVICE."

install -m 644 "$SRC_DIR/sparrow-capture.service" /etc/systemd/system/sparrow-capture.service
systemctl daemon-reload
systemctl enable --now sparrow-capture.service

echo "==> Done. Follow logs with: journalctl -u sparrow-capture -f"
