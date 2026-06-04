#!/usr/bin/env bash
set -euo pipefail

APP_USER="${APP_USER:-israel}"
APP_DIR="${APP_DIR:-/home/${APP_USER}/dev/omoikane/virtual_assistant/backend}"
APP_ENV="${APP_ENV:-/home/${APP_USER}/dev/omoikane/virtual_assistant/backend/.env}"

SERVICE_NAME="virtual-assistant"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

if [ $EUID -ne 0 ]; then
  echo "Run as root: sudo ./install-service.sh"
  exit 1
fi

echo "Installing ${SERVICE_NAME} service..."
echo "  User:   ${APP_USER}"
echo "  Dir:    ${APP_DIR}"
echo "  Env:    ${APP_ENV}"

cat > "$SERVICE_FILE" <<SERVICEEOF
[Unit]
Description=Virtual Assistant Backend
After=network.target network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${APP_USER}
Group=${APP_USER}
WorkingDirectory=${APP_DIR}

EnvironmentFile=-${APP_ENV}

ExecStart=${APP_DIR}/.venv/bin/python ${APP_DIR}/main.py

Restart=always
RestartSec=10

StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SERVICEEOF

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"

echo "✓ Service installed and started."
echo "  Status: systemctl status ${SERVICE_NAME}"
echo "  Logs:   journalctl -fu ${SERVICE_NAME}"
