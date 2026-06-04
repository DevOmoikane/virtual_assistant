#!/usr/bin/env bash
set -euo pipefail

REPO="DevOmoikane/virtual_assistant"
RUNNER_VERSION="${RUNNER_VERSION:-2.322.0}"
RUNNER_DIR="/home/israel/actions-runner"

if [ $EUID -eq 0 ]; then
  echo "Do not run as root. Run as the user that will own the runner."
  exit 1
fi

if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "Usage: GITHUB_TOKEN=ghp_xxx ./install-runner.sh"
  echo "Get a PAT from https://github.com/settings/tokens (repo scope)"
  exit 1
fi

sudo apt-get update
sudo apt-get install -y curl jq

mkdir -p "$RUNNER_DIR"
cd "$RUNNER_DIR"

ARCH=$(uname -m)
case "$ARCH" in
  aarch64) PACKAGE="actions-runner-linux-arm64-${RUNNER_VERSION}.tar.gz" ;;
  armv7l)  PACKAGE="actions-runner-linux-arm-${RUNNER_VERSION}.tar.gz" ;;
  x86_64)  PACKAGE="actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" ;;
  *)       echo "Unsupported arch: $ARCH"; exit 1 ;;
esac

if [ ! -f ".runner" ]; then
  curl -sSLO "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${PACKAGE}"
  echo "Extracting ${PACKAGE}..."
  tar xzf "$PACKAGE"
  rm "$PACKAGE"

  echo "Configuring runner for ${REPO}..."
  ./config.sh --url "https://github.com/${REPO}" --token "$GITHUB_TOKEN" \
              --name "rpi5" --labels "self-hosted,rpi5,linux,arm64" \
              --unattended --replace
fi

echo "Installing runner as a systemd service..."
sudo ./svc.sh install
sudo ./svc.sh start

echo "✓ Runner installed and started. Check status with: sudo ./svc.sh status"
