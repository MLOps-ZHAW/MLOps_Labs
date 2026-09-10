#!/bin/bash
# Compute Engine startup script for the MLOps tutorial VM.
# Runs automatically as root on first boot. Installs everything Parts 1-6
# of README_VertexAI.md need, so students never install anything themselves.
set -euo pipefail

apt-get update
apt-get install -y docker.io git jq google-cloud-cli

systemctl enable --now docker

# Debian's own repos only ship docker.io (the engine) - the "docker compose"
# v2 CLI plugin has to be fetched separately.
mkdir -p /usr/local/lib/docker/cli-plugins
curl -fSL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64" \
    -o /usr/local/lib/docker/cli-plugins/docker-compose
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
