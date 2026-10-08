#!/usr/bin/env bash
# From your laptop: sync this repo to a DO GPU box and run setup_instance_do.sh.
#
# Usage:
#   SSH_KEY=./key WITH_EVAL=1 ./tools/bootstrap_do_gpu.sh 162.243.72.54
#
# Requires root SSH. Add your public key on the droplet first (see README / plan).

set -euo pipefail

IP="${1:?usage: $0 <droplet-ip>}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KEY="${SSH_KEY:-${HOME}/.ssh/id_ed25519}"
USER="${SSH_USER:-root}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/bonsai}"
WITH_EVAL="${WITH_EVAL:-1}"

if [ ! -f "$KEY" ]; then
  echo "FATAL: SSH private key not found at $KEY (set SSH_KEY=...)"
  exit 1
fi

SSH=(ssh -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new "${USER}@${IP}")
RSYNC=(rsync -az --delete-delay
  --exclude .git/
  --exclude .venv/
  --exclude venv/
  --exclude runs/
  --exclude __pycache__/
  --exclude .tox/
  --exclude key
  --exclude key.pub
  -e "ssh -i '${KEY}' -o BatchMode=yes -o StrictHostKeyChecking=accept-new")

echo "== checking SSH to ${USER}@${IP} =="
"${SSH[@]}" "whoami && nvidia-smi -L"

echo "== syncing repo to ${REMOTE_ROOT} =="
"${SSH[@]}" "mkdir -p ${REMOTE_ROOT}"
"${RSYNC[@]}" "${REPO_ROOT}/" "${USER}@${IP}:${REMOTE_ROOT}/"

if [ -f "${REPO_ROOT}/.env" ]; then
  echo "== copying .env (HF_TOKEN) =="
  scp -i "$KEY" -o BatchMode=yes "${REPO_ROOT}/.env" "${USER}@${IP}:${REMOTE_ROOT}/.env"
else
  echo "WARN: no local .env — create ${REMOTE_ROOT}/.env on the box with HF_TOKEN before Hub pulls"
fi

echo "== running setup_instance_do.sh (WITH_EVAL=${WITH_EVAL}) =="
"${SSH[@]}" "cd ${REMOTE_ROOT} && WITH_EVAL=${WITH_EVAL} bash tools/setup_instance_do.sh"

echo "== gpu provenance =="
"${SSH[@]}" "PYTHON=${REMOTE_ROOT%/bonsai}/venv/bin/python bash ${REMOTE_ROOT}/tools/gpu_env.sh"

echo "BOOTSTRAP_DONE ssh -i ${KEY} ${USER}@${IP}"
