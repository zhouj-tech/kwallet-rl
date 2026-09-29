#!/bin/bash

REMOTE="root@connect.westc.seetacloud.com"
PORT="37431"
REMOTE_DIR="/root/autodl-tmp/kwallet-rl"

cd /Users/qiubi/kwallet-rl || exit 1

rsync -av \
  --exclude ".git/" \
  --exclude ".venv/" \
  --exclude "__pycache__/" \
  --exclude ".DS_Store" \
  --exclude "logs/" \
  --exclude "results/" \
  --exclude "reports/" \
  --exclude "archive/" \
  -e "ssh -p ${PORT}" \
  ./ \
  ${REMOTE}:${REMOTE_DIR}/

echo "Code synced to remote."
