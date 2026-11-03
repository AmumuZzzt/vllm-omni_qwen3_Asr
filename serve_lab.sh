#!/usr/bin/env bash
# Launch the lab Qwen3-ASR omni server (patched v0.32.0 checkout).
# Usage: bash /root/vllm-work/lab2/serve_lab.sh [port]
set -euo pipefail

PORT="${1:-8000}"
MODEL=/opt/qwen3-asr/models/Qwen3-ASR-0.6B
SRC=/root/vllm-work/src/vllm-omni
LOG=/root/vllm-work/lab2/logs/server_${PORT}.log

source /root/vllm-work/env.sh
export PYTHONPATH="$SRC"
mkdir -p "$(dirname "$LOG")"

cd "$SRC"
nohup python -m vllm_omni.entrypoints.cli.main serve "$MODEL" \
  --omni --host 127.0.0.1 --port "$PORT" --trust-remote-code \
  --gpu-memory-utilization 0.6 --max-model-len 8192 \
  > "$LOG" 2>&1 &

echo "launched pid=$! port=$PORT log=$LOG"
echo "wait for: $(grep -c 'Application startup complete' "$LOG" 2>/dev/null || echo 0) (poll the log)"
