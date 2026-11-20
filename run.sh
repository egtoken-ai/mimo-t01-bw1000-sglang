#!/usr/bin/env bash
# Install declared extras, replay patches, start SGLang and wait for readiness.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
source ./env.sh

PID_FILE="$ROOT/server.pid"
LOG_FILE="$ROOT/server.log"

if [[ -f requirements-extra.txt ]]; then
  echo "[run] installing declared extras"
  pip install -q --no-deps -r requirements-extra.txt
fi

shopt -s nullglob
for patch_file in patches/*.patch; do
  if patch -p1 -R -f -s --dry-run -d "$PATCH_ROOT" < "$patch_file" >/dev/null 2>&1; then
    echo "[run] skip $patch_file (already applied)"
  else
    echo "[run] applying $patch_file"
    patch -p1 -f -s -d "$PATCH_ROOT" < "$patch_file"
  fi
done

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "[run] existing server PID $(cat "$PID_FILE")"
  exit 0
fi

if curl -sf --noproxy '*' "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "[run] port $PORT already has a healthy server; stop it before a fresh launch" >&2
  exit 1
fi

CMD=(sglang serve
  --model-path "$MODEL_PATH"
  --served-model-name "$SERVED_NAME"
  --host 0.0.0.0
  --port "$PORT"
  --tp-size "$TP"
  --pp-size "$PP"
  --context-length "$MAX_LEN"
  --mem-fraction-static "$MEM_FRAC")

if [[ "$TP" -gt 1 && -n "${NCCL_PORT:-}" ]]; then
  CMD+=(--nccl-port "$NCCL_PORT")
fi
if [[ "$NNODES" -gt 1 ]]; then
  CMD+=(--nnodes "$NNODES" --node-rank "$NODE_RANK" --dist-init-addr "$DIST_ADDR")
fi
read -r -a EXTRA <<< "$EXTRA_ARGS"
CMD+=("${EXTRA[@]}")

printf '[run] %q ' "${CMD[@]}" | tee server.log
printf '\n' | tee -a server.log
"${CMD[@]}" >> server.log 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

if [[ "$NODE_RANK" -ne 0 ]]; then
  wait "$SERVER_PID"
  exit $?
fi

for ((elapsed = 0; elapsed < LOAD_TIMEOUT; elapsed += 5)); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[run] server exited; see $LOG_FILE" >&2
    tail -n 120 "$LOG_FILE" >&2 || true
    exit 1
  fi
  if curl -sf --noproxy '*' "http://127.0.0.1:${PORT}/health" >/dev/null; then
    echo "[run] server up after ${elapsed}s (PID $SERVER_PID)"
    exit 0
  fi
  sleep 5
done

echo "[run] server not ready after ${LOAD_TIMEOUT}s; see $LOG_FILE" >&2
kill "$SERVER_PID" 2>/dev/null || true
exit 1
