#!/usr/bin/env bash
# Record the effective runtime after run.sh has installed extras and replayed patches.
set -euo pipefail
source ./env.sh

{
  echo "timestamp: $(date -Is)"
  echo "hostname: $(hostname)"
  echo "image: $IMAGE"
  echo "model_id: $MODEL_ID"
  echo "model_path: $MODEL_PATH"
  echo "framework: $FRAMEWORK"
  echo "precision: $PRECISION"
  echo "card: $CARD"
  echo "tp: $TP ep: $EP pp: $PP"
  echo "dtk_env: ${DTK_VERSION:-unknown}"
  hy-smi
  python3 - <<'PY'
import sys
import torch
import transformers
import sglang

print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("torch.hip:", torch.version.hip)
print("transformers:", transformers.__version__)
print("sglang:", sglang.__version__)
print("device_count:", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    prop = torch.cuda.get_device_properties(index)
    print(f"device[{index}]: {torch.cuda.get_device_name(index)} {prop.total_memory / 2**30:.3f} GiB")
PY
  pip list 2>/dev/null | grep -iE '^(torch|triton|transformers|tokenizers|sglang|vllm|flash|fla|lmslim|ray)|das' || true
  echo "--- model files ---"
  du -sh "$MODEL_PATH"
  sha256sum "$MODEL_PATH/config.json" \
    "$MODEL_PATH/generation_config.json" \
    "$MODEL_PATH/tokenizer_config.json" \
    "$MODEL_PATH/chat_template.jinja"
  echo "--- server info ---"
  curl -sS --noproxy '*' "http://127.0.0.1:${PORT}/get_server_info"
  echo
} 2>&1 | tee env.log
