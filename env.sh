#!/usr/bin/env bash
# Single source of truth for the MiMo T01 submission.
export MODEL_ID="XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B"
export MODEL_PATH="/public/home/maxkonger/SothisAI/model/Aihub/MiMo-V2.6-Distill-Qwen-9B/main/MiMo-V2.6-Distill-Qwen-9B"
export SERVED_NAME="MiMo-V2.6-Distill-Qwen-9B"
# Set IMAGE from the SCNet image page before submission; the container does not expose its registry tag.
export IMAGE="${IMAGE:-UNVERIFIED}"
export FRAMEWORK="sglang"
export CARD="BW1000"
export PRECISION="bf16"
export PORT="${PORT:-30000}"
export TP="${TP:-1}"
export EP=1
export PP=1
export MAX_LEN=131072
export MEM_FRAC=0.85
export NNODES=1
export NODE_RANK=0
export DIST_ADDR="${DIST_ADDR:-10.0.0.1:50000}"
export LOAD_TIMEOUT=1800
export PATCH_ROOT="${PATCH_ROOT:-$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')}"
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-fa3}"
export EXTRA_ARGS="--reasoning-parser mimo --dtype bfloat16 --kv-cache-dtype bfloat16 --max-running-requests 128 --chunked-prefill-size 8192 --max-prefill-tokens 16384 --disable-radix-cache --attention-backend ${ATTENTION_BACKEND} --page-size 64 --mamba-scheduler-strategy no_buffer --cuda-graph-max-bs 128 --disable-custom-all-reduce --enable-metrics"
