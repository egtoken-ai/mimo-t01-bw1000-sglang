#!/usr/bin/env bash
# Required T01 performance matrix: 2 input lengths x 3 concurrencies x 2 rounds.
set -uo pipefail
source ./env.sh
mkdir -p bench
rm -f bench/round1.jsonl bench/round2.jsonl bench/failures.txt bench/status.tsv

MODULE=sglang.bench_serving
if ! python3 -c "import ${MODULE}" >/dev/null 2>&1; then
  echo "benchmark module ${MODULE} is unavailable" | tee bench/failures.txt
  exit 1
fi

for INPUT_LEN in 4096 16384; do
  for CONCURRENCY in 1 8 64; do
    PROMPTS=$(( CONCURRENCY * 2 > 8 ? CONCURRENCY * 2 : 8 ))
    for ROUND in 1 2; do
      TAG="round${ROUND}_${INPUT_LEN}_1024_c${CONCURRENCY}"
      echo "[bench] $TAG"
      python3 -m "$MODULE" \
        --backend sglang \
        --host 127.0.0.1 \
        --port "$PORT" \
        --model "$MODEL_PATH" \
        --served-model-name "$SERVED_NAME" \
        --tokenizer "$MODEL_PATH" \
        --dataset-name random-ids \
        --random-input-len "$INPUT_LEN" \
        --random-output-len 1024 \
        --random-range-ratio 1 \
        --max-concurrency "$CONCURRENCY" \
        --num-prompts "$PROMPTS" \
        --seed 1 \
        --output-file "bench/round${ROUND}.jsonl" \
        > "bench/${TAG}.log" 2>&1
      CODE=$?
      printf '%s\t%s\n' "$TAG" "$CODE" >> bench/status.tsv
      if [[ "$CODE" -ne 0 ]]; then
        echo "[bench] FAIL $TAG code=$CODE" | tee -a bench/failures.txt
      fi
    done
  done
done

python3 - <<'PY' | tee bench/summary.md
import json
from pathlib import Path

columns = [
    "random_input_len", "random_output_len", "max_concurrency", "completed",
    "duration", "request_throughput", "output_throughput", "total_throughput",
    "mean_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p99_tpot_ms",
    "mean_e2e_latency_ms",
]
print("| " + " | ".join(columns) + " |")
print("|" + " --- |" * len(columns))
path = Path("bench/round2.jsonl")
if not path.exists():
    print("| no round2 results |" + " - |" * (len(columns) - 1))
else:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cells = [
            f"{row[key]:.2f}" if isinstance(row.get(key), float) else str(row.get(key, "-"))
            for key in columns
        ]
        print("| " + " | ".join(cells) + " |")
PY

echo "[bench] status:"
cat bench/status.tsv
if [[ -s bench/failures.txt ]]; then
  echo "[bench] completed with failures; see bench/failures.txt" >&2
  exit 1
fi
echo "[bench] done"
