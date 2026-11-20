# Public Accuracy Evidence

These results come from public GSM8K and MMLU samples. They are provisional and are
not an official contest result: the official baseline file and its fixed subsets
were not available, so `verify.py` (the official entry point) reports `blocked` and
`verify_result.json` stays unchanged.

## Results

Both runs used greedy decoding (`temperature=0`), `max_tokens=1024`, thinking
enabled, and public data only.

| Run | Sample | GSM8K | MMLU | Aggregate |
| --- | --- | --- | --- | --- |
| 24 per metric | seed 20261120 | 23/24 (95.83%) | 20/24 (83.33%) | `public_verify_result_24.json` |
| 64 per metric | seed 20261120 | 57/64 (89.06%) | 54/64 (84.38%) | `public_verify_result.json` |
| 64 per metric, repeat | same items | 57/64 (89.06%) | 54/64 (84.38%) | `public_verify_items_run2_64.jsonl` |

The MMLU 64-item sample covers all 57 subjects plus seven additional items.
Comparing the two 64-item runs, 0 of 128 item verdicts changed and one response
text differed. There were no request failures, timeouts or parse errors.

## Files

- `public_verify_result.json` - 64+64 aggregate.
- `public_verify_items.jsonl` - 64+64 per-item responses and verdicts (128 rows).
- `public_verify_result_24.json`, `public_verify_items_24.jsonl` - 24+24 run (48 rows).
- `public_verify_items_run2_64.jsonl` - repeated 64+64 per-item run (128 rows).
- `samples/` - fixed seed 20261120 samples and `manifest.json` with upstream
  sources and full-dataset SHA-256 values.
- `public_verify.py` - regenerates the run from `samples/`.

## Reproducing

Start the service with `bash run.sh`, then:

```bash
source ./env.sh
PUBLIC_SAMPLE_SIZE=64 PUBLIC_OUTPUT_DIR=accuracy-evidence/runs python3 accuracy-evidence/public_verify.py
```

New output lands in `runs/` so the recorded results above stay untouched.
