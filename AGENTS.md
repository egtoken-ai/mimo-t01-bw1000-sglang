# Repository Guidelines

## Project Structure & Module Organization

This is the T01 submission for `MiMo-V2.6-Distill-Qwen-9B` on one BW1000 card with SGLang. `env.sh` is the source of truth for model identity, paths, precision, and serving parameters. Its companion scripts produce `env.log`, `server.log`, `smoke.log`, `verify_result.json`, and `bench/`. `accuracy-evidence/` holds provisional public-data runs; `docs/cookbook/` holds usage notes. Keep official baselines and contest subsets in `baselines/` and `data/`. Files under `submission/dev-evidence/` are local development material.

## Build, Test, and Development Commands

Run commands from the repository root on the SCNet host:

```bash
bash run.sh                         # start SGLang and wait for /health
bash env_check.sh                   # record hardware and package versions
bash smoke.sh                       # health, models, single/multi-turn, streaming
python3 verify.py --selfcheck       # check official pass/fail arithmetic
source ./env.sh && python3 verify.py  # official GSM8K/MMLU gate
bash bench.sh                       # six load tiers, two rounds, results in bench/
```

For public-data diagnostics, run `source ./env.sh && PUBLIC_SAMPLE_SIZE=64 python3 accuracy-evidence/public_verify.py`. Add `requirements-extra.txt` only when the image needs pinned extra packages.

## Coding Style & Naming Conventions

Use two-space indentation in Bash and four spaces in Python. Name Python functions and variables with `snake_case`, classes with `PascalCase`, environment variables in uppercase, and documentation files in lowercase with hyphens. Keep model and service settings in `env.sh`; avoid duplicating them in scripts. No formatter or linter is configured. Check shell edits with `bash -n` and Python edits with `python3 -m py_compile`.

## Testing Guidelines

There is no unit-test suite. Run smoke checks against the live service and assert semantic answers; a healthy endpoint does not prove correct inference. Keep the same card, precision, attention backend, page size, and workload when comparing results. `verify.py` must return `blocked` when official assets are missing. Never label public samples as official contest results.

## Commit & Pull Request Guidelines

Follow the existing Conventional Commits style, for example `docs: clarify benchmark setup` or `feat: add smoke coverage`. Pull requests should state the model, card, image, framework, precision, parallelism, commands run, evidence paths, and known limitations. Link the relevant issue and report failures honestly. Do not claim a target was met without a fresh result from that configuration.

## Security & Configuration Tips

Never commit passwords, access tokens, SSH keys, model weights, or private data. Keep host-specific values in environment variables and redact credentials from logs. Mark unknown image or platform details as `UNVERIFIED`. Rebuild and inspect the submission archive after changing packaged files.
