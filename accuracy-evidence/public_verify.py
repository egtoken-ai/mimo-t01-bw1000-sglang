#!/usr/bin/env python3
"""Run a deterministic public-only GSM8K/MMLU compatibility check."""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from verify import TASK_RUNNERS

MAX_ITEMS = int(os.getenv("PUBLIC_SAMPLE_SIZE", "24"))
MAX_TOKENS = int(os.getenv("PUBLIC_MAX_TOKENS", "1024"))
THINKING = os.getenv("PUBLIC_THINKING", "true").lower() == "true"
OUTPUT_DIR = Path(os.getenv("PUBLIC_OUTPUT_DIR", str(HERE / "runs")))


def run_one(kind: str, item: dict) -> dict:
    extra = {"chat_template_kwargs": {"enable_thinking": THINKING}}
    detail = TASK_RUNNERS[kind](item, MAX_TOKENS, extra)
    return {"kind": kind, **detail}


def main() -> int:
    sources = {
        "gsm8k": HERE / "samples" / f"gsm8k_{MAX_ITEMS}.jsonl",
        "mmlu": HERE / "samples" / f"mmlu_{MAX_ITEMS}.jsonl",
    }
    items = []
    for kind, path in sources.items():
        data = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(data) != MAX_ITEMS:
            raise RuntimeError(f"{kind}: expected {MAX_ITEMS} items, got {len(data)}")
        items.extend((kind, item) for item in data)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda pair: run_one(*pair), items))

    results.sort(key=lambda item: (item["kind"], item.get("id") or ""))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"public_verify_items_{MAX_ITEMS}.jsonl"
    output_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in results),
        encoding="utf-8",
    )

    summary = {}
    for kind in sources:
        rows = [item for item in results if item["kind"] == kind]
        summary[kind] = {
            "correct": sum(item["correct"] for item in rows),
            "total": len(rows),
            "accuracy_pct": round(100.0 * sum(item["correct"] for item in rows) / len(rows), 2),
        }
    result = {
        "status": "provisional",
        "official_acceptance": False,
        "thinking": THINKING,
        "max_tokens": MAX_TOKENS,
        "sample_size_per_metric": MAX_ITEMS,
        "metrics": summary,
        "evidence": str(output_path),
        "note": "Public datasets only; official baseline and fixed contest subsets are still required.",
    }
    (OUTPUT_DIR / f"public_verify_result_{MAX_ITEMS}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
