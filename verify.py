#!/usr/bin/env python3
"""Evaluate fixed GSM8K/MMLU subsets against an immutable official baseline."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TOL = 0.02
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def check(got: float, metric: dict) -> dict:
    lower_is_better = metric.get("lower_is_better", False)
    worse = got - metric["ref"] if lower_is_better else metric["ref"] - got
    tolerance = TOL * metric.get("scale", metric["ref"])
    return {
        "got": round(got, 4),
        "ref": metric["ref"],
        "worse": round(worse, 4),
        "tol": round(tolerance, 4),
        "pass": worse <= tolerance + 1e-9,
    }


def request_chat(prompt: str, max_tokens: int, extra: dict) -> str:
    body = {
        "model": os.environ["SERVED_NAME"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        **extra,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{os.environ['PORT']}/v1/chat/completions",
        json.dumps(body, ensure_ascii=False).encode(),
        {"Content-Type": "application/json"},
    )
    with OPENER.open(request, timeout=3600) as response:
        payload = json.load(response)
    message = payload["choices"][0]["message"]
    return (message.get("content") or "").split("</think>")[-1].strip()


def gsm8k_detail(item: dict, max_tokens: int, extra: dict) -> dict:
    response = request_chat(
        item["question"] + "\nPlease reason step by step, and put your final answer within \\boxed{}.",
        max_tokens,
        extra,
    )
    found = re.findall(r"\\boxed\{([^{}]*)\}", response) or re.findall(r"-?\d[\d,]*(?:\.\d+)?", response)
    try:
        correct = bool(found) and float(found[-1].replace(",", "").strip(" $\\")) == float(item["answer"])
    except ValueError:
        correct = False
    return {"id": item.get("id"), "expected": str(item["answer"]), "response": response, "correct": correct}


def gsm8k(item: dict, max_tokens: int, extra: dict) -> bool:
    return gsm8k_detail(item, max_tokens, extra)["correct"]


def mmlu_detail(item: dict, max_tokens: int, extra: dict) -> dict:
    options = "\n".join(f"{key}. {value}" for key, value in zip("ABCD", item["choices"]))
    response = request_chat(f"{item['question']}\n{options}\nAnswer with the option letter only.", max_tokens, extra)
    found = re.findall(r"\b[ABCD]\b", response)
    return {
        "id": item.get("id"),
        "subject": item.get("subject"),
        "expected": item["answer"],
        "response": response,
        "correct": bool(found) and found[0] == item["answer"],
    }


def mmlu(item: dict, max_tokens: int, extra: dict) -> bool:
    return mmlu_detail(item, max_tokens, extra)["correct"]


TASK_RUNNERS = {"gsm8k": gsm8k_detail, "mmlu": mmlu_detail}
EVALUATORS = {"gsm8k": gsm8k, "mmlu": mmlu}


def evaluate(name: str, metric: dict, extra: dict) -> float:
    if name not in EVALUATORS:
        raise ValueError(f"missing evaluator: {name}")
    path = Path(metric["subset"])
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(items) != metric["n"]:
        raise ValueError(f"{name}: expected {metric['n']} items, got {len(items)}")
    with ThreadPoolExecutor(max_workers=16) as pool:
        hits = list(pool.map(lambda item: EVALUATORS[name](item, metric["max_tokens"], extra), items))
    return 100.0 * sum(hits) / len(items)


def selfcheck() -> None:
    accuracy = {"ref": 80.0, "scale": 100}
    wer = {"ref": 5.0, "scale": 100, "lower_is_better": True}
    assert not check(77.9, accuracy)["pass"]
    assert check(78.0, accuracy)["pass"] and check(85.0, accuracy)["pass"]
    assert not check(7.1, wer)["pass"] and check(4.0, wer)["pass"]
    print("selfcheck OK")


def write_blocked(path: Path, missing: list[str]) -> None:
    path.write_text(json.dumps({
        "status": "blocked",
        "passed": False,
        "missing": missing,
        "detail": "Official contest baseline/subset is required; no fallback baseline is used.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0

    model_id = os.environ["MODEL_ID"]
    baseline_path = Path("baselines") / f"{model_id.replace('/', '__')}.json"
    output_path = Path("verify_result.json")
    if not baseline_path.exists():
        write_blocked(output_path, [str(baseline_path)])
        print(f"missing official baseline: {baseline_path}", file=sys.stderr)
        return 2

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    metrics = baseline.get("metrics", {})
    missing = [name for name in ("gsm8k", "mmlu") if name not in metrics]
    missing += [metric["subset"] for metric in metrics.values() if not Path(metric["subset"]).exists()]
    if missing:
        write_blocked(output_path, missing)
        print("missing official assets: " + ", ".join(missing), file=sys.stderr)
        return 2

    extra = baseline.get("extra_body", {})
    detail = {name: check(evaluate(name, metric, extra), metric) for name, metric in metrics.items()}
    passed = all(item["pass"] for item in detail.values())
    result = {
        "model_id": model_id,
        "card": os.environ.get("CARD"),
        "framework": os.environ.get("FRAMEWORK"),
        "precision": os.environ.get("PRECISION"),
        "detail": detail,
        "passed": passed,
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
