#!/usr/bin/env bash
# Validate health, model identity, single-turn, multi-turn, streaming and reasoning.
set -euo pipefail
source ./env.sh
exec > >(tee smoke.log) 2>&1

python3 - <<'PY'
import json
import os
import re
import urllib.request

base = f"http://127.0.0.1:{os.environ['PORT']}"
model = os.environ['SERVED_NAME']
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get(path):
    with opener.open(base + path, timeout=60) as response:
        assert response.status == 200, (path, response.status)
        return response.read()


def chat(messages, max_tokens=512):
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    request = urllib.request.Request(
        base + "/v1/chat/completions",
        json.dumps(body, ensure_ascii=False).encode(),
        {"Content-Type": "application/json"},
    )
    with opener.open(request, timeout=600) as response:
        assert response.status == 200
        return json.load(response)


def stream(messages, max_tokens=256):
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    request = urllib.request.Request(
        base + "/v1/chat/completions",
        json.dumps(body, ensure_ascii=False).encode(),
        {"Content-Type": "application/json"},
    )
    chunks = []
    reasoning = []
    done = False
    with opener.open(request, timeout=600) as response:
        assert response.status == 200
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                done = True
                break
            item = json.loads(payload)
            delta = item.get("choices", [{}])[0].get("delta", {})
            chunks.append(delta.get("content") or "")
            reasoning.append(delta.get("reasoning_content") or "")
    assert done, "stream did not terminate with [DONE]"
    return "".join(chunks).strip(), "".join(reasoning).strip()


print("== health")
get("/health")
print("HTTP 200")

print("== /v1/models")
models = json.loads(get("/v1/models"))
ids = [item["id"] for item in models["data"]]
assert model in ids, (model, ids)
print(json.dumps(models, ensure_ascii=False))

print("== /metrics")
get("/metrics")
print("HTTP 200")

print("== single-turn")
single = chat([
    {"role": "user", "content": "计算 17 + 25，最后一行只输出答案：数字。"},
])
message = single["choices"][0]["message"]
answer = message.get("content") or ""
usage = single.get("usage") or {}
assert re.search(r"\b42\b", answer), answer
assert (usage.get("reasoning_tokens") or 0) > 0 or message.get("reasoning_content"), usage
print(json.dumps(single, ensure_ascii=False))

print("== multi-turn")
multi = chat([
    {"role": "user", "content": "请记住，我的名字是小海。"},
    {"role": "assistant", "content": "好的。"},
    {"role": "user", "content": "我叫什么名字？只回答名字。"},
])
multi_answer = multi["choices"][0]["message"].get("content") or ""
assert "小海" in multi_answer, multi_answer
print(json.dumps(multi, ensure_ascii=False))

print("== streaming")
stream_answer, stream_reasoning = stream([
    {"role": "user", "content": "用一句中文回答：你好。"},
])
assert stream_answer, "stream produced no final content"
print(json.dumps({
    "content": stream_answer,
    "reasoning_content_present": bool(stream_reasoning),
    "terminated": True,
}, ensure_ascii=False))

print("smoke OK")
PY
