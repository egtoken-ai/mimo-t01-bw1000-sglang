# Cookbook: MiMo-V2.6-Distill-Qwen-9B on BW1000 with SGLang

本卡验证过的组合：`--attention-backend fa3 --page-size 64`，配合 `--disable-radix-cache`、`--mamba-scheduler-strategy no_buffer` 与 `--reasoning-parser mimo`。

## 1. 启动服务

```bash
source ./env.sh
bash run.sh
bash env_check.sh   # 可选，记录环境到 env.log
```

服务就绪后 `http://127.0.0.1:30000/health` 返回 200。

## 2. 基本问答

```bash
curl -s http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"MiMo-V2.6-Distill-Qwen-9B","messages":[{"role":"user","content":"用一句话介绍 DCU。"}],"temperature":0,"max_tokens":256}'
```

默认未开启思考时，答案在 `choices[0].message.content`。

## 3. 开启思考

`chat_template_kwargs` 在原始 HTTP 请求中是顶层字段，用 OpenAI SDK 时通过 `extra_body` 传入；思考内容在 `message.reasoning_content`，最终答案在 `message.content`。

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="MiMo-V2.6-Distill-Qwen-9B",
    messages=[{"role": "user", "content": "计算 17 + 25。"}],
    temperature=0,
    max_tokens=512,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)
message = response.choices[0].message
print(message.reasoning_content or "")
print(message.content or "")
```

## 4. 多轮对话

按顺序追加 `messages` 即可，服务端无需额外参数：

```python
messages = [
    {"role": "user", "content": "请记住，我的名字是小海。"},
    {"role": "assistant", "content": "好的。"},
    {"role": "user", "content": "我叫什么名字？只回答名字。"},
]
```

注意本部署关闭了 radix cache，多轮请求不会复用前缀 KV，长对话的 TTFT 随上下文增长。

## 5. 流式输出

请求加 `"stream": true`，响应为 SSE：

```bash
curl -N -s http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"MiMo-V2.6-Distill-Qwen-9B","messages":[{"role":"user","content":"用一句中文回答：你好。"}],"temperature":0,"max_tokens":256,"stream":true}'
```

逐行读取 `data: ` 后的 JSON，取 `choices[0].delta.content` 累加，最后一行是 `data: [DONE]`。

## 6. 图像输入

模型带 27 层视觉塔，可用 OpenAI 兼容的 `image_url` 内容块传图：

```json
{
  "model": "MiMo-V2.6-Distill-Qwen-9B",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
    {"type": "text", "text": "What color is the square? Answer with one word."}
  ]}],
  "temperature": 0,
  "max_tokens": 64
}
```

## 7. 自检与验证

```bash
python verify.py --selfcheck   # 判分逻辑自检，不访问服务
bash smoke.sh                  # 健康、模型列表、单轮、多轮、流式
```

## 8. 已知限制

- 仅验证 FA3 + page size 64；triton 在本卡该 page size 下输出异常。
- 关闭 radix cache，多轮前缀不复用。
- 单卡验证，TP>1 未测。
