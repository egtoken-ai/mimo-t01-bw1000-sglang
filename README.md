# MiMo-V2.6-Distill-Qwen-9B on BW1000 + SGLang 0.5.12

## 1. 简介

本文档记录 `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B` 在 `BW1000 + SGLang 0.5.12`（精度 `bf16`）环境的适配与验证结果。

模型约 9.4B 参数，`Qwen3_5ForConditionalGeneration` 架构：32 层文本层中 24 层线性注意力与 8 层全注意力交替（每 4 层 1 层全注意力），另带 27 层视觉塔，支持文本与图像输入，权重 BF16、四个分片约 18 GB。

与官方部署方式的核心差异：权重、结构、tokenizer、chat template 均未改动，没有软件包升级，也没有代码 patch，直接使用镜像内自带的 SGLang；差异集中在服务参数上，注意力后端选 FA3（triton 在本卡 page size 64 下输出异常），关闭 radix cache，线性注意力层使用 `no_buffer` 调度，并接入 `--reasoning-parser mimo` 让思考内容从 `reasoning_content` 返回。

相关获取地址：

- 权重下载（超算互联网 AI 社区）：https://www.scnet.cn/ui/aihub/models/sugon_scnet/MiMo-V2.6-Distill-Qwen-9B/files/tree/main
- 镜像：`UNVERIFIED`（容器内未暴露 registry tag，回填方法见第 8 节）
- 提交仓库：https://github.com/egtoken-ai/mimo-t01-bw1000-sglang

参考文档：

- cookbook：`docs/cookbook/mimo-v2.6-bw-sglang.md`（启动、思考请求、多轮、流式、图像输入与验证）
- 赛事 T01 条目：模型规格与官方 `sglang serve` 启动方式

仓库内容：

| 文件 | 作用 | 产出 |
| --- | --- | --- |
| `env.sh` | 模型、镜像、框架、卡型、并行方式与启动参数，其余脚本读这里的变量 | — |
| `env_check.sh` | 采集卡型与重点组件版本 | `env.log` |
| `patches/` | 代码改动（本件为空，无改动） | — |
| `run.sh` | 模型启动脚本：装依赖 + 重放 patch + 拉起服务，轮询 `/health` | `server.log` |
| `smoke.sh` | Smoke 测试：健康检查、单轮 / 多轮 / 流式问答 | `smoke.log` |
| `verify.py` | 精度验证脚本：按官方基线文件跑评测并判定 | `verify_result.json` |
| `bench.sh` | 性能测试脚本：6 档负载，每档连测两次 | `bench/` |
| `accuracy-evidence/` | 公开数据补充验证：逐题回答、汇总、固定样本与生成脚本 | — |
| `docs/cookbook/` | cookbook 条目：启动、思考、多轮、流式、图像输入 | — |

一键复现：

```bash
git clone https://github.com/egtoken-ai/mimo-t01-bw1000-sglang.git && cd mimo-t01-bw1000-sglang
bash run.sh                             # 装依赖 + 重放 patch + 起服务（重复执行自动跳过已完成的步骤）
bash env_check.sh                       # 记录验证环境 → env.log
bash smoke.sh                           # Smoke 测试 → smoke.log
source env.sh && python verify.py       # 精度测评 → verify_result.json
bash bench.sh                           # 性能测试 → bench/
```

## 2. 验证环境

镜像：`UNVERIFIED`（本次为超算互联网 Notebook 容器，容器内未暴露 registry tag，`env.sh` 与 `env.log` 同记该值）

重点组件（取自 `env.log`）：

| 组件 | 版本 | 备注 |
| --- | --- | --- |
| DTK | `2604` | 镜像自带 |
| `torch` | `2.10.0+das.opt1.dtk2604.2608131652.g28498c` | 镜像自带，HIP 6.3.26113 |
| `triton` | `3.6.0+das185.dtk2604.torch2100.2608131719.gb796b1` | 镜像自带 |
| `sglang` | `0.5.12+das185.dtk2604.torch2100.2608121934.gcbf025` | 镜像自带 |
| `sglang-kernel` | `0.4.2.post2+das185.dtk2604.torch2100.2608121934.gcbf025` | 镜像自带 |
| `transformers` | `5.6.0` | 镜像自带，未升级 |
| `flash_attn` | `2.8.3+das185.dtk2604.torch2100.2608132155.g9febea` | 镜像自带，FA3 后端 |

- 卡型：`BW1000`
- 卡数：`1`（单机 TP=1，PP=1，EP=1）
- 模型路径：`/public/home/maxkonger/SothisAI/model/Aihub/MiMo-V2.6-Distill-Qwen-9B/main/MiMo-V2.6-Distill-Qwen-9B`
- 服务端口：`30000`

环境自检（贴 `env.log` 关键部分原文）：

```text
image: UNVERIFIED
dtk_env: 2604
python: 3.10.12
torch: 2.10.0
torch.hip: 6.3.26113
transformers: 5.6.0
sglang: 0.5.12
device_count: 1
device[0]: BW 63.984 GiB
aiter 0.1.5+das185.dtk2604.torch2100.2608131714.gb0979d
flash_attn 2.8.3+das185.dtk2604.torch2100.2608132155.g9febea
sglang 0.5.12+das185.dtk2604.torch2100.2608121934.gcbf025
sglang-kernel 0.4.2.post2+das185.dtk2604.torch2100.2608121934.gcbf025
tokenizers 0.22.2
torch 2.10.0+das.opt1.dtk2604.2608131652.g28498c
transformers 5.6.0
triton 3.6.0+das185.dtk2604.torch2100.2608131719.gb796b1
```

超算互联网平台（参赛算力统一使用超算互联网，必填）：

| 项 | 值 |
| --- | --- |
| 区域 | 未记录（控制台实例详情页未截图保存） |
| Notebook 规格 | 15 vCPU / 59 GB 内存 / `/public/home` 文件存储（`NOTEBOOK_CPU_NUM=15`、`NOTEBOOK_RAM_SIZE=59GB`）；系统盘容量未记录 |
| 加速卡与显存 | `BW1000` × `1`；`64GB`（`NOTEBOOK_ACCELERATOR_MEMORY=64GB`，`hy-smi` 报 63.984 GiB） |
| 开发镜像 | `UNVERIFIED`（容器内未暴露 tag） |
| 模型获取方式 | AI 社区模型库 `sugon_scnet/MiMo-V2.6-Distill-Qwen-9B` 克隆到 `/public/home/maxkonger/SothisAI/model/Aihub/MiMo-V2.6-Distill-Qwen-9B/main/MiMo-V2.6-Distill-Qwen-9B` |
| 服务映射 | 无（本次直接使用容器内 `127.0.0.1:30000`，未配置对外代理） |
| 平台证据 | `platform/`（本次未随包提交，见第 8 节） |

## 3. 适配优化方案

无代码改动，直接使用镜像内版本。

## 4. 服务启动脚本

`env.sh` 中与启动相关的配置：

```bash
export MODEL_PATH="/public/home/maxkonger/SothisAI/model/Aihub/MiMo-V2.6-Distill-Qwen-9B/main/MiMo-V2.6-Distill-Qwen-9B"
export SERVED_NAME="MiMo-V2.6-Distill-Qwen-9B"
export FRAMEWORK="sglang"
export PORT=30000
export TP=1
export EP=1
export PP=1
export MAX_LEN=131072       # 须容纳 16K 输入 + 1K 输出的压测
export MEM_FRAC=0.85
export EXTRA_ARGS="--reasoning-parser mimo --dtype bfloat16 --kv-cache-dtype bfloat16 --max-running-requests 128 --chunked-prefill-size 8192 --max-prefill-tokens 16384 --disable-radix-cache --attention-backend fa3 --page-size 64 --mamba-scheduler-strategy no_buffer --cuda-graph-max-bs 128 --disable-custom-all-reduce --enable-metrics"
# 单卡部署，无需 HIP_VISIBLE_DEVICES 等设备可见性环境变量
```

启动：

```bash
bash run.sh
```

`run.sh` 展开后实际执行的完整命令（取自 `server.log` 首行）：

```bash
sglang serve \
  --model-path /public/home/maxkonger/SothisAI/model/Aihub/MiMo-V2.6-Distill-Qwen-9B/main/MiMo-V2.6-Distill-Qwen-9B \
  --served-model-name MiMo-V2.6-Distill-Qwen-9B \
  --tp-size 1 --pp-size 1 \
  --context-length 131072 \
  --mem-fraction-static 0.85 \
  --host 0.0.0.0 --port 30000 \
  --reasoning-parser mimo \
  --dtype bfloat16 --kv-cache-dtype bfloat16 \
  --max-running-requests 128 \
  --chunked-prefill-size 8192 --max-prefill-tokens 16384 \
  --disable-radix-cache \
  --attention-backend fa3 --page-size 64 \
  --mamba-scheduler-strategy no_buffer \
  --cuda-graph-max-bs 128 \
  --disable-custom-all-reduce --enable-metrics
```

重点参数说明（只列与默认值不同、或对能否跑通 / 性能有影响的参数）：

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| `--tp-size` | `1` | 权重 17.64 GB，单卡 64 GB 可容纳，无需切分 |
| `--context-length` | `131072` | 覆盖 16K + 1K 压测；模型 config 上限 262144，压到一半留出 KV 空间 |
| `--mem-fraction-static` | `0.85` | 实测 KV 池 30.19 GB、`max_total_num_tokens=989312`，满足 128 并发 |
| `--reasoning-parser` | `mimo` | 思考内容从 `reasoning_content` 返回 |
| `--attention-backend` / `--page-size` | `fa3` / `64` | 本卡上输出正确；`triton` 在 page size 64 下输出异常，见第 8 节 |
| `--disable-radix-cache` | 开 | 本件按关闭前缀缓存验证与交付，压测为随机输入 |
| `--mamba-scheduler-strategy` | `no_buffer` | 线性注意力层按需分配状态，128 并发下稳定 |
| `--max-running-requests` / `--cuda-graph-max-bs` | `128` / `128` | 并发上限与 CUDA Graph 批次对齐 |
| `--disable-custom-all-reduce` | 开 | 单卡无通信，关闭以减少初始化开销 |
| 环境变量 | 无 | 单卡部署，未设置设备可见性等变量 |

启动耗时（`server.log` 时间戳）：进程起于 15:27:35，权重加载 3.42 s，CUDA Graph 捕获 4.06 s，15:28:28 打印 `The server is fired up and ready to roll!`，全程约 53 s。显存占用（`/get_server_info` 的 `memory_usage`）：权重 17.64 GB、KV cache 30.19 GB、graph 0.56 GB。

多机部署：未使用。本次为单机单卡，`NNODES=1`、`NODE_RANK=0`，不需要 `DIST_ADDR` 组网。

## 5. Smoke 测试

```bash
bash smoke.sh
```

`smoke.sh` 依次执行以下检查（任一接口非 2xx 即失败），下列命令为等价的手工复现方式；脚本内部用 Python 的 urllib 发同样的请求，并在单轮 / 多轮请求中带 `chat_template_kwargs.enable_thinking=true`：

```bash
# 健康检查
curl -sf http://127.0.0.1:30000/health -o /dev/null -w "HTTP %{http_code}\n"

# 模型列表
curl -sf http://127.0.0.1:30000/v1/models

# 简单问答
curl -sf http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiMo-V2.6-Distill-Qwen-9B",
    "messages": [{"role": "user", "content": "计算 17 + 25，最后一行只输出答案：数字。"}],
    "temperature": 0,
    "max_tokens": 512,
    "chat_template_kwargs": {"enable_thinking": true}
  }'

# 另有 /metrics、多轮（先告知名字再问「我叫什么名字？」）与流式（"stream": true），见 smoke.sh
```

验证结果（贴 `smoke.log` 关键部分）：

- `/health` 返回 `200`
- `/v1/models` 返回 `MiMo-V2.6-Distill-Qwen-9B`，`max_model_len=131072`
- `/metrics` 返回 `200`
- 单轮问答回复：「42」，`usage.reasoning_tokens=2`，`finish_reason=stop`
- 多轮：正确回答「小海。」
- 流式：逐 token 返回「你好！有什么我可以帮你的吗？」，以 `data: [DONE]` 结束

补充观察（未写入 `smoke.sh`）：向同一接口发送一张 224×224 纯色图片并询问颜色，返回 `red`，说明视觉塔在本次部署下可推理；该项未做图集精度评测。

## 6. 精度测评

精度验证脚本 `verify.py` 按官方基线文件 `baselines/XiaomiMiMo__MiMo-V2.6-Distill-Qwen-9B.json` 写死的配置调用服务跑评测，与官方 GPU 基线对比，较基线劣化 ≤2% 判 PASS（百分制即 2 pt），提升不罚。

```bash
python verify.py --selfcheck            # 先确认判分逻辑未被改动
source env.sh && python verify.py       # 跑评测，结果写入 verify_result.json
```

`--selfcheck` 实测输出 `selfcheck OK`，判分逻辑未被改动。

评测配置（取自官方基线文件，参赛者不改）：

| 数据集 | 子集题数 | max_tokens | 解码 | 思考开关 | 判分方式 |
| --- | --- | --- | --- | --- | --- |
| GSM8K | 未下发（官方基线文件缺失） | 未下发 | greedy（temperature=0） | 未下发 | 取 `\boxed{}` 或末尾数字，与答案数值相等 |
| MMLU | 未下发（官方基线文件缺失） | 未下发 | greedy（temperature=0） | 未下发 | 取首个 A–D 选项字母 |

官方基线文件与其固定子集未取得，`verify.py` 不使用非官方数据顶替，直接返回 blocked。结果（`verify_result.json` 原文）：

```json
{
  "status": "blocked",
  "passed": false,
  "missing": [
    "baselines/XiaomiMiMo__MiMo-V2.6-Distill-Qwen-9B.json"
  ],
  "detail": "Official contest baseline/subset is required; no fallback baseline is used."
}
```

| 数据集 | 测试集 | 本次提交 | 官方 GPU 基线 | 劣化 | 容差 | 判定 |
| --- | --- | --- | --- | --- | --- | --- |
| GSM8K | 公开抽样 24 题（seed 20261120） | 23/24（95.83%） | 未下发 | 无法计算 | ≤2 pt | 补充测试，不参与官方判定 |
| GSM8K | 公开抽样 64 题（seed 20261120） | 57/64（89.06%） | 未下发 | 无法计算 | ≤2 pt | 补充测试，不参与官方判定 |
| MMLU | 公开抽样 24 题（seed 20261120） | 20/24（83.33%） | 未下发 | 无法计算 | ≤2 pt | 补充测试，不参与官方判定 |
| MMLU | 公开抽样 64 题（seed 20261120，覆盖全部 57 个学科） | 54/64（84.38%） | 未下发 | 无法计算 | ≤2 pt | 补充测试，不参与官方判定 |

结论：上表已填入公开数据测试结果；官方验收仍为 BLOCKED，未能判定 PASS 或 FAIL，原因是官方基线文件与固定 GSM8K/MMLU 子集未下发。取得后放入 `baselines/` 与 `data/` 并重跑 `verify.py` 即可，脚本无需修改。

公开测试使用 greedy 解码（`temperature=0`）、`max_tokens=1024`、开启思考，数据与逐题回答保存在 `accuracy-evidence/`。同批 64+64 复跑一次，总分一致，128 题判分无变化，仅 1 条回答文本不同。该结果不参与官方判定，也不作为官方成绩。

> T02–T04 的评测器、数据集与判定口径随分题材料下发，表格列按实际指标调整；本件为 T01，沿用上述口径。

## 7. 性能参考

性能测试脚本 `bench.sh` 使用 SGLang 自带压测客户端（SGLang / vLLM 服务通用），随机 token 输入、固定输入输出长度，负载如下：

| 输入 | 输出 | 并发 |
| --- | --- | --- |
| 4K（4096） | 1K（1024） | 1 / 8 / 64 |
| 16K（16384） | 1K（1024） | 1 / 8 / 64 |

每档连测两次，第一次预热，**取第二次数据**；请求数为并发的 2 倍（至少 8 个）。

```bash
bash bench.sh                           # 文本模型
```

单档命令（`bench.sh` 内部对 6 档 × 2 次循环执行）：

```bash
python3 -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port 30000 \
  --model "$MODEL_PATH" --served-model-name MiMo-V2.6-Distill-Qwen-9B \
  --tokenizer "$MODEL_PATH" \
  --dataset-name random-ids \
  --random-input-len 4096 --random-output-len 1024 --random-range-ratio 1 \
  --max-concurrency 8 --num-prompts 16 --seed 1 \
  --output-file bench/round2.jsonl
```

第二次结果（`bench/summary.md` 原文）：

| random_input_len | random_output_len | max_concurrency | completed | duration | request_throughput | output_throughput | total_throughput | mean_ttft_ms | p99_ttft_ms | mean_tpot_ms | p99_tpot_ms | mean_e2e_latency_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 | 1024 | 1 | 8 | 171.61 | 0.05 | 47.74 | 238.68 | 317.96 | 327.99 | 20.65 | 20.67 | 21447.70 |
| 4096 | 1024 | 8 | 16 | 41.14 | 0.39 | 398.24 | 1991.20 | 1824.80 | 2580.44 | 18.31 | 19.72 | 20557.10 |
| 4096 | 1024 | 64 | 128 | 115.13 | 1.11 | 1138.50 | 5692.52 | 12854.52 | 21254.13 | 43.67 | 55.17 | 57532.69 |
| 16384 | 1024 | 1 | 8 | 187.50 | 0.04 | 43.69 | 742.73 | 1638.50 | 1764.15 | 21.31 | 21.46 | 23433.81 |
| 16384 | 1024 | 8 | 16 | 67.85 | 0.24 | 241.46 | 4104.79 | 8575.77 | 12530.94 | 24.77 | 31.03 | 33911.82 |
| 16384 | 1024 | 64 | 128 | 345.53 | 0.37 | 379.34 | 6448.73 | 68259.07 | 170938.56 | 78.36 | 126.69 | 148417.20 |

- 原始数据：`bench/round1.jsonl`、`bench/round2.jsonl`、各档日志 `bench/round*_*.log`、`bench/status.tsv`（12 次调用全部 `exit 0`）
- 本结果作为该件「模型 × 卡型 × 框架 × 精度」的性能基准；后续优化在同卡型、同框架、同负载下对比
- T03（生成类）/ T04（语音类）的性能测法后续随分题材料提供，本件为 T01 文本模型，不适用

## 8. 注意事项

- **不支持的负载**：本次未覆盖超过 128 的路由并发、超过 131072 的上下文，以及 TP>1 的切分部署。131072 是部署配置值，模型 config 上限为 262144，未在此卡上验证更长上下文；6 档压测本身全部出数，无失败档位。
- **修改的模型参数**：无。`config.json`、`generation_config.json`、tokenizer、chat template 均未改动，`env.log` 记录了这四个文件的 SHA-256；第 6 节的公开数据结果即为未改参数下的输出。
- **已知限制**：镜像内 `transformers` 为 5.6.0，检查点 `config.json` 标注 `transformers_version: 5.12.1`，实际加载与全部验证均在 5.6.0 上完成且未报错（SGLang 走自身的 `Qwen3_5ForConditionalGeneration` 实现）。官方精度未验收（基线文件与固定子集未下发，`verify_result.json` 为 blocked）；镜像 tag、Notebook 区域与 `platform/` 截图为材料缺口，回填方式见本节最后一条；全程关闭 radix cache，第 7 节数字均为无前缀复用条件下的结果，未做开启前后的 A/B 对照。
- **踩坑与规避**：先用 `--attention-backend triton --page-size 64` 启动，服务能起且 `/health` 正常，但「12+7」的请求出现重复循环直到打满 `max_tokens`、`content` 为空；换成 `--attention-backend fa3` 后输出正确。结论是无错误码的健康检查不足以证明推理正确，必须带语义断言。另：`run.sh` 在端口已有健康服务时直接报错退出，复跑前需先停掉旧进程或删除 `server.pid`。
- **其他**：本件为单卡 TP1，启动约 53 s，`LOAD_TIMEOUT` 保持 1800 s 无需调整；`server.log` 中有 `torchcodec` 缺失的 import 提示与 NUMA balancing 告警，前者只影响未使用的多模态处理器分支，后者是平台内核设置，本次推理未出现对应报错。镜像回填方法：在 SCNet 控制台镜像库找到本次镜像，取得 `仓库地址:tag` 后更新 `env.sh` 的 `IMAGE` 并重跑 `bash env_check.sh` 刷新 `env.log`，同时补齐 `platform/` 下的实例创建页、环境自检与模型管理截图。
