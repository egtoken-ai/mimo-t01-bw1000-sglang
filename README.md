# MiMo-V2.6-Distill-Qwen-9B on BW1000 + SGLang 0.5.12

## 1. 简介

本文档记录 `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B` 在 `BW1000 + SGLang 0.5.12`（精度 `bf16`）环境的适配与验证结果。

模型以 Qwen3.5-9B 为基座做 SFT，约 9.4B 参数，`Qwen3_5ForConditionalGeneration` 架构：32 层文本层中 24 层线性注意力与 8 层全注意力交替（每 4 层 1 层全注意力），另带 27 层视觉塔，因此同时支持文本和图像输入。权重 BF16，四个分片约 18 GB。

与官方部署方式的核心差异：权重、结构、tokenizer、chat template 均未改动，没有软件包升级，也没有代码 patch，直接使用镜像内自带的 SGLang。适配工作集中在服务参数上——注意力后端选用 FA3，关闭 radix cache，线性注意力层使用 `no_buffer` 调度策略，并接入 `--reasoning-parser mimo` 让思考内容从 `reasoning_content` 返回。

相关获取地址：

- 权重下载（超算互联网 AI 社区）：https://www.scnet.cn/ui/aihub/models/sugon_scnet/MiMo-V2.6-Distill-Qwen-9B/files/tree/main
- 镜像：`UNVERIFIED`（容器内未暴露 registry tag，提交前按第 8 节从 SCNet 镜像库回填）
- 提交仓库：https://github.com/egtoken-ai/mimo-t01-bw1000-sglang

参考文档：

- 赛事详情中的 T01 条目（模型规格与启动方式）

仓库内容：

| 文件 | 作用 | 产出 |
| --- | --- | --- |
| `env.sh` | 模型、镜像、框架、卡型、并行方式与启动参数，其余脚本读这里的变量 | — |
| `env_check.sh` | 采集卡型与重点组件版本 | `env.log` |
| `patches/` | 代码改动（本件为空） | — |
| `run.sh` | 启动服务：重放 patch（如有）+ 拉起 SGLang + 轮询 `/health` | `server.log` |
| `smoke.sh` | 健康检查、单轮、多轮、流式、思考 | `smoke.log` |
| `verify.py` | 按官方基线跑 GSM8K/MMLU 并判定 | `verify_result.json` |
| `bench.sh` | 6 档负载 × 2 轮性能测试 | `bench/` |
| `accuracy-evidence/` | 公开数据精度结果、固定样本与生成脚本（补充材料，非官方评测） | — |

一键复现：

```bash
bash run.sh
bash env_check.sh
bash smoke.sh
source env.sh && python verify.py
bash bench.sh
```

## 2. 验证环境

镜像：`UNVERIFIED`。本次运行环境为超算互联网 Notebook 容器，容器内没有暴露 registry tag，因此 `env.sh` 与 `env.log` 中该字段记为 `UNVERIFIED`；提交前需从 SCNet 镜像库页面把完整地址回填到 `env.sh` 的 `IMAGE` 与本节。

重点组件（取自 `env.log`）：

| 组件 | 版本 | 备注 |
| --- | --- | --- |
| DTK | `2604` | 镜像自带 |
| `torch` | `2.10.0+das.opt1.dtk2604.2608131652.g28498c` | 镜像自带，HIP 6.3.26113 |
| `triton` | `3.6.0+das185.dtk2604.torch2100.2608131719.gb796b1` | 镜像自带 |
| `sglang` | `0.5.12+das185.dtk2604.torch2100.2608121934.gcbf025` | 镜像自带 |
| `sglang-kernel` | `0.4.2.post2+das185.dtk2604.torch2100.2608121934.gcbf025` | 镜像自带 |
| `transformers` | `5.6.0` | 镜像自带 |
| `flash_attn` | `2.8.3+das185.dtk2604.torch2100.2608132155.g9febea` | 镜像自带，FA3 后端 |
| `tokenizers` | `0.22.2` | 镜像自带 |
| `aiter` | `0.1.5+das185.dtk2604.torch2100.2608131714.gb0979d` | 镜像自带 |

- 卡型：`BW1000`
- 卡数：1（单机 TP=1，PP=1，EP=1）
- 模型路径：`/public/home/maxkonger/SothisAI/model/Aihub/MiMo-V2.6-Distill-Qwen-9B/main/MiMo-V2.6-Distill-Qwen-9B`
- 服务端口：`30000`

环境自检（`env.log` 节选）：

```text
image: UNVERIFIED
tp: 1 ep: 1 pp: 1
dtk_env: 2604
python: 3.10.12
torch: 2.10.0
torch.hip: 6.3.26113
transformers: 5.6.0
sglang: 0.5.12
device_count: 1
device[0]: BW 63.984 GiB
```

设备状态采样（`hy-smi`，启动前空闲状态）：

```text
HCU     Temp     AvgPwr     Perf     PwrCap     VRAM%      HCU%      Mode
0       54.0C    82.0W      manual   1000.0W    87%        0.0%      Normal
```

超算互联网平台：

| 项 | 值 |
| --- | --- |
| 区域 | 未记录（控制台实例详情页未截图保存） |
| Notebook 规格 | 15 vCPU / 59 GB 内存 / 系统盘与 `/public/home` 文件存储（取自容器环境变量 `NOTEBOOK_CPU_NUM=15`、`NOTEBOOK_RAM_SIZE=59GB`） |
| 加速卡与显存 | `BW1000` × 1；64 GB（`NOTEBOOK_ACCELERATOR_MEMORY=64GB`，`hy-smi` 报告 63.984 GiB） |
| 开发镜像 | `UNVERIFIED`（容器内未暴露 tag，与第 2 节镜像字段同源） |
| 模型获取方式 | AI 社区模型库 `sugon_scnet/MiMo-V2.6-Distill-Qwen-9B` 克隆到 `/public/home/maxkonger/SothisAI/model/Aihub/MiMo-V2.6-Distill-Qwen-9B/main/MiMo-V2.6-Distill-Qwen-9B` |
| 服务映射 | 本次验证直接使用容器内 `127.0.0.1:30000`，未配置对外代理映射 |
| 平台证据 | `platform/`（本次未随包提交） |

## 3. 适配优化方案

### 3.1 基础镜像

基于镜像内已有的 `sglang 0.5.12 + DTK 2604` 环境。该版本已包含 `Qwen3_5ForConditionalGeneration` 实现，可以直接加载本检查点，无需改动模型代码。

### 3.2 软件包升级

推理时未额外安装或升级任何包，因此不提供 `requirements-extra.txt`。镜像内 `transformers` 自带版本为 5.6.0，而检查点 `config.json` 标注 `transformers_version: 5.12.1`；实际加载与全部验证均在 5.6.0 上完成，SGLang 走的是自身的 `Qwen3_5ForConditionalGeneration` 实现，未因版本差异报错。

### 3.3 代码改动

无代码改动，直接使用镜像内版本。`patches/` 为空目录。

模型权重、`config.json`、`generation_config.json`、tokenizer 与 chat template 均未修改；`env.log` 中记录了这几个文件的 SHA-256。

### 3.4 升级步骤

无需安装与重放，直接进入第 4 节启动服务。

## 4. 服务启动脚本

`env.sh` 中与启动相关的配置：

```bash
export MODEL_PATH="/public/home/maxkonger/SothisAI/model/Aihub/MiMo-V2.6-Distill-Qwen-9B/main/MiMo-V2.6-Distill-Qwen-9B"
export SERVED_NAME="MiMo-V2.6-Distill-Qwen-9B"
export FRAMEWORK="sglang"
export CARD="BW1000"
export PRECISION="bf16"
export PORT="30000"
export TP=1
export EP=1
export PP=1
export MAX_LEN=131072
export MEM_FRAC=0.85
export EXTRA_ARGS="--reasoning-parser mimo --dtype bfloat16 --kv-cache-dtype bfloat16 --max-running-requests 128 --chunked-prefill-size 8192 --max-prefill-tokens 16384 --disable-radix-cache --attention-backend fa3 --page-size 64 --mamba-scheduler-strategy no_buffer --cuda-graph-max-bs 128 --disable-custom-all-reduce --enable-metrics"
```

启动：

```bash
bash run.sh
```

`run.sh` 展开后实际执行的完整命令（`server.log` 首行）：

```bash
sglang serve \
  --model-path /public/home/maxkonger/SothisAI/model/Aihub/MiMo-V2.6-Distill-Qwen-9B/main/MiMo-V2.6-Distill-Qwen-9B \
  --served-model-name MiMo-V2.6-Distill-Qwen-9B \
  --host 0.0.0.0 --port 30000 \
  --tp-size 1 --pp-size 1 \
  --context-length 131072 --mem-fraction-static 0.85 \
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

重点参数说明：

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| `--tp-size` | `1` | 权重 17.64 GB，单卡 64 GB 可容纳，无需切分 |
| `--context-length` | `131072` | 覆盖 16K 输入 + 1K 输出压测；模型 config 上限 262144，压到一半留出 KV 空间 |
| `--mem-fraction-static` | `0.85` | 实测 KV 池 30.19 GB、`max_total_num_tokens=989312`，满足 128 并发 |
| `--attention-backend` | `fa3` | 本卡上实测输出正确；`triton` 在 page size 64 下输出异常，见第 8 节 |
| `--page-size` | `64` | 与 FA3 配套 |
| `--disable-radix-cache` | 开 | 本件按关闭前缀缓存的方式验证与交付；压测为随机输入，该负载下不使用前缀复用 |
| `--mamba-scheduler-strategy` | `no_buffer` | 线性注意力层按需分配状态，128 并发下稳定 |
| `--disable-custom-all-reduce` | 开 | 单卡无通信，关闭自定义 AllReduce 减少初始化开销 |
| `--max-running-requests` | `128` | 与 `--cuda-graph-max-bs 128` 对齐 |
| `--reasoning-parser` | `mimo` | 思考内容经 `reasoning_content` 返回 |

启动耗时（`server.log` 时间戳）：进程起于 15:27:35，权重加载 3.42 s（15:28:12 至 15:28:15），CUDA Graph 捕获 4.06 s，15:28:28 打印 `The server is fired up and ready to roll!`，全程约 53 s。

显存占用（`/get_server_info` 的 `memory_usage`）：权重 17.64 GB，KV cache 30.19 GB，graph 0.56 GB。

`run.sh` 会先检查 `/health`：已有健康服务时直接拒绝重复启动，避免误占端口。多机部署未使用，单机即可完整复现。

## 5. Smoke 测试

```bash
bash smoke.sh
```

`smoke.sh` 依次检查 `/health`、`/v1/models`、`/metrics`、单轮问答、多轮记忆、流式输出，任一接口非 2xx 即失败。单轮与多轮请求均带 `chat_template_kwargs.enable_thinking=true`，并断言单轮返回了思考 token。

`smoke.log` 实测结果：

- `/health` 返回 `200`
- `/v1/models` 返回 `MiMo-V2.6-Distill-Qwen-9B`，`max_model_len=131072`
- `/metrics` 返回 `200`
- 单轮「计算 17 + 25」回复 `42`，`usage.reasoning_tokens=2`，`finish_reason=stop`
- 多轮：先告知「我的名字是小海」，再问「我叫什么名字？」，回复 `小海。`
- 流式：返回 `你好！有什么我可以帮你的吗？`，以 `data: [DONE]` 结束

此外单独验证了图像输入（未写入 `smoke.sh`）：向接口发送一张 224×224 的纯色图片并询问颜色，返回 `red`，说明视觉塔在本次部署下可正常推理。该项属于补充观察，未做图集精度评测。

## 6. 精度测评

精度验证脚本 `verify.py` 按官方基线文件 `baselines/XiaomiMiMo__MiMo-V2.6-Distill-Qwen-9B.json` 写死的配置跑评测，与官方 GPU 基线对比，劣化 ≤2 pt 判 PASS。

```bash
python verify.py --selfcheck
source env.sh && python verify.py
```

`--selfcheck` 实测通过（`selfcheck OK`），判分逻辑未被改动。

**官方评测当前不具备条件。** 官方基线文件与随基线下发的固定子集未取得，`verify.py` 不会用非官方数据顶替，因此直接返回 blocked。`verify_result.json` 原文：

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

| 数据集 | 官方 GPU 基线 | 本次提交 | 劣化 | 容差 | 判定 |
| --- | --- | --- | --- | --- | --- |
| GSM8K | 未下发 | 未测（官方子集缺失） | — | ≤2 pt | BLOCKED |
| MMLU | 未下发 | 未测（官方子集缺失） | — | ≤2 pt | BLOCKED |

### 6.1 公开数据兼容性验证（补充，非官方口径）

在官方数据到位之前，用公开数据跑了两组确定性抽样，参数为 greedy 解码（`temperature=0`）、`max_tokens=1024`、开启思考，样本由固定 seed `20261120` 生成，公开样本与生成脚本在 `accuracy-evidence/`。

| 数据集 | 题数 | 正确 | 准确率 |
| --- | ---: | ---: | ---: |
| GSM8K | 24 | 23 | 95.83% |
| MMLU | 24 | 20 | 83.33% |
| GSM8K | 64 | 57 | 89.06% |
| MMLU | 64 | 54 | 84.38% |

64 题版本中 MMLU 覆盖全部 57 个学科。同一批 64+64 样本复跑一次，两轮总分完全一致，128 道题的判分结果无一处变化，仅 1 条回答文本不同；两轮均无请求失败、超时或解析异常。

结论：服务在本卡上的数值链路可用，判分链路可重复；但由于官方基线与固定子集缺失，本节不构成对 ≤2 pt 门槛的判定，上述数字也不应视为官方成绩。

## 7. 性能参考

`bench.sh` 使用 SGLang 自带压测客户端，`random-ids` 数据集，固定输入输出长度，seed 1；每档两轮，第一轮预热，取第二轮。请求数为并发的 2 倍（至少 8 个）。

| 输入 | 输出 | 并发 | 请求数 |
| --- | --- | --- | --- |
| 4096 | 1024 | 1 / 8 / 64 | 8 / 16 / 128 |
| 16384 | 1024 | 1 / 8 / 64 | 8 / 16 / 128 |

单档命令（`bench.sh` 内部循环执行）：

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

12 次调用全部 `exit 0`（`bench/status.tsv`），无失败档位。

几点观察：单请求解码约 20–21 ms/token，4K 与 16K 输入的 TPOT 接近，说明线性注意力层在长输入下没有明显退化；并发 64 时输出吞吐由 4K 的 47.74 tok/s 提升到 1138.50 tok/s（约 24 倍），但 TPOT 升到 43.67 ms，16K × 64 时 TTFT 达 68.3 s，该档更适合作为吞吐上限参考，不宜作为交互式服务的延迟指标。

- 原始数据：`bench/round1.jsonl`、`bench/round2.jsonl`、`bench/*.log`、`bench/status.tsv`
- 本结果作为该「模型 × 卡型 × 框架 × 精度」的性能基准；后续优化应在同卡型、同框架、同负载下对比

## 8. 注意事项

- **镜像 tag 与平台截图待补**：容器内没有暴露 registry tag，`env.sh`、`env.log` 与本文第 2 节的镜像字段记为 `UNVERIFIED`，区域一项记为未记录。回填方法：在 SCNet 控制台镜像库找到本次使用的镜像，取得 `仓库地址:tag` 后更新 `env.sh` 的 `IMAGE` 并重跑 `bash env_check.sh` 刷新 `env.log`；同时补齐 `platform/` 下的实例创建页、环境自检与模型管理截图。这是当前提交已知的材料缺口，不影响第 4 至 7 节的复现命令与数据。
- **官方精度未验收**：官方基线文件与其固定 GSM8K/MMLU 子集未下发，`verify.py` 返回 blocked，`verify_result.json` 保持 blocked 原文。取得官方文件后放入 `baselines/` 与 `data/` 再跑即可，脚本无需修改。第 6.1 节是公开数据补充验证，不参与官方判定。
- **不支持的负载**：本次未覆盖超过 128 的路由并发、超过 131072 的上下文，以及 TP>1 的切分部署。131072 是部署配置值，模型 config 上限为 262144，未在此卡上验证更长上下文。
- **修改的模型参数**：无。`config.json`、`generation_config.json`、tokenizer、chat template 均未改动。
- **踩坑与规避 1：注意力后端**：先用 `--attention-backend triton --page-size 64` 启动，服务能起且 `/health` 正常，但同一道「12+7」的请求出现重复循环直到打满 `max_tokens`，`content` 为空。换成 `--attention-backend fa3` 后输出正确。结论是无错误码的健康检查不足以证明推理正确，必须带语义断言。
- **前缀缓存的取舍**：本次全程关闭 radix cache，因此第 7 节所有性能数字都是无前缀复用条件下的结果。未做开启前后的 A/B 对照，也就无法给出关闭带来的吞吐或延迟代价；若要多轮长对话复用前缀，需要单独测量再决定是否开启。
- **踩坑与规避 3：重复启动**：`run.sh` 在端口已有健康服务时直接报错退出，避免两个进程争抢同一块卡的显存；复跑前需先停掉旧进程或删除 `server.pid`。
- **压测口径**：第 7 节数字来自 `sglang.bench_serving` 的 `random-ids` 合成负载，未走 chat template，与真实业务 prompt 的分布不同；TTFT/TPOT 会随 prompt 长度分布变化，跨实现对比时应确认客户端与负载一致。
- **其他**：`server.log` 中有 `torchcodec` 缺失的 import 提示与 NUMA balancing 开启的告警。前者只影响未使用的多模态处理器分支，后者是平台内核设置，本次推理未出现对应报错。
