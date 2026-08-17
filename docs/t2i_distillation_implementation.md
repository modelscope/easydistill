# T2I 蒸馏方法实现文档

> 本文档是 **实现/开发者指南**：描述 easydistill T2I 文生图蒸馏管线的架构设计、
> 模块边界、阶段逻辑、后端协议和扩展指南。可运行的配置示例与字段速查表见使用指南：
> [English](t2i_distillation.md) · [中文](t2i_distillation_zh.md)。

---

## 1. 架构总览

### 1.1 管线全景

T2I 蒸馏管线的核心目标:将种子文本 prompt 转化为多模态 SFT 训练数据
(prompt → image 对),其中图像由教师 T2I 模型生成,经 VLM 评测和质量过滤后
组装为 ShareGPT 格式。

```
seed prompt
    │
    ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ prompt_optimize  │ ──> │  t2i_generate    │ ──> │   t2i_eval      │ ──> │ quality_filter   │ ──> │ build_t2i_sft    │
│ T2IPromptOptim   │     │ T2IGenerationOp  │     │ T2IImageEval    │     │ run_quality_     │     │ T2ISFTBuilder    │
│ (LLM 重写)       │     │ (教师生成图像)    │     │ (VLM 4维评分)   │     │ filter_stage     │     │ (多模态SFT格式)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘     └──────────────────┘     └──────────────────┘
    backend              t2i_backend              eval_backend              (纯函数)                  (纯函数)
    qwen3.7-max          Qwen-Image/Wanx           qwen3.6-plus
```

5 个阶段可选可配,唯一约束:最后阶段必须是 `build_t2i_sft`。

### 1.2 三后端分离设计

管线使用三个独立后端,各自有不同的 API 协议:

| 后端 | 配置键 | 基类 | 用途 | 默认模型 |
|------|--------|------|------|---------|
| 文本模型 | `backend` | `ModelBackend` | prompt 优化(LLM 文本生成) | qwen3.7-max |
| VLM 评测 | `eval_backend` | `ModelBackend` | 图像质量评分(多模态) | qwen3.6-plus |
| T2I 生成 | `t2i_backend` | `T2IBackend` | 教师图像生成 | Qwen-Image / wanx2.1-t2i-turbo |

**设计决策:为什么 T2I 后端单独抽象?**

T2I 模型 API(图像生成)与 chat completions(文本对话)协议完全不同:
- chat: `POST /chat/completions` → `{choices: [{message: {content}}]}`
- T2I: `POST /images/generations` → `{data: [{url}]}` 或异步任务轮询

如果复用 `ModelBackend`,会导致接口污染(图像 URL 无法通过 text 字段传递)。
因此 `T2IBackend` 作为平行抽象存在,operator 代码通过后端类型隔离保持 clean。

`eval_backend` 可选:未配置时复用 `backend`(但推荐分离,因为 VLM 和纯文本模型
可能不是同一个)。

### 1.3 类继承关系

```
BaseDistillationPipeline (ABC)            ← pipeline/base.py
  └── T2IDistillationPipeline             ← pipeline/t2i_distillation.py

PromptGenerationOperator (ABC)            ← operators/prompt_base.py
  └── T2IPromptOptimizer                  ← operators/t2i/prompt_optimizer.py

Operator (ABC)                            ← operators/base.py
  └── T2ISFTBuilder                       ← operators/t2i/sft_builder.py

T2IGenerationOperator (独立类)             ← operators/t2i/t2i_generation.py
  (不继承 PromptGenerationOperator,因为 T2IBackend 协议不同)

LLMJudgeEvaluator (ABC)                   ← eval/base.py
  └── T2IImageEvaluator                   ← eval/t2i.py

T2IBackend (ABC)                          ← backends/t2i_base.py
  ├── PAIDiffusionBackend                 ← backends/pai_diffusion_backend.py
  ├── WanxBackend                         ← backends/wanx_backend.py
  └── QwenImageBackend                    ← backends/qwen_image_backend.py
```

---

## 2. 核心管线

### 2.1 BaseDistillationPipeline(抽象基类)

`pipeline/base.py` 定义了管线的骨架:

- `__init__`: 接收 backend、pipeline_config(阶段列表)、dataset_config 等
- `run()`: 从 `dataset_config["input_path"]` 加载数据,调用 `_run_stages()`
- `run_with_data(data)`: 直接传入数据(测试用)
- `_run_stages(data)`: 逐阶段调度,每阶段调用 `_dispatch_stage` + 保存中间输出
- `_dispatch_stage(stage_name, ...)`: **抽象方法**,子类实现阶段分发

**约束**: 最后阶段必须匹配 `_last_stage`(T2I 管线为 `"build_t2i_sft"`)。

中间输出:每个阶段可通过 `output_path` 保存 JSONL 快照,支持断点续跑。

### 2.2 T2IDistillationPipeline

`pipeline/t2i_distillation.py` 实现五阶段调度:

```python
class T2IDistillationPipeline(BaseDistillationPipeline):
    _last_stage = "build_t2i_sft"
    _default_eval_metrics = ["prompt_consistency", "aesthetic_quality",
                             "detail_richness", "artifact_absence"]
```

构造函数接收三后端 + 四配置:
- `backend`: 文本模型(prompt 优化用)
- `t2i_backend`: T2I 后端(图像生成用)
- `eval_backend`: VLM 后端(评测用,缺省复用 backend)
- `pipeline_config`: 阶段列表
- `generation_config` / `sft_config` / `eval_config`: 全局默认配置

### 2.3 阶段调度机制

`_dispatch_stage` 按 `stage_name` 路由到对应处理函数:

| stage_name | 处理函数 | 所在文件 |
|-----------|---------|---------|
| `prompt_optimize` | `_run_prompt_optimize_stage` | t2i_distillation.py |
| `t2i_generate` | `_run_t2i_generate_stage` | t2i_distillation.py |
| `t2i_eval` | `run_t2i_eval_stage` | pipeline/common.py |
| `quality_filter` | `run_quality_filter_stage` | pipeline/common.py |
| `build_t2i_sft` | `run_build_t2i_sft_stage` | pipeline/common.py |

配置合并:阶段级 config 覆盖全局 config(`{**global, **stage}`),实现
"全局默认 + 阶段覆盖"的配置模式。

---

## 3. 阶段实现

### 3.1 Stage 1: prompt_optimize

**模块**: `T2IPromptOptimizer` (operators/t2i/prompt_optimizer.py)

**继承**: `PromptGenerationOperator[Dict, Dict]`

**输入**: `[{id, prompt}]`
**输出**: `[{id, raw_prompt, optimized_prompt, ...metadata}]`

**方法**: 用 LLM 将简短种子 prompt 重写为详细 T2I prompt。优化模板
(`T2I_PROMPT_OPTIMIZE_PROMPT`)包含 7 条指导原则:
1. 描述主体及其属性
2. 构图、取景、 camera angle
3. 艺术风格/媒介(照片级真实、油画、动漫、3D 渲染)
4. 光照、色彩、氛围
5. 质量增强词(highly detailed, 8K, sharp focus)
6. 控制在 30-80 词
7. 保留原始意图

**关键实现**:
- `_build_requests`: 将每行 prompt 填入模板,构造 `GenerationRequest`
- `_parse_result`: 从 `<answer>...</answer>` 标签提取优化后 prompt
- 模板可自定义: `prompt_template`(内联)或 `prompt_template_file`(文件)
- 并发: 继承 `PromptGenerationOperator` 的 `max_workers` 线程池

**设计决策**: 保留 `raw_prompt` 在输出中,用于后续溯源和 Δ 计算
(IntentDistill 的解释鸿沟度量需要对比 raw_prompt 和 optimized_prompt)。

### 3.2 Stage 2: t2i_generate

**模块**: `T2IGenerationOperator` (operators/t2i/t2i_generation.py)

**继承**: **不继承** `PromptGenerationOperator`(独立类)

**输入**: `[{id, optimized_prompt, ...}]`
**输出**: `[{id, optimized_prompt, image_urls, t2i_model, ...}]`

**设计决策:为什么不继承 PromptGenerationOperator?**

`PromptGenerationOperator` 的抽象假设是"调用 chat completions API 并解析文本响应"。
T2I 生成的 API 协议完全不同(返回图像 URL 而非文本),强行继承会导致大量 override
和空实现。因此 `T2IGenerationOperator` 自包含并发和重试逻辑,直接调用
`T2IBackend.generate_image()`。

**关键实现**:
- `_generate_one(row)`: 单行生成,带指数退避重试
  - 重试条件: `TimeoutError`, `ConnectionError`, `RateLimitError` 等
  - 退避策略: `base * 2^(attempt-1) * (0.5 + random())`,上限 `retry_max_wait`
  - 失败处理: `raise_on_error=True` 抛异常,否则返回 None 跳过
- `run(data)`: 根据 `max_workers` 选择顺序或并发执行
  - 并发: `ThreadPoolExecutor`,保持结果顺序(通过 index 映射)
- `prompt_key`: 可配置从哪個字段读取 prompt(默认 `optimized_prompt`)
- 额外参数透传: `seed`, `negative_prompt`, `infer_steps`, `cfg_scale` 等
  通过 `_extra_kwargs` 传给后端(Qwen-Image 特有参数)

### 3.3 Stage 3: t2i_eval

**模块**: `T2IImageEvaluator` (eval/t2i.py)

**继承**: `LLMJudgeEvaluator`

**输入**: `[{id, optimized_prompt, image_urls}]`
**输出**: `[{id, ..., prompt_consistency, aesthetic_quality, detail_richness, artifact_absence}]`

**方法**: VLM-as-judge,对每张生成图打 4 维分数(0-9 整数)。各维度含义见使用指南中的“评测维度”章节;本文档只关注实现细节。

**关键实现**:
- `_extract_sample`: 从行中提取 prompt(优先 `optimized_prompt`)和首张图 URL
- `_extract_images`: 只评估每行的第一张图(`image_urls[:1]`)
- 评分模板: 每个维度有独立 prompt 模板(`T2I_EVAL_PROMPTS`),
  指示 VLM 输出 `<score>N</score>` 标签
- 可自定义: `eval.prompts_file` 指向自定义 YAML 模板
- 并发: 继承 `LLMJudgeEvaluator` 的线程池

**公共函数** `run_t2i_eval_stage`(pipeline/common.py):
- 构建 eval_samples(提取 prompt + image_urls)
- 调用 evaluator.run()
- 将分数合并回原行(按 id 匹配)

### 3.4 Stage 4: quality_filter

**函数**: `run_quality_filter_stage` (pipeline/common.py)

**输入**: 带 eval 分数的行
**输出**: 过滤后的行子集

**两遍过滤**:

1. **min_scores 阈值过滤**: 每个维度设最低分,低于阈值的行被淘汰
   - `require_all_metrics=True`(默认): 缺少任一维度分数的行也被淘汰
   - 例: `min_scores: {prompt_consistency: 6, aesthetic_quality: 5}`

2. **top-k / top-ratio 选择**: 按多维度均分排序,保留前 K 条或前 N%
   - `keep_top_k: 1000` → 保留均分最高的 1000 条
   - `keep_top_ratio: 0.8` → 保留均分前 80%
   - 均分计算: `compute_average_score(row, eval_metrics)`,跳过 None 值

**设计决策**: 两遍串行——先阈值粗筛,再排序精选。这样 min_scores 淘汰的
低质数据不参与排序,避免"低分但未达阈值"的数据挤占 top-k 名额。

### 3.5 Stage 5: build_t2i_sft

**模块**: `T2ISFTBuilder` (operators/t2i/sft_builder.py)

**继承**: `Operator[List[Dict], List[SFTSample]]`

**输入**: `[{id, optimized_prompt, image_urls, ...eval_scores}]`
**输出**: `[{messages: [...], metadata: {...}}]`(SFTSample.model_dump())

**方法**: 将 prompt-image 对转化为多模态 SFT 训练样本:
- user message: `optimized_prompt`(或 `prompt`)文本
- assistant message: 多模态列表 `[{type: "image_url", image_url: {url: ...}}]`

**配置**:
- `skip_empty`: 跳过无图或空 prompt 的行(默认 True)
- `min_prompt_length`: 最短 prompt 字符数(默认 0)
- `max_images_per_prompt`: 每样本最多图片数(默认 1,只用首图)
- `system_prompt`: 可选系统 prompt

**metadata 保留**: SFT 样本的 metadata 包含溯源信息:
- `source`: "t2i_distillation"
- `t2i_model`: 生成模型名
- `raw_prompt`: 原始种子 prompt
- `request_id`: 原始行 id
- 4 个 eval 分数(如果存在): 用于训练时加权采样

---

## 4. 后端实现

### 4.1 T2IBackend(抽象接口)

`backends/t2i_base.py` 定义 T2I 后端的抽象接口:

```python
class T2IBackend(ABC):
    @abstractmethod
    def generate_image(self, prompt, model_id=None, size="1024*1024",
                       n=1, **kwargs) -> ImageGenerationResult:
        ...

    def health_check(self) -> bool: ...
    def close(self) -> None: ...
    def __enter__(self): return self
    def __exit__(self, ...): self.close()
```

返回 `ImageGenerationResult`,包含 `prompt`, `image_urls`, `model`, `usage`,
`metadata`。支持上下文管理器协议。

### 4.2 PAIDiffusionBackend(PAI-EAS 部署)

`backends/pai_diffusion_backend.py`,支持 **sync + async 双模式自动检测**:

**Sync 模式**(SD/Flux via vLLM):
- `POST /images/generations` 立即返回 `{data: [{url | b64_json}]}`
- OpenAI 兼容协议

**Async 模式**(Qwen-Image on EAS):
- `POST /images/generations` 返回 `{task_id: "..."}`
- 轮询 `GET /tasks/{task_id}/status` 直到 `completed`
- 下载 `GET /tasks/{task_id}/image`

**自动检测**: 响应含 `task_id` → async,否则 sync。一套代码兼容两种协议。

**关键参数**:
- `endpoint_url`: PAI-EAS 端点(含 `/v1` 后缀)
- `token`: 认证令牌
- `auth_prefix`: `"Bearer "`(默认)或 `""`(EAS 原始 token)
- `output_dir`: 异步模式图片保存目录(None 则返回 base64 data URL)
- `poll_interval` / `max_poll_wait`: 异步轮询参数

**透传参数**: `seed`, `negative_prompt`, `infer_steps`, `cfg_scale` 等
Qwen-Image 特有参数通过 `**kwargs` 透传到 API payload。

**健康检查**: 先试 `GET /models`(OpenAI 兼容),失败则检查 base URL
(部分 EAS 部署无 `/models` 端点)。

### 4.3 WanxBackend(通义万相云 API)

`backends/wanx_backend.py`,基于 dashscope SDK。配置示例(如 API key、默认模型)见使用指南;这里只说明实现要点:
- 异步任务: 提交 → 轮询 → 返回 OSS URL
- 默认模型: `wanx2.1-t2i-turbo`

### 4.4 QwenImageBackend(Qwen-Image 云 API)

`backends/qwen_image_backend.py`,基于 dashscope SDK,协议与 Wanx 相同。配置示例见使用指南;实现要点:
- 异步任务: 提交 → 轮询 → 返回 OSS URL
- 默认模型: `qwen-image2.0-pro`
- 可通过 `model_id` 覆盖为其他 Qwen-Image 模型(如 `qwen-image`)

---

## 5. 数据流

### 5.1 数据格式演进

每阶段输出是下阶段输入,字段逐步累加:

```
Stage 0 (种子):
  {id, prompt}

Stage 1 (prompt_optimize):
  {id, raw_prompt, optimized_prompt}

Stage 2 (t2i_generate):
  {id, raw_prompt, optimized_prompt, image_urls, t2i_model, [t2i_usage]}

Stage 3 (t2i_eval):
  {id, raw_prompt, optimized_prompt, image_urls, t2i_model,
   prompt_consistency, aesthetic_quality, detail_richness, artifact_absence}

Stage 4 (quality_filter):
  (Stage 3 的子集,字段不变)

Stage 5 (build_t2i_sft):
  {messages: [
     {role: "user", content: optimized_prompt},
     {role: "assistant", content: [{type: "image_url", image_url: {url: ...}}]}
   ],
   metadata: {source, t2i_model, raw_prompt, request_id, [eval_scores]}}
```

### 5.2 SFT 输出格式

最终输出兼容 LLaMA-Factory 和 ms-swift 多模态训练:

```json
{
  "messages": [
    {"role": "user", "content": "A cat on the moon, cinematic lighting, 8K"},
    {"role": "assistant", "content": [
      {"type": "image_url", "image_url": {"url": "outputs/t2i_images/task_abc.png"}}
    ]}
  ],
  "metadata": {
    "source": "t2i_distillation",
    "t2i_model": "Qwen-Image",
    "raw_prompt": "一只在月球上的猫",
    "request_id": "1",
    "prompt_consistency": 8,
    "aesthetic_quality": 7,
    "detail_richness": 6,
    "artifact_absence": 8
  }
}
```

---

## 6. 配置体系

> 本节展示完整管线的 YAML 结构与阶段覆盖机制。字段含义速查和可运行示例见使用指南。

### 6.1 YAML 配置结构

完整管线配置(以 `advanced_t2i_distill_pai_diffusion.yaml` 为例):

```yaml
job_type: advanced_t2i_distill    # CLI 分发键

backend:                             # 文本模型(prompt优化)
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  model_id: qwen3.7-max

eval_backend:                        # VLM(评测,可选)
  type: pai_token
  model_id: qwen3.6-plus

t2i_backend:                         # T2I后端(图像生成)
  type: pai_diffusion
  endpoint_url: ${EAS_ENDPOINT_URL}
  token: ${EAS_TOKEN}
  model_id: Qwen-Image

generation:                          # 全局T2I默认参数
  size: "1328*1328"
  n: 1
  max_workers: 4
  seed: 42
  infer_steps: 50
  cfg_scale: 4

eval:                                # 全局评测默认参数
  metrics: [prompt_consistency, aesthetic_quality, detail_richness, artifact_absence]
  temperature: 0.0

sft:                                 # 全局SFT默认参数
  skip_empty: true
  min_prompt_length: 5

pipeline:                            # 阶段列表(有序)
  - stage: prompt_optimize
    config: {temperature: 0.7, max_tokens: 1024}
    output_path: outputs/stage1.jsonl
  - stage: t2i_generate
    config: {max_workers: 3}
    output_path: outputs/stage2.jsonl
  - stage: t2i_eval
    output_path: outputs/stage3.jsonl
  - stage: quality_filter
    config:
      min_scores: {prompt_consistency: 6, aesthetic_quality: 5}
      keep_top_ratio: 0.8
    output_path: outputs/stage4.jsonl
  - stage: build_t2i_sft
    config: {}

dataset:
  input_path: examples/seed_t2i_prompts.jsonl
  output_path: outputs/t2i_sft_final.jsonl
  prompt_key: prompt
```

### 6.2 配置模板

`configs/t2i/` 下 10 个模板:

| 配置 | 后端 | 阶段数 | 用途 |
|------|------|--------|------|
| `prompt_optimize_pai_token.yaml` | —(文本 LLM) | 1 (optimize) | 仅 prompt 优化 |
| `t2i_generation_wanx.yaml` | Wanx | 1 (generate) | 仅生成图片 |
| `t2i_generation_qwen_image.yaml` | Qwen-Image | 1 (generate) | 仅生成图片 |
| `t2i_distill_wanx.yaml` | Wanx | 2 (generate→SFT) | 最简流程 |
| `t2i_distill_qwen_image.yaml` | Qwen-Image | 2 (generate→SFT) | 最简流程 |
| `t2i_distill_pai_diffusion.yaml` | PAI-Diffusion | 2 | 最简流程 |
| `advanced_t2i_distill_wanx.yaml` | Wanx | 5 (全) | 完整管线 |
| `advanced_t2i_distill_qwen_image.yaml` | Qwen-Image | 5 (全) | 完整管线 |
| `advanced_t2i_distill_pai_diffusion.yaml` | PAI-Diffusion | 5 (全) | 完整管线 |

---

## 7. CLI 入口

> 本节说明 CLI 如何按 `job_type` 分发到 runner。各 job 的可运行 YAML 示例见使用指南的“独立操作”与“高级 T2I 蒸馏流水线”章节。

### 7.1 job_type 分发

`easydistill --config <yaml>` 根据 `job_type` 字段分发到对应 runner:

| job_type | runner 函数 | 说明 |
|----------|------------|------|
| `t2i_distill` | `run_t2i_distill` | 基础流程:种子→生成→SFT |
| `prompt_optimize` | `run_prompt_optimize` | 单阶段:仅 prompt 优化 |
| `t2i_generation` | `run_t2i_generation` | 单阶段:仅图像生成 |
| `t2i_eval` | `run_t2i_eval` | 单阶段:仅 VLM 评测 |
| `advanced_t2i_distill` | `run_advanced_t2i_distill` | 完整 5 阶段管线 |

### 7.2 runner 实现

单阶段 runner(`run_prompt_optimize` / `run_t2i_generation` / `run_t2i_eval`):
- 加载配置 → 构建对应后端 → 调用单个 operator → 保存结果

完整管线 runner(`run_advanced_t2i_distill`):
- 加载配置 → 构建三后端(backend + t2i_backend + eval_backend)
- 构造 `T2IDistillationPipeline` → 调用 `pipeline.run()`

基础 runner(`run_t2i_distill`):
- 加载配置 → 构建 t2i_backend → T2IGenerationOperator → T2ISFTBuilder
- 不走管线调度,直接串联两步

---

## 8. 扩展指南

### 8.1 新增 T2I 后端

1. 继承 `T2IBackend`,实现 `generate_image()`:
   ```python
   class MyT2IBackend(T2IBackend):
       def generate_image(self, prompt, model_id=None, size="1024*1024",
                          n=1, **kwargs) -> ImageGenerationResult:
           # 调用你的 API
           return ImageGenerationResult(prompt=prompt, image_urls=[...], model=...)
   ```
2. 在 `cli/backend_factory.py` 的 `build_t2i_backend()` 注册新类型
3. YAML 配置 `t2i_backend.type: my_t2i`

`T2IGenerationOperator` 无需修改——它通过 `T2IBackend` 接口操作,
与新后端的具体 API 无关。

### 8.2 新增评测维度

1. 在 `configs/prompts/t2i_eval_prompts.yaml` 添加新维度模板:
   ```yaml
   text_rendering: |
     You are an expert judge evaluating text rendering...
     <score>{N}</score>
     Text Prompt: {instruction}
     Generated Image: {output}
   ```
2. YAML 配置 `eval.metrics` 加入 `text_rendering`
3. `quality_filter` 的 `min_scores` 可设新维度阈值

`T2IImageEvaluator` 会自动加载新维度模板并评分——评测器是 metric-driven 的,
新增维度无需改代码。

### 8.3 新增管线阶段

1. 在 `pipeline/common.py` 实现阶段处理函数:
   ```python
   def run_my_stage(backend, stage_config, data, eval_metrics):
       # 处理 data,返回新 data
       return new_data
   ```
2. 在 `T2IDistillationPipeline._dispatch_stage` 注册:
   ```python
   elif stage_name == "my_stage":
       data = run_my_stage(self.backend, stage_config, data, eval_metrics)
   ```
3. YAML `pipeline` 列表加入 `{stage: my_stage, config: {...}}`

阶段间数据通过 JSONL 中间输出衔接,支持断点续跑。

---

## 附录:文件索引

| 文件 | 职责 |
|------|------|
| `pipeline/base.py` | 管线抽象基类 |
| `pipeline/t2i_distillation.py` | T2I 五阶段管线 |
| `pipeline/common.py` | 共享阶段函数(eval/filter/SFT) |
| `operators/t2i/prompt_optimizer.py` | T2I prompt 优化 |
| `operators/t2i/t2i_generation.py` | T2I 图像生成 |
| `operators/t2i/sft_builder.py` | 多模态 SFT 构造 |
| `eval/t2i.py` | VLM-as-judge 评测 |
| `backends/t2i_base.py` | T2I 后端抽象 |
| `backends/pai_diffusion_backend.py` | PAI-EAS 后端(sync+async) |
| `backends/wanx_backend.py` | 通义万相后端 |
| `backends/qwen_image_backend.py` | Qwen-Image 后端 |
| `cli/runners/t2i.py` | CLI runner(5 个入口) |
| `prompts.py` | T2I 优化/评测 prompt 模板 |
| `configs/t2i/*.yaml` | 10 个配置模板 |
| `configs/prompts/t2i_eval_prompts.yaml` | 评测维度模板 |
