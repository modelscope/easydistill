# T2V/I2V 文生视频蒸馏

EasyDistill 2 支持 T2V（文生视频）与 I2V（图生视频）蒸馏：将种子文本提示（可选地以首帧图像为条件）蒸馏为多模态 SFT 训练数据，其中 assistant 回复是生成的视频。核心流程为：

**种子提示 → 提示词优化（抽取 → 组合） → 教师视频生成 → 视频评估（VLM / omni / VBench） → 质量过滤 → 多模态 SFT 数据集**

输出格式为 `{messages: [{role: user, content: optimized_prompt}, {role: assistant, content: [video_url]}]}`，兼容 LLaMA-Factory 与 ms-swift 多模态训练。I2V 样本的 user 轮中会额外携带条件首帧图像。

## 架构总览

T2V 蒸馏使用**三个后端槽位**，且可跨来源自由混搭（例如 VLM 走 PAI-Token、视频模型走 EAS 自部署）：

| 后端 | 配置键 | 类型 | 用途 |
|------|--------|------|------|
| LLM/VLM 后端 | `backend` | `ModelBackend` | 两段式提示词优化（种子含 I2V 行时需要 VLM） |
| 评估后端 | `eval_backend` | `ModelBackend` | 视频评估（VLM 裁判 / omni 检查器）。缺省时回落到 `backend` |
| T2V 后端 | `t2v_backend` | `T2VBackend` | 视频生成（T2V 与 I2V 两种模式统一在一个后端内） |

带 `first_frame_image` 字段的行以 **I2V 模式**运行，不带的行以普通 **T2V** 运行。两类行可以混在同一批数据中——每个阶段都逐行分流。

## 支持的视频后端

| 后端 | 配置 `type` | 协议 | 说明 |
|------|-------------|------|------|
| PAI-Token 网关视频 | `pai_token_video` | DashScope 风格异步（提交 + 轮询） | 如 `happyhorse-1.1-t2v` / `happyhorse-1.1-i2v` / `wan2.7-t2v` |
| PAI-EAS 自部署视频 | `pai_video` | `legacy`（同步/异步）或 `sglang`（`/v1/videos`） | PAI-EAS 上自部署的视频模型 |

### PAI-Token 视频后端

```yaml
t2v_backend:
  type: pai_token_video
  api_key: ${PAI_TOKEN_API_KEY}
  base_url: ${PAI_TOKEN_BASE_URL}
  model_id: happyhorse-1.1-t2v      # 不带 first_frame_image 的行
  i2v_model_id: happyhorse-1.1-i2v  # 带 first_frame_image 的行
  output_dir: outputs/t2v_videos    # 视频下载到本地的目录
```

网关代理 DashScope 风格的视频合成接口：请求提交异步任务（`X-DashScope-Async: enable`），后端轮询直到返回视频 URL 并下载到 `output_dir`。真机调试中确认的注意事项：

- `generation.duration` 必须是**整数**秒（常见范围 3–15）。
- I2V 首帧通过 `input.media`（`{"type": "first_frame", "url": ...}`）传递；仍使用旧字段的模型可设 `i2v_image_field: img_url`。
- `generation.resolution` 支持的档位因模型而异（如 `wan2.7-t2v` 仅接受 `720P`/`1080P`），不支持的档位在提交时即被 API 拒绝。
- 远端视频 URL **会过期**（通常 24 小时）——务必设置 `output_dir`，让数据行携带持久的本地路径。

### PAI-EAS 视频后端

```yaml
t2v_backend:
  type: pai_video
  endpoint_url: ${EAS_VIDEO_ENDPOINT_URL}
  token: ${EAS_VIDEO_TOKEN}
  protocol: sglang            # 或 legacy
  auth_prefix: ""             # sglang 部署使用裸 token 鉴权
  t2v_task: t2va
  i2v_task: fl2va
  sglang_short_edge: 768
  output_dir: outputs/t2v_videos
```

一个后端类支持两种协议：

- **`legacy`** —— 服务自有 JSON API；同步与异步（任务 ID 轮询）两种响应形态自动识别。
- **`sglang`** —— sglang 服务栈的 videos API：`POST /v1/videos` → 轮询 `GET /v1/videos/{id}` → 下载 `GET /v1/videos/{id}/content`。必须设置 `output_dir`。I2V 行的首帧通过 `reference_url` 传递，必须是 **EAS 服务可访问的 http(s) URL**（本地路径会被拒绝）。

## 数据格式

### 输入：种子提示

输入行为带 `prompt` 字段的 JSON 对象，I2V 行额外携带 `first_frame_image`：

```jsonl
{"id": "1", "prompt": "一只猫在月球表面缓慢行走，扬起细小的尘埃"}
{"id": "4", "prompt": "画面中的场景逐渐入夜，远处雷峰塔的灯光亮起", "first_frame_image": "examples/seed_t2v_first_frame.jpg"}
```

`first_frame_image` 接受本地文件路径、`file://` URL、`http(s)://` URL 与 data URL；本地图片在发送给 VLM 或后端 API 前会归一化为 base64 data URL（sglang 路径除外，它要求 http(s) URL）。

**仅 T2V 行支持行级分辨率控制**：

- `resolution` — 按行覆盖配置的分辨率档位（`480P` / `720P` / `1080P`），或设为 `auto`，由抽取阶段通过 LLM 从提示中推断宽高比并写入 `ratio`。
- `ratio` — 按行覆盖配置的宽高比（如 `16:9`、`9:16`）；`resolution: auto` 推断出的值也写入该字段。

```jsonl
{"id": "2", "prompt": "夕阳下的城市延时摄影，画面拉向远处的天际线", "resolution": "auto"}
{"id": "3", "prompt": "一只橘猫在窗台上打哈欠", "resolution": "480P", "ratio": "4:3"}
```

**I2V 行一律跟随首帧**：这类行上的 `resolution` / `ratio` 会被忽略（记录一条警告）。如需拦截异常首帧，可在 `generation` 下设置 `i2v_frame_check` 为 `off`（默认）| `warn` | `skip` | `raise`，并用 `i2v_frame_min_edge`（最小短边，默认 256）和 `i2v_frame_max_aspect`（最大长宽比，默认 3.0）调参；`skip` 会丢弃该行并记录警告。http(s) 首帧无法在本地检查，一律放行。

覆盖全部四种输入形态（默认、显式 `resolution` + `ratio`、`auto`、I2V）的可运行示例见 `examples/seed_t2v_prompts.jsonl`。

### 输出：多模态 SFT

```jsonl
{
  "messages": [
    {"role": "user", "content": "A cinematic 3D CG render of an adult gray tabby cat ... slowly pushes in with small amplitude."},
    {"role": "assistant", "content": [
      {"type": "video_url", "video_url": {"url": "outputs/t2v_videos/365be208-....mp4"}}
    ]}
  ],
  "metadata": {
    "source": "t2v_distillation",
    "t2v_model": "happyhorse-1.1-t2v",
    "t2v_mode": "t2v",
    "raw_prompt": "一只猫在月球表面缓慢行走，扬起细小的尘埃",
    "request_id": "1",
    "prompt_consistency": 3, "visual_quality": 3, "subject_consistency": 4,
    "motion_quality": 2, "temporal_execution": 2, "camera_accuracy": 4
  }
}
```

I2V 样本采用图像 + 视频消息对：user 轮同时携带首帧图像与提示词，`metadata.t2v_mode` 为 `"i2v"`。评估分数保存在 `metadata` 中，下游训练可以直接重新过滤而无需重跑评估。

## 提示词优化（抽取 → 组合）

第一阶段（**extract**）将种子提示（I2V 行以首帧图像为依据）解析为结构化 JSON 草稿：主体、外观、动作、场景、光照、镜头、时序节拍。第二阶段（**compose**）按**caption schema** 将草稿改写为最终提示词。每行恰好两次模型调用；草稿保留在 `draft` 列中便于检查。

系统内置一份通用的、模型无关的默认 schema。每个视频模型都有厂商文档说明的偏好提示词风格（语言、长度、结构）——**生产运行时，请基于目标模型的官方提示词指南编写 schema** 并配置给该阶段：

```yaml
- stage: prompt_optimize
  config:
    schema_file: path/to/your_model_caption_schema.txt   # 或内联：schema: |
```

## 视频评估

评估是 `eval.checkers` 下的可组合检查器链；每个条目可用 `enabled` 开关，检查器之间故障隔离。评分为 **0–4 分制**，每个维度输出三列：`<dim>`、`<dim>_confidence`、`<dim>_reason`。

| 检查器 | `type` | 评什么 | 依赖 |
|--------|--------|--------|------|
| VLM 裁判 | `vlm` | 稀疏抽帧可见的方面：`prompt_consistency`、`visual_quality`、`subject_consistency`、`first_frame_consistency`（仅 I2V） | `eval_backend` 上的 VLM；本地 OpenCV 抽帧 |
| Omni 检查器 | `omni` | 需要完整视频的动态质量：`motion_quality`、`temporal_execution`、`camera_accuracy` | 视频原生 omni 模型（如 `qwen3.5-omni-plus`）；`video_transport: auto` 按大小自动选 URL 或 base64 |
| VBench | `vbench` | 通过 [VBench](https://github.com/Vchitect/VBench) 计算客观指标（`vbench_*` 列） | 独立 venv 中 `pip install vbench`（见下文）；多数维度需 GPU；环境不满足时优雅跳过并记录 `vbench_skipped_reason` |

维度池定义在 `configs/eval/t2v/vlm_dimensions.yaml` 与 `configs/eval/t2v/omni_dimensions.yaml`（两池刻意不相交）。各来源分数保持原始值，不做跨检查器归一化。

### VBench 环境搭建

VBench 作为**外部工具**使用——仓库中不 vendor 任何 VBench 代码。请将其安装到独立 virtualenv 中（Python ≤ 3.11；其固定版本的依赖与 easydistill 冲突）：

```bash
python3.10 -m venv /path/to/vbench_env
/path/to/vbench_env/bin/pip install vbench "setuptools<81"
```

然后把检查器指向该 CLI：

```yaml
- type: vbench
  enabled: true
  vbench_bin: /path/to/vbench_env/bin/vbench   # 或环境变量 VBENCH_BIN
  dimensions: [motion_smoothness, dynamic_degree, imaging_quality, temporal_flickering]
```

模型 checkpoint 首次使用时自动下载。多数维度需要 GPU（默认 `require_gpu: true`）。也支持从 repo 克隆运行（`vbench_repo` + `python_executable`）；两者同时配置时 `vbench_bin` 优先。环境不可用时，检查器会把原因记入 `vbench_skipped_reason`，不会让流水线失败。

## 高级 T2V 蒸馏流水线

```bash
export PAI_TOKEN_API_KEY=your_key
export PAI_TOKEN_BASE_URL=https://your-endpoint/v1
easydistill --config configs/pipeline/advanced_t2v_distill_pai_token.yaml

# EAS 自部署视频模型（LLM/VLM 仍可走任意后端）：
export EAS_VIDEO_ENDPOINT_URL=https://your-video-service.pai-eas.aliyuncs.com
export EAS_VIDEO_TOKEN=your_token
easydistill --config configs/pipeline/advanced_t2v_distill_pai_eas.yaml
```

流水线阶段：

1. `prompt_optimize` —— 两段式抽取 → 组合提示词增强。
2. `t2v_generate` —— 通过 T2V 后端生成视频（I2V 行使用 `i2v_model_id`）。
3. `t2v_eval` —— 可组合视频评估（VLM / omni / VBench）。
4. `quality_filter` —— 按各指标 `min_scores` 或对 `eval.metrics` 求均值做 top-k/top-ratio 过滤。
5. `build_t2v_sft` —— 转换为多模态 SFT 数据集。

每个阶段把中间产物写入 `pipeline[].output_path`。

### 断点续跑 / 崩溃恢复

视频生成慢且昂贵，因此高开销阶段支持可选的断点续跑：

```yaml
- stage: t2v_generate
  config:
    resume: true
  output_path: outputs/t2v_stage2_generated.jsonl
```

开启 `resume: true` 后，原样重跑同一条命令会从该阶段的 `output_path` 复用已完成的行，只跑缺失/失败的行。`t2v_generate` 还会**每生成完一条视频立即逐行落盘**，因此阶段中途崩溃也不会重新生成已完成的视频。行按 `id` 匹配（缺失时用内容哈希）；上次失败的行（无 `video_urls`）会自动重试。要强制全量重跑（例如修改生成参数后），删除对应阶段的输出文件即可。

## 独立运行的单阶段任务

单阶段 job 复用相同的配置节（由 `job_type` 选择阶段）：`t2v_distill`（种子 → 视频 → SFT，无优化/评估）、`t2v_prompt_optimize`、`t2v_generation`、`t2v_eval`。

```bash
# 基础 T2V 蒸馏（无优化 / 评估）
easydistill --config configs/basic/t2v_distill_pai_token.yaml
```

## 配置参考

| 字段 | 说明 |
|------|------|
| `backend.type` | LLM/VLM 后端：`pai_token`、`pai_eas` 或 `openai` |
| `eval_backend` | 可选的独立评估后端（缺省回落到 `backend`） |
| `t2v_backend.type` | 视频后端：`pai_token_video` 或 `pai_video` |
| `t2v_backend.model_id` / `i2v_model_id` | T2V / I2V 模型 ID |
| `t2v_backend.output_dir` | 视频本地下载目录（强烈建议设置） |
| `t2v_backend.protocol` | 仅 `pai_video`：`legacy` 或 `sglang` |
| `t2v_backend.i2v_image_field` | 仅 `pai_token_video`：`media`（默认）或 `img_url` |
| `generation.resolution` / `ratio` | 分辨率档位（如 `720P`）与宽高比 |
| `generation.duration` | 视频时长秒数（整数） |
| `generation.watermark` | 支持的服务上关闭水印 |
| `generation.max_workers` | 并发生成数（默认 1——单副本 EAS 请保持 1） |
| `prompt_optimize.schema` / `schema_file` | 目标模型的 caption schema（默认使用内置通用版） |
| `eval.checkers` | 检查器链：`vlm` / `omni` / `vbench` 条目，各带 `enabled` 开关 |
| `eval.metrics` | quality_filter 求均值使用的指标（默认：`prompt_consistency`、`visual_quality`、`subject_consistency`） |
| `quality_filter.min_scores` | 各指标最低分（0–4 分制） |
| `sft.skip_empty` / `min_prompt_length` / `max_videos_per_prompt` | SFT 构建选项 |
| `pipeline[].config.resume` | 阶段级断点续跑开关（见上文） |
| `dataset.input_path` / `output_path` | 输入种子 / 最终 SFT 输出 |

## 安装

```bash
pip install -e .          # 核心安装（基于 httpx 的视频后端已包含）
pip install -e ".[all]"   # 全部可选依赖
```

VLM 检查器的抽帧依赖 OpenCV（`opencv-python-headless`），默认依赖中已包含。
