# 后端

EasyDistill 2 将所有模型调用委托给 `ModelBackend`。CLI 开箱支持三种后端类型，它们都使用 OpenAI chat-completions 协议：

| 后端类型 | 适用场景 |
|---|---|
| `openai` | 任意 OpenAI 兼容端点：OpenAI API、Azure OpenAI、vLLM、llama.cpp server、Ollama OpenAI 兼容模式等。 |
| `pai_token` | 阿里云 PAI-Token 服务。 |
| `pai_eas` | 阿里云 PAI-EAS 自部署模型。 |

后端通过每个配置文件顶层 `backend:` 下的 `type` 字段选择。

## 通用后端字段

| 字段 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `type` | 是 | `openai` | 后端类型：`openai`、`pai_token` 或 `pai_eas`。 |
| `model_id` | 否 | 后端相关 | 传入聊天端点的模型 ID。若省略，OpenAI/EAS 后端会尝试 list models；PAI-Token 必须显式设置 `model_id` 或 `PAI_TOKEN_MODEL_ID` 环境变量。 |
| `timeout` | 否 | `120.0` | 请求超时时间（秒）。 |
| `max_retries` | 否 | `0` | OpenAI 客户端级别的瞬态失败重试次数。默认由 `TextGenerationOperator` 等算子处理重试，因此通常保持为 `0` 以避免重复重试。 |

## `openai` 后端

`openai` 后端最通用，使用官方 `openai` Python 客户端调用任意 `/v1/chat/completions` 端点。

### 所需凭据

可在配置中设置，也可通过环境变量设置：

| 配置键 | 环境变量 | 说明 |
|---|---|---|
| `api_key` | `OPENAI_API_KEY` | 作为 Bearer token 发送的 API key。 |
| `base_url` | `OPENAI_BASE_URL` | 端点 base URL。若均未设置，默认为 `https://api.openai.com/v1`。 |

### 示例：OpenAI API

```yaml
backend:
  type: openai
  api_key: ${OPENAI_API_KEY}
  base_url: https://api.openai.com/v1
  model_id: gpt-4o-mini
```

### 示例：vLLM 或本地 OpenAI 兼容服务

```yaml
backend:
  type: openai
  api_key: dummy  # vLLM 通常不校验 key
  base_url: http://localhost:8000/v1
  model_id: Qwen/Qwen2.5-7B-Instruct
```

### 示例：Azure OpenAI

```yaml
backend:
  type: openai
  api_key: ${AZURE_OPENAI_API_KEY}
  base_url: https://your-resource.openai.azure.com/v1
  model_id: gpt-4o
```

## `pai_token` 后端

PAI-Token 提供 OpenAI 兼容的聊天补全端点，需要提供 API key 与显式指定的 model_id。

| 配置键 | 环境变量 | 默认值 | 说明 |
|---|---|---|---|
| `api_key` | `PAI_TOKEN_API_KEY` | — | PAI-Token API key。 |
| `base_url` | `PAI_TOKEN_BASE_URL` | `https://cn-beijing.pai-token.aliyuncs.com/v1` | PAI-Token 端点 base URL。 |
| `model_id` | `PAI_TOKEN_MODEL_ID` | — | 调用的模型。必须显式设置，例如 `kimi-k2.6` 或 `qwen2.5-72b-instruct`。 |

```yaml
backend:
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  model_id: qwen2.5-72b-instruct
```

## `pai_eas` 后端

PAI-EAS 托管的自部署模型通过 OpenAI 兼容端点暴露。需要服务端点 URL 与访问 token。

| 配置键 | 环境变量 | 说明 |
|---|---|---|
| `endpoint_url` | `EAS_ENDPOINT_URL` | 服务 URL，例如 `https://<service>-<id>.cn-beijing.pai-eas.aliyuncs.com/v1`。以 `/v1/chat/completions` 结尾的 URL 会自动规范化。 |
| `token` | `EAS_TOKEN` | PAI-EAS 访问 token。 |
| `model_id` | — | 可选模型 ID。多数 EAS 部署会忽略该值。 |

```yaml
backend:
  type: pai_eas
  endpoint_url: https://your-service.cn-beijing.pai-eas.aliyuncs.com/v1
  token: ${EAS_TOKEN}
```

## 健康检查

CLI 启动时会对后端进行轻量级健康检查：

- `openai` 与 `pai_eas`：尝试通过 OpenAI `models.list()` 列出可用模型。
- `pai_token`：检查是否提供了 API key 与 base_url（PAI-Token 不暴露 model list 端点）。

若检查失败，CLI 会在运行任何流水线阶段前报错退出。

## 在配置中选择后端

每个配置文件都有顶层 `backend` 节。同一流水线配置只需复制并修改 `backend` 节即可切换到不同服务商。示例配置默认提供 `pai_token` 与 `pai_eas` 版本；若要使用通用 OpenAI 端点，将 `type` 改为 `openai` 并设置 `api_key`/`base_url` 即可。

## T2I 后端

文生图任务使用独立的后端抽象（`T2IBackend`），通过顶层 `t2i_backend:` 节配置。支持的 T2I 后端列表、配置示例与环境变量说明，请见专门的 T2I 文生图蒸馏指南：[t2i_distillation_zh.md](t2i_distillation_zh.md)。

## T2V 后端

文生/图生视频任务使用独立的后端抽象（`T2VBackend`），通过顶层 `t2v_backend:` 节配置，一个后端同时支持 T2V 与 I2V 两种模式。支持的视频后端（`pai_token_video`、`pai_video`）、协议细节与配置示例，请见专门的 T2V 文生视频蒸馏指南：[t2v_distillation_zh.md](t2v_distillation_zh.md)。
