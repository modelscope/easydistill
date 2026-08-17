# Backends

EasyDistill 2 delegates all model calls to a `ModelBackend`. The CLI supports three backend types out of the box, all of which speak the OpenAI chat-completions protocol:

| Backend type | Use case |
|---|---|
| `openai` | Any OpenAI-compatible endpoint: OpenAI API, Azure OpenAI, vLLM, llama.cpp server, Ollama in OpenAI compatibility mode, etc. |
| `pai_token` | Alibaba Cloud PAI-Token service. |
| `pai_eas` | Self-deployed models on Alibaba Cloud PAI-EAS. |

The backend is selected by the `type` field under the top-level `backend:` section of every config.

## Common backend fields

| Field | Required | Default | Description |
|---|---|---|---|
| `type` | Yes | `openai` | Backend type: `openai`, `pai_token`, or `pai_eas`. |
| `model_id` | No | Backend-specific | Model ID passed to the chat endpoint. If omitted, OpenAI/EAS backends try to list models; PAI-Token requires an explicit `model_id` or `PAI_TOKEN_MODEL_ID` env var. |
| `timeout` | No | `120.0` | Request timeout in seconds. |
| `max_retries` | No | `0` | Number of OpenAI client-level retries on transient failures. Operators such as `TextGenerationOperator` handle retries by default, so this is usually left at `0` to avoid compounding. |

## `openai` backend

The `openai` backend is the most generic. It uses the official `openai` Python client to call any `/v1/chat/completions` endpoint.

### Required credentials

Either set in the config or via environment variables:

| Config key | Environment variable | Description |
|---|---|---|
| `api_key` | `OPENAI_API_KEY` | API key sent as a bearer token. |
| `base_url` | `OPENAI_BASE_URL` | Base URL of the endpoint. Defaults to `https://api.openai.com/v1` if neither is set. |

### Example: OpenAI API

```yaml
backend:
  type: openai
  api_key: ${OPENAI_API_KEY}
  base_url: https://api.openai.com/v1
  model_id: gpt-4o-mini
```

### Example: vLLM or local OpenAI-compatible server

```yaml
backend:
  type: openai
  api_key: dummy  # vLLM usually does not check the key
  base_url: http://localhost:8000/v1
  model_id: Qwen/Qwen2.5-7B-Instruct
```

### Example: Azure OpenAI

```yaml
backend:
  type: openai
  api_key: ${AZURE_OPENAI_API_KEY}
  base_url: https://your-resource.openai.azure.com/v1
  model_id: gpt-4o
```

## `pai_token` backend

PAI-Token exposes an OpenAI-compatible chat completion endpoint. It requires an API key and an explicit model ID.

| Config key | Environment variable | Default | Description |
|---|---|---|---|
| `api_key` | `PAI_TOKEN_API_KEY` | — | PAI-Token API key. |
| `base_url` | `PAI_TOKEN_BASE_URL` | `https://cn-beijing.pai-token.aliyuncs.com/v1` | PAI-Token endpoint base URL. |
| `model_id` | `PAI_TOKEN_MODEL_ID` | — | Model to call. Must be set explicitly, e.g. `kimi-k2.6` or `qwen2.5-72b-instruct`. |

```yaml
backend:
  type: pai_token
  api_key: ${PAI_TOKEN_API_KEY}
  model_id: qwen2.5-72b-instruct
```

## `pai_eas` backend

PAI-EAS hosts self-deployed models behind an OpenAI-compatible endpoint. It requires the service endpoint URL and an access token.

| Config key | Environment variable | Description |
|---|---|---|
| `endpoint_url` | `EAS_ENDPOINT_URL` | Service URL, e.g. `https://<service>-<id>.cn-beijing.pai-eas.aliyuncs.com/v1`. URLs ending in `/v1/chat/completions` are normalized automatically. |
| `token` | `EAS_TOKEN` | PAI-EAS access token. |
| `model_id` | — | Optional model ID. Many EAS deployments ignore this value. |

```yaml
backend:
  type: pai_eas
  endpoint_url: https://your-service.cn-beijing.pai-eas.aliyuncs.com/v1
  token: ${EAS_TOKEN}
```

## Health checks

When the CLI starts, it runs a lightweight health check on the backend:

- `openai` and `pai_eas`: try to list available models via the OpenAI `models.list()` call.
- `pai_token`: checks that an API key and base URL are present (PAI-Token does not expose the model list endpoint).

If the check fails, the CLI exits with an error before running any pipeline stage.

## Choosing a backend in a config

Every config file has a top-level `backend` section. The same pipeline config can be copied and only the `backend` section changed to target a different provider. Example configs are provided for `pai_token` and `pai_eas`; to use a generic OpenAI endpoint, change `type` to `openai` and set `api_key`/`base_url` accordingly.

## T2I backends

Text-to-image jobs use a separate backend abstraction (`T2IBackend`) configured under the top-level `t2i_backend:` section. For a list of supported T2I backends, configuration examples, and environment variables, see the dedicated T2I distillation guide: [t2i_distillation.md](t2i_distillation.md).

## T2V backends

Text/image-to-video jobs use a separate backend abstraction (`T2VBackend`) configured under the top-level `t2v_backend:` section, supporting both T2V and I2V modes in one backend. For supported video backends (`pai_token_video`, `pai_video`), protocol details, and configuration examples, see the dedicated T2V distillation guide: [t2v_distillation.md](t2v_distillation.md).
