# OpenAI-style text chat API

`script/openai_api/server.py` is an independent Flask adapter around a **dedicated** little-gemma UNIX socket. It implements a text-only subset of [Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create), including streaming. It does not depend on the voice dashboard, Whisper or Piper.

## Start

Start an engine with its own socket, without `-sys` (request messages supply the system prompt):

```sh
../little-gemma/build/run-cuda-i8 -m /path/to/model.gguf -s /tmp/lg-api.sock -think 0
```

In another terminal, from little-gemma-tools:

```sh
python3 -m venv .venv-api
.venv-api/bin/pip install -r script/openai_api/requirements.txt
.venv-api/bin/python script/openai_api/server.py \
  --socket /tmp/lg-api.sock --model little-gemma --port 8082
```

The default bind address is `127.0.0.1`. For LAN access, add `--host 0.0.0.0`. Set `LG_API_KEY` in the adapter environment to require `Authorization: Bearer <key>`; it is optional for local use. The adapter is a development service, with one active request per process and no queue. Use a single process/threaded server: multiple WSGI workers do not share the admission lock. Do not connect it to a socket held by voicecat; the engine serves one connection at a time. It does not launch, stop or reconfigure the engine.

## Call

```sh
curl http://localhost:8082/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"little-gemma","messages":[{"role":"user","content":"What is a prairie?"}]}'
```

With the OpenAI Python client installed in the calling application:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8082/v1", api_key="local")
response = client.chat.completions.create(
    model="little-gemma",
    messages=[
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": "What is a prairie?"},
    ],
)
print(response.choices[0].message.content)

for chunk in client.chat.completions.create(
    model="little-gemma",
    messages=[{"role": "user", "content": "Explain crop rotation."}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

Use the configured key instead of `local` when authentication is enabled.

## Supported behavior and limits

- `GET /v1/models` lists the configured model alias; `POST /v1/chat/completions` returns JSON or SSE with the usual role/content deltas, finish reason and `[DONE]` terminator. `GET /health` reports adapter health, socket-file existence and request occupancy; file existence is not an engine readiness probe.
- Messages accept `system`, `developer`, `user`, `assistant`, and string or text-only content parts. Developer maps to the engine's system role. The last message must be a user message. Each request gets a fresh connection and re-prefills its supplied history; no hidden state is retained between HTTP requests. Include prior assistant replies in later requests to preserve history.
- History uses native Gemma turn markers inside a binary `T` text frame. This preserves newlines without accidentally submitting extra turns. The socket supplies the initial user opener; a leading system/developer message therefore follows an empty user turn. Prior assistant messages are prefilled as supplied, not regenerated. Literal engine delimiters (`<|`, `|>`, NUL) in message text are rejected to keep role framing unambiguous.
- Input is limited to **3,500 UTF-8 bytes including serialized history markers**. This conservative limit stays below the engine's fixed 4,096-token prompt buffer without estimating tokens. `--max-input-bytes` can lower it. It is not a statement about the model's architectural context capacity. The dedicated engine should have no additional `-sys` prefix.
- `stream`, `stop` (up to four strings), and `n=1` are supported. Stop text and protocol/thought-channel markers are removed before delivery, including when they cross socket-read boundaries. Engine generation-cap termination is reported as `finish_reason="length"`.
- Sampling is configured on the engine. Per-request `temperature`, `top_p`, token limits (`max_tokens`, `max_completion_tokens`), tool calls, multimodal inputs, structured outputs, log probabilities, token usage and other endpoints are **not implemented**. Unsupported non-null fields return an explicit error rather than silently being ignored. Exact usage is omitted because the socket does not expose it; `stream_options.include_usage=true` is rejected.
- Busy requests return 429, connection failures 503, truncated replies 502, and non-streaming response timeouts 504. During an already-started SSE response, upstream failure produces an error event and closes without a success finish chunk or `[DONE]`. A disconnected client releases the socket/admission lock. `--timeout` bounds generation (default 120 seconds).

Run socket/protocol tests without a model:

```sh
.venv-api/bin/python -m unittest discover -s tests -p test_openai_api.py
```

These exercise request isolation, role history/newlines, UTF-8 and marker boundaries, SSE, custom stop strings, authentication, invalid input, upstream truncation, concurrent admission and stream cancellation. Real-engine and SDK smoke-test artifacts belong in the protected research repository.
