# @ts-dspy/openai

OpenAI provider for [TS-DSPy](https://github.com/ardada2468/LLMTypeSafe), built on
the official `openai` SDK. Supports streaming, native JSON-schema structured
outputs, and any OpenAI-compatible endpoint via `baseURL`.

```bash
npm install @ts-dspy/core @ts-dspy/openai
```

```ts
import { OpenAILM } from '@ts-dspy/openai';
import { configure } from '@ts-dspy/core';

const lm = new OpenAILM({
  apiKey: process.env.OPENAI_API_KEY,
  // model: 'gpt-5.2',              // defaults to a current model
  // baseURL: 'https://...',        // Azure, a proxy, or a compatible server
  // timeout: 30_000,
  // maxRetries: 2,
});

configure({ lm });
```

Per-call `timeout` and `retries` map onto the SDK's own request options, so
retries honour `retry-after` headers rather than being re-implemented here.
`temperature` and `top_p` are sent only when you set them, which keeps reasoning
models working.

## OpenAI-compatible endpoints

Ollama, LM Studio, vLLM, Groq, Together, and OpenRouter all speak the OpenAI
chat-completions API. `OpenAICompatibleLM` points at them without inheriting the
defaults that are only correct for `api.openai.com`:

```ts
import { OpenAICompatibleLM } from '@ts-dspy/openai';

const lm = new OpenAICompatibleLM({
  baseURL: 'http://localhost:11434/v1', // required
  model: 'llama3.2', // required
  // apiKey defaults to a placeholder for local servers that ignore it
  // supportsStructuredOutput: false,   // default
  // supportsFunctionCalling: false,    // default
  // supportsVision: false,             // default
  // maxContextLength: 8192,            // default
});
```

`supportsStructuredOutput` is the flag that matters. `Predict` branches on it, and
at OpenAI's `true` every call ships a `response_format` of
`{ type: 'json_schema', strict: true }` that most compatible servers reject
outright. Left at `false`, structured output goes through the prompt-based
fallback and the reply is still validated against the signature.

| Endpoint   | `baseURL`                        | Key                  | Capabilities to set                                            |
| ---------- | -------------------------------- | -------------------- | -------------------------------------------------------------- |
| Ollama     | `http://localhost:11434/v1`      | none                 | Defaults; raise `maxContextLength` to match the model.         |
| LM Studio  | `http://localhost:1234/v1`       | none                 | Defaults.                                                      |
| vLLM       | `http://localhost:8000/v1`       | none                 | `supportsStructuredOutput: true` with guided decoding enabled. |
| Groq       | `https://api.groq.com/openai/v1` | `GROQ_API_KEY`       | `supportsFunctionCalling: true` on the tool-capable models.    |
| Together   | `https://api.together.xyz/v1`    | `TOGETHER_API_KEY`   | `supportsStructuredOutput: true` on models listing JSON mode.  |
| OpenRouter | `https://openrouter.ai/api/v1`   | `OPENROUTER_API_KEY` | Per routed model; the safe default is to set nothing.          |

Those URLs are exported as `OPENAI_COMPATIBLE_BASE_URLS`, keyed `ollama`,
`lmstudio`, `vllm`, `groq`, `together`, and `openrouter`.

Unlike `OpenAILM`, `OPENAI_API_KEY` is deliberately not read from the environment:
`baseURL` here is by definition somewhere other than `api.openai.com`, so
forwarding an OpenAI credential to it would be a leak rather than a convenience.
Pass the endpoint's own key as `apiKey`.

Set `supportsStreaming: false` for a server that rejects the streaming request
outright — some older builds and strict proxies refuse the `stream_options` field
the OpenAI SDK sends. `generateStream` then makes one plain call and yields the
whole reply as a single chunk, so callers written against the streaming API keep
working.

Requires Node.js 22+. Ships ESM and CommonJS.

Full documentation: <https://github.com/ardada2468/LLMTypeSafe#readme>

## License

MIT
