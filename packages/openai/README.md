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

## Options

| Option            | Default                     | Notes                                                                 |
| ----------------- | --------------------------- | --------------------------------------------------------------------- |
| `apiKey`          | `OPENAI_API_KEY`            |                                                                       |
| `model`           | `DEFAULT_OPENAI_MODEL`      | Per call: `options.model`.                                            |
| `baseURL`         | `https://api.openai.com/v1` | Azure, a proxy, or a compatible server.                               |
| `organization`    | —                           |                                                                       |
| `project`         | —                           |                                                                       |
| `timeout`         | SDK default (10 min)        | Milliseconds. Per call: `options.timeout`.                            |
| `maxRetries`      | `2`                         | Retried inside the SDK, honouring `retry-after`. Per call: `retries`. |
| `reasoningEffort` | unset                       | `'none'`–`'max'`; sent as `reasoning_effort` only when set.           |

`temperature` and `top_p` are sent only when you set them, which keeps reasoning
models working; `maxTokens` is sent as `max_completion_tokens`.

## Reasoning effort

Set a default on the constructor and override it per call through
`providerOptions.openai`:

```ts
const lm = new OpenAILM({ model: 'gpt-5.2', reasoningEffort: 'low' });

await predict.forward(inputs, {
  providerOptions: { openai: { reasoningEffort: 'high' } },
});
```

Any other key under `providerOptions.openai` is merged into the Chat Completions
request body unchanged (`{ openai: { service_tier: 'flex' } }`), as an escape
hatch for parameters ts-dspy does not model. Leave `reasoningEffort` unset for
models without reasoning; they reject the parameter.

Servers that return reasoning text (`reasoning_content` from DeepSeek or vLLM,
`reasoning` from OpenRouter) have it surfaced as `ChatResult.reasoning` and, on
streams, as `metadata.reasoningDelta` chunks, never mixed into the answer.

## Usage, prompt caching and reasoning tokens

OpenAI caches long prompt prefixes automatically. `lm.getUsage()` (and per-call
usage) reports `cachedTokens` from `prompt_tokens_details.cached_tokens` and
`reasoningTokens` from `completion_tokens_details.reasoning_tokens`, for both
streaming and non-streaming calls. Both are subsets of the prompt and completion
totals.

## Embeddings

```ts
import { OpenAIEmbedder } from '@ts-dspy/openai';
import { cosineSimilarity } from '@ts-dspy/core';

const embedder = new OpenAIEmbedder({
  // model: 'text-embedding-3-small', // default
  // dimensions: 512,                  // shorten vectors (text-embedding-3-*)
  // batchSize: 512,                   // inputs per request, capped at 2048
});
const [a, b] = await embedder.embed(['refund policy', 'returns within 30 days']);
cosineSimilarity(a, b);
```

`OpenAIEmbedder` implements core's `Embedder`: inputs are batched, results come
back in input order, `embed(texts, { signal })` honours cancellation, and
failures raise the same `RateLimitError`/`AuthError`/… as `OpenAILM`. Pass
`client` to share an existing `OpenAI` instance.

## Retries

The SDK retries 408/409/429/5xx and connection errors itself (`maxRetries`,
default 2). For a policy that spans providers — or fallback to another model —
use core's middleware and turn the SDK's off:

```ts
import { wrapLM, retry } from '@ts-dspy/core';

const lm = wrapLM(new OpenAILM({ maxRetries: 0 }), retry({ maxRetries: 4 }));
```

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
