# @ts-dspy/ai-sdk

Run [TS-DSPy](https://github.com/ardada2468/LLMTypeSafe) programs on any
[Vercel AI SDK](https://ai-sdk.dev) language model — every provider the AI SDK
supports, its gateway, and its middleware, without a dedicated ts-dspy package
for each.

```bash
npm install @ts-dspy/core @ts-dspy/ai-sdk ai @ai-sdk/openai
```

```ts
import { openai } from '@ai-sdk/openai';
import { configure } from '@ts-dspy/core';
import { AISDKLM } from '@ts-dspy/ai-sdk';

configure({ lm: new AISDKLM({ model: openai('gpt-5.2') }) });
```

`AISDKLM` implements the full core language-model interface on top of
`generateText` and `streamText`:

- **Chat**, including image parts (URLs, data URIs and base64).
- **Streaming** via `chatStream`, with tool-argument deltas in chunk metadata and
  the final usage, finish reason and tool calls on the last chunk.
- **Tool calling**: core `ToolSpec`s become AI SDK tools _without_ `execute`,
  so the calls come back to ts-dspy and `ReAct` stays in charge of running them.
- **Structured output** through `Output.object` and `jsonSchema()`.
- **Usage**, including cached and reasoning tokens when the provider reports
  them, recorded for per-call usage and pricing.
- **Cancellation** with `signal`, plus `timeout` and `retries` per call.
- **Errors** mapped onto core's taxonomy — `RateLimitError` (with
  `retryAfterMs`), `AuthError`, `ContextLengthError`, `ContentFilterError`,
  `TimeoutError`, `LMError` — from the AI SDK's `APICallError`, unwrapping its
  `RetryError`.

## Capabilities

The AI SDK does not say what a model can do, so the defaults assume a current
frontier chat model: streaming, structured output, tool calling and vision, with
a 128k-token context. Turn off what yours lacks:

```ts
new AISDKLM({
  model: ollama('llama3.2'),
  capabilities: {
    supportsFunctionCalling: false,
    supportsVision: false,
    maxContextLength: 8192,
  },
});
```

Other options: `provider` (the name used in usage keys and pricing; defaults to
the model's own, e.g. `openai`), `maxTokens`, `maxRetries`, `headers`, and
`providerOptions`, all applied to every request.

Requires Node.js 22+ and `ai` 7. Ships ESM and CommonJS (`ai` itself is
ESM-only; the CommonJS build loads it with `require(esm)`, available from Node
22.12).

Full documentation: <https://github.com/ardada2468/LLMTypeSafe#readme>

## License

MIT
