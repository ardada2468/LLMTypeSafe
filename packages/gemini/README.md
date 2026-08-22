# @ts-dspy/gemini

Google Gemini provider for [TS-DSPy](https://github.com/ardada2468/LLMTypeSafe),
built on the current `@google/genai` SDK. Supports streaming, native
`responseSchema` structured outputs, and both the Gemini API and Vertex AI.

```bash
npm install @ts-dspy/core @ts-dspy/gemini
```

```ts
import { GeminiLM } from '@ts-dspy/gemini';
import { configure } from '@ts-dspy/core';

const lm = new GeminiLM({
  apiKey: process.env.GEMINI_API_KEY,
  // model: 'gemini-3.5-flash',   // defaults to a current model
});

configure({ lm });
```

For Vertex AI, authenticate with Application Default Credentials and pass the
project instead of an API key:

```ts
const lm = new GeminiLM({ vertexai: true, project: 'my-project', location: 'us-central1' });
```

Safety settings default to `BLOCK_MEDIUM_AND_ABOVE` across all four harm
categories; override with `safetySettings`. A blocked prompt raises an error
naming the block reason rather than returning empty text.

### Timeouts, retries, and cancellation

`timeout` and `maxRetries` are accepted at construction, matching the OpenAI and
Anthropic providers:

```ts
const lm = new GeminiLM({ apiKey, timeout: 20_000, maxRetries: 2 });
```

Per-call `timeout`, `retries`, and `signal` override those defaults:

```ts
const controller = new AbortController();
await lm.generate('...', { signal: controller.signal, timeout: 30_000, retries: 1 });
```

Gemini has a single `abortSignal` slot, so a caller-supplied signal and a timeout
are combined with `AbortSignal.any()` for each request — whichever fires first
ends the call.

Retries are the one place this provider does more than pass options along.
`@google/genai` reads its retry policy from client-level options only, so a
per-call `retries` cannot be expressed through it; its wrapper also replaces API
errors with generic ones (losing the status code) and keeps retrying after an
abort. This provider therefore runs the loop itself, retrying 408/409/429/5xx and
transport failures with exponential backoff, and never retrying an abort or a
client error. `maxRetries` defaults to 2, matching the other two SDKs, and
`timeout` bounds each attempt rather than the whole sequence — again as they do.

Requires Node.js 22+. Ships ESM and CommonJS.

Full documentation: <https://github.com/ardada2468/LLMTypeSafe#readme>

## License

MIT
