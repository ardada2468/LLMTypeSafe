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

Requires Node.js 22+. Ships ESM and CommonJS.

Full documentation: <https://github.com/ardada2468/LLMTypeSafe#readme>

## License

MIT
