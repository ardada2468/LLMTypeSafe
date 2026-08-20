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

Requires Node.js 22+. Ships ESM and CommonJS.

Full documentation: <https://github.com/ardada2468/LLMTypeSafe#readme>

## License

MIT
