# @ts-dspy/core

Signatures, modules, and runtime validation for building type-safe LLM
applications in TypeScript. This package has no provider of its own — pair it
with [`@ts-dspy/openai`](https://www.npmjs.com/package/@ts-dspy/openai),
[`@ts-dspy/gemini`](https://www.npmjs.com/package/@ts-dspy/gemini), or
[`@ts-dspy/anthropic`](https://www.npmjs.com/package/@ts-dspy/anthropic).

```bash
npm install @ts-dspy/core @ts-dspy/openai
```

```ts
import { Signature, InputField, OutputField, Predict, configure } from '@ts-dspy/core';
import { OpenAILM } from '@ts-dspy/openai';

class AnswerQuestion extends Signature {
  @InputField({ description: 'the question' })
  question!: string;

  @OutputField({ description: 'a concise answer' })
  answer!: string;

  @OutputField({ description: 'confidence 0-1', type: 'number' })
  confidence!: number;
}

configure({ lm: new OpenAILM({ apiKey: process.env.OPENAI_API_KEY }) });

const result = await new Predict(AnswerQuestion).forward({
  question: 'What is the capital of France?',
});
// result.confidence is a number, validated at runtime
```

Class signatures require `experimentalDecorators` in your `tsconfig.json`.
Requires Node.js 22+. Ships ESM and CommonJS.

## Testing

Test doubles ship with the package, under `@ts-dspy/core/testing`. Neither
touches the network, so a suite needs no API key.

```ts
import { MockLM, CassetteLM } from '@ts-dspy/core/testing';

// Scripted replies, and a record of every call.
const lm = new MockLM({ responses: ['answer: Paris\nconfidence: 0.95'] });
const result = await new Predict(AnswerQuestion, lm).forward({ question: 'Capital?' });
expect(lm.lastPrompt()).toContain('Capital?');

// Or replay a cassette recorded once against a real provider.
const replayed = new CassetteLM({ path: 'cassettes/answer-question.json' });
```

Record a cassette by wrapping a live model once, then commit the JSON file:

```ts
const recorder = CassetteLM.record('cassettes/answer-question.json', new OpenAILM({ apiKey }));
await new Predict(AnswerQuestion, recorder).forward({ question: 'Capital?' });
```

A cassette is an array of `{ key, request, response }` entries keyed by a hash
of the request, so a changed prompt shows up as a readable diff. `mode: 'auto'`
replays what the file has and records the rest.

Full documentation: <https://github.com/ardada2468/LLMTypeSafe#readme>

## License

MIT
