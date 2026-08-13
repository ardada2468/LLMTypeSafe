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

Full documentation: <https://github.com/ardada2468/LLMTypeSafe#readme>

## License

MIT
