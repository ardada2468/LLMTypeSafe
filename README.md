# TS-DSPy

[![CI](https://github.com/ardada2468/LLMTypeSafe/actions/workflows/ci.yml/badge.svg)](https://github.com/ardada2468/LLMTypeSafe/actions/workflows/ci.yml)
[![npm](https://img.shields.io/npm/v/@ts-dspy/core.svg)](https://www.npmjs.com/package/@ts-dspy/core)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Build LLM applications in TypeScript by declaring the shape of the input and
output you want, instead of hand-writing prompts and parsing replies. Inspired by
[DSPy](https://github.com/stanfordnlp/dspy).

Model output is **validated at runtime** against the shape you declared. If a
field you declared as a number comes back as prose, you get a `ValidationError`
naming the field — not a string masquerading as a number.

```bash
npm install @ts-dspy/core @ts-dspy/openai
```

```ts
import { Signature, InputField, OutputField, Predict, configure } from '@ts-dspy/core';
import { OpenAILM } from '@ts-dspy/openai';

class AnswerQuestion extends Signature {
  static description = 'Answer a factual question concisely.';

  @InputField({ description: 'the question to answer' })
  question!: string;

  @OutputField({ description: 'a concise answer' })
  answer!: string;

  @OutputField({ description: 'confidence between 0 and 1', type: 'number' })
  confidence!: number;
}

configure({ lm: new OpenAILM({ apiKey: process.env.OPENAI_API_KEY }) });

const result = await new Predict(AnswerQuestion).forward({
  question: 'What is the capital of France?',
});

console.log(result.answer); // "Paris"
console.log(result.confidence); // 0.98 — a number, verified at runtime
```

## Packages

| Package                                    | Provider                                                     |
| ------------------------------------------ | ------------------------------------------------------------ |
| [`@ts-dspy/core`](packages/core)           | Signatures, modules, validation. No provider.                |
| [`@ts-dspy/openai`](packages/openai)       | OpenAI, via the official `openai` SDK                        |
| [`@ts-dspy/gemini`](packages/gemini)       | Google Gemini, via `@google/genai` (Gemini API or Vertex AI) |
| [`@ts-dspy/anthropic`](packages/anthropic) | Anthropic Claude, via `@anthropic-ai/sdk`                    |

Install core plus whichever providers you use. Each provider defaults to a current
model for that vendor; pass `model` to pin one yourself.

Requires Node.js 22 or newer. Packages ship both ESM and CommonJS builds.

## Concepts

### Signatures

A signature declares a task's inputs and outputs. Use a class with decorators when
you want descriptions and types:

```ts
class AnalyzeReview extends Signature {
  static description = 'Analyze a product review.';

  @InputField({ description: 'the review text' })
  review!: string;

  @OutputField({ description: 'positive, negative, or neutral' })
  sentiment!: string;

  @OutputField({ description: 'rating from 1 to 5', type: 'int' })
  rating!: number;

  @OutputField({ description: 'key themes', type: 'string[]' })
  themes!: string[];

  @OutputField({ description: 'follow-up question', required: false })
  followUp?: string;
}
```

Or a string, for quick work: `'question -> answer: string, confidence: float'`.

Field types: `string` (default), `number`/`float`, `int`/`integer`,
`boolean`/`bool`, `string[]`, `number[]`, `array`/`list`, `object`/`json`.
Set `required: false` to make a field optional.

Class signatures need `experimentalDecorators` in your `tsconfig.json`.

### Modules

- **`Predict`** — one call, validated against the signature.
- **`ChainOfThought`** — reasons in free text first, then answers; the result adds a `reasoning` field.
- **`RespAct`** — a reason-and-act loop that calls the tools you provide until it can answer.

All three accept per-call options that are passed through to the provider SDK:

```ts
await predict.forward({ question: '...' }, { temperature: 0, timeout: 30_000, retries: 2 });
```

When a provider supports native structured output, `Predict` and `ChainOfThought`
use it — the model is constrained to your schema rather than merely asked for it —
and fall back to parsing labelled text otherwise.

### Validation

```ts
import { ValidationError } from '@ts-dspy/core';

try {
  const result = await predict.forward({ question: '...' });
} catch (error) {
  if (error instanceof ValidationError) {
    for (const issue of error.issues) {
      console.error(`${issue.field} (${issue.expected}): ${issue.message}`);
    }
    console.error('raw model output:', error.rawOutput);
  }
}
```

Coercion is deliberately lenient — models emit text, so `"42"` satisfies a
`number` field and `"a, b, c"` satisfies a `string[]`. What is not lenient is
failure: anything that cannot be coerced throws rather than silently passing
through.

### Output types

Decorators record fields at runtime, so TypeScript cannot infer per-field types
from the class. Results are therefore typed loosely by default. Name the shape
when you want precise types:

```ts
type ReviewAnalysis = { sentiment: string; rating: number; themes: string[] };

const analysis = await new Predict<typeof AnalyzeReview, ReviewAnalysis>(AnalyzeReview).forward(
  { review }
);

analysis.themes.join(', '); // typed as string[]
```

Runtime validation comes from the signature either way.

### Tools

```ts
const agent = new RespAct(AnswerQuestion, {
  tools: {
    search: {
      description: 'Search the web. Input: a query string. Returns snippets.',
      function: async (query: string) => search(query),
    },
  },
  maxSteps: 8,
  onEvent: (event) => console.log(event),
});
```

Tool descriptions are what the model uses to decide when to call each tool, so
they earn the detail. Never pass model output to `eval()` — see
[`examples/utils.ts`](examples/utils.ts) for a bounded arithmetic evaluator.

## Examples

```bash
git clone https://github.com/ardada2468/LLMTypeSafe.git
cd LLMTypeSafe
npm install
npm run build

export OPENAI_API_KEY="sk-..."
npm run example:openai
```

See [`examples/`](examples) for OpenAI, Gemini, Anthropic, and tool-use programs.

## Development

```bash
npm install
npm run build       # tsup, per package
npm test            # vitest
npm run lint
npm run typecheck
```

Releases run on [changesets](https://github.com/changesets/changesets): add one
with `npx changeset` when you change a published package.

## License

MIT — see [LICENSE](LICENSE).
