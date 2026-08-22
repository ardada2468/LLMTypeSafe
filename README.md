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

## Downloads

[![total downloads](https://img.shields.io/npm/dt/@ts-dspy/core?label=total%20downloads&color=2a78d6)](https://www.npmjs.com/package/@ts-dspy/core)
[![@ts-dspy/core](https://img.shields.io/npm/dm/@ts-dspy/core?label=%40ts-dspy%2Fcore&color=2a78d6)](https://www.npmjs.com/package/@ts-dspy/core)
[![@ts-dspy/openai](https://img.shields.io/npm/dm/@ts-dspy/openai?label=%40ts-dspy%2Fopenai&color=2a78d6)](https://www.npmjs.com/package/@ts-dspy/openai)
[![@ts-dspy/gemini](https://img.shields.io/npm/dm/@ts-dspy/gemini?label=%40ts-dspy%2Fgemini&color=2a78d6)](https://www.npmjs.com/package/@ts-dspy/gemini)
[![@ts-dspy/anthropic](https://img.shields.io/npm/dm/@ts-dspy/anthropic?label=%40ts-dspy%2Fanthropic&color=2a78d6)](https://www.npmjs.com/package/@ts-dspy/anthropic)

<!-- <img src="assets/npm-downloads.svg" alt="Monthly npm downloads for @ts-dspy/core" width="840"> -->

The chart is regenerated from the
[public npm downloads API](https://github.com/npm/registry/blob/main/docs/download-counts.md)
on the 2nd of each month by
[`.github/workflows/npm-downloads-chart.yml`](.github/workflows/npm-downloads-chart.yml).
To refresh it by hand: `npm run chart:downloads`.

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
`boolean`/`bool`, `string[]`, `number[]`, `array`/`list`, `object`/`json`, `enum`.
Set `required: false` to make a field optional.

An `enum` field pins the answer to a closed set, so the model cannot invent a
fourth value that still passes validation:

```ts
@OutputField({ description: 'overall sentiment', type: 'enum', values: ['positive', 'negative', 'neutral'] })
sentiment!: string;
```

String signatures declare the same thing inline, pipe-separated:
`'review -> sentiment: enum(positive|negative|neutral)'`.

Class signatures need `experimentalDecorators` in your `tsconfig.json`.

Or zod schemas, which need no decorators and infer their own types — see
[Zod signatures](#zod-signatures).

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

### Zod signatures

`signature()` builds a signature from zod schemas. The shape lives in the type
system rather than in runtime metadata, so results are inferred exactly — no
type argument, and no `experimentalDecorators`:

```ts
import { z } from 'zod';
import { signature, Predict } from '@ts-dspy/core';

const AnalyzeReview = signature({
  description: 'Analyze a product review.',
  input: z.object({ review: z.string() }),
  output: z.object({
    sentiment: z.enum(['positive', 'negative', 'neutral']),
    rating: z.number().int().min(1).max(5),
    themes: z.array(z.string()),
    followUp: z.string().optional(),
  }),
});

const r = await new Predict(AnalyzeReview).forward({ review });

r.sentiment; // 'positive' | 'negative' | 'neutral' — inferred
r.themes.join(', '); // string[]
r.followUp?.trim(); // string | undefined
```

The zod schema is the validator, so anything you can express is enforced:
enums, unions, nested objects, numeric bounds, string formats, and
object-level refinements — none of which the flat decorator field-type list can
spell. Input keys are typed too, so a misspelt input is a compile error.

On the provider structured-output path the same schema becomes the JSON Schema
sent to the model, rewritten for OpenAI strict mode: every property in
`required`, `additionalProperties: false` at every level of nesting, and
optional fields expressed as `type: [base, 'null']`.

Decorator and string signatures keep working unchanged; this is a third form,
not a replacement.

### Output types

A zod signature infers its output type, as above. Decorators record fields at
runtime, so TypeScript cannot infer per-field types from the class — results
from a class signature are typed loosely by default. Name the shape when you
want precise types:

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

### Testing

`@ts-dspy/core/testing` ships the test doubles the library's own suite uses, so
you never have to hand-roll a fake model. Nothing there touches the network.

```ts
import { MockLM, CassetteLM } from '@ts-dspy/core/testing';

const lm = new MockLM({ responses: ['answer: Paris\nconfidence: 0.95'] });
const result = await new Predict(AnswerQuestion, lm).forward({ question: 'Capital?' });

lm.lastPrompt(); // the prompt the module actually sent
```

`CassetteLM` records real provider replies into a JSON file once, then replays
them forever:

```ts
// Once, with a key:
const recorder = CassetteLM.record('cassettes/answer.json', new OpenAILM({ apiKey }));
await new Predict(AnswerQuestion, recorder).forward({ question: 'Capital?' });

// In CI, with no key and no network:
const replay = CassetteLM.replay('cassettes/answer.json');
```

Cassettes are an array of `{ key, request, response }` entries keyed by a hash
of the request, so they review like any other fixture and an unrecorded request
fails loudly instead of calling out.

### Tracing

`configure({ tracing: true })` records every module invocation: the prompt as
sent, the raw reply, the parsed output, the tokens that call cost, and how long
it took. `inspectHistory(n)` reads the last `n` back out of a bounded in-memory
buffer — including inside a `catch`, which is where you usually want it.

```ts
import { clearHistory, configure, inspectHistory } from '@ts-dspy/core';

configure({ lm, tracing: true });

try {
  await triage.forward({ ticket });
} catch (error) {
  const [failed] = inspectHistory(1);
  console.error(failed.rawLMInput, '\n---\n', failed.rawLMOutput);
}
```

Multi-step modules record each call under `calls`, so a `ChainOfThought` trace
holds both the reasoning prompt and the final one. The buffer keeps 100 entries
by default (`traceHistorySize`), and `clearHistory()` empties it.

Pass `onTrace` to forward entries to Langfuse, OpenTelemetry, or your own
logger as they are recorded — the library never prints:

```ts
configure({ lm, tracing: true, onTrace: (entry) => logger.debug(entry) });
```

Tracing is off by default and costs a single boolean check while off.

### Caching

`configure({ cache: true })` replays a previous answer instead of paying for a
repeated prompt. It is off by default, because replaying an old answer changes
what a program does.

```ts
import { configure, MemoryCache, type Cache } from '@ts-dspy/core';

configure({ cache: true }); // process-wide LRU, 1000 entries
configure({ cache: new MemoryCache({ maxSize: 10_000 }) }); // or size it yourself
```

`generate`, `chat`, and `generateStructured` are all cached for every provider.
The key is a SHA-256 hash of the provider, the model, the prompt or messages,
the sampling parameters (`temperature`, `topP`, `maxTokens`, `stopSequences`,
and the penalties), and the JSON schema on structured calls — so two calls that
differ in any of those never collide. Errors are never cached.

A cache hit costs nothing, and `getUsage()` says so: hits land in a `cacheHits`
counter and stay out of `requestCount` and the token totals, so cost accounting
still reflects real provider traffic.

`cache` also accepts your own implementation. Both methods may be async, so
Redis, SQLite, or a directory of files fits without a wrapper:

```ts
const redisCache: Cache = {
  async get(key) {
    const hit = await redis.get(key);
    return hit === null ? undefined : JSON.parse(hit);
  },
  async set(key, value) {
    await redis.set(key, JSON.stringify(value), { EX: 86_400 });
  },
};

configure({ cache: redisCache });
```

## Examples

```bash
git clone https://github.com/ardada2468/LLMTypeSafe.git
cd LLMTypeSafe
npm install
npm run build

export OPENAI_API_KEY="sk-..."
npm run example:openai
npm run example:zod
```

See [`examples/`](examples) for OpenAI, Gemini, Anthropic, zod-signature, and
tool-use programs.

## Development

```bash
npm install
npm run build              # tsup, per package (core first — providers need its types)
npm test                   # vitest
npm run lint
npm run typecheck          # packages and examples
npm run format:check
npm run verify:packaging   # pack, install, and import as a real consumer would
```

Every one of these runs in CI on each pull request, across Node 22, 24, and 26.
The release workflow runs the same set before it can publish, so the release path
cannot drift from the path changes are reviewed through.

`verify:packaging` is the one worth knowing about: it packs the tarballs,
installs them into a throwaway project, and imports them from both ESM and
CommonJS. Builds, type checks, and unit tests all run against workspace symlinks,
so none of them can see a wrong dependency range — which is how 0.4.2 shipped
providers depending on a version of `@ts-dspy/core` that release did not satisfy.

Releases run on [changesets](https://github.com/changesets/changesets): add one
with `npx changeset` when you change a published package, or
`npx changeset --empty` for a change that intentionally ships no release. CI
fails a pull request that changes a published package without one.

Publishing uses npm trusted publishing, so there is no npm token to hold or
rotate. See [RELEASING.md](RELEASING.md).

## License

MIT — see [LICENSE](LICENSE).
