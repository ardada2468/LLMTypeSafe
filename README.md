# TS-DSPy

[![CI](https://github.com/ardada2468/LLMTypeSafe/actions/workflows/ci.yml/badge.svg)](https://github.com/ardada2468/LLMTypeSafe/actions/workflows/ci.yml)
[![npm](https://img.shields.io/npm/v/@ts-dspy/core.svg)](https://www.npmjs.com/package/@ts-dspy/core)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Build LLM applications in TypeScript by declaring the shape of the input and
output you want, instead of hand-writing prompts and parsing replies. A port of
[DSPy](https://github.com/stanfordnlp/dspy) 3.x: signatures, modules, adapters
and optimizers, with TypeScript types and runtime validation.

Model output is **validated at runtime** against the shape you declared. If a
field you declared as a number comes back as prose, you get a `ValidationError`
naming the field — not a string masquerading as a number.

Programs also **optimize themselves from data**, the part of DSPy that makes it
more than a prompt template library. Give an optimizer a trainset and a metric
and it tunes every predictor in your program — few-shot demos
(`BootstrapFewShot`, `KNNFewShot`), instructions (`COPRO`, `GEPA`), or both at
once (`MIPROv2`, `SIMBA`) — and hands back a compiled copy you can save to JSON.

> **Upgrading from 0.6?** 0.7 has breaking changes — `ChainOfThought` is one
> call, `RespAct` is now `ReAct`, prompts belong to adapters. See
> [MIGRATION.md](MIGRATION.md).

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

Or skip the class entirely: `new Predict('question -> answer, confidence: float')`.

## Features

- **Signatures three ways** — decorator classes, `'question -> answer: int'`
  strings, or zod schemas with fully inferred result types.
- **Runtime validation and self-repair** — typed `ValidationError`s naming each
  failing field, and opt-in `repairAttempts` that show the model its mistake.
- **Adapters** — `ChatAdapter`, `JSONAdapter` (native structured output),
  `XMLAdapter` and `TwoStepAdapter`, chosen per call, module, scope or globally.
- **Modules** — `Predict`, single-call `ChainOfThought`, a `ReAct` agent with
  native tool calling, `BestOfN`, `Refine`, `Parallel`, `MultiChainComparison`,
  `ProgramOfThought` and `Retrieve`.
- **Optimizers** — `LabeledFewShot`, `BootstrapFewShot`,
  `BootstrapFewShotWithRandomSearch`, `KNNFewShot`, `Ensemble`, `COPRO`,
  `MIPROv2`, `SIMBA` and `GEPA`, all seeded and reproducible.
- **Composition and persistence** — programs are plain classes holding modules;
  `namedPredictors()`, `deepCopy()`, `save()` / `load()`.
- **Scoped settings** — `settings.context()` overrides the model, adapter,
  cache or callbacks for one async scope, safely under concurrency.
- **Observability** — callbacks for module, LM, tool and adapter events, exact
  per-call token usage and cost, tracing, and OpenTelemetry spans.
- **LM middleware** — `wrapLM(lm, retry(), fallback([...]), rateLimit(...))`, plus
  `circuitBreaker`, `concurrencyLimit` and a `budget` cap for production.
- **Streaming** — per-field streaming from any module, and `streamify()` for
  whole programs.
- **Providers** — OpenAI (and any OpenAI-compatible server), Anthropic,
  Gemini, and every Vercel AI SDK model; reasoning controls, prompt caching and
  embedders included.
- **Integrations** — MCP tools in agents and programs served over MCP,
  OpenTelemetry tracing.
- **Testing** — `MockLM` and record/replay `CassetteLM`, no network needed.

## Packages

| Package                                    | Provider                                                                   |
| ------------------------------------------ | -------------------------------------------------------------------------- |
| [`@ts-dspy/core`](packages/core)           | Signatures, modules, validation. No provider.                              |
| [`@ts-dspy/openai`](packages/openai)       | OpenAI, plus any OpenAI-compatible endpoint, via the official `openai` SDK |
| [`@ts-dspy/gemini`](packages/gemini)       | Google Gemini, via `@google/genai` (Gemini API or Vertex AI)               |
| [`@ts-dspy/anthropic`](packages/anthropic) | Anthropic Claude, via `@anthropic-ai/sdk`                                  |
| [`@ts-dspy/ai-sdk`](packages/ai-sdk)       | Any Vercel AI SDK language model, via `ai`                                 |
| [`@ts-dspy/mcp`](packages/mcp)             | Model Context Protocol: MCP tools in `ReAct`, programs served as MCP tools |
| [`@ts-dspy/otel`](packages/otel)           | OpenTelemetry spans for modules, LM calls and tools (GenAI conventions)    |

Install core plus whichever providers you use. Each provider defaults to a current
model for that vendor; pass `model` to pin one yourself. The last three are
integrations rather than vendors: `@ts-dspy/ai-sdk` wraps any AI SDK model,
`@ts-dspy/mcp` connects programs to MCP servers in both directions, and
`@ts-dspy/otel` traces them.

`@ts-dspy/openai` also exports `OpenAICompatibleLM` for the many servers that speak
the OpenAI chat-completions API — Ollama, LM Studio, vLLM, Groq, Together, and
OpenRouter. It requires `baseURL` and `model`, defaults the API key to a
placeholder for local servers that ignore it, and takes model capabilities from
config rather than assuming OpenAI's:

```ts
import { OpenAICompatibleLM } from '@ts-dspy/openai';

const lm = new OpenAICompatibleLM({
  baseURL: 'http://localhost:11434/v1',
  model: 'llama3.2',
});
```

`npm run example:ollama` runs it end to end against a local Ollama, with no cloud
key involved.

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

| Module                                        | What it does                                                                                                                              |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `Predict`                                     | One call, validated against the signature.                                                                                                |
| `ChainOfThought`                              | Adds a leading `reasoning` output, so the model thinks before it answers — in the same call.                                              |
| `ReAct`                                       | A tool-using agent over any signature; see [Tools](#tools). (`RespAct` is a deprecated alias.)                                            |
| `BestOfN(module, { n, rewardFn, threshold })` | Runs a module up to `n` times at a sampling temperature and returns the first prediction whose reward reaches `threshold`, else the best. |
| `Refine(module, { n, rewardFn, threshold })`  | `BestOfN`, plus model-written advice after each miss, passed to the next attempt as a hint.                                               |
| `Parallel({ concurrency })`                   | `.run([[module, inputs], …])` runs different modules side by side; results in input order.                                                |
| `MultiChainComparison(sig, { m })`            | Compares `m` reasoning attempts and produces a `rationale` plus the outputs.                                                              |
| `ProgramOfThought(sig)` _(experimental)_      | The model writes JavaScript, run in a fresh `node:vm` context, with errors fed back. Not a security boundary.                             |
| `Retrieve({ k })`                             | Returns `{ passages }` from the configured retriever — see [Retrieval](#retrieval).                                                       |

Every module accepts per-call options that are passed through to the provider SDK:

```ts
await predict.forward({ question: '...' }, { temperature: 0, timeout: 30_000, retries: 2 });
```

Pass `signal` to cancel a call that is already in flight — a React effect tearing
down, or a server request whose client disconnected. It composes with `timeout`:
whichever fires first ends the call.

```ts
const controller = new AbortController();
// React: return () => controller.abort() from the effect.

await predict.forward({ question: '...' }, { signal: controller.signal, timeout: 30_000 });
```

A module resolves its model when it runs — `options.lm`, then its own model,
then the [scoped](#scoped-settings) one, then `configure()`'s — so modules can
be built at import time, before anything is configured.

```ts
const refine = new Refine(new ChainOfThought('product, notes -> headline'), {
  n: 3,
  threshold: 1,
  rewardFn: (inputs, pred) => (String(pred.get('headline')).split(' ').length <= 8 ? 1 : 0),
});
```

### Adapters

How a signature becomes a prompt, and a reply becomes fields, is an adapter's
job, as in DSPy. `ChatAdapter` writes every field under a `[[ ## field ## ]]`
marker; `JSONAdapter` asks for one JSON object and, when a provider supports
native structured output, constrains decoding to your schema; `XMLAdapter` uses
`<field>` tags; `TwoStepAdapter({ extractionLM })` lets a reasoning model answer
freely and a second model extract the fields. Demos and `History` inputs travel
as real user/assistant turns.

With nothing chosen, modules use `JSONAdapter` on a structured-output model and
`ChatAdapter` (which retries in JSON when a reply ignores its format) otherwise.
Choose one per call, per module, per scope, or globally — in that order of
precedence:

```ts
await predict.forward(inputs, { adapter: new XMLAdapter() });
new Predict(QA, { adapter: new JSONAdapter() });
await settings.context({ adapter: new ChatAdapter() }, () => program.forward(inputs));
configure({ adapter: new TwoStepAdapter({ extractionLM: cheapLM }) });
```

Signature helpers: `History` inputs (`type: 'history'`, zod `history()`, or
`history: history` in a string) carry earlier turns of a conversation;
`extendSignature(sig, { prependOutputs, appendInputs, removeOutputs, … })`
derives a new signature, as `ChainOfThought` and `ReAct` do internally.

### Composition and save/load

A program is an ordinary class that holds modules in its fields — directly, in
arrays, or in records. Nothing needs registering:

```ts
import { ChainOfThought, Module, Predict, type Prediction } from '@ts-dspy/core';

class DraftAndPolish extends Module {
  draft = new ChainOfThought('question -> answer');
  polish = new Predict('question, draft -> answer');

  async forward(inputs: { question: string }): Promise<Prediction & Record<string, any>> {
    const { answer } = await this.draft.forward(inputs);
    return this.polish.forward({ question: inputs.question, draft: answer });
  }
}

const program = new DraftAndPolish();
program.namedPredictors(); // [['draft', ChainOfThought], ['polish', Predict]]
```

Those predictor paths are what optimizers tune and what persistence keys on.
`dumpState()` / `loadState(state)` round-trip every predictor's demos and
instructions as plain JSON; `save(path)` / `load(path)` do the same through a
file (Node only). Loading into a program of a different shape is an error, not
a silent partial load. `deepCopy()` and `resetCopy()` clone a program, and
`Predict` has `getInstructions()` / `setInstructions()` alongside
`getDemos()` / `setDemos()`.

```ts
await compiled.save('program.json');
const restored = await new DraftAndPolish().load('program.json');
```

Writing a custom module that calls a model directly? Use
`this.resolveLM(options)` rather than `this.lm`, which is `undefined` unless the
module was given a model explicitly.

### Scoped settings

`configure()` sets process-wide defaults. `settings.context(overrides, fn)`
(alias `withSettings`) overrides them for one async scope, backed by
`AsyncLocalStorage`, so concurrent requests can use different models, adapters,
caches or callbacks without interfering:

```ts
import { configure, settings } from '@ts-dspy/core';

configure({ lm: defaultLM, cache: true });

app.post('/answer', (req, res) =>
  settings.context({ lm: req.user.premium ? bigLM : smallLM }, async () => {
    res.json(await program.forward(req.body));
  })
);
```

`lm`, `adapter`, `cache`, `tracing`, `onTrace`, `callbacks`, `trackUsage`,
`pricing` and `retriever` can all be scoped; `traceHistorySize` is global only.

### Callbacks and usage

A callback is an object with any of `onModuleStart`/`onModuleEnd`,
`onLMStart`/`onLMEnd`, `onToolStart`/`onToolEnd` and
`onAdapterFormat`/`onAdapterParse`. Every event carries a `callId` and the
`parentCallId` of the invocation it ran inside, correct under concurrency; end
events add `durationMs`, `error`, and — for modules and LM calls — `usage`.

```ts
configure({
  callbacks: [
    {
      onLMEnd: (e) => log.info({ model: e.model, ms: e.durationMs, usage: e.usage }),
      onToolStart: (e) => log.debug(`tool ${e.toolName}`, e.args),
    },
  ],
});
```

Usage is collected per invocation, exactly, keyed by `provider/model`. Turn it
on with `trackUsage` (tracing or any callback also enables it), and add a
`pricing` table — ts-dspy ships none, since price lists go stale — for cost:

```ts
configure({
  lm,
  trackUsage: true,
  // Your own prices, per million tokens (these numbers are placeholders).
  pricing: { 'openai/gpt-5.2': { input: 2, output: 8, cachedInput: 0.2 } },
});

const prediction = await program.forward(inputs);
prediction.getLMUsage();
// { 'openai/gpt-5.2': { promptTokens, completionTokens, cachedTokens, reasoningTokens, totalCost, … } }
```

### LM middleware

`wrapLM(lm, ...middleware)` returns a model whose calls pass through each
middleware in turn, outermost first. Four cover the common cases:

```ts
import { wrapLM, retry, fallback, defaults, rateLimit } from '@ts-dspy/core';

const lm = wrapLM(
  new OpenAILM({ apiKey }),
  rateLimit({ requestsPerMinute: 500 }),
  retry({ maxRetries: 4, initialDelayMs: 500 }), // rate limits, timeouts, 5xx
  fallback([new AnthropicLM({ apiKey: anthropicKey })]), // tried when OpenAI fails
  defaults({ temperature: 0 }) // unless a call sets its own
);
```

A middleware is `(request, next) => Promise<unknown>`: inspect or rewrite the
request, call `next` (or not), and return the result — enough for logging,
redaction, routing or a custom cache.

Three more guard a production deployment:

```ts
import { wrapLM, budget, circuitBreaker, concurrencyLimit, fallback } from '@ts-dspy/core';

const primary = new OpenAILM({ apiKey });
const lm = wrapLM(
  primary,
  budget({ usage: primary, maxCost: 25, maxRequests: 10_000 }), // hard stop for a runaway loop
  concurrencyLimit({ maxConcurrent: 16 }), // at most 16 calls in flight; the rest queue
  fallback([new AnthropicLM({ apiKey: anthropicKey })]),
  circuitBreaker({ failureThreshold: 5, resetTimeoutMs: 30_000 }) // fail fast while OpenAI is down
);
```

- `circuitBreaker` opens after `failureThreshold` consecutive rate-limit,
  timeout or 5xx failures and rejects with `CircuitOpenError` without calling
  the model, so a `fallback` outside it takes over at once instead of waiting
  out a timeout per request. After `resetTimeoutMs` it lets one probe through.
  `onStateChange` reports transitions for alerting.
- `concurrencyLimit` bounds calls in flight (a stream holds its slot until it
  ends), where `rateLimit` spaces out when calls start.
- `budget` throws `BudgetExceededError` once `maxRequests`, `maxTokens` or
  `maxCost` is reached. Limits are checked before each call, so calls already
  in flight can overshoot; `maxCost` needs `configure({ pricing })`.

### Streaming

`stream()` runs the same prediction as `forward()`, but yields the output fields
as they fill in:

```ts
for await (const partial of predict.stream({ question: '...' })) {
  render(partial.answer); // grows as tokens arrive
}
```

Each yield is a snapshot of the fields parsed so far. The **last yield** is the
complete output, validated against the signature exactly as `forward()` validates
it — a stream that ends in something the signature rejects still throws a
`ValidationError`. The generator's **return value** is the `Prediction` wrapper,
for callers who drive `next()` by hand, since `for await` discards return values.

Only that last snapshot is guaranteed to match the declared types — coercion
belongs to validation, so a field declared `number` may still be the raw `'0.'`
the model is part-way through writing. Snapshots are typed as `PartialOutput<T>`
accordingly: every field optional, and possibly still a string.

```ts
const stream = predict.stream({ question: '...' }, { signal: controller.signal });

let step = await stream.next();
while (!step.done) step = await stream.next();

step.value.confidence; // the validated Prediction
```

The adapter reads each partial reply: `JSONAdapter` (the default for providers
with native structured output) through an incremental parser — exported as
`parsePartialJson` — that recovers the fields present in a document truncated
mid-string or mid-key, `ChatAdapter` section by `[[ ## field ## ]]` section.
`ChainOfThought` streams its `reasoning` like any other field. A model or adapter
that cannot stream falls back to a single call, yielded once, rather than
failing. Pass an `AbortSignal` to cancel; `break`ing out of the loop closes the
provider's stream.

To stream a whole program — a hand-written multi-step `forward` included — use
`streamify`. It yields `status` events as modules, LM calls and tools start and
finish, `field` events (with `delta`s) as each predictor writes, and a final
`prediction`:

```ts
import { streamify } from '@ts-dspy/core';

for await (const event of streamify(program)({ topic })) {
  if (event.type === 'field' && event.delta) process.stdout.write(event.delta);
  if (event.type === 'status') console.log(`\n[${event.message}]`);
}
```

### Batch and concurrency

Every module inherits `batch()`, a bounded worker pool over a list of inputs.
Results come back **in input order**, whatever order the calls finished in, and a
failing input is captured rather than thrown — one bad row does not destroy a
ten-thousand-row job.

```ts
const results = await predict.batch(rows, {
  concurrency: 16, // in flight at once; defaults to 8
  onProgress: (done, total) => bar.update(done / total),
});

for (const [i, result] of results.entries()) {
  if (result.status === 'fulfilled') save(rows[i], result.value);
  else quarantine(rows[i], result.reason);
}
```

Pass `stopOnError: true` to reject the whole batch on the first failure instead,
or a `signal` to stop starting new inputs on cancellation — both reject rather
than returning the inputs that already finished, so wrap a cancellable batch in a
`try`. Every other option is passed through to each underlying call. The pool
itself is exported as `mapWithConcurrency(items, worker, options)` for anything
else with a rate limit.

### Evaluation

`evaluate` runs a program over a dataset of `Example`s and grades every
prediction, so a prompt or signature change comes with a number attached rather
than a hunch.

```ts
import { evaluate, exactMatch, Example, Predict, formatReport } from '@ts-dspy/core';

const dataset = [
  new Example({ question: 'Capital of France?', answer: 'Paris' }).withInputs('question'),
  new Example({ question: 'Capital of Japan?', answer: 'Tokyo' }).withInputs('question'),
];

const report = await evaluate(new Predict(AnswerQuestion), dataset, exactMatch, {
  concurrency: 8,
});

report.score; // mean across every example
report.results; // per example: inputs, expected, prediction, score, feedback?, error?
report.usage; // tokens, latency and (with pricing) cost for this run only

console.log(formatReport(report));
```

A `Metric` is one function type shared by `evaluate` and every optimizer:
`(example, prediction, trace?, predName?, predTrace?)` returning a `number`, a
`boolean`, or `{ score, feedback }` — sync or async. Most metrics use only the
first two arguments; optimizers pass the run's `trace` while bootstrapping, and
`GEPA` passes `predName`/`predTrace` when it wants feedback aimed at one
predictor. `metricScore()` and `metricFeedback()` normalise any result. Built
in: `exactMatch`, `normalizedMatch`, `numericMatch`, `fieldAccuracy` for
per-field partial credit, and `tokenF1` for free text.

An example whose program or metric throws is recorded as a zero with the error
attached and the run continues — an evaluation that dies on row 40 of 500 tells
you nothing. Usage is collected exactly from the calls the run makes, so other
work on the same model is never counted.

### Optimizers

Every optimizer takes a metric, works on the predictor paths of any program
(`namedPredictors()`), returns a compiled **copy** without touching the
student, accepts `{ seed, concurrency, onProgress }`, and records what it did
(`results` / `detailedResults`) for inspection.

| Optimizer                          | Tunes                  | In brief                                                                                       |
| ---------------------------------- | ---------------------- | ---------------------------------------------------------------------------------------------- |
| `LabeledFewShot`                   | demos                  | `k` labelled examples, no model calls.                                                         |
| `BootstrapFewShot`                 | demos                  | Runs the trainset (optionally through a `teacher`), keeps passing runs as per-predictor demos. |
| `BootstrapFewShotWithRandomSearch` | demos                  | Several bootstrapped candidates, scored on the valset; keeps the best.                         |
| `KNNFewShot`                       | demos, per call        | Embeds each call's inputs and uses the `k` nearest training examples.                          |
| `Ensemble`                         | —                      | Runs several programs and reduces their outputs (majority vote by default).                    |
| `COPRO`                            | instructions           | `breadth` proposals over `depth` rounds, each conditioned on earlier scores.                   |
| `MIPROv2`                          | instructions and demos | Grounded instruction proposals plus demo sets, searched with a seeded TPE sampler.             |
| `SIMBA`                            | instructions and demos | Mini-batch ascent: finds high-variance examples, adds demos or self-reflective rules.          |
| `GEPA`                             | instructions           | Reflective evolution from metric feedback, with a Pareto front of candidates.                  |

```ts
import { MIPROv2 } from '@ts-dspy/core';

const optimizer = new MIPROv2({
  metric: routedCorrectly,
  promptModel: strongLM, // writes candidate instructions; defaults to the configured model
  auto: 'light', // or 'medium' / 'heavy', or null with numCandidates + numTrials
});

const compiled = await optimizer.compile(program, { trainset, valset });
compiled.miproResults.bestScore;
await compiled.save('router.json');
```

`GEPA` learns most from a metric that explains itself:

```ts
import { GEPA, type Metric } from '@ts-dspy/core';

const metric: Metric = (example, prediction) => {
  const ok = prediction.get('sku') === example.get('sku');
  return {
    score: ok ? 1 : 0,
    feedback: ok ? 'Correct.' : `SKU should be "${example.get('sku')}" in upper case.`,
  };
};

const gepa = new GEPA({ metric, reflectionLM: strongLM, auto: 'light' }); // or maxMetricCalls / maxFullEvals
const optimized = await gepa.compile(program, { trainset, valset });
optimized.detailedResults.paretoFront;
```

Runnable versions: [`examples/optimizer.ts`](examples/optimizer.ts)
(`LabeledFewShot`, `BootstrapFewShot` with a teacher),
[`examples/mipro.ts`](examples/mipro.ts), [`examples/gepa.ts`](examples/gepa.ts)
and [`examples/rag-knn.ts`](examples/rag-knn.ts) (`KNNFewShot`).

### Retrieval

`Retriever` is a one-method interface (`retrieve(query, k)` → `Passage[]`).
`InMemoryVectorStore` implements it over any `Embedder` — `OpenAIEmbedder` and
`GeminiEmbedder` ship with their providers — and `Retrieve` uses whichever
retriever is configured:

```ts
import { InMemoryVectorStore, Retrieve, configure } from '@ts-dspy/core';
import { OpenAIEmbedder } from '@ts-dspy/openai';

const store = new InMemoryVectorStore(new OpenAIEmbedder({ apiKey }));
await store.addAll(documents);
configure({ lm, retriever: store });

const { passages } = await new Retrieve({ k: 3 }).forward({ query: 'refund policy' });
```

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

### Self-repair

A model that fails validation has often understood the task and merely fumbled
the shape. Pass `repairAttempts` to spend that many extra round-trips telling it
exactly what went wrong before giving up:

```ts
const result = await predict.forward({ question: '...' }, { repairAttempts: 1 });
```

The rejected reply goes back as an assistant turn, followed by a user turn naming
each failing field with its declared type and the value that actually arrived.
It works with every adapter, native structured output included, and every
module built on `Predict`. The default is `0`, so
validation failures throw immediately unless you opt in; once the attempts are
spent the last `ValidationError` is rethrown. Attempts are capped at 10, and the
loop stops early if an attempt reproduces the previous failure exactly.

### Provider errors

A failed provider call throws a typed subclass of `LMError` whenever the SDK
gives us enough to tell, so retry logic is a `catch` on a class rather than a
sniff at an HTTP status number:

```ts
import { RateLimitError, ContextLengthError, LMError } from '@ts-dspy/core';

try {
  return await predict.forward({ question });
} catch (error) {
  if (error instanceof RateLimitError) return queue.retryLater(question);
  if (error instanceof ContextLengthError)
    return predict.forward({ question: shorten(question) });
  if (error instanceof LMError) log.error(error.provider, error.status, error.cause);
  throw error;
}
```

| Error                | Thrown when                                                                      |
| -------------------- | -------------------------------------------------------------------------------- |
| `RateLimitError`     | A rate or quota limit was hit. Worth retrying after a backoff.                   |
| `AuthError`          | The key is missing, wrong, or not entitled to the model. Retrying will not help. |
| `ContextLengthError` | The prompt did not fit the context window.                                       |
| `ContentFilterError` | Safety classifiers declined the request or the reply. Carries `category`.        |
| `TimeoutError`       | The request timed out before a reply arrived.                                    |
| `LMError`            | Anything else. Every class above extends it, so existing handlers still work.    |

What each SDK reports differs, and the classification follows that rather than
pretending otherwise: OpenAI's `code` is the only dependable signal for a
context-length overflow, Anthropic's typed `error.type` union is the cleanest of
the three, and Gemini reports nothing but an HTTP status.

Content filtering is the exception — all three providers answer `200 OK` and
leave a marker in the body (`stop_reason: 'refusal'`, `finishReason: 'SAFETY'`,
`finish_reason: 'content_filter'`), so `ContentFilterError` comes from
inspecting the response. `AnthropicRefusalError` is now a deprecated alias of
`ContentFilterError` — an alias of that class rather than a subclass, so an
`instanceof` check under the old name now matches any provider's filter. Test
`error.provider` to tell them apart.

### Provider features

- **Reasoning**: OpenAI `reasoningEffort`, Anthropic `thinking` (adaptive or
  budgeted) and `effort`, Gemini `thinking`. Reasoning text arrives as
  `ChatResult.reasoning` and as `metadata.reasoningDelta` on stream chunks.
- **Prompt caching**: Anthropic `cachePrompt: true` marks the system prompt and
  conversation prefix as cache breakpoints.
- **`providerOptions`**: per-call, provider-keyed options deep-merged into the
  raw request — `{ providerOptions: { anthropic: { thinking: false } } }`.
- **Usage** reports `cachedTokens` and `reasoningTokens` for all three vendors.
- **Embeddings**: `OpenAIEmbedder` and `GeminiEmbedder` implement core's
  `Embedder`.

See each provider's README for details.

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
const agent = new ReAct(AnswerQuestion, {
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

Give a tool a `parameters` schema — JSON Schema or Zod — and it takes named,
validated arguments instead of one string:

```ts
const agent = new ReAct(AnswerQuestion, {
  tools: {
    flights: {
      description: 'Find flights between two airports on a date.',
      parameters: z.object({ from: z.string(), to: z.string(), date: z.string() }),
      function: ({ from, to, date }) => search(from, to, date),
    },
  },
});
```

Each step is a `Predict` over the signature's inputs plus the `trajectory` so
far, choosing `next_thought`, `next_tool_name` (one of the tools, or `finish`)
and `next_tool_args`; then a `ChainOfThought` extracts the signature's outputs
from the whole trajectory. The prompt comes from the signature through the
adapter, so any inputs work — not just `question`. Running out of `maxSteps` is
not an error: as in DSPy, the agent extracts its answer from what it has, and
the result's `steps` and `trajectory` show what happened. `agent.react` and
`agent.extract` are ordinary predictors that optimizers can tune.

Tools can also be `Tool` instances — `new Tool({ name, description, parameters, fn })`
— passed as an array, which is what [`@ts-dspy/mcp`](#integrations)'s
`mcpTools()` returns. `finish` is reserved as a tool name.

Against a provider reporting `supportsFunctionCalling: true` — all three of ours
do — tools are declared in the request instead and the model's calls come back
as structured data, so several tools can run in one turn. Both paths run the
same tools and emit the same events; pass `forceTextMode: true` to keep a
tool-capable model on the adapter loop.

Tool descriptions are what the model uses to decide when to call each tool, so
they earn the detail. Never pass model output to `eval()` — see
[`examples/utils.ts`](examples/utils.ts) for a bounded arithmetic evaluator.

### Images

`ChatMessage.content` is `string | ContentPart[]`, so one message can carry text
and images together. Declare an image input with `@ImageField`, the `image` type
in a string signature, or `image()` in a zod signature, and every module sends it
as a real image part:

```ts
import { Signature, ImageField, OutputField, Predict, image, signature } from '@ts-dspy/core';

class ReadSign extends Signature {
  static description = 'Read the sign in the photo.';

  @ImageField({ description: 'photo of the sign' })
  photo!: string;

  @OutputField({ description: 'the words on the sign' })
  words!: string;
}

const { words } = await new Predict(ReadSign).forward({ photo: 'data:image/png;base64,...' });

// The same input in a zod signature:
const ReadSignZ = signature({
  input: z.object({ photo: image().describe('photo of the sign') }),
  output: z.object({ words: z.string() }),
});
```

`imagePart()` accepts an `https://` URL, a `data:` URI, or an explicit
`{ kind: 'base64', data, mediaType }` source. OpenAI receives `image_url` parts,
Anthropic `image` blocks, and Gemini `inlineData` (Gemini fetches no arbitrary
web URLs, so pass bytes or a Files API URI). Only user turns can carry an image,
so system and assistant content is flattened to text. A plain string content
behaves exactly as it did, and `supportsVision` is reported per model instead of
being hardcoded to `true`.

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

Modules that make several calls — a repair round trip, a `ReAct` loop — record
each one under `calls`, each with its own exact usage. Entries carry an `id` and
the `parentId` of the enclosing module call, so a composite program's entries
form a tree. The buffer keeps 100 entries by default (`traceHistorySize`), and
`clearHistory()` empties it.

Pass `onTrace` to forward entries to Langfuse or your own logger as they are
recorded — the library never prints. (For OpenTelemetry, use
[`@ts-dspy/otel`](#integrations), which builds real spans from callbacks.)

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

## Integrations

**MCP** — [`@ts-dspy/mcp`](packages/mcp) turns any Model Context Protocol
server's tools into `ReAct` tools, and serves a ts-dspy program as an MCP tool:

```ts
import { ReAct } from '@ts-dspy/core';
import { connectMcp, mcpTools, serveProgram } from '@ts-dspy/mcp';

const client = await connectMcp({
  command: 'npx',
  args: ['-y', '@modelcontextprotocol/server-filesystem', '.'],
});
const agent = new ReAct('question -> answer', {
  tools: await mcpTools(client, { prefix: 'fs_' }),
});

const server = serveProgram(new ChainOfThought('question -> answer'), { name: 'qa' });
```

**OpenTelemetry** — [`@ts-dspy/otel`](packages/otel) creates nested
`ts-dspy.module`, `chat <model>` and `execute_tool` spans with GenAI semantic
conventions. Prompt and completion capture is opt-in.

```ts
import { instrument } from '@ts-dspy/otel';

instrument(); // uses the global tracer provider; or instrument({ tracer, captureContent: true })
```

**Vercel AI SDK** — [`@ts-dspy/ai-sdk`](packages/ai-sdk) runs ts-dspy programs
on any AI SDK `LanguageModel`, with streaming, tools, structured output and
usage:

```ts
import { openai } from '@ai-sdk/openai';
import { AISDKLM } from '@ts-dspy/ai-sdk';

configure({ lm: new AISDKLM({ model: openai('gpt-5.2') }) });
```

## Examples

```bash
git clone https://github.com/ardada2468/LLMTypeSafe.git
cd LLMTypeSafe
npm install
npm run build

export OPENAI_API_KEY="sk-..."
npm run example:openai
```

| Script              | File                                                                | Shows                                                 |
| ------------------- | ------------------------------------------------------------------- | ----------------------------------------------------- |
| `example:openai`    | [`basic-usage.ts`](examples/basic-usage.ts)                         | `Predict`, `ChainOfThought`, typed fields, validation |
| `example:anthropic` | [`basic-anthropic-example.ts`](examples/basic-anthropic-example.ts) | The same on Claude                                    |
| `example:gemini`    | [`basic-gemini-example.ts`](examples/basic-gemini-example.ts)       | The same on Gemini                                    |
| `example:zod`       | [`zod-signature.ts`](examples/zod-signature.ts)                     | Zod signatures with inferred types                    |
| `example:ollama`    | [`ollama-local.ts`](examples/ollama-local.ts)                       | A local model through `OpenAICompatibleLM`            |
| `example:tools`     | [`tool-demo.ts`](examples/tool-demo.ts)                             | `ReAct` with tools                                    |
| `example:optimizer` | [`optimizer.ts`](examples/optimizer.ts)                             | `LabeledFewShot`, `BootstrapFewShot` with a teacher   |
| `example:save-load` | [`composition-save-load.ts`](examples/composition-save-load.ts)     | Composite programs, `save`/`load`, `settings.context` |
| `example:rag`       | [`rag-knn.ts`](examples/rag-knn.ts)                                 | `InMemoryVectorStore`, `Retrieve`, `KNNFewShot`       |
| `example:mipro`     | [`mipro.ts`](examples/mipro.ts)                                     | `MIPROv2`                                             |
| `example:gepa`      | [`gepa.ts`](examples/gepa.ts)                                       | `GEPA` with a feedback metric                         |
| `example:refine`    | [`refine.ts`](examples/refine.ts)                                   | `BestOfN` and `Refine`                                |
| `example:streamify` | [`streamify.ts`](examples/streamify.ts)                             | Streaming a multi-step program                        |
| `example:mcp`       | [`mcp-react.ts`](examples/mcp-react.ts)                             | MCP server tools in `ReAct`                           |
| `example:otel`      | [`otel-tracing.ts`](examples/otel-tracing.ts)                       | OpenTelemetry spans to the console                    |
| `example:ai-sdk`    | [`ai-sdk.ts`](examples/ai-sdk.ts)                                   | `AISDKLM` via the AI Gateway (`AI_GATEWAY_API_KEY`)   |

Each example checks for the key it needs and explains how to set it.

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
