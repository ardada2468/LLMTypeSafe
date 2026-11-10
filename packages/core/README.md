# @ts-dspy/core

Signatures, modules, adapters, optimizers and runtime validation for building
type-safe LLM programs in TypeScript — a port of
[DSPy](https://github.com/stanfordnlp/dspy) 3.x. This package has no provider of
its own: pair it with [`@ts-dspy/openai`](https://www.npmjs.com/package/@ts-dspy/openai),
[`@ts-dspy/anthropic`](https://www.npmjs.com/package/@ts-dspy/anthropic),
[`@ts-dspy/gemini`](https://www.npmjs.com/package/@ts-dspy/gemini) or
[`@ts-dspy/ai-sdk`](https://www.npmjs.com/package/@ts-dspy/ai-sdk).

```bash
npm install @ts-dspy/core @ts-dspy/openai
```

```ts
import { Signature, InputField, OutputField, Predict, configure } from '@ts-dspy/core';
import { OpenAILM } from '@ts-dspy/openai';

class AnswerQuestion extends Signature {
  static description = 'Answer the question concisely.';

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

> Upgrading from 0.6? See the
> [migration guide](https://github.com/ardada2468/LLMTypeSafe/blob/main/MIGRATION.md).

## Signatures

Three interchangeable forms:

```ts
import { signature, history, image } from '@ts-dspy/core';
import { z } from 'zod';

// String: types, optional markers, quoted descriptions, instructions.
const QA = signature('question -> answer, confidence: float', 'Answer concisely.');

// Zod: no decorators, result types inferred exactly.
const Review = signature({
  input: z.object({ review: z.string(), photo: image().optional() }),
  output: z.object({ sentiment: z.enum(['positive', 'negative', 'neutral']) }),
});

// Decorator class: as above.
```

Inputs can be images (`@ImageField`, `image`, `image()`) and conversation
`History` (`history: history`, `history()`). `extendSignature(sig, options)`
derives a signature with fields added or removed.

## Modules

| Export                 | Purpose                                                                                             |
| ---------------------- | --------------------------------------------------------------------------------------------------- |
| `Predict`              | One call, validated against the signature. Options: `{ lm, demos, instructions, adapter }`.         |
| `ChainOfThought`       | `Predict` with a leading `reasoning` output, in the same call.                                      |
| `ReAct`                | Tool-using agent: a `react` step predictor plus an `extract` step. `RespAct` is a deprecated alias. |
| `BestOfN`, `Refine`    | Sample a module until a reward function reaches a threshold; `Refine` adds feedback hints.          |
| `Parallel`             | Run `[module, inputs]` pairs concurrently, results in input order.                                  |
| `MultiChainComparison` | Compare several reasoning attempts and answer with a `rationale`.                                   |
| `ProgramOfThought`     | _(experimental)_ The model writes JavaScript, run in a `node:vm` sandbox.                           |
| `Retrieve`             | `{ passages }` from a `Retriever` such as `InMemoryVectorStore`.                                    |

Every module has `forward(inputs, options?)`, `batch(rows, { concurrency })`,
and — `Predict` and its subclasses — `stream(inputs)` for per-field partial
output. `streamify(program)` streams any program as status, field and
prediction events.

Write your own by extending `Module` and holding sub-modules in fields:

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
```

`namedPredictors()`, `namedSubModules()`, `deepCopy()` and `resetCopy()` walk
the tree; `dumpState()` / `loadState()` and `save(path)` / `load(path)` persist
every predictor's demos and instructions. A module that calls a model directly
should use `this.resolveLM(options)`: models resolve at call time, from
`options.lm`, the module, the active scope, then `configure()`.

## Adapters

`ChatAdapter` (DSPy's `[[ ## field ## ]]` format), `JSONAdapter` (native
structured output where available), `XMLAdapter` and
`TwoStepAdapter({ extractionLM })`. The default is `JSONAdapter` for
structured-output models and `ChatAdapter` otherwise. Choose one per call
(`{ adapter }`), per module, per scope or globally. Subclass `Adapter` for your
own format.

## Settings, callbacks and usage

```ts
import { configure, settings } from '@ts-dspy/core';

configure({
  lm,
  adapter, // default prompt format
  cache: true, // or a Cache implementation
  tracing: true, // inspectHistory(), onTrace
  callbacks: [{ onLMEnd: (e) => log(e.model, e.usage) }],
  trackUsage: true, // prediction.getLMUsage()
  pricing: { 'openai/gpt-5.2': { input: 2, output: 8 } }, // your own prices, per 1M tokens
  retriever, // for Retrieve
});

// Override any of them for one async scope (AsyncLocalStorage-backed):
await settings.context({ lm: otherLM }, () => program.forward(inputs));
```

Callbacks receive module, LM, tool and adapter start/end events with
`callId`/`parentCallId`. Usage is exact per invocation, keyed by
`provider/model`, with cached and reasoning tokens and an optional `totalCost`.

## LM middleware

```ts
import { wrapLM, retry, fallback, defaults, rateLimit } from '@ts-dspy/core';

const lm = wrapLM(primary, rateLimit({ requestsPerMinute: 500 }), retry(), fallback([backup]));
```

Middleware is `(request, next) => Promise<unknown>`, outermost first.

For production, `circuitBreaker()` fails fast while a model is down,
`concurrencyLimit({ maxConcurrent })` bounds calls in flight, and
`budget({ usage: lm, maxCost })` stops a runaway loop with a
`BudgetExceededError`.

## Evaluation and optimizers

```ts
import { evaluate, exactMatch, MIPROv2 } from '@ts-dspy/core';

const report = await evaluate(program, devset, exactMatch, { concurrency: 8 });

const compiled = await new MIPROv2({ metric: exactMatch, auto: 'light' }).compile(program, {
  trainset,
  valset: devset,
});
await compiled.save('program.json');
```

A `Metric` is `(example, prediction, trace?, predName?, predTrace?)` returning a
number, a boolean, or `{ score, feedback }`. Optimizers: `LabeledFewShot`,
`BootstrapFewShot`, `BootstrapFewShotWithRandomSearch`, `KNNFewShot`,
`Ensemble`, `COPRO`, `MIPROv2`, `SIMBA` and `GEPA`. Each returns a compiled
copy, leaves the student untouched, and is reproducible with `seed`.

## Errors

`ValidationError` (with per-field `issues` and `rawOutput`), and `LMError` with
the subclasses `RateLimitError`, `AuthError`, `ContextLengthError`,
`ContentFilterError` and `TimeoutError`. Pass `repairAttempts` to let the model
correct a reply that failed validation.

## Testing

Test doubles ship with the package, under `@ts-dspy/core/testing`. Neither
touches the network, so a suite needs no API key.

```ts
import { MockLM, CassetteLM } from '@ts-dspy/core/testing';

// Scripted replies, and a record of every call.
const lm = new MockLM({ responses: ['answer: Paris\nconfidence: 0.95'] });
const result = await new Predict(AnswerQuestion, lm).forward({ question: 'Capital?' });
expect(lm.lastPrompt()).toContain('Capital?');

// Or compute replies from the request.
const echo = new MockLM({ respond: (messages) => `answer: ${messages.length} messages` });

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
