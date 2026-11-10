# Migrating from 0.6 to 0.7

0.7 brings ts-dspy much closer to DSPy 3.x: prompts are built by adapters,
`ChainOfThought` makes one call, the agent module is `ReAct`, and modules
resolve their model at call time. Most programs need only a few changes, and
most of those show up as type errors or test failures, not silent behaviour
changes.

The breaking changes, roughly in order of how likely they are to affect you:

1. [`ChainOfThought` makes one call](#1-chainofthought-makes-one-call)
2. [`RespAct` is now `ReAct`, with a new protocol](#2-respact-is-now-react)
3. [Prompts belong to adapters: `buildPrompt` and friends are gone](#3-prompts-belong-to-adapters)
4. [`Predict` talks to models through `chat`](#4-predict-uses-chat)
5. [Lazy model resolution: `resolveLM` in custom modules](#5-lazy-model-resolution)
6. [One `Metric` type, with richer return values](#6-one-metric-type)
7. [`Example` and `Prediction` are proxies](#7-example-and-prediction-are-proxies)
8. [Tracing internals: `beginTrace` and `TraceSpan`](#8-tracing-internals)
9. [`evaluate`'s `lm` option is ignored](#9-evaluates-lm-option-is-ignored)
10. [Token counts: Anthropic prompt tokens, Gemini completion tokens](#10-token-counts)
11. [String signatures are validated](#11-string-signatures-are-validated)

For everything that is new rather than changed, see the [README](README.md) and
the package changelogs.

---

## 1. `ChainOfThought` makes one call

In 0.6, `ChainOfThought` made two calls: one to reason in free text, a second
to answer with that reasoning in context. In 0.7 it makes **one** call against
the signature with a leading `reasoning` output field, exactly as DSPy does.

The result still carries `reasoning`, so most code is unaffected. What changes:

- **Call counts and cost** halve. Anything asserting on the number of calls, or
  on `trace.calls.length`, needs updating.
- **Scripted test replies**: a `MockLM` scripted with a reasoning reply and then
  an answer reply now needs a single reply containing both. Re-record cassettes.
- **Streaming**: `reasoning` streams like any other field, instead of arriving
  complete before the answer starts.
- **Repair** now regenerates the reasoning too, since it is part of the same
  reply.

```ts
// Before (0.6): two scripted replies
const lm = new MockLM({
  responses: ['The train covers 60 km in 0.75 h.', 'answer: 80 km/h'],
});

// After (0.7): one reply with both fields
const lm = new MockLM({
  responses: [
    '[[ ## reasoning ## ]]\n60 km in 0.75 h.\n\n[[ ## answer ## ]]\n80 km/h\n\n[[ ## completed ## ]]',
  ],
});
```

(`MockLM` replies in the older `field: value` form still parse — `ChatAdapter`
falls back to it — but the marker format above is what a real model is asked
for.)

## 2. `RespAct` is now `ReAct`

`RespAct` was renamed `ReAct`, matching DSPy. `RespAct`, `RespActOptions` and
`RespActEvent` remain as deprecated aliases and will be removed in a future
major release.

```ts
// Before
import { RespAct } from '@ts-dspy/core';
const agent = new RespAct('question -> answer', { tools, maxSteps: 6 });

// After
import { ReAct } from '@ts-dspy/core';
const agent = new ReAct('question -> answer', { tools, maxSteps: 6 });
```

The rename is the easy part. The agent was also rebuilt after DSPy's:

**A new protocol.** 0.6 asked the model for `Thought:` / `Action:` /
`Final Answer:` text and parsed it. 0.7 runs two ordinary predictors:

- `agent.react` — a `Predict` over the signature's inputs plus a `trajectory`,
  producing `next_thought`, `next_tool_name` (a tool, or `finish`) and
  `next_tool_args`;
- `agent.extract` — a `ChainOfThought` over the inputs plus the whole
  trajectory, producing the signature's outputs.

Both go through the adapter, so any signature works (not just `question`), and
optimizers can tune both. On a model with native tool calling the `react` step
uses it, as before. Scripted `MockLM` replies written for the old text protocol
need rewriting — script `next_thought` / `next_tool_name` / `next_tool_args`
replies, then an extract reply with `reasoning` and the outputs — or switch the
test to the native tool path.

**Running out of `maxSteps` is not an error.** 0.6 threw when the loop ended
without a final answer. 0.7 moves on to the extract step with the trajectory it
has, as DSPy does:

```ts
// Before
try {
  const result = await agent.forward({ question });
} catch (error) {
  // "RespAct exceeded maximum steps (6) …"
  return fallbackAnswer(question);
}

// After: always returns; inspect how it got there
const result = await agent.forward({ question });
result.steps; // steps taken; equal to maxSteps when the budget ran out
result.trajectory; // { thought_0, tool_name_0, tool_args_0, observation_0, … }
result.reasoning; // the extract step's reasoning
```

**`finish` is reserved.** A tool named `finish` is rejected; rename it.

**New options.** `tools` may also be an array of `Tool` instances (as
`@ts-dspy/mcp`'s `mcpTools()` returns), and `adapter` sets the prompt format
for every step. `onEvent` and `forceTextMode` work as before.

## 3. Prompts belong to adapters

How a signature becomes a prompt is now an adapter's job, as in DSPy. The
default for a model without native structured output is `ChatAdapter`, whose
format differs from 0.6's:

- a **system message** lists the input and output fields with types and
  descriptions, shows the `[[ ## field ## ]]` structure, and ends with the
  instructions;
- **demos** (and `History` inputs) are real user/assistant turns, not text
  pasted into one prompt;
- the **user message** carries the inputs under `[[ ## field ## ]]` markers.

Models with native structured output use `JSONAdapter` by default. Choose an
adapter explicitly with `configure({ adapter })`, `settings.context({ adapter }, fn)`,
`new Predict(sig, { adapter })` or `forward(inputs, { adapter })`.

These exports were **removed**:

| Removed                                       | Use instead                                                                        |
| --------------------------------------------- | ---------------------------------------------------------------------------------- |
| `buildPrompt(sig, inputs, demos?)`            | `new ChatAdapter().format({ signature, inputs, demos })` (returns `ChatMessage[]`) |
| `buildPromptContent(sig, inputs, demos?)`     | the same — image inputs become image parts of the user message                     |
| `renderDemos(sig, demos)`                     | pass `demos` to `adapter.format(...)`; they become user/assistant turns            |
| `buildRepairObservation(error)`               | `buildRepairInstruction(error, closing)`                                           |
| `RenderDemosOptions` and related option types | —                                                                                  |

```ts
// Before
import { buildPrompt } from '@ts-dspy/core';
const prompt = buildPrompt(QA, { question: 'Capital of France?' });
const reply = await lm.generate(prompt);

// After
import { ChatAdapter } from '@ts-dspy/core';
const adapter = new ChatAdapter();
const messages = adapter.format({ signature: QA, inputs: { question: 'Capital of France?' } });
const reply = await lm.chat(messages);
const fields = adapter.parse(QA, reply);

// Or let the adapter do the whole round trip, with validation and repair:
const outputs = await adapter.call(lm, {
  signature: QA,
  inputs: { question: 'Capital of France?' },
});
```

`parseOutput` is still exported. Snapshot tests of prompt text, and recorded
cassettes, need regenerating.

## 4. `Predict` uses `chat`

In 0.6, `Predict` sent a single prompt string through `lm.generate()` on a
text-only model. In 0.7 every adapter except `JSONAdapter`'s native path calls
`lm.chat(messages)`; `JSONAdapter` on a structured-output model still calls
`lm.generateStructured()`.

If you implement `ILanguageModel` yourself, make sure `chat` is real rather than
a stub — extending `BaseLM` and implementing `chat` is the simplest route.
Cassettes recorded on 0.6 hold `generate` requests and will not match; record
them again.

## 5. Lazy model resolution

Modules no longer resolve their model when they are constructed. A module
built before `configure()` used to throw; now it works, and the model is chosen
at call time: `options.lm`, then the module's own model, then the active
`settings.context` scope's, then `configure()`'s.

As a consequence `Module.lm` is now `ILanguageModel | undefined` — it holds only
a model given explicitly. Custom modules that called `this.lm` directly must
resolve it instead:

```ts
// Before
class Summarize extends Module {
  async forward(inputs: { text: string }, options?: LLMCallOptions) {
    const summary = await this.lm.generate(`Summarize:\n${inputs.text}`, options);
    return new Prediction({ summary });
  }
}

// After
class Summarize extends Module {
  async forward(inputs: { text: string }, options?: LLMCallOptions) {
    const lm = this.resolveLM(options); // options.lm → this.lm → scope → configure()
    const summary = await lm.generate(`Summarize:\n${inputs.text}`, options);
    return new Prediction({ summary });
  }
}
```

Better still, compose existing modules (a `Predict` field) so the adapter,
tracing, callbacks, usage and optimizers all apply.

## 6. One `Metric` type

0.6 defined `Metric` twice — once for `evaluate` (returning `MetricScore`) and
once for the optimizers (returning `MetricResult`). 0.7 defines it once:

```ts
type MetricResult = number | boolean | { score: number; feedback?: string };

type Metric = (
  example: Example,
  prediction: Prediction,
  trace?: MetricTraceStep[],
  predName?: string,
  predTrace?: MetricTraceStep[]
) => MetricResult | Promise<MetricResult>;
```

- Existing two-argument metrics keep working with both `evaluate` and every
  optimizer.
- `MetricScore` is a deprecated alias of `MetricResult`.
- A metric may now return `{ score, feedback }`. `evaluate` reads the score and
  keeps `feedback` on each result; `GEPA` learns from the feedback.
- Optimizers pass `trace` while bootstrapping, so a metric can be stricter then.
- Code that **consumes** metric results must handle the object form. Use
  `metricScore()` (and `metricFeedback()`) rather than `Number(result)`:

```ts
// Before
const score = typeof result === 'boolean' ? Number(result) : result;

// After
import { metricScore, metricFeedback } from '@ts-dspy/core';
const score = metricScore(result);
const why = metricFeedback(result);
```

## 7. `Example` and `Prediction` are proxies

Both classes now keep their data behind a `Proxy` instead of defining a
property per field. Field access (`example.question`, `prediction.answer`) works
as before. The differences:

- **Methods win.** A field named `get`, `set`, `toObject`, `trace` (and so on)
  no longer replaces the method or property — read it with `get('trace')`.
- **`Object.keys`, spreading and `for…in`** see the data fields, not internal
  state such as `_data` (which no longer exists).
- **`JSON.stringify`** produces the data alone, via the new `toJSON()`:

```ts
const prediction = await predict.forward({ question });

// Before: {"_data":{"answer":"Paris"},"answer":"Paris"} (plus trace when tracing)
// After:  {"answer":"Paris"}
JSON.stringify(prediction);
```

- **Assigning a new field on an `Example`** (`example.label = 'x'`) now writes to
  its data, so `get('label')`, `toObject()` and `withInputs()` see it.
  `Prediction` fields remain read-only.

## 8. Tracing internals

`Module.traced()` keeps its signature, but the lower-level helpers changed — only
code importing from `@ts-dspy/core`'s internals is affected:

```ts
// Before
const span = beginTrace(moduleId, lm, inputs);
span.endCall(rawOutput);

// After
const span = beginTrace(moduleId, inputs, frame?);
span.endCall(rawOutput, usage?); // usage of exactly this call
```

Usage is no longer measured by differencing the model's lifetime counters, so
it is exact under concurrency. `TraceEntry` gains `id` and `parentId` (the
enclosing module call), and each `LMCall.usage` is that call's alone. For
cross-cutting observation prefer `configure({ callbacks })`, or
`@ts-dspy/otel` for OpenTelemetry.

## 9. `evaluate`'s `lm` option is ignored

`evaluate(program, dataset, metric, { lm })` used `lm` to difference usage
counters. Usage is now collected exactly from every call the run makes, whichever
model serves it, so the option is deprecated and ignored. Remove it:

```ts
// Before
const report = await evaluate(program, devset, metric, { lm, concurrency: 8 });

// After
const report = await evaluate(program, devset, metric, { concurrency: 8 });
report.usage.byModel; // per provider/model
```

## 10. Token counts

Two providers now report totals that include tokens they previously left out:

- **Anthropic**: `promptTokens` now **includes** cache reads and cache writes.
  `cachedTokens` reports the cache-read portion.
- **Gemini**: `completionTokens` now **includes** thought tokens.
  `reasoningTokens` reports that portion.

Dashboards or budgets built on 0.6 numbers will see these figures rise for
cached Anthropic prompts and thinking Gemini models. Nothing is billed
differently — the numbers are now complete.

## 11. String signatures are validated

`Signature.parseStringSignature` — and so every string signature — now throws a
`TsDspyError` for malformed input instead of producing a half-parsed signature:
a missing or repeated `->`, an empty side, or an empty or duplicate field name.

```ts
new Predict('question -> answer -> confidence'); // throws: repeated "->"
new Predict('question, question -> answer'); // throws: duplicate field name
```

String signatures also gained features: quoted descriptions, `name?:` optional
markers, and `signature('q -> a', 'instructions')`.

---

# 0.7 → 0.8

0.8 is a hardening release: no API was removed, but several behaviours that
were wrong, or silently lossy, are now strict. Each is listed with what to do
if you relied on the old behaviour.

## Errors

- **Cancelled calls are `TimeoutError`.** OpenAI, Anthropic and Gemini used to
  surface an aborted `signal` (and Gemini a timed-out request) as a plain
  `LMError`. They now raise `TimeoutError`, as its documentation always said.
  The `retry()` middleware does not retry a call whose own signal aborted.
- **OpenAI refusals are `ContentFilterError`** (`category: 'refusal'`), not an
  empty string. A response with no `choices` is an `LMError`.
- **`retry()` also retries dropped connections** (`ECONNRESET`, DNS failures,
  `fetch failed`). Pass `retryOn` to restore the narrower policy.
- **Optimizers throw when every example fails at the provider** — a bad key or
  an outage — instead of returning an unoptimised program with a score of 0.

## Parsing

- **Numbers.** `1,5` is no longer read as `15`, and `0x10` no longer as `16`;
  both fail validation. Thousands separators (`1,234.5`) still work.
- **Arrays.** A `string[]` value that starts with `[` but is not valid JSON
  fails validation instead of being split on commas into fragments.
- **`ChatAdapter` markers** are recognised only at the start of a line, as in
  DSPy. A reply that puts two markers on one line no longer parses as two
  fields.
- **Text fields** accept a nested JSON object or array and store its JSON text.
- **`XMLAdapter`** falls back to label parsing when a reply contains none of the
  signature's tags, rather than reporting every field missing.

## Providers

- **Anthropic sampling.** `temperature` and `topP` are sent only to models that
  accept them (Claude Opus 4.6 / Sonnet 4.6 and earlier). Newer models reject
  them, so they are dropped rather than failing the call.
- **Anthropic forced tool choice.** `toolChoice: 'required'` or a named tool
  becomes `auto` on models that reject a forced choice.
- **`OpenAICompatibleLM`** no longer forwards `OPENAI_ORG_ID` /
  `OPENAI_PROJECT_ID` from the environment. Pass `organization` / `project`
  explicitly if your endpoint wants them.

## Caching and cassettes

- **Cache keys** now include `tools` and `toolChoice` when present. Cached
  entries for tool-calling requests are not reused; everything else is.
- **Cassette keys** now include `tools`, `toolChoice` and `providerOptions`
  when present. Re-record cassettes whose calls used any of them.

## Observability

- **OpenTelemetry error messages** are recorded only with
  `captureContent: true`, because they quote model output. Without it a failed
  span carries `error.type` and a status naming the error class.
- **`gen_ai.provider.name`** for Gemini is now `gcp.gemini`.

## ReAct

- The repeat guard no longer blocks retrying a tool call that **failed**; it
  still refuses to repeat one that succeeded.
