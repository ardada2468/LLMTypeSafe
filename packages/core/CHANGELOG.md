# @ts-dspy/core

## 0.6.0

### Minor Changes

- c3f4ee8: Add `signal` to `LLMCallOptions` so an in-flight provider call can be cancelled.

  `timeout` bounds how long a call may take, but there was no way to drop one whose
  answer nobody is waiting for any more — a React component that unmounted, or a
  server request whose client hung up. `LLMCallOptions.signal` takes any
  `AbortSignal`; the call rejects as soon as it aborts. OpenAI and Anthropic pass it
  straight to their SDK request options. Gemini exposes a single `abortSignal` slot,
  so a caller-supplied signal and the timeout signal are combined with
  `AbortSignal.any()`, built freshly per request.

  The Gemini provider also gains the reliability options the other two already had.
  `GeminiConfig` now accepts `timeout` and `maxRetries` at construction, and per-call
  `retries` is honoured instead of being silently ignored. `@google/genai` reads its
  retry policy from client-level options only — a per-call `retries` cannot be
  expressed through it, and its wrapper replaces API errors with generic ones and
  keeps retrying after an abort — so the provider runs the loop itself: exponential
  backoff on 408/409/429/5xx and transport failures, never on an abort or a client
  error, with the status code preserved on the resulting `LMError`. `maxRetries`
  defaults to 2, as the other two SDKs do, so a Gemini instance built with no
  options now rides out a transient failure the way the others already did.

  Aborting mid-stream now rejects with an `LMError` and increments `errorCount` on
  the OpenAI and Gemini providers, matching Anthropic; previously the raw SDK error
  escaped `generateStream`/`chatStream` uncounted.

- 95eedf2: Add `Module.batch()` and a bounded worker pool.

  Every module — `Predict`, `ChainOfThought`, `RespAct` — now inherits
  `batch(inputs, options)`, which runs a list of inputs with at most
  `concurrency` calls in flight (eight by default). Results come back in
  input order regardless of the order the calls finished in, which is the
  detail hand-rolled loops get wrong: `Promise.all` over fixed-size slices
  stalls each slice on its slowest call, and a queue that pushes results as
  they settle loses the correspondence between row and answer.

  By default a per-input failure is captured rather than thrown, in the shape
  of `Promise.allSettled` — `{ status: 'fulfilled', value }` or
  `{ status: 'rejected', reason }` — so one bad row does not destroy a
  ten-thousand-row job. `stopOnError: true` rejects the whole batch on the
  first failure instead — with the lowest-indexed failure, not whichever one
  landed first — `onProgress` fires as inputs settle, and an `AbortSignal`
  stops new inputs from starting. Both of those reject rather than returning
  the inputs that already finished. A `concurrency` that is not a
  positive integer throws `RangeError` rather than hanging forever. Every
  other option is passed through to each underlying call unchanged.

  The pool underneath is exported as `mapWithConcurrency(items, worker,
options)` with the same guarantees, for rate-limited work that has nothing
  to do with a module.

- 8b13de5: Add `evaluate`, a harness for measuring a program against a dataset, plus the
  built-in metrics that grade it.

  Until now there was no way to tell whether a signature or prompt change made
  things better or worse, which also made optimisation impossible: an optimiser
  is only as good as the number it is climbing. `evaluate(program, dataset,
metric, options)` runs the program over `Example` records — the class has always
  split inputs from outputs via `withInputs()`, which is exactly the split an
  evaluation needs — and returns a report carrying the aggregate score, the
  per-example results, and the tokens and latency the run consumed.

  Failures are recorded, not thrown: an example whose program or metric throws
  comes back as a zero-score result with the error attached, and the run
  continues. An evaluation that dies on row 40 of 500 tells you nothing. Examples
  run with bounded concurrency, defaulting to four in flight.

  Built-in metrics cover the usual grading shapes — `exactMatch`,
  `normalizedMatch` for case- and whitespace-insensitive text, `numericMatch` for
  a tolerance, `fieldAccuracy` for per-field partial credit on a multi-output
  signature, and `tokenF1` for free-text answers — with `matchMetric`,
  `fieldAccuracyMetric`, and `tokenF1Metric` as the configurable factories behind
  them. A `Metric` is just `(example, prediction) => number | boolean`, so a
  metric of your own is a one-line function.

  Usage is obtained by diffing the language model's own counters around the run,
  so it reflects the calls the evaluation made and nothing else. There is still no
  cost figure: a built-in price table goes stale, and the last one reported numbers
  wrong by more than an order of magnitude. `formatReport` renders a report as
  plain text for the caller to print, since the library itself never writes to a
  console.

- b13fe66: Add few-shot demos and optimizers, so a program can improve itself from data
  rather than from prompt edits.

  `Predict` now accepts demos — `new Predict(Sig, { demos })`, or `withDemos()` for
  a configured copy — and renders them into the prompt as worked examples before
  the real input, in the same labelled `field: value` shape the parser reads back.
  A prompt built without demos is byte-for-byte what it was before.

  Two optimizers turn a labelled trainset into those demos. `LabeledFewShot`
  selects _k_ of your own labels and makes no model calls at all. `BootstrapFewShot`
  runs the module over the trainset, scores each attempt with a metric, and
  promotes the runs that passed into demos; a `teacher` option generates them with
  a stronger model that the cheaper student then imitates, so you pay for the
  strong model once, at compile time.

  Both are deterministic given a seed, so a compiled program can be reproduced and
  tested. Trainset runs use bounded concurrency, and an example whose attempt
  throws is skipped rather than failing the whole compile. Progress is reported
  through an optional callback.

  `Predict` also gains `withLM()`, and `renderDemos()` is exported for inspecting
  the few-shot text a set of demos produces.

- 0232800: Add module-level streaming: `Predict.stream()` (inherited and extended by
  `ChainOfThought`) returns an async generator of progressively-filled output
  fields, so a field can be rendered as its tokens arrive. Every provider already
  implemented `chatStream`, but nothing above the provider layer exposed it, which
  left the capability unreachable from a module.

  Each yield is a snapshot of the fields parsed so far; the last yield is the
  complete output, validated against the signature exactly as `forward()`
  validates it, and the generator's return value is the `Prediction` wrapper. A
  stream that ends in something the signature rejects still throws a
  `ValidationError`, so streaming does not opt out of the runtime checks. Only the
  final snapshot is guaranteed to match the declared types, since coercion belongs
  to validation, and snapshots are typed as `PartialOutput<T>` to say so rather
  than claiming a field is a `number` while the model is still writing `'0.'`.

  Both of `complete()`'s paths are covered. Providers with native structured
  output stream JSON, read by a new dependency-free incremental parser exported as
  `parsePartialJson`, which recovers the fields present in a document truncated
  mid-string, mid-key, or after a comma without throwing. Everything else streams
  labelled text through the existing `parseOutput` heuristics over an accumulating
  buffer.

  Models that do not support streaming, or that omit the optional `chatStream`,
  fall back to a single non-streaming call yielded once rather than failing.
  `stream()` also accepts an `AbortSignal`, and abandoning the generator early
  closes the underlying provider stream.

- c7db627: Native tool calling, end to end.

  All three providers reported `supportsFunctionCalling: true` while implementing
  nothing, and `RespAct` ran ReAct purely by text prompting — regex-extracting
  `Action:`/`Action Input:` from raw completions. That capped every tool at exactly
  one string argument, ruled out parallel calls, and left the loop at the mercy of
  the model formatting its output correctly. The flag is now honest.

  `LLMCallOptions` gains `tools` and `toolChoice`, and `ILanguageModel` gains
  `chatWithTools`, which returns text, tool calls, and a normalised finish reason
  from one turn. `BaseLM` supplies a text-only default, so the capability flag —
  not feature detection — is what callers branch on. Each provider translates the
  declarations into its own request shape (OpenAI `tools`/`tool_calls`, Anthropic
  `input_schema`/`tool_use`, Gemini `functionDeclarations`/`functionCall`) and
  reads the calls back out.

  `RespAct` uses that path whenever the model supports it and tools are declared,
  and keeps the text-parsing loop as the fallback for local models and providers
  without native tool calling — the same task completes either way. Tools can now
  declare a JSON Schema or Zod schema for their arguments and receive a validated
  object instead of a single string; bare functions and `{ description, function }`
  keep working unchanged. Parallel tool calls in one turn are executed and reported
  individually, and the whole `RespActEvent` surface stays meaningful on both
  paths. `forceTextMode` pins a tool-capable model to the text loop.

  **Breaking:** `ToolCall` is reshaped for cross-provider use. It was a copy of
  OpenAI's encoding — a required `id`, a `type: 'function'` literal, and a nested
  `function.arguments` JSON _string_ — which no other provider can populate
  faithfully. It is now `{ id?, name, arguments, rawArguments? }`, where
  `arguments` is always a parsed object and `id` is optional because Gemini's
  function calls have none. The dead `ChatMessage.functionCall` field is removed;
  `ChatMessage` gains `toolCallId` to correlate a tool result with its call.

  That correlation also fixes a silent role collapse in all three converters:
  `tool` and `function` turns were downgraded to `user` text, and Anthropic could
  then merge a tool result into the preceding user turn. Anthropic additionally
  dropped `tool_use` blocks on the floor (`textOf` keeps only `text` blocks) and
  ignored `input_json_delta` while streaming; both are now surfaced.

- ae35bd2: Close the gaps left where the 0.6 features met each other.

  Images now reach the model through `Predict` and `ChainOfThought`. A signature
  declaring an `image` input previously had it flattened to an `[image: …]`
  placeholder before the request was built, so the model never saw the picture;
  the prompt now travels as chat content whenever a field is declared `image`,
  and as a plain string otherwise. Structured output over an image asks for the
  schema in the prompt, since the provider methods that constrain decoding accept
  only a string.

  Every provider now overrides `cacheScope()`. Two clients differing only in
  `maxTokens`, `safetySettings`, `baseURL`, or declared capabilities hashed to the
  same cache key, so one could be served a reply the other's configuration would
  never have produced.

  `AnthropicRefusalError` is a subclass of `ContentFilterError` rather than an
  alias of it. As an alias, `instanceof AnthropicRefusalError` also matched OpenAI
  and Gemini content filters; as a subclass, a cross-provider `catch` on
  `ContentFilterError` still works and narrowing to Anthropic means Anthropic
  again.

- 3c950b4: Publish the testing utilities as `@ts-dspy/core/testing`, and add record/replay
  cassettes.

  `MockLM` already existed but was never exported, so every consumer of a library
  that sells runtime validation had to hand-roll a fake model before it could test
  anything. It now ships under a subpath export, with `import`/`require`
  conditions and types for both, and `scripts/verify-packaging.js` imports it from
  both module systems the way a real consumer would.

  `MockLM` also gains `chatStream`/`generateStream`, so it no longer advertises
  capabilities it lacks; its existing API is unchanged.

  `CassetteLM` is new: point it at a JSON file and it replays recorded provider
  replies deterministically, or, given a live model and `mode: 'record'`, captures
  them. Cassettes are a plain array of `{ key, request, response }` entries keyed
  by a hash of the request, so they diff and review like any other fixture. The
  intended shape is to record once against a real provider and then run CI forever
  with no API key and no flake.

- 57e3ad2: Make `configure({ cache })` real. The flag has been exported and unit-tested
  since 0.1, but nothing read it — enabling it bought you nothing.

  `BaseLM` now wraps `generate`, `chat`, and `generateStructured`, so all three
  providers inherit caching without a line of provider code. The key is a SHA-256
  hash of the provider, the model (including a per-call `model` override), the
  prompt or messages, the sampling parameters — `temperature`, `topP`,
  `maxTokens`, `stopSequences`, `frequencyPenalty`, `presencePenalty` — and the
  JSON schema on structured calls, with object keys sorted so property order does
  not split an entry. Transport options such as `timeout` and `retries` are
  excluded, because they cannot change the answer. Errors are never cached: a
  transient 429 must not pin a failure to a prompt for the life of the process.

  Cache hits are kept out of usage accounting. `UsageStats` gains a `cacheHits`
  counter, and a hit increments only that — `requestCount` and the token totals
  keep reflecting real provider traffic, so a figure multiplied by a published
  price stays honest.

  `cache` now accepts an implementation as well as a boolean. `Cache` allows async
  `get`/`set`, so a Redis-, SQLite-, or disk-backed store fits without a wrapper,
  and the new LRU `MemoryCache` — the default for `cache: true`, with a
  configurable `maxSize` — is exported for callers who want to size it themselves.
  `getCache()` and `clearCache()` are exported alongside the existing
  `isCacheEnabled()`.

  BREAKING: caching now defaults to off rather than on. The old default was inert,
  so no behaviour regresses, but a process-wide cache that replays answers for
  repeated prompts changes what a program does — sampling stops varying, agent
  loops stop exploring — so it is opt-in. Call `configure({ cache: true })` to
  turn it on.

- 992718d: Emit strict-mode-correct JSON Schema, and add an `enum` field type.

  `buildOutputJsonSchema` used to describe an `object`/`json` field as
  `{ type: 'object', additionalProperties: true }` and a bare `array`/`list` field
  as `{ type: 'array', items: {} }`. OpenAI's strict structured output rejects
  both: it requires `additionalProperties: false` on every object in the document,
  nested ones included, and will not accept an empty `items` schema. Any signature
  with such a field was therefore refused by the API on the provider path. Objects
  now emit a closed, empty object and arrays declare `items: { type: 'string' }`.

  Two consequences worth knowing, both documented on the site. One schema is built
  per signature and handed to whichever provider is configured, so these shapes
  land everywhere, not only on OpenAI. Strict mode cannot express a free-form
  object at all, which means a bare `object` field is now pinned to `{}` on every
  provider with structured output — declare the keys you want as their own
  signature fields instead. And a bare `array` now tells the provider its elements
  are strings, so a list of figures arrives as `['1', '2']`; declare `number[]`
  when the elements have a type worth naming. Only the text path, taken when a
  model reports `supportsStructuredOutput: false`, is unchanged.

  The new `enum` field type pins an output to a closed set, so the model cannot
  invent a fourth value that still passes validation. Declare members with
  `@OutputField({ type: 'enum', values: ['positive', 'negative', 'neutral'] })`,
  or inline in a string signature as `sentiment: enum(positive|negative|neutral)`
  — pipe-separated, because commas already separate fields. Matching trims and
  ignores case, in the same lenient spirit as the other coercions, and returns the
  declared spelling; anything outside the set is a `ValidationError` that names the
  members. The set is emitted into the provider schema as `enum`, so it constrains
  decoding rather than only the check afterwards, and it is named in the prompt on
  the text path, where nothing else could carry it. An optional enum admits `null`
  into its member list so `type: [base, 'null']` and `enum` do not contradict each
  other. An enum with no members, or a malformed inline declaration such as
  `enum(a|b`, throws rather than degrading to an unconstrained string.

- fca1917: Make `configure({ tracing: true })` do something. Tracing was a flag nothing
  read, and `Prediction.trace` was a field nothing populated — so when a signature
  misbehaved there was no way to see the prompt that had actually been sent.

  Every module invocation now records a `TraceEntry` while tracing is on: the
  prompt, the raw reply, the parsed output, the token usage attributable to that
  invocation, its duration, and the module's id. Multi-step modules
  (`ChainOfThought`, `RespAct`) record each language-model call individually under
  `calls`. Failed invocations are recorded too, with the error attached, because a
  `ValidationError` is exactly when the prompt matters.

  New `inspectHistory(n?)` returns the last `n` entries from a bounded in-memory
  ring buffer — 100 by default, configurable via `traceHistorySize`. New
  `clearHistory()` empties it. `configure({ onTrace })` forwards each entry to
  Langfuse, OpenTelemetry, or your own logger as it is recorded; a handler that
  throws is ignored, so instrumentation cannot fail the run it instruments.

  Tracing stays off by default and costs a single boolean check when off — nothing
  is timed, copied, or stored.

- 9965514: Add a typed error taxonomy so callers can branch on a class instead of sniffing
  HTTP status numbers.

  `RateLimitError`, `AuthError`, `ContextLengthError`, `ContentFilterError` and
  `TimeoutError` now join `LMError`, which they all extend — existing `catch (e) {
if (e instanceof LMError) }` handlers keep working unchanged. A shared
  `classify(status, { type, code })` helper in core picks the class, and each
  provider's `toLMError()` delegates to it with whatever discriminators its SDK
  actually supplies: OpenAI's `code` (the only dependable signal for a
  context-length overflow), Anthropic's typed `error.type` union, and, for Gemini,
  nothing but an HTTP status.

  Content filtering is a 200-response condition on all three providers rather than
  a thrown SDK error, so `ContentFilterError` comes from response inspection.
  `AnthropicRefusalError` is now a deprecated alias of `ContentFilterError`. It is
  an alias of that class rather than a subclass of it, so two things change:
  constructing one directly now takes `(provider, message, options)` instead of
  `(category, explanation)`, and an `instanceof` check under the old name also
  matches an OpenAI or Gemini content filter. Test `error.provider` to tell them
  apart.

  Three bugs fixed along the way:

  - Gemini's `toLMError()` coerced any `status` with `Number(...)`, producing
    `status: NaN` for errors carrying a non-numeric one, such as a Node system
    error.
  - OpenAI never checked `finish_reason === 'content_filter'`, so a filtered
    completion was returned as an empty string with no error at all.
  - Gemini never checked `finishReason === 'MAX_TOKENS'`, so a truncated
    structured reply fell through to `JSON.parse` and surfaced as a misleading
    "not valid JSON" error. It also never checked
    `candidates[].finishReason === 'SAFETY'`.

- b3b0df9: Add opt-in validation self-repair to `Predict` and `ChainOfThought`.

  Runtime validation is the point of this library, but until now a response that
  failed it was simply thrown away. Models frequently produce a nearly-correct
  answer — a number written as prose, a required field left off the end — that one
  more round-trip would fix.

  Call options gain `repairAttempts`, defaulting to `0` so existing behaviour is
  unchanged. When it is greater than zero, a `ValidationError` triggers a follow-up
  prompt naming every failing field with its declared type and the value that
  actually arrived, and the result is re-validated. Once the attempts are spent the
  last `ValidationError` is rethrown, carrying the usual `issues` and `rawOutput`.
  The value is capped at 10, and the loop stops early when an attempt reproduces
  the previous failure exactly — the next prompt would be byte-identical, so
  against a deterministic model the remaining calls cannot do better.

  Both of `Predict`'s paths are covered: the provider's native structured-output
  mode and the labelled-text fallback. `ChainOfThought` repairs the answering step
  only, reusing the reasoning it already has rather than regenerating it.

  `RespAct` already recovered from a malformed `Final Answer` inside its reasoning
  loop. That prompt now comes from the same shared helper as the new `Predict`
  path, so there is one repair wording rather than two that can drift apart. The
  helper is exported as `buildRepairPrompt`, `buildRepairObservation`,
  `describeValidationIssues` and `listFailingFields`.

- 628deed: Send images, not just text. `ChatMessage.content` is widened from `string` to
  `string | ContentPart[]`, where a `ContentPart` is either text or an image
  carried as an `https://` URL, a `data:` URI, or base64 plus a media type. Plain
  strings remain valid content and behave exactly as before, so text-only code —
  `generate()`, `generateStructured()`, and every module — is untouched.

  Signature inputs can now be declared as images with `@ImageField` (or the
  `image` type in a string signature), and the new `buildPromptContent()` renders
  such a signature as content parts, returning a plain string when every input is
  text. `buildPrompt()` still returns a string, rendering an image input as an
  `[image: image/png]` placeholder.

  Each provider maps parts onto its own SDK shape: OpenAI `image_url` parts (only
  on user turns, since system and assistant messages accept text alone),
  Anthropic `image` blocks with a base64 or URL source, and Gemini `inlineData` or
  `fileData`. Anthropic's merging of consecutive same-role turns now concatenates
  block arrays rather than strings; it previously merged only when both turns were
  strings, which silently skipped the merge for image turns and produced two
  adjacent user messages that the Messages API rejects.

  Widening `ChatMessage.content` is a breaking change to a public type — code
  that treats it as a `string` without narrowing will need a narrowing step. Per
  the pre-1.0 convention this ships as a minor.

  `supportsVision` is reported per model rather than hardcoded to `true`: false
  for `gpt-3.5`, `o1-mini` and `o3-mini`, for `claude-3-5-haiku` and older Claude
  models, and for Gemini embedding models.

- e75c7fb: Add zod-native signatures: a third signature form, alongside decorated classes
  and the string shorthand, built with the new `signature()` factory from a pair of
  `z.object()` schemas.

  Decorators record fields at runtime, so TypeScript could never infer per-field
  types from a signature class — callers had to hand-write a `TOutput` type
  argument to get real types back, and the decorator field-type list had no
  spelling for an enum, a union, a nested object, or a numeric bound. A zod
  signature carries its shape in the type system instead, so `Predict`,
  `ChainOfThought` and `RespAct` infer the result type with no type argument, and
  the input keys are typed too. It also needs no `experimentalDecorators`, which
  was a real adoption barrier for projects that cannot enable it.

  The caller's zod schema is used verbatim as the validator, so every constraint
  they express is enforced. Text responses are still coerced leniently — `"42"`
  satisfies a number field, `"a, b"` satisfies a `string[]` — with the coercion
  applied at the object level so optionality, defaults and object-level
  refinements survive. On the provider structured-output path the schema is
  converted with `z.toJSONSchema()` and then rewritten for OpenAI strict mode:
  every property listed in `required`, `additionalProperties: false` on every
  object including nested ones, and optional fields expressed as
  `type: [base, 'null']`.

  Decorator and string signatures are unchanged.

### Patch Changes

- bb1a1c1: Expand npm keywords so the packages surface for the searches people actually
  run — `zod`, `json-schema`, `structured-outputs`, `validation`, `type-safe`,
  `tool-calling`, and per-provider terms like `gpt`, `claude`, and `gemini-api`.

## 0.5.0

### Minor Changes

- ff65bdb: Modernize the whole toolchain and enforce the type-safety guarantee at runtime.

  **Breaking: output validation now throws.** `parseOutput` validates model output
  against the signature with zod and throws `ValidationError` instead of silently
  coercing. Previously a field declared `number` could hold the string
  `"not_a_number"`, and a missing field became `null`, while TypeScript insisted
  otherwise. To migrate: catch `ValidationError` (it carries per-field `issues` and
  the `rawOutput`), or mark fields `required: false` if they are genuinely optional.

  **Breaking: provider packages depended on the wrong core.** `@ts-dspy/openai` and
  `@ts-dspy/gemini` at 0.4.2 declared `@ts-dspy/core@^0.3.0`, so installs resolved a
  stale core. Ranges are now correct and kept in sync by changesets.

  **Breaking: packaging.** Packages are ESM-first with a proper `exports` map and
  dual ESM/CJS builds plus type definitions for both, declare `engines.node >= 22`
  (Node 20 reached end of life in April 2026), and ship `LICENSE`. `@ts-dspy/core` no longer depends on `reflect-metadata` (it was
  imported but never used) and no longer imports `node:fs`, so it runs in edge and
  browser runtimes.

  **Breaking: removed unused surface.** `Module.save`/`Module.load`/`Module.compiled`
  (load always threw), and the unreferenced `ModuleConfig`, `MetricFunction`, and
  `OptimizerOptions` types.

  **New: `@ts-dspy/anthropic`.** Claude provider on the official `@anthropic-ai/sdk`,
  defaulting to `claude-opus-5`, with structured outputs, streaming, and explicit
  handling for `stop_reason: "refusal"`.

  **Providers rewritten on their official SDKs.** OpenAI moves from hand-rolled
  `fetch` to the `openai` SDK, defaulting to `gpt-5.2` (was `gpt-4`); it sends
  `max_completion_tokens` and omits `temperature`/`top_p` unless you set them, so
  reasoning models work. Gemini moves from the deprecated `@google/generative-ai` to
  `@google/genai`, defaulting to `gemini-3.5-flash` (was `gemini-2.0-flash`, now end
  of life), with Vertex AI support. Both gain streaming, native JSON-schema
  structured outputs, and real token usage.

  **Fixed:** `GeminiLM.chat()` mutated the caller's message array via `pop()`;
  Gemini's safety-block check was unreachable because it ran after the call that
  threw; OpenAI's usage reported a `totalCost` computed from hardcoded GPT-3.5
  pricing (wrong by roughly 20x) and a `maxContextLength` hardcoded to 4096 for every
  model — cost estimation is removed rather than left wrong. Field names are now
  escaped before regex interpolation, so a field like `cost($)` parses correctly, and
  the special case for fields named `answer` is gone.

  **Structured outputs are used when available.** `Predict` and `ChainOfThought` call
  the provider's native JSON-schema mode when it has one, and fall back to parsing
  labelled text otherwise. `LLMCallOptions` (including `timeout` and `retries`, which
  previously did nothing) now flows through every module to the provider SDK.

  `RespAct` accepts an `lm` option and an `onEvent` callback, replacing the
  `console.warn` calls it used to make.
