# @ts-dspy/anthropic

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

### Patch Changes

- bb1a1c1: Expand npm keywords so the packages surface for the searches people actually
  run — `zod`, `json-schema`, `structured-outputs`, `validation`, `type-safe`,
  `tool-calling`, and per-provider terms like `gpt`, `claude`, and `gemini-api`.
- Updated dependencies [c3f4ee8]
- Updated dependencies [95eedf2]
- Updated dependencies [8b13de5]
- Updated dependencies [bb1a1c1]
- Updated dependencies [b13fe66]
- Updated dependencies [0232800]
- Updated dependencies [c7db627]
- Updated dependencies [ae35bd2]
- Updated dependencies [3c950b4]
- Updated dependencies [57e3ad2]
- Updated dependencies [992718d]
- Updated dependencies [fca1917]
- Updated dependencies [9965514]
- Updated dependencies [b3b0df9]
- Updated dependencies [628deed]
- Updated dependencies [e75c7fb]
  - @ts-dspy/core@0.6.0

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

### Patch Changes

- Updated dependencies [ff65bdb]
  - @ts-dspy/core@0.5.0
