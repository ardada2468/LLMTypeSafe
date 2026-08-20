# @ts-dspy/openai

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
