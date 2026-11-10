---
'@ts-dspy/core': minor
'@ts-dspy/openai': minor
'@ts-dspy/anthropic': minor
'@ts-dspy/gemini': minor
'@ts-dspy/ai-sdk': minor
'@ts-dspy/mcp': minor
'@ts-dspy/otel': minor
---

Production hardening. See MIGRATION.md ("0.7 → 0.8") for the behaviour changes.

**New in core**

- LM middleware `circuitBreaker` (`CircuitOpenError`), `concurrencyLimit` and `budget` (`BudgetExceededError`).
- `evaluate` options `maxErrors` and `failureScore`; `callOptions.signal` now cancels the run.
- `retryAfterMsFrom(headers)`, `buildCacheKey`, and `ToolCall.providerMetadata`.

**Fixed in core**

- Async callbacks and `onTrace` handlers that reject no longer cause unhandled rejections.
- A failing cache backend reads as a miss and never discards a reply; cache keys include `tools` and `toolChoice`.
- `retry`: a `NaN` limit no longer retries forever, a `retry-after` beyond `setTimeout`'s range is honoured, and dropped connections are retried. `rateLimit` returns the slots of calls aborted while queued. `defaults` ignores an explicit `undefined`.
- `Module.batch` passes its `signal` to `forward`.
- JSON replies with a code fence inside a string value parse correctly. Number coercion no longer reads `1,5` as 15 or accepts hex; array coercion no longer splits JSON syntax into elements. `ChatAdapter` markers are recognised only at the start of a line and parse in linear time. Model-chosen keys cannot reach a prototype.
- `ReAct` accepts tool arguments written as a JSON object and lets a failed tool call be retried.
- `ProgramOfThought` caps captured output at 64k characters.
- Optimizers throw when every example fails at the provider, and no longer pass the student's sampling options to proposer or reflection calls. `MIPROv2` returns the student when its own demos beat everything searched. `GEPA` rejects a checkpoint written for different instructions and applies `failureScore` consistently. `maxBootstrappedDemos: 0` bootstraps nothing.
- `tokenF1` and `exactMatch` no longer treat distinct objects or dates as equal.
- Cassettes are written atomically and keyed on `tools`, `toolChoice` and `providerOptions`.

**Providers and integrations**

- OpenAI: a refusal raises `ContentFilterError`, an empty `choices` raises `LMError`, a cancelled call raises `TimeoutError`, and `RateLimitError.retryAfterMs` is populated. `OpenAICompatibleLM` no longer sends `OPENAI_ORG_ID` / `OPENAI_PROJECT_ID` to third-party hosts.
- Anthropic: structured-output schemas are stripped of unsupported constraints; `temperature` / `top_p` are sent only to models that accept them; a forced tool choice becomes `auto` on models that reject one; a cancelled call raises `TimeoutError`; `retryAfterMs` is populated.
- Gemini: function-call thought signatures and ids survive a tool round trip; timeouts and cancellations raise `TimeoutError`; no more SDK `console.warn` on tool-calling turns.
- AI SDK: per-call `providerOptions` are honoured.
- MCP: `serveProgram` cancels the program when the client cancels the call.
- OTel: error messages are recorded only with `captureContent`; token usage is recorded on failed spans; Gemini is reported as `gcp.gemini`.
