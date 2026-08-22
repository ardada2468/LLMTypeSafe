---
'@ts-dspy/core': minor
---

Make `configure({ tracing: true })` do something. Tracing was a flag nothing
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
