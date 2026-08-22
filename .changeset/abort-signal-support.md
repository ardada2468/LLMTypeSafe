---
'@ts-dspy/anthropic': minor
'@ts-dspy/gemini': minor
'@ts-dspy/openai': minor
'@ts-dspy/core': minor
---

Add `signal` to `LLMCallOptions` so an in-flight provider call can be cancelled.

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
