---
'@ts-dspy/anthropic': minor
'@ts-dspy/gemini': minor
'@ts-dspy/openai': minor
'@ts-dspy/core': minor
---

Add a typed error taxonomy so callers can branch on a class instead of sniffing
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
