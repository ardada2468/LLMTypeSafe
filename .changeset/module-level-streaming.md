---
'@ts-dspy/core': minor
---

Add module-level streaming: `Predict.stream()` (inherited and extended by
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
