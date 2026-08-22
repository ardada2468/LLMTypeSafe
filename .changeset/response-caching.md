---
'@ts-dspy/core': minor
---

Make `configure({ cache })` real. The flag has been exported and unit-tested
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
