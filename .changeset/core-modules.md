---
'@ts-dspy/core': minor
---

New modules, retrieval and program streaming, after DSPy 3:

- **`BestOfN(module, { n, rewardFn, threshold, failCount?, temperature? })`**: runs a deep copy of the module up to `n` times, each rollout's calls stamped with `temperature` (default 1.0) and a distinct `metadata.rolloutId` — through the module's own models, the scope's model and `options.lm`, so it works through hand-written `forward` methods. Returns the first prediction reaching `threshold`, else the best (a NaN reward ranks lowest); tolerates `failCount` failures, and an abort stops it at once. Nested rollouts compose their ids (`"1/0"`), so an inner `BestOfN` still samples afresh.
- **`Refine`**: `BestOfN` plus, after each failed attempt, an `OfferFeedback` call whose per-predictor advice reaches the next attempt as a `hint_` input. Hints travel in an async-local scope, and the feedback predictor is invisible to `namedPredictors()`/`dumpState()`.
- **`Parallel({ concurrency, maxErrors, returnFailed, signal })`** with `.run([[module, inputs, options?], …])`, results in input order.
- **`MultiChainComparison(signature, { m = 3 })`**: compares `m` reasoning attempts (`completions`) and produces a `rationale` plus the outputs.
- **`ProgramOfThought(signature, { maxIters = 3, timeoutMs })`** _(experimental)_: the model writes JavaScript, run in a fresh `node:vm` context with its own microtask queue under the timeout and no host functions exposed (loaded lazily; still not a security boundary) with errors fed back for regeneration. `runSandboxed` and `extractCode` are exported.
- **Retrieval**: `Retriever` and `Passage` types, `InMemoryVectorStore` over any `Embedder`, a `Retrieve` module returning `{ passages }`, and `configure({ retriever })` / `settings.context({ retriever })`.
- **`streamify(program, { statusMessageProvider?, includeFinal? })`**: runs any program as an async stream of `status`, per-predictor `field` (with deltas) and `prediction` events. Every `Predict` streams while a streamified program is listening, with back-pressure; breaking out aborts the run and closes provider streams.
- **`Predict.stream` is now `forward` with a stream listener**: it emits module callbacks, records trace entries, attributes usage, passes the trace span to the LM call and honours `repairAttempts` — previously it did none of these.
- **Cache keys** include `options.metadata.rolloutId` when it is set; keys without one are unchanged.
- `Adapter.completeFrom(lm, messages, completion, input)` finishes a prediction whose first reply arrived some other way (parse, validate, repair).
