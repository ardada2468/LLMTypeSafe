---
'@ts-dspy/core': minor
'@ts-dspy/gemini': patch
---

Runtime foundations:

- **Scoped settings**: `settings.context(overrides, fn)` / `withSettings`, backed by `AsyncLocalStorage`, so concurrent requests can use different models, caches, callbacks and adapters.
- **Lazy LM resolution (breaking)**: modules resolve their model at call time (`options.lm` → module → context → global) and can be constructed before `configure()`.
- **Composition and persistence**: `namedSubModules()`, `namedPredictors()`, `deepCopy()`, `resetCopy()`, `dumpState()`/`loadState()`, `save()`/`load()`; `Predict` gains optimizable `instructions`.
- **Callbacks**: module, LM, tool and adapter start/end events with correct parent ids under concurrency. Trace entries gain `id`/`parentId`.
- **Exact per-call usage**: `Prediction.getLMUsage()`, `trackUsage`, optional `pricing` for `totalCost`, and `cachedTokens`/`reasoningTokens` in `UsageStats`. Trace and evaluation usage no longer diff shared counters.
- **LM middleware**: `wrapLM(lm, retry(), fallback([...]), defaults({...}), rateLimit({...}))`. Gemini now retries through the shared backoff helper.
- **Contracts**: `Tool` (accepted by `RespAct`), `Embedder` + `cosineSimilarity`, and an experimental `Adapter` interface.
