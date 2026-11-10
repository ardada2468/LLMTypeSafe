---
'@ts-dspy/core': minor
---

Add production LM middleware: `circuitBreaker` (fail fast with `CircuitOpenError` while a model keeps failing), `concurrencyLimit` (bound calls in flight, streams included) and `budget` (throw `BudgetExceededError` once a request, token or cost limit is reached).
