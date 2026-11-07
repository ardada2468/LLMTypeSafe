---
'@ts-dspy/core': minor
---

Bug fixes and test infrastructure:

- Object and array inputs now render as JSON in prompts for decorator-class and string signatures (previously `[object Object]`).
- `parseStringSignature` throws a `TsDspyError` for malformed signatures (missing or repeated `->`, empty sides, empty or duplicate field names).
- `RespAct` throws the new `MaxStepsError` when it runs out of steps, and nudges the model back to the format when a reply has neither an Action nor a Final Answer.
- `Example` and `Prediction` store their data behind a Proxy, so fields named `get`, `toObject` or `trace` no longer shadow methods. Both gain `toJSON()`.
- `Metric`/`MetricResult` are defined once (`MetricScore` is a deprecated alias); the duplicate `mapWithConcurrency` and the dead `ISignature` type are removed.
- `evaluate` throws up front when no example designates input keys and no `inputKeys` option is given.
