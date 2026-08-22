---
'@ts-dspy/core': minor
---

Add opt-in validation self-repair to `Predict` and `ChainOfThought`.

Runtime validation is the point of this library, but until now a response that
failed it was simply thrown away. Models frequently produce a nearly-correct
answer — a number written as prose, a required field left off the end — that one
more round-trip would fix.

Call options gain `repairAttempts`, defaulting to `0` so existing behaviour is
unchanged. When it is greater than zero, a `ValidationError` triggers a follow-up
prompt naming every failing field with its declared type and the value that
actually arrived, and the result is re-validated. Once the attempts are spent the
last `ValidationError` is rethrown, carrying the usual `issues` and `rawOutput`.
The value is capped at 10, and the loop stops early when an attempt reproduces
the previous failure exactly — the next prompt would be byte-identical, so
against a deterministic model the remaining calls cannot do better.

Both of `Predict`'s paths are covered: the provider's native structured-output
mode and the labelled-text fallback. `ChainOfThought` repairs the answering step
only, reusing the reasoning it already has rather than regenerating it.

`RespAct` already recovered from a malformed `Final Answer` inside its reasoning
loop. That prompt now comes from the same shared helper as the new `Predict`
path, so there is one repair wording rather than two that can drift apart. The
helper is exported as `buildRepairPrompt`, `buildRepairObservation`,
`describeValidationIssues` and `listFailingFields`.
