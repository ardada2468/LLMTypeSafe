---
'@ts-dspy/core': minor
---

Optimizers, after DSPy 3.x. Every optimizer works on predictor paths (`namedPredictors()`), returns a compiled copy without touching the student, takes `{ seed, concurrency, onProgress }`, and exposes what it did for inspection (`results` / `detailedResults`, also attached to the returned program as a non-enumerable property).

- **`BootstrapFewShotWithRandomSearch`**: zero-shot, labelled-only, unshuffled and seeded shuffled bootstrap candidates, scored on the valset; returns the best with `candidatePrograms` (scores and states) attached.
- **`KNNFewShot`**: a `KNNProgram` wrapper that picks the `k` training examples nearest each call's inputs (any `Embedder`, cosine similarity) as demos, optionally bootstrapping them first.
- **`Ensemble`** and **`majority()`**: run several programs (or a seeded sample of `size`) and reduce their predictions, by default with a majority vote over the output fields.
- **`COPRO`**: per-predictor instruction search, `breadth` proposals over `depth` rounds, each round conditioned on earlier attempts and their scores.
- **`MIPROv2`**: bootstrapped demo-set candidates and grounded instruction proposals (LM dataset summary, program description, demos, a random tip) per predictor, searched with a seeded TPE sampler (`TPESampler`, pure TypeScript) over minibatches with periodic full evaluations; `auto: 'light' | 'medium' | 'heavy'` presets.
- **`SIMBA`**: stochastic mini-batch ascent — runs at a spread of temperatures, finds high-variability examples, and improves by appending a demo from a successful trajectory or a self-reflective rule to the instructions.
- **`GEPA`**: reflective prompt evolution — Pareto-front candidate selection over per-example valset scores, round-robin predictor reflection on train minibatches with metric feedback, merges of frontier candidates sharing an ancestor, metric-call budgets (`auto`, `maxMetricCalls`, `maxFullEvals`), checkpoint/resume through `logDir`, and `detailedResults` with candidates, scores, the Pareto front and lineage.
- **`BootstrapFewShot` assigns demos per predictor**: a `Module` student is traced and each predictor gets demos of its own inputs and outputs from passing runs (identical behaviour for a single `Predict`); a teacher now replaces the model on every predictor. Structural `DemoModule` students keep the previous behaviour.
- **Metrics may return `{ score, feedback }`** and receive `(example, prediction, trace?, predName?, predTrace?)`. `evaluate` reads the score and keeps `feedback` on each result; `metricScore()`/`metricFeedback()` normalise any result.
- Shared helpers: `evaluateProgram`, `runWithTrace`, `bootstrapTraces`, `proposeInstructions`, `summarizeDataset`, `describeProgram`, `sampleMinibatch`, plus `createRng`/`shuffled`. `Predict.getSignature()` returns the signature a predictor sends.
