---
'@ts-dspy/core': minor
---

Add `evaluate`, a harness for measuring a program against a dataset, plus the
built-in metrics that grade it.

Until now there was no way to tell whether a signature or prompt change made
things better or worse, which also made optimisation impossible: an optimiser
is only as good as the number it is climbing. `evaluate(program, dataset,
metric, options)` runs the program over `Example` records — the class has always
split inputs from outputs via `withInputs()`, which is exactly the split an
evaluation needs — and returns a report carrying the aggregate score, the
per-example results, and the tokens and latency the run consumed.

Failures are recorded, not thrown: an example whose program or metric throws
comes back as a zero-score result with the error attached, and the run
continues. An evaluation that dies on row 40 of 500 tells you nothing. Examples
run with bounded concurrency, defaulting to four in flight.

Built-in metrics cover the usual grading shapes — `exactMatch`,
`normalizedMatch` for case- and whitespace-insensitive text, `numericMatch` for
a tolerance, `fieldAccuracy` for per-field partial credit on a multi-output
signature, and `tokenF1` for free-text answers — with `matchMetric`,
`fieldAccuracyMetric`, and `tokenF1Metric` as the configurable factories behind
them. A `Metric` is just `(example, prediction) => number | boolean`, so a
metric of your own is a one-line function.

Usage is obtained by diffing the language model's own counters around the run,
so it reflects the calls the evaluation made and nothing else. There is still no
cost figure: a built-in price table goes stale, and the last one reported numbers
wrong by more than an order of magnitude. `formatReport` renders a report as
plain text for the caller to print, since the library itself never writes to a
console.
