---
'@ts-dspy/core': minor
---

Add few-shot demos and optimizers, so a program can improve itself from data
rather than from prompt edits.

`Predict` now accepts demos — `new Predict(Sig, { demos })`, or `withDemos()` for
a configured copy — and renders them into the prompt as worked examples before
the real input, in the same labelled `field: value` shape the parser reads back.
A prompt built without demos is byte-for-byte what it was before.

Two optimizers turn a labelled trainset into those demos. `LabeledFewShot`
selects _k_ of your own labels and makes no model calls at all. `BootstrapFewShot`
runs the module over the trainset, scores each attempt with a metric, and
promotes the runs that passed into demos; a `teacher` option generates them with
a stronger model that the cheaper student then imitates, so you pay for the
strong model once, at compile time.

Both are deterministic given a seed, so a compiled program can be reproduced and
tested. Trainset runs use bounded concurrency, and an example whose attempt
throws is skipped rather than failing the whole compile. Progress is reported
through an optional callback.

`Predict` also gains `withLM()`, and `renderDemos()` is exported for inspecting
the few-shot text a set of demos produces.
