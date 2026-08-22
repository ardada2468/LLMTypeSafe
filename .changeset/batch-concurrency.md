---
'@ts-dspy/core': minor
---

Add `Module.batch()` and a bounded worker pool.

Every module — `Predict`, `ChainOfThought`, `RespAct` — now inherits
`batch(inputs, options)`, which runs a list of inputs with at most
`concurrency` calls in flight (eight by default). Results come back in
input order regardless of the order the calls finished in, which is the
detail hand-rolled loops get wrong: `Promise.all` over fixed-size slices
stalls each slice on its slowest call, and a queue that pushes results as
they settle loses the correspondence between row and answer.

By default a per-input failure is captured rather than thrown, in the shape
of `Promise.allSettled` — `{ status: 'fulfilled', value }` or
`{ status: 'rejected', reason }` — so one bad row does not destroy a
ten-thousand-row job. `stopOnError: true` rejects the whole batch on the
first failure instead — with the lowest-indexed failure, not whichever one
landed first — `onProgress` fires as inputs settle, and an `AbortSignal`
stops new inputs from starting. Both of those reject rather than returning
the inputs that already finished. A `concurrency` that is not a
positive integer throws `RangeError` rather than hanging forever. Every
other option is passed through to each underlying call unchanged.

The pool underneath is exported as `mapWithConcurrency(items, worker,
options)` with the same guarantees, for rate-limited work that has nothing
to do with a module.
