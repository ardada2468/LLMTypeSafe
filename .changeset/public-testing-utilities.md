---
'@ts-dspy/core': minor
---

Publish the testing utilities as `@ts-dspy/core/testing`, and add record/replay
cassettes.

`MockLM` already existed but was never exported, so every consumer of a library
that sells runtime validation had to hand-roll a fake model before it could test
anything. It now ships under a subpath export, with `import`/`require`
conditions and types for both, and `scripts/verify-packaging.js` imports it from
both module systems the way a real consumer would.

`MockLM` also gains `chatStream`/`generateStream`, so it no longer advertises
capabilities it lacks; its existing API is unchanged.

`CassetteLM` is new: point it at a JSON file and it replays recorded provider
replies deterministically, or, given a live model and `mode: 'record'`, captures
them. Cassettes are a plain array of `{ key, request, response }` entries keyed
by a hash of the request, so they diff and review like any other fixture. The
intended shape is to record once against a real provider and then run CI forever
with no API key and no flake.
