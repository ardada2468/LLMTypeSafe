---
'@ts-dspy/core': minor
---

Adapters and signature types:

- **Adapters**: `ChatAdapter` (DSPy's `[[ ## field ## ]]` format, falling back to JSON when a reply ignores it), `JSONAdapter` (native structured output when available), `XMLAdapter` and `TwoStepAdapter({ extractionLM })`, on an `Adapter` base class with `format`/`parse`/`parsePartial`/`call`. Chosen per call (`options.adapter`), per module (`new Predict(sig, { adapter })`), per scope or globally; the default is `JSONAdapter` for structured-output models and `ChatAdapter` otherwise. `Predict.forward` and `Predict.stream` go through the adapter; demos and history are real user/assistant turns, and a repair is an assistant turn plus a corrective user turn. `onAdapterFormat`/`onAdapterParse` callbacks fire.
- **ChainOfThought is one call (breaking)**: the signature gains a leading `reasoning` output; `reasoning` streams like any other field.
- **`ReAct` (breaking)**: `RespAct` is renamed `ReAct` (the old name remains as a deprecated alias) and rebuilt after DSPy: `react` step predictor over inputs + `trajectory` → `next_thought`/`next_tool_name`/`next_tool_args`, then a `ChainOfThought` `extract` step. Works with any signature, not just `question`. Running out of `maxSteps` extracts from the trajectory instead of throwing `MaxStepsError`. Results carry `reasoning`, `trajectory` and `steps`.
- **Signature types**: `History` inputs (`type: 'history'`, zod `history()`, string `history: history`); zod `image()`; optional outputs resolve to `undefined` or their zod default; string signatures take quoted descriptions and `name?:` optional markers; `signature('q -> a', 'instructions')`; `extendSignature()` and `toSignatureClass()`.
- **Removed (breaking)**: `buildPrompt`, `buildPromptContent`, `renderDemos` and their option types (prompts belong to adapters), and `buildRepairObservation` (replaced by `buildRepairInstruction`).
