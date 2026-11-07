---
'@ts-dspy/core': minor
'@ts-dspy/openai': minor
'@ts-dspy/anthropic': minor
'@ts-dspy/gemini': minor
---

Provider upgrades:

- **Reasoning controls**: OpenAI `reasoningEffort`, Anthropic `thinking` (adaptive or budgeted) and `effort`, Gemini `thinking` (`thinkingConfig`). Reasoning text is surfaced as `ChatResult.reasoning` and as `metadata.reasoningDelta` on stream chunks.
- **`providerOptions`** on `LLMCallOptions`: per-call, provider-keyed options deep-merged into the raw request.
- **Prompt caching**: Anthropic `cachePrompt` marks the system prompt and conversation prefix as cache breakpoints.
- **Usage**: `cachedTokens` and `reasoningTokens` are reported for all three providers, streaming and non-streaming. Anthropic `promptTokens` now includes cache reads/writes; Gemini `completionTokens` now includes thought tokens.
- **Embeddings**: `OpenAIEmbedder` and `GeminiEmbedder` implement core's `Embedder`.
- Consistent exports across providers and expanded READMEs.
