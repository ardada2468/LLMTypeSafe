---
'@ts-dspy/anthropic': minor
'@ts-dspy/gemini': minor
'@ts-dspy/openai': minor
'@ts-dspy/core': minor
---

Close the gaps left where the 0.6 features met each other.

Images now reach the model through `Predict` and `ChainOfThought`. A signature
declaring an `image` input previously had it flattened to an `[image: …]`
placeholder before the request was built, so the model never saw the picture;
the prompt now travels as chat content whenever a field is declared `image`,
and as a plain string otherwise. Structured output over an image asks for the
schema in the prompt, since the provider methods that constrain decoding accept
only a string.

Every provider now overrides `cacheScope()`. Two clients differing only in
`maxTokens`, `safetySettings`, `baseURL`, or declared capabilities hashed to the
same cache key, so one could be served a reply the other's configuration would
never have produced.

`AnthropicRefusalError` is a subclass of `ContentFilterError` rather than an
alias of it. As an alias, `instanceof AnthropicRefusalError` also matched OpenAI
and Gemini content filters; as a subclass, a cross-provider `catch` on
`ContentFilterError` still works and narrowing to Anthropic means Anthropic
again.
