---
'@ts-dspy/openai': minor
---

Add `OpenAICompatibleLM`, a first-class provider for the many servers that speak
the OpenAI chat-completions API — Ollama, LM Studio, vLLM, Groq, Together, and
OpenRouter. `OpenAILM` could already be pointed at them through `baseURL`, but
every default it carries is wrong once you leave `api.openai.com`.

`baseURL` and `model` are now required, since `gpt-5.2` means nothing to Ollama.
The API key defaults to a placeholder for local servers that want an
`Authorization` header but ignore its contents, which removes the confusing auth
failure a user with no `OPENAI_API_KEY` hit before a request was ever sent. Model
capabilities come from config with conservative defaults instead of `OpenAILM`'s
hardcoded optimism: `supportsStructuredOutput` matters most, because `Predict`
branches on it and a wrongly-`true` value makes every call ship a strict
JSON-schema `response_format` that most compatible servers reject outright. The
context window is configurable too, rather than falling through a `gpt-*` prefix
table that never matches `llama-3.3-70b` and silently reports 128k.

Also exports `OPENAI_COMPATIBLE_BASE_URLS` with the known-good endpoint URLs, and
adds `examples/ollama-local.ts` (`npm run example:ollama`), which runs end to end
with no cloud key.
