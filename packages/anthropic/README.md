# @ts-dspy/anthropic

Anthropic Claude provider for
[TS-DSPy](https://github.com/ardada2468/LLMTypeSafe), built on the official
`@anthropic-ai/sdk`. Supports streaming and native JSON-schema structured
outputs.

```bash
npm install @ts-dspy/core @ts-dspy/anthropic
```

```ts
import { AnthropicLM } from '@ts-dspy/anthropic';
import { configure } from '@ts-dspy/core';

const lm = new AnthropicLM({
  apiKey: process.env.ANTHROPIC_API_KEY,
  // model: 'claude-opus-5',   // defaults to the current Opus
  // maxTokens: 16_000,        // the Messages API requires a value
});

configure({ lm });
```

## Options

| Option        | Default                   | Notes                                                                   |
| ------------- | ------------------------- | ----------------------------------------------------------------------- |
| `apiKey`      | `ANTHROPIC_API_KEY`       |                                                                         |
| `model`       | `DEFAULT_ANTHROPIC_MODEL` | Per call: `options.model`.                                              |
| `baseURL`     | Anthropic API             | A gateway or proxy.                                                     |
| `timeout`     | SDK default               | Milliseconds. Per call: `options.timeout`.                              |
| `maxRetries`  | `2`                       | Retried inside the SDK, honouring `retry-after`. Per call: `retries`.   |
| `maxTokens`   | `16_000`                  | Non-streaming default; streams default to `64_000`.                     |
| `thinking`    | unset                     | `{ budgetTokens?, display? }`, or `false` to disable.                   |
| `effort`      | unset                     | `output_config.effort`: `'low'`–`'max'`.                                |
| `cachePrompt` | `false`                   | Adds `cache_control` breakpoints to the system prompt and conversation. |

Per call, `providerOptions.anthropic` accepts `thinking`, `effort` and
`cachePrompt`; any other key is merged into the Messages request body as-is
(`{ anthropic: { metadata: { user_id: 'u-1' } } }`).

## Extended thinking

```ts
// Current models: adaptive thinking, depth steered by effort.
const lm = new AnthropicLM({ thinking: { display: 'summarized' }, effort: 'high' });

// Older models (before Claude 4.6) take a fixed budget instead:
const legacy = new AnthropicLM({ model: 'claude-haiku-4-5', thinking: { budgetTokens: 4096 } });

// Per call:
await lm.chat(messages, { providerOptions: { anthropic: { thinking: false } } });
```

`thinking: {}` sends `{ type: 'adaptive' }`; adding `budgetTokens` sends
`{ type: 'enabled', budget_tokens }`, which current models reject. With a
budget, `max_tokens` is raised to `budget + 1024` if it would not exceed it, as
the API requires. While thinking is on, `temperature` and `top_p` are not sent.

Thinking text never appears in `content`. `chatWithTools` returns it as
`reasoning`, and streams emit `metadata.reasoningDelta` chunks plus the full
`metadata.reasoning` on the final chunk. Current models return empty thinking
text unless `display: 'summarized'` is set.

## Prompt caching

```ts
const lm = new AnthropicLM({ cachePrompt: true });
```

This marks the system prompt with `cache_control: { type: 'ephemeral' }` and, in
a multi-turn request, the turn before the last — so instructions and few-shot
turns are cached while the newest message is not. Prefixes shorter than the
model's minimum cacheable length are silently not cached. Usage reports cache
reads as `cachedTokens`; `promptTokens` includes cache reads and writes, and
thinking tokens appear as `reasoningTokens` (streaming and non-streaming).

## Embeddings

Anthropic has no embeddings endpoint. Pair `AnthropicLM` with `OpenAIEmbedder`
from `@ts-dspy/openai` or `GeminiEmbedder` from `@ts-dspy/gemini`.

## Retries

The SDK retries 408/409/429/5xx itself (`maxRetries`, default 2). For a policy
across providers, use core's middleware with the SDK's retries off:

```ts
import { wrapLM, retry } from '@ts-dspy/core';

const lm = wrapLM(new AnthropicLM({ maxRetries: 0 }), retry({ maxRetries: 4 }));
```

## Refusals

Claude's safety classifiers decline a request with a normal HTTP 200 response
carrying `stop_reason: "refusal"`, not an HTTP error. This provider surfaces that
as a typed error rather than an empty string:

```ts
import { AnthropicRefusalError } from '@ts-dspy/anthropic';

try {
  await lm.generate(prompt);
} catch (error) {
  if (error instanceof AnthropicRefusalError) {
    console.log(`Declined: ${error.category ?? 'unspecified'}`);
  }
}
```

System messages in a conversation are lifted into the API's top-level `system`
parameter, and consecutive same-role turns are merged to satisfy the API's
alternation requirement. `temperature` and `top_p` are never sent together.

Requires Node.js 22+. Ships ESM and CommonJS.

Full documentation: <https://github.com/ardada2468/LLMTypeSafe#readme>

## License

MIT
