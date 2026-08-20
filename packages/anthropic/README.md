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
