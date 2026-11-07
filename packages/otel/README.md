# @ts-dspy/otel

[OpenTelemetry](https://opentelemetry.io) instrumentation for
[TS-DSPy](https://github.com/ardada2468/LLMTypeSafe). Every module invocation,
language-model call and tool call becomes a span, nested exactly as the calls
were — including when invocations run concurrently.

```bash
npm install @ts-dspy/core @ts-dspy/otel @opentelemetry/api
```

```ts
import { NodeSDK } from '@opentelemetry/sdk-node';
import { instrument } from '@ts-dspy/otel';

new NodeSDK({/* exporter, resource, ... */}).start();
instrument();
```

`instrument()` registers an `OpenTelemetryCallback` globally, next to any
callbacks already configured. To scope it instead, register the callback
yourself:

```ts
import { configure, settings } from '@ts-dspy/core';
import { OpenTelemetryCallback } from '@ts-dspy/otel';

const otel = new OpenTelemetryCallback({ tracer: myTracer });
configure({ callbacks: [otel] });
// or: settings.context({ callbacks: [otel] }, () => program.forward(input));
```

## Spans

| Span                     | Kind     | Key attributes                                                                                                                                                                                                                        |
| ------------------------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ts-dspy.module <Class>` | INTERNAL | `ts-dspy.module.id`, `ts-dspy.module.class`, `ts-dspy.usage.input_tokens` / `output_tokens` (all calls beneath it)                                                                                                                    |
| `chat <model>`           | CLIENT   | `gen_ai.operation.name`, `gen_ai.provider.name` (and `gen_ai.system`), `gen_ai.request.model`, `gen_ai.request.temperature` / `max_tokens` / `top_p`, `gen_ai.usage.input_tokens` / `output_tokens`, `gen_ai.response.finish_reasons` |
| `execute_tool <name>`    | INTERNAL | `gen_ai.operation.name`, `gen_ai.tool.name`                                                                                                                                                                                           |

LM and tool spans follow the
[GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/).
A failed call records the exception, sets `error.type`, and marks the span
`ERROR`.

Spans are parented from ts-dspy's own call tracking rather than from
OpenTelemetry's context, so the hierarchy is right under any concurrency and
with or without a context manager. A top-level program call is parented to the
span active at the time — an HTTP request span, say.

## Content capture

Prompts, completions, module inputs and outputs, and tool arguments and results
are **not** recorded by default: they routinely contain personal data. Opt in
with `captureContent: true`, which records them as `gen_ai.input.messages`,
`gen_ai.output.messages`, `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result`
and `ts-dspy.module.inputs` / `outputs`. Base64 image payloads are omitted even
then.

Requires Node.js 22+. Ships ESM and CommonJS.

Full documentation: <https://github.com/ardada2468/LLMTypeSafe#readme>

## License

MIT
