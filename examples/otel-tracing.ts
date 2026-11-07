/**
 * OpenTelemetry tracing with @ts-dspy/otel.
 *
 * `instrument()` registers a callback that turns every module invocation,
 * language-model call and tool call into a span, nested exactly as the calls
 * were. Here the spans go to a console exporter; in production you would
 * register an OTLP exporter (or `@opentelemetry/sdk-node`) instead.
 *
 *   export OPENAI_API_KEY="sk-..."
 *   npm run example:otel
 *
 * Install in your own project with:
 *
 *   npm install @ts-dspy/otel @opentelemetry/api @opentelemetry/sdk-trace-base
 */
import { trace } from '@opentelemetry/api';
import {
    BasicTracerProvider,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
} from '@opentelemetry/sdk-trace-base';
import { ChainOfThought, ReAct, configure } from '@ts-dspy/core';
import { OpenAILM } from '@ts-dspy/openai';
import { instrument, uninstrument } from '@ts-dspy/otel';
import { evaluateArithmetic, requireEnv, section } from './utils';

async function main(): Promise<void> {
    const apiKey = requireEnv('OPENAI_API_KEY');

    // A tracer provider that prints each span as it ends.
    const provider = new BasicTracerProvider({
        spanProcessors: [new SimpleSpanProcessor(new ConsoleSpanExporter())],
    });
    trace.setGlobalTracerProvider(provider);

    // Resolves its tracer through the global provider registered above. Pass
    // `{ tracer }` to use a specific one, and `captureContent: true` to also
    // record prompts, completions and tool arguments (off by default, since
    // they routinely contain personal data).
    instrument();

    configure({ lm: new OpenAILM({ apiKey }) });

    try {
        section('ChainOfThought: one module span, one chat span');
        const cot = new ChainOfThought('question -> answer');
        const { answer } = await cot.forward({ question: 'What is 17 squared?' });
        console.log(`answer: ${answer}`);

        section('ReAct: module, chat and execute_tool spans');
        const agent = new ReAct('question -> answer: float', {
            tools: {
                calculator: {
                    description: 'Evaluate an arithmetic expression, e.g. "(3 + 4) * 2".',
                    function: (expression: string) => evaluateArithmetic(expression),
                },
            },
            maxSteps: 4,
        });
        const result = await agent.forward({ question: 'What is 12.5% of 3,240?' });
        console.log(`answer: ${result.answer}`);
    } finally {
        uninstrument();
        await provider.shutdown();
    }
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});
