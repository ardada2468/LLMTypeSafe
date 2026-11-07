import {
    BasicTracerProvider,
    InMemorySpanExporter,
    SimpleSpanProcessor,
    type ReadableSpan,
} from '@opentelemetry/sdk-trace-base';
import { SpanKind, SpanStatusCode, context, trace } from '@opentelemetry/api';
import {
    BaseLM,
    Module,
    Predict,
    ReAct,
    Tool,
    configure,
    getCallbacks,
    settings,
    type ChatMessage,
    type ChatResult,
    type LLMCallOptions,
    type ModelCapabilities,
} from '@ts-dspy/core';
import { MockLM } from '@ts-dspy/core/testing';
import { z } from 'zod';
import { OpenTelemetryCallback, instrument, uninstrument } from './index';

/** Replies from a script of `ChatResult`s, reports native tool calling and token usage. */
class ScriptedLM extends BaseLM {
    constructor(
        private readonly script: ChatResult[],
        private readonly delayMs = 0
    ) {
        super('scripted', 'scripted-1');
    }

    async chat(messages: ChatMessage[], options?: LLMCallOptions): Promise<string> {
        return (await this.chatWithTools(messages, options)).content;
    }

    async chatWithTools(
        _messages?: ChatMessage[],
        _options?: LLMCallOptions
    ): Promise<ChatResult> {
        if (this.delayMs) await new Promise((resolve) => setTimeout(resolve, this.delayMs));
        const next = this.script.shift();
        if (!next) throw new Error('script exhausted');
        this.recordUsage({ promptTokens: 11, completionTokens: 7, cachedTokens: 3 });
        return next;
    }

    getCapabilities(): ModelCapabilities {
        return {
            supportsStreaming: false,
            supportsStructuredOutput: false,
            supportsFunctionCalling: true,
            supportsVision: false,
            maxContextLength: 8192,
            supportedFormats: ['text'],
        };
    }
}

const answer = (text: string) => `[[ ## answer ## ]]\n${text}\n\n[[ ## completed ## ]]`;

let exporter: InMemorySpanExporter;
let provider: BasicTracerProvider;

function setup(captureContent = false): OpenTelemetryCallback {
    exporter = new InMemorySpanExporter();
    provider = new BasicTracerProvider({ spanProcessors: [new SimpleSpanProcessor(exporter)] });
    return new OpenTelemetryCallback({ tracer: provider.getTracer('test'), captureContent });
}

/** Run `fn` with `callback` registered for its scope only. */
function traced<T>(callback: OpenTelemetryCallback, fn: () => Promise<T>): Promise<T> {
    return settings.context({ callbacks: [callback] }, fn);
}

function spans(): ReadableSpan[] {
    return exporter.getFinishedSpans();
}

function byName(name: string): ReadableSpan[] {
    return spans().filter((span) => span.name === name);
}

function parentOf(span: ReadableSpan): ReadableSpan | undefined {
    const parentId = span.parentSpanContext?.spanId;
    return spans().find((candidate) => candidate.spanContext().spanId === parentId);
}

afterEach(async () => {
    configure({ callbacks: [] });
    await provider?.shutdown();
});

describe('OpenTelemetryCallback', () => {
    it('creates a module span with a nested LM span carrying GenAI attributes', async () => {
        const callback = setup();
        const lm = new MockLM({ responses: [answer('Paris')] });
        const predict = new Predict('question -> answer', lm);

        await traced(callback, () =>
            predict.forward({ question: 'Capital?' }, { temperature: 0.2, maxTokens: 64 })
        );

        const [moduleSpan] = byName('ts-dspy.module Predict');
        const [lmSpan] = byName('chat mock-model');
        expect(moduleSpan).toBeDefined();
        expect(lmSpan).toBeDefined();

        expect(moduleSpan.kind).toBe(SpanKind.INTERNAL);
        expect(moduleSpan.attributes['ts-dspy.module.class']).toBe('Predict');
        expect(moduleSpan.attributes['ts-dspy.module.id']).toBe(predict.moduleId);
        expect(moduleSpan.parentSpanContext).toBeUndefined();

        expect(parentOf(lmSpan)).toBe(moduleSpan);
        expect(lmSpan.spanContext().traceId).toBe(moduleSpan.spanContext().traceId);
        expect(lmSpan.kind).toBe(SpanKind.CLIENT);
        expect(lmSpan.attributes).toMatchObject({
            'gen_ai.operation.name': 'chat',
            'gen_ai.provider.name': 'mock',
            'gen_ai.system': 'mock',
            'gen_ai.request.model': 'mock-model',
            'gen_ai.request.temperature': 0.2,
            'gen_ai.request.max_tokens': 64,
            'gen_ai.response.model': 'mock-model',
            'ts-dspy.module.id': predict.moduleId,
        });
        expect(typeof lmSpan.attributes['gen_ai.usage.input_tokens']).toBe('number');
        expect(typeof lmSpan.attributes['gen_ai.usage.output_tokens']).toBe('number');
        // Content capture is off by default.
        expect(lmSpan.attributes['gen_ai.input.messages']).toBeUndefined();
        expect(lmSpan.attributes['gen_ai.output.messages']).toBeUndefined();
        expect(moduleSpan.attributes['ts-dspy.module.inputs']).toBeUndefined();
        expect(lmSpan.status.code).not.toBe(SpanStatusCode.ERROR);
    });

    it('nests LM and tool spans under a ReAct run, with usage and finish reasons', async () => {
        const callback = setup();
        const lm = new ScriptedLM([
            {
                content: '',
                toolCalls: [{ id: 'c1', name: 'add', arguments: { a: 2, b: 3 } }],
                finishReason: 'tool_calls',
            },
            { content: 'It is 5', finishReason: 'stop' },
        ]);
        const add = Tool.fromFunction(({ a, b }: { a: number; b: number }) => a + b, {
            name: 'add',
            description: 'Add two numbers',
            parameters: z.object({ a: z.number(), b: z.number() }),
        });
        const agent = new ReAct('question -> answer', { lm, tools: [add] });

        const result = await traced(callback, () => agent.forward({ question: '2 + 3?' }));
        expect(result.answer).toBe('It is 5');

        const [root] = byName('ts-dspy.module ReAct');
        const lmSpans = byName('chat scripted-1');
        const [toolSpan] = byName('execute_tool add');
        expect(root.parentSpanContext).toBeUndefined();
        expect(lmSpans).toHaveLength(2);

        for (const span of lmSpans) {
            // Under the ReAct module span, or a predictor nested inside it.
            let ancestor = parentOf(span);
            while (ancestor && ancestor !== root) ancestor = parentOf(ancestor);
            expect(ancestor).toBe(root);
            expect(span.attributes).toMatchObject({
                'gen_ai.provider.name': 'scripted',
                'gen_ai.usage.input_tokens': 11,
                'gen_ai.usage.output_tokens': 7,
                'gen_ai.usage.cache_read.input_tokens': 3,
                'ts-dspy.lm.method': 'chatWithTools',
            });
        }
        expect(
            lmSpans.map((span) => span.attributes['gen_ai.response.finish_reasons'])
        ).toEqual([['tool_calls'], ['stop']]);

        expect(toolSpan.attributes).toMatchObject({
            'gen_ai.operation.name': 'execute_tool',
            'gen_ai.tool.name': 'add',
        });
        let ancestor = parentOf(toolSpan);
        while (ancestor && ancestor !== root) ancestor = parentOf(ancestor);
        expect(ancestor).toBe(root);

        // The module span totals every call beneath it.
        expect(root.attributes['ts-dspy.usage.input_tokens']).toBe(22);
        expect(root.attributes['ts-dspy.usage.output_tokens']).toBe(14);
    });

    it('parents spans correctly when invocations run concurrently', async () => {
        const callback = setup();

        class Pipeline extends Module {
            constructor(
                private readonly label: string,
                private readonly delayMs: number
            ) {
                super();
                this.first = new Predict(
                    'question -> answer',
                    new ScriptedLM([{ content: answer(`${label}-1`) }], delayMs)
                );
                this.second = new Predict(
                    'question -> answer',
                    new ScriptedLM([{ content: answer(`${label}-2`) }], delayMs)
                );
            }
            first: Predict;
            second: Predict;

            async forward(inputs: Record<string, any>) {
                const a = await this.first.forward(inputs);
                return this.second.forward({ question: `${this.label}:${a.answer}` });
            }
        }

        // Interleaved on purpose: the slower pipeline's calls start first and
        // finish last, so any "most recent open span" heuristic would misparent.
        const slow = new Pipeline('slow', 15);
        const fast = new Pipeline('fast', 1);
        const instant = new Pipeline('instant', 0);
        await traced(callback, () =>
            Promise.all([
                slow.forward({ question: 'q' }),
                fast.forward({ question: 'q' }),
                instant.forward({ question: 'q' }),
            ])
        );

        const roots = byName('ts-dspy.module Pipeline');
        expect(roots).toHaveLength(3);
        // Three separate traces.
        expect(new Set(roots.map((span) => span.spanContext().traceId)).size).toBe(3);

        for (const root of roots) {
            const children = spans().filter((span) => parentOf(span) === root);
            expect(children.map((span) => span.name)).toEqual([
                'ts-dspy.module Predict',
                'ts-dspy.module Predict',
            ]);
            for (const child of children) {
                const grandchildren = spans().filter((span) => parentOf(span) === child);
                expect(grandchildren.map((span) => span.name)).toEqual(['chat scripted-1']);
                expect(grandchildren[0].spanContext().traceId).toBe(root.spanContext().traceId);
            }
        }
        // 3 pipelines × (1 + 2 predictors + 2 LM calls).
        expect(spans()).toHaveLength(15);
    });

    it('records errors with recordException and an ERROR status', async () => {
        const callback = setup();
        const lm = new ScriptedLM([]);
        const predict = new Predict('question -> answer', lm);

        await expect(
            traced(callback, () => predict.forward({ question: 'q' }))
        ).rejects.toThrow('script exhausted');

        const [moduleSpan] = byName('ts-dspy.module Predict');
        const [lmSpan] = byName('chat scripted-1');
        for (const span of [moduleSpan, lmSpan]) {
            expect(span.status).toEqual({
                code: SpanStatusCode.ERROR,
                message: 'script exhausted',
            });
            expect(span.attributes['error.type']).toBe('Error');
            const [exception] = span.events;
            expect(exception.name).toBe('exception');
            expect(exception.attributes?.['exception.message']).toBe('script exhausted');
        }
    });

    it('records a failing tool on its own span', async () => {
        const callback = setup();
        const broken = Tool.fromFunction(
            () => {
                throw new TypeError('nope');
            },
            { name: 'broken' }
        );
        await traced(callback, () => broken.call({}).catch(() => undefined));

        const [toolSpan] = byName('execute_tool broken');
        expect(toolSpan.status.code).toBe(SpanStatusCode.ERROR);
        expect(toolSpan.attributes['error.type']).toBe('TypeError');
    });

    it('captures prompts, completions and tool payloads only when asked', async () => {
        const callback = setup(true);
        const lm = new MockLM({ responses: [answer('Paris')] });
        await traced(callback, () =>
            new Predict('question -> answer', lm).forward({ question: 'Capital?' })
        );
        const add = Tool.fromFunction(({ a }: { a: number }) => a * 2, { name: 'double' });
        await traced(callback, () => add.call({ a: 21 }));

        const [moduleSpan] = byName('ts-dspy.module Predict');
        const [lmSpan] = byName('chat mock-model');
        const [toolSpan] = byName('execute_tool double');

        expect(JSON.parse(String(moduleSpan.attributes['ts-dspy.module.inputs']))).toEqual({
            question: 'Capital?',
        });
        expect(JSON.parse(String(moduleSpan.attributes['ts-dspy.module.outputs']))).toEqual({
            answer: 'Paris',
        });
        const input = JSON.parse(String(lmSpan.attributes['gen_ai.input.messages']));
        expect(JSON.stringify(input)).toContain('Capital?');
        const output = JSON.parse(String(lmSpan.attributes['gen_ai.output.messages']));
        expect(output[0]).toMatchObject({ role: 'assistant' });
        expect(JSON.stringify(output)).toContain('Paris');
        expect(toolSpan.attributes['gen_ai.tool.call.arguments']).toBe('{"a":21}');
        expect(toolSpan.attributes['gen_ai.tool.call.result']).toBe('42');
    });

    it('omits base64 image payloads from captured messages', async () => {
        const callback = setup(true);
        const lm = new MockLM({
            responses: [answer('a cat')],
            capabilities: { supportsVision: true },
        });
        await traced(callback, () =>
            new Predict('question: string, photo: image -> answer', lm).forward({
                question: 'What is this?',
                photo: `data:image/png;base64,${'A'.repeat(5000)}`,
            })
        );
        const [lmSpan] = byName('chat mock-model');
        const captured = String(lmSpan.attributes['gen_ai.input.messages']);
        expect(captured).toContain('What is this?');
        expect(captured).toContain('[omitted]');
        expect(captured).not.toContain('AAAA');
    });

    it('parents a top-level call to the active OpenTelemetry span', async () => {
        const callback = setup();
        const tracer = provider.getTracer('app');
        const request = tracer.startSpan('http.request');
        const lm = new MockLM({ responses: [answer('x')] });

        // No context manager is registered in this test, so pass the parent
        // explicitly through the callback's view of the active context.
        const activeSpy = vi
            .spyOn(context, 'active')
            .mockReturnValue(trace.setSpan(context.active(), request));
        try {
            await traced(callback, () =>
                new Predict('question -> answer', lm).forward({ question: 'q' })
            );
        } finally {
            activeSpy.mockRestore();
            request.end();
        }

        const [moduleSpan] = byName('ts-dspy.module Predict');
        expect(moduleSpan.parentSpanContext?.spanId).toBe(request.spanContext().spanId);
        expect(moduleSpan.spanContext().traceId).toBe(request.spanContext().traceId);
    });
});

describe('instrument', () => {
    it('registers one callback globally, alongside existing ones, and can be undone', async () => {
        const other = { onModuleStart: vi.fn() };
        configure({ callbacks: [other] });
        exporter = new InMemorySpanExporter();
        provider = new BasicTracerProvider({
            spanProcessors: [new SimpleSpanProcessor(exporter)],
        });

        instrument({ tracer: provider.getTracer('test') });
        const second = instrument({ tracer: provider.getTracer('test') });
        expect(getCallbacks()).toEqual([other, second]);

        await new Predict(
            'question -> answer',
            new MockLM({ responses: [answer('y')] })
        ).forward({ question: 'q' });
        expect(byName('ts-dspy.module Predict')).toHaveLength(1);
        expect(other.onModuleStart).toHaveBeenCalledTimes(1);

        uninstrument();
        expect(getCallbacks()).toEqual([other]);
    });

    it('defaults to the global tracer provider', () => {
        const callback = instrument();
        expect(callback).toBeInstanceOf(OpenTelemetryCallback);
        uninstrument();
    });
});
