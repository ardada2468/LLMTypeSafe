import { APICallError } from 'ai';
import { MockLanguageModelV4, convertArrayToReadableStream } from 'ai/test';
import {
    AuthError,
    BaseLM,
    ContentFilterError,
    ContextLengthError,
    LMError,
    Predict,
    RateLimitError,
    ReAct,
    TimeoutError,
    Tool,
    configure,
    type ChatMessage,
} from '@ts-dspy/core';
import { z } from 'zod';
import { AISDKLM, DEFAULT_AI_SDK_CAPABILITIES, toAISDKPrompt, toLMError } from './index';

type GenerateResult = Awaited<ReturnType<MockLanguageModelV4['doGenerate']>>;

function usage(
    input: number,
    output: number,
    extra: { cacheRead?: number; reasoning?: number } = {}
) {
    return {
        inputTokens: {
            total: input,
            noCache: input - (extra.cacheRead ?? 0),
            cacheRead: extra.cacheRead,
            cacheWrite: undefined,
        },
        outputTokens: {
            total: output,
            text: output - (extra.reasoning ?? 0),
            reasoning: extra.reasoning,
        },
    };
}

function textResult(
    text: string,
    finish: GenerateResult['finishReason']['unified'] = 'stop',
    tokens = usage(10, 5)
): GenerateResult {
    return {
        content: [{ type: 'text', text }],
        finishReason: { unified: finish, raw: finish },
        usage: tokens,
        warnings: [],
    };
}

function mockModel(results: GenerateResult[]): MockLanguageModelV4 {
    return new MockLanguageModelV4({
        provider: 'openai.responses',
        modelId: 'gpt-test',
        doGenerate: results,
    });
}

describe('AISDKLM', () => {
    it('is a BaseLM named after the wrapped model', () => {
        const lm = new AISDKLM({ model: mockModel([]) });
        expect(lm).toBeInstanceOf(BaseLM);
        expect(lm.getModelName()).toBe('gpt-test');
        expect((lm as any).provider).toBe('openai');
        expect(new AISDKLM({ model: mockModel([]), provider: 'custom' })['provider']).toBe(
            'custom'
        );
        expect(new AISDKLM({ model: 'openai/gpt-5.2' })['provider']).toBe('gateway');
    });

    it('chats, sending system turns as instructions and recording usage', async () => {
        const model = mockModel([
            textResult('Hello!', 'stop', usage(12, 4, { cacheRead: 8, reasoning: 2 })),
        ]);
        const lm = new AISDKLM({ model });

        const reply = await lm.chat(
            [
                { role: 'system', content: 'Be terse.' },
                { role: 'user', content: 'Hi' },
            ],
            { temperature: 0.3, maxTokens: 50, topP: 0.9, stopSequences: ['END'] }
        );

        expect(reply).toBe('Hello!');
        const call = model.doGenerateCalls[0];
        expect(call.prompt[0]).toMatchObject({ role: 'system', content: 'Be terse.' });
        expect(call.prompt[1]).toMatchObject({
            role: 'user',
            content: [{ type: 'text', text: 'Hi' }],
        });
        expect(call).toMatchObject({
            temperature: 0.3,
            maxOutputTokens: 50,
            topP: 0.9,
            stopSequences: ['END'],
        });

        expect(lm.getUsage()).toMatchObject({
            promptTokens: 12,
            completionTokens: 4,
            totalTokens: 16,
            cachedTokens: 8,
            reasoningTokens: 2,
            requestCount: 1,
        });
    });

    it('sends image parts from URLs, data URIs and base64', async () => {
        // Declaring https image support keeps the AI SDK from downloading the URL.
        const model = new MockLanguageModelV4({
            supportedUrls: { 'image/*': [/^https:\/\//] },
            doGenerate: [textResult('A cat.')],
        });
        const lm = new AISDKLM({ model });
        // AI SDK 7 deprecates the `image` prompt part and warns on every use;
        // images must go as `file` parts so a library never prints that.
        const warn = vi.spyOn(process, 'emitWarning').mockImplementation(() => {});

        await lm.chat([
            {
                role: 'user',
                content: [
                    { type: 'text', text: 'Describe these.' },
                    {
                        type: 'image',
                        source: { kind: 'url', url: 'https://example.com/cat.png' },
                    },
                    {
                        type: 'image',
                        source: { kind: 'url', url: 'data:image/jpeg;base64,/9j/AAAA' },
                    },
                    {
                        type: 'image',
                        source: { kind: 'base64', data: 'iVBORw0K', mediaType: 'image/png' },
                    },
                ],
            },
        ]);

        const content = (model.doGenerateCalls[0].prompt[0] as any).content;
        expect(content[0]).toMatchObject({ type: 'text', text: 'Describe these.' });
        expect(content.slice(1).map((part: any) => part.type)).toEqual([
            'file',
            'file',
            'file',
        ]);
        expect(content[1].mediaType).toMatch(/^image/);
        expect(JSON.stringify(content[1].data)).toContain('https://example.com/cat.png');
        expect(content[2].mediaType).toBe('image/jpeg');
        expect(content[3].mediaType).toBe('image/png');

        const deprecations = warn.mock.calls.filter((call) =>
            JSON.stringify(call).includes('image')
        );
        warn.mockRestore();
        expect(deprecations).toEqual([]);
    });

    it('declares tools without execute and returns the calls', async () => {
        const model = mockModel([
            {
                content: [
                    { type: 'text', text: 'Checking.' },
                    {
                        type: 'tool-call',
                        toolCallId: 'call_1',
                        toolName: 'weather',
                        input: JSON.stringify({ city: 'Paris' }),
                    },
                ],
                finishReason: { unified: 'tool-calls', raw: 'tool_calls' },
                usage: usage(20, 8),
                warnings: [],
            },
        ]);
        const lm = new AISDKLM({ model });

        const result = await lm.chatWithTools(
            [{ role: 'user', content: 'Weather in Paris?' }],
            {
                tools: [
                    {
                        name: 'weather',
                        description: 'Current weather',
                        parameters: {
                            type: 'object',
                            properties: { city: { type: 'string' } },
                            required: ['city'],
                        },
                    },
                ],
                toolChoice: { name: 'weather' },
            }
        );

        expect(result).toEqual({
            content: 'Checking.',
            toolCalls: [{ id: 'call_1', name: 'weather', arguments: { city: 'Paris' } }],
            finishReason: 'tool_calls',
        });
        const call = model.doGenerateCalls[0];
        expect(call.tools).toEqual([
            expect.objectContaining({
                type: 'function',
                name: 'weather',
                description: 'Current weather',
                inputSchema: expect.objectContaining({ required: ['city'] }),
            }),
        ]);
        expect(call.toolChoice).toEqual({ type: 'tool', toolName: 'weather' });
    });

    it('replays tool calls and results, synthesising ids that match by name', () => {
        const messages: ChatMessage[] = [
            { role: 'system', content: 'sys' },
            { role: 'user', content: 'go' },
            {
                role: 'assistant',
                content: '',
                toolCalls: [
                    { id: 'a1', name: 'add', arguments: { a: 1 } },
                    { name: 'mul', arguments: { b: 2 } },
                ],
            },
            { role: 'tool', name: 'add', toolCallId: 'a1', content: '1' },
            { role: 'tool', name: 'mul', content: '2' },
            { role: 'system', content: 'late system note' },
        ];
        const prompt = toAISDKPrompt(messages);

        expect(prompt.instructions).toBe('sys');
        expect(prompt.allowSystemInMessages).toBe(true);
        const assistant = prompt.messages[1] as any;
        const mulId = assistant.content[1].toolCallId;
        expect(assistant.content).toEqual([
            { type: 'tool-call', toolCallId: 'a1', toolName: 'add', input: { a: 1 } },
            { type: 'tool-call', toolCallId: mulId, toolName: 'mul', input: { b: 2 } },
        ]);
        expect((prompt.messages[2] as any).content[0]).toMatchObject({
            toolCallId: 'a1',
            output: { type: 'text', value: '1' },
        });
        expect((prompt.messages[3] as any).content[0].toolCallId).toBe(mulId);
        expect(prompt.messages[4]).toEqual({ role: 'system', content: 'late system note' });
    });

    it('generates structured output through a JSON schema', async () => {
        const model = mockModel([textResult('{"answer":"Paris","confidence":0.9}')]);
        const lm = new AISDKLM({ model });
        const schema = {
            type: 'object',
            properties: { answer: { type: 'string' }, confidence: { type: 'number' } },
            required: ['answer', 'confidence'],
            additionalProperties: false,
        };

        const result = await lm.generateStructured<{ answer: string }>(
            'Capital of France?',
            schema
        );

        expect(result).toEqual({ answer: 'Paris', confidence: 0.9 });
        expect(model.doGenerateCalls[0].responseFormat).toMatchObject({ type: 'json', schema });
    });

    it('reports a truncated structured reply as an LMError', async () => {
        const lm = new AISDKLM({ model: mockModel([textResult('{"answer":"Par', 'length')]) });
        await expect(lm.generateStructured('q', { type: 'object' })).rejects.toThrow(
            /truncated/
        );
        expect(lm.getUsage().errorCount).toBe(1);
    });

    it('reports a structured reply truncated before any text as an LMError', async () => {
        // No text at all: the AI SDK resolves `generateText` and only throws
        // NoOutputGeneratedError when `output` is read.
        const lm = new AISDKLM({
            model: mockModel([
                {
                    content: [],
                    finishReason: { unified: 'length', raw: 'length' },
                    usage: usage(10, 0),
                    warnings: [],
                },
            ]),
        });
        const error: Error = await lm
            .generateStructured<Error>('q', { type: 'object' })
            .catch((e: Error) => e);
        expect(error).toBeInstanceOf(LMError);
        expect(error.message).toMatch(/truncated/);
        expect(lm.getUsage().errorCount).toBe(1);
    });

    it('reports a structured reply with no output for another reason as an LMError', async () => {
        const lm = new AISDKLM({
            model: mockModel([
                {
                    content: [],
                    finishReason: { unified: 'other', raw: 'unknown' },
                    usage: usage(10, 0),
                    warnings: [],
                },
            ]),
        });
        const error: Error = await lm
            .generateStructured<Error>('q', { type: 'object' })
            .catch((e: Error) => e);
        expect(error).toBeInstanceOf(LMError);
        expect(error.message).not.toMatch(/truncated/);
    });

    it('drives Predict with native structured output', async () => {
        const model = mockModel([textResult('{"answer":"4"}')]);
        const predict = new Predict('question -> answer', new AISDKLM({ model }));
        const result = await predict.forward({ question: '2+2?' });
        expect(result.answer).toBe('4');
        expect(model.doGenerateCalls[0].responseFormat?.type).toBe('json');
    });

    it('drives ReAct through native tool calling', async () => {
        const model = mockModel([
            {
                content: [
                    {
                        type: 'tool-call',
                        toolCallId: 'c1',
                        toolName: 'add',
                        input: JSON.stringify({ a: 2, b: 3 }),
                    },
                ],
                finishReason: { unified: 'tool-calls', raw: undefined },
                usage: usage(5, 5),
                warnings: [],
            },
            textResult('The sum is 5'),
            // The extract step, through the JSON adapter.
            textResult('{"reasoning":"2 + 3 = 5","answer":"The sum is 5"}'),
        ]);
        const add = Tool.fromFunction(({ a, b }: { a: number; b: number }) => a + b, {
            name: 'add',
            description: 'Add two numbers',
            parameters: z.object({ a: z.number(), b: z.number() }),
        });
        const agent = new ReAct('question -> answer', {
            lm: new AISDKLM({ model }),
            tools: [add],
        });

        const result = await agent.forward({ question: '2 + 3?' });

        expect(result.answer).toBe('The sum is 5');
        expect(result.trajectory.observation_0).toBe('5');
        // The second turn carries the call and its result back to the model.
        const second = model.doGenerateCalls[1].prompt;
        expect(second.some((message) => message.role === 'tool')).toBe(true);
    });

    it('streams text and finishes with usage and tool calls', async () => {
        const model = new MockLanguageModelV4({
            modelId: 'stream-model',
            doStream: {
                stream: convertArrayToReadableStream([
                    { type: 'stream-start', warnings: [] },
                    { type: 'text-start', id: 't' },
                    { type: 'text-delta', id: 't', delta: 'Hel' },
                    { type: 'text-delta', id: 't', delta: 'lo' },
                    { type: 'text-end', id: 't' },
                    { type: 'tool-input-start', id: 'c1', toolName: 'echo' },
                    { type: 'tool-input-delta', id: 'c1', delta: '{"x":1}' },
                    { type: 'tool-input-end', id: 'c1' },
                    { type: 'tool-call', toolCallId: 'c1', toolName: 'echo', input: '{"x":1}' },
                    {
                        type: 'finish',
                        finishReason: { unified: 'tool-calls', raw: undefined },
                        usage: usage(7, 3),
                    },
                ]),
            },
        });
        const lm = new AISDKLM({ model });

        const chunks = [];
        for await (const chunk of lm.chatStream([{ role: 'user', content: 'hi' }], {
            tools: [{ name: 'echo', parameters: { type: 'object' } }],
        })) {
            chunks.push(chunk);
        }

        expect(
            chunks
                .filter((chunk) => !chunk.done)
                .map((chunk) => chunk.content)
                .join('')
        ).toBe('Hello');
        expect(chunks.some((chunk) => chunk.metadata?.toolInputDelta)).toBe(true);
        const last = chunks.at(-1)!;
        expect(last.done).toBe(true);
        expect(last.usage).toEqual({ promptTokens: 7, completionTokens: 3, totalTokens: 10 });
        expect(last.metadata).toEqual({
            finishReason: 'tool_calls',
            toolCalls: [{ id: 'c1', name: 'echo', arguments: { x: 1 } }],
        });
        expect(lm.getUsage()).toMatchObject({
            promptTokens: 7,
            completionTokens: 3,
            requestCount: 1,
        });
    });

    it('surfaces a stream error as an LMError without logging it', async () => {
        const error = vi.spyOn(console, 'error').mockImplementation(() => {});
        const model = new MockLanguageModelV4({
            doStream: {
                stream: convertArrayToReadableStream([
                    { type: 'stream-start', warnings: [] },
                    {
                        type: 'error',
                        error: new APICallError({
                            message: 'overloaded',
                            url: 'https://api.example.com',
                            requestBodyValues: {},
                            statusCode: 429,
                        }),
                    },
                ]),
            },
        });
        const lm = new AISDKLM({ model, maxRetries: 0 });

        const consume = async () => {
            for await (const _ of lm.chatStream([{ role: 'user', content: 'hi' }])) {
                // drain
            }
        };
        await expect(consume()).rejects.toBeInstanceOf(RateLimitError);
        expect(lm.getUsage().errorCount).toBe(1);
        expect(error).not.toHaveBeenCalled();
        error.mockRestore();
    });

    it('throws ContentFilterError for a filtered reply', async () => {
        const lm = new AISDKLM({ model: mockModel([textResult('', 'content-filter')]) });
        await expect(lm.chat([{ role: 'user', content: 'x' }])).rejects.toBeInstanceOf(
            ContentFilterError
        );
    });

    it('maps API errors to core error classes', async () => {
        const failing = (
            statusCode: number,
            extra: Partial<ConstructorParameters<typeof APICallError>[0]> = {}
        ) =>
            new MockLanguageModelV4({
                doGenerate: async () => {
                    throw new APICallError({
                        message: `HTTP ${statusCode}`,
                        url: 'https://api.example.com',
                        requestBodyValues: {},
                        statusCode,
                        isRetryable: false,
                        ...extra,
                    });
                },
            });

        const call = (model: MockLanguageModelV4) =>
            new AISDKLM({ model, maxRetries: 0 }).chat([{ role: 'user', content: 'x' }]);

        await expect(call(failing(401))).rejects.toBeInstanceOf(AuthError);
        await expect(
            call(failing(400, { data: { error: { code: 'context_length_exceeded' } } }))
        ).rejects.toBeInstanceOf(ContextLengthError);

        const rateLimited = await call(
            failing(429, { responseHeaders: { 'retry-after': '3' } })
        ).catch((error) => error);
        expect(rateLimited).toBeInstanceOf(RateLimitError);
        expect(rateLimited.retryAfterMs).toBe(3000);
        expect(rateLimited.status).toBe(429);

        const generic = await call(failing(500)).catch((error) => error);
        expect(generic).toBeInstanceOf(LMError);
        expect(generic.status).toBe(500);
        expect(generic.provider).toBe('mock-provider');
    });

    it('unwraps the AI SDK RetryError to classify the last attempt', async () => {
        let attempts = 0;
        const model = new MockLanguageModelV4({
            doGenerate: async () => {
                attempts += 1;
                throw new APICallError({
                    message: 'slow down',
                    url: 'https://api.example.com',
                    requestBodyValues: {},
                    statusCode: 429,
                    isRetryable: true,
                    responseHeaders: { 'retry-after-ms': '1' },
                });
            },
        });
        const lm = new AISDKLM({ model });
        const error = await lm
            .chat([{ role: 'user', content: 'x' }], { retries: 1 })
            .catch((caught) => caught);
        expect(attempts).toBe(2);
        expect(error).toBeInstanceOf(RateLimitError);
    });

    it('passes the abort signal and reports an abort as TimeoutError', async () => {
        const controller = new AbortController();
        const model = new MockLanguageModelV4({
            doGenerate: async ({ abortSignal }) =>
                new Promise((_, reject) => {
                    abortSignal?.addEventListener('abort', () => {
                        const error = new Error('aborted');
                        error.name = 'AbortError';
                        reject(error);
                    });
                }),
        });
        const lm = new AISDKLM({ model, maxRetries: 0 });

        const pending = lm.chat([{ role: 'user', content: 'x' }], {
            signal: controller.signal,
        });
        // Abort once the request is in flight, not before it has started.
        setTimeout(() => controller.abort(), 10);
        await expect(pending).rejects.toBeInstanceOf(TimeoutError);
    });

    it('reports default capabilities and accepts overrides', () => {
        expect(new AISDKLM({ model: mockModel([]) }).getCapabilities()).toEqual(
            DEFAULT_AI_SDK_CAPABILITIES
        );
        const local = new AISDKLM({
            model: mockModel([]),
            capabilities: { supportsFunctionCalling: false, maxContextLength: 8192 },
        });
        expect(local.getCapabilities()).toMatchObject({
            supportsFunctionCalling: false,
            supportsStreaming: true,
            maxContextLength: 8192,
        });
    });

    it('applies configured defaults and provider options', async () => {
        const model = mockModel([textResult('ok')]);
        const lm = new AISDKLM({
            model,
            maxTokens: 256,
            headers: { 'x-test': '1' },
            providerOptions: { openai: { reasoningEffort: 'low' } },
        });
        await lm.generate('hi');
        const call = model.doGenerateCalls[0];
        expect(call.maxOutputTokens).toBe(256);
        expect(call.headers).toMatchObject({ 'x-test': '1' });
        expect(call.providerOptions).toEqual({ openai: { reasoningEffort: 'low' } });
    });

    it('merges per-call provider options over the configured ones', async () => {
        const model = mockModel([textResult('ok')]);
        const lm = new AISDKLM({
            model,
            providerOptions: { openai: { reasoningEffort: 'low', user: 'u' } },
        });
        await lm.chat([{ role: 'user', content: 'hi' }], {
            providerOptions: { openai: { reasoningEffort: 'high' }, anthropic: { x: 1 } },
        } as any);
        expect(model.doGenerateCalls[0].providerOptions).toEqual({
            openai: { reasoningEffort: 'high', user: 'u' },
            anthropic: { x: 1 },
        });
    });

    it('attributes usage and cache hits to the bound model when a per-call model is ignored', async () => {
        configure({ trackUsage: true, cache: true });
        try {
            const model = mockModel([textResult('{"answer":"A"}')]);
            const predict = new Predict('question -> answer', new AISDKLM({ model }));

            const first = await predict.forward({ question: 'Q' }, { model: 'other-model' });
            const second = await predict.forward({ question: 'Q' }, { model: 'other-model' });

            expect(model.doGenerateCalls).toHaveLength(1);
            expect(Object.keys(first.getLMUsage())).toEqual(['openai/gpt-test']);
            expect(second.getLMUsage()).toEqual({
                'openai/gpt-test': expect.objectContaining({ cacheHits: 1, requestCount: 0 }),
            });
        } finally {
            configure({ cache: false, trackUsage: false });
        }
    });

    it('toLMError passes LMErrors through and wraps anything else', () => {
        const existing = new LMError('x', 'y');
        expect(toLMError(existing, 'p')).toBe(existing);
        const wrapped = toLMError('boom', 'p');
        expect(wrapped).toBeInstanceOf(LMError);
        expect(wrapped.message).toBe('[p] boom');
    });
});
