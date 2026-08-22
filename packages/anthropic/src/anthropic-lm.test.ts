import { LMError } from '@ts-dspy/core';
import {
    AnthropicLM,
    AnthropicRefusalError,
    toAnthropicMessages,
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_MAX_TOKENS,
} from './anthropic-lm';

const mocks = vi.hoisted(() => {
    class MockAPIError extends Error {
        status: number;
        constructor(status: number, message: string) {
            super(message);
            this.status = status;
        }
    }
    return { create: vi.fn(), stream: vi.fn(), MockAPIError };
});

const { MockAPIError } = mocks;

vi.mock('@anthropic-ai/sdk', () => ({
    default: class {
        messages = { create: mocks.create, stream: mocks.stream };
        constructor(public options: unknown) {}
    },
    APIError: mocks.MockAPIError,
}));

function message(text: string, extra: Record<string, unknown> = {}) {
    return {
        content: [{ type: 'text', text }],
        stop_reason: 'end_turn',
        usage: { input_tokens: 14, output_tokens: 9 },
        ...extra,
    };
}

beforeEach(() => {
    mocks.create.mockReset();
    mocks.stream.mockReset();
});

describe('AnthropicLM', () => {
    it('defaults to the current Opus model with no date suffix', () => {
        expect(new AnthropicLM({ apiKey: 'k' }).getModelName()).toBe(DEFAULT_ANTHROPIC_MODEL);
        expect(DEFAULT_ANTHROPIC_MODEL).toBe('claude-opus-5');
        expect(DEFAULT_ANTHROPIC_MODEL).not.toMatch(/\d{8}$/);
    });

    it('concatenates text blocks from the response', async () => {
        mocks.create.mockResolvedValue({
            content: [
                { type: 'text', text: 'Hello ' },
                { type: 'thinking', thinking: 'ignored' },
                { type: 'text', text: 'there' },
            ],
            stop_reason: 'end_turn',
            usage: { input_tokens: 1, output_tokens: 2 },
        });

        expect(await new AnthropicLM({ apiKey: 'k' }).generate('Hi')).toBe('Hello there');
    });

    it('always sends the required max_tokens', async () => {
        mocks.create.mockResolvedValue(message('ok'));
        await new AnthropicLM({ apiKey: 'k' }).generate('Hi');

        expect(mocks.create.mock.calls[0][0].max_tokens).toBe(DEFAULT_MAX_TOKENS);
    });

    it('records token usage', async () => {
        mocks.create.mockResolvedValue(message('ok'));
        const lm = new AnthropicLM({ apiKey: 'k' });
        await lm.generate('Hi');

        expect(lm.getUsage()).toMatchObject({
            promptTokens: 14,
            completionTokens: 9,
            totalTokens: 23,
        });
    });

    describe('refusals', () => {
        it('throws instead of returning empty content', async () => {
            mocks.create.mockResolvedValue({
                content: [],
                stop_reason: 'refusal',
                stop_details: { type: 'refusal', category: 'cyber', explanation: 'declined' },
                usage: { input_tokens: 5, output_tokens: 0 },
            });

            const lm = new AnthropicLM({ apiKey: 'k' });
            await expect(lm.generate('Hi')).rejects.toThrow(AnthropicRefusalError);
        });

        it('surfaces the refusal category', async () => {
            mocks.create.mockResolvedValue({
                content: [],
                stop_reason: 'refusal',
                stop_details: { type: 'refusal', category: 'bio' },
                usage: { input_tokens: 5, output_tokens: 0 },
            });

            await expect(new AnthropicLM({ apiKey: 'k' }).generate('Hi')).rejects.toMatchObject(
                {
                    category: 'bio',
                }
            );
        });

        it('tolerates a null stop_details', async () => {
            mocks.create.mockResolvedValue({
                content: [],
                stop_reason: 'refusal',
                stop_details: null,
                usage: { input_tokens: 5, output_tokens: 0 },
            });

            await expect(new AnthropicLM({ apiKey: 'k' }).generate('Hi')).rejects.toThrow(
                AnthropicRefusalError
            );
        });
    });

    describe('sampling parameters', () => {
        it('sends nothing when the caller sets nothing', async () => {
            mocks.create.mockResolvedValue(message('ok'));
            await new AnthropicLM({ apiKey: 'k' }).generate('Hi');

            const body = mocks.create.mock.calls[0][0];
            expect(body).not.toHaveProperty('temperature');
            expect(body).not.toHaveProperty('top_p');
        });

        it('never sends temperature and top_p together', async () => {
            mocks.create.mockResolvedValue(message('ok'));
            await new AnthropicLM({ apiKey: 'k' }).generate('Hi', {
                temperature: 0.5,
                topP: 0.9,
            });

            const body = mocks.create.mock.calls[0][0];
            expect(body.temperature).toBe(0.5);
            expect(body).not.toHaveProperty('top_p');
        });

        it('falls back to top_p when only top_p is given', async () => {
            mocks.create.mockResolvedValue(message('ok'));
            await new AnthropicLM({ apiKey: 'k' }).generate('Hi', { topP: 0.9 });

            expect(mocks.create.mock.calls[0][0].top_p).toBe(0.9);
        });

        it('maps timeout and retries to SDK request options', async () => {
            mocks.create.mockResolvedValue(message('ok'));
            await new AnthropicLM({ apiKey: 'k' }).generate('Hi', {
                timeout: 3000,
                retries: 4,
            });

            expect(mocks.create.mock.calls[0][1]).toEqual({ timeout: 3000, maxRetries: 4 });
        });
    });

    describe('generateStructured', () => {
        it('uses output_config.format with the supplied schema', async () => {
            mocks.create.mockResolvedValue(message('{"answer":"Paris"}'));
            const schema = { type: 'object', properties: { answer: { type: 'string' } } };

            const result = await new AnthropicLM({ apiKey: 'k' }).generateStructured(
                'Q',
                schema
            );

            expect(result).toEqual({ answer: 'Paris' });
            expect(mocks.create.mock.calls[0][0].output_config).toEqual({
                format: { type: 'json_schema', schema },
            });
        });

        it('does not use assistant prefill', async () => {
            mocks.create.mockResolvedValue(message('{"a":1}'));
            await new AnthropicLM({ apiKey: 'k' }).generateStructured('Q', {});

            const messages = mocks.create.mock.calls[0][0].messages;
            expect(messages.at(-1).role).toBe('user');
        });

        it('raises a clear error when the response is truncated', async () => {
            mocks.create.mockResolvedValue(message('{"a"', { stop_reason: 'max_tokens' }));

            await expect(
                new AnthropicLM({ apiKey: 'k' }).generateStructured('Q', {})
            ).rejects.toThrow(/truncated/);
        });

        it('checks for a refusal before parsing', async () => {
            mocks.create.mockResolvedValue({
                content: [],
                stop_reason: 'refusal',
                stop_details: { type: 'refusal', category: 'cyber' },
                usage: { input_tokens: 1, output_tokens: 0 },
            });

            await expect(
                new AnthropicLM({ apiKey: 'k' }).generateStructured('Q', {})
            ).rejects.toThrow(AnthropicRefusalError);
        });
    });

    describe('streaming', () => {
        it('yields text deltas then a final usage chunk', async () => {
            const events = [
                { type: 'content_block_delta', delta: { type: 'text_delta', text: 'Hel' } },
                { type: 'content_block_delta', delta: { type: 'text_delta', text: 'lo' } },
                { type: 'message_stop' },
            ];
            mocks.stream.mockReturnValue({
                async *[Symbol.asyncIterator]() {
                    yield* events;
                },
                finalMessage: async () => message('Hello'),
            });

            const chunks = [];
            for await (const chunk of new AnthropicLM({ apiKey: 'k' }).generateStream('Hi')) {
                chunks.push(chunk);
            }

            expect(chunks.filter((c) => !c.done).map((c) => c.content)).toEqual(['Hel', 'lo']);
            expect(chunks.at(-1)).toMatchObject({
                done: true,
                usage: { promptTokens: 14, completionTokens: 9, totalTokens: 23 },
            });
        });

        it('forwards tool-argument deltas and the assembled calls', async () => {
            const events = [
                { type: 'content_block_delta', delta: { type: 'text_delta', text: 'ok' } },
                {
                    type: 'content_block_delta',
                    index: 1,
                    delta: { type: 'input_json_delta', partial_json: '{"a":' },
                },
                {
                    type: 'content_block_delta',
                    index: 1,
                    delta: { type: 'input_json_delta', partial_json: '1}' },
                },
            ];
            mocks.stream.mockReturnValue({
                async *[Symbol.asyncIterator]() {
                    yield* events;
                },
                finalMessage: async () =>
                    message('ok', {
                        content: [
                            { type: 'text', text: 'ok' },
                            { type: 'tool_use', id: 'toolu_1', name: 'add', input: { a: 1 } },
                        ],
                        stop_reason: 'tool_use',
                    }),
            });

            const chunks = [];
            for await (const chunk of new AnthropicLM({ apiKey: 'k' }).generateStream('Hi')) {
                chunks.push(chunk);
            }

            // Argument fragments are not text, so they travel in metadata.
            expect(chunks.filter((c) => !c.done).map((c) => c.content)).toEqual(['ok', '', '']);
            expect(chunks[1].metadata).toEqual({
                toolInputDelta: { index: 1, partialJson: '{"a":' },
            });
            expect(chunks.at(-1)?.metadata).toEqual({
                toolCalls: [{ id: 'toolu_1', name: 'add', arguments: { a: 1 } }],
            });
        });
    });

    describe('tool calling', () => {
        it('sends tool declarations as name/description/input_schema', async () => {
            mocks.create.mockResolvedValue(message('ok'));

            await new AnthropicLM({ apiKey: 'k' }).chatWithTools(
                [{ role: 'user', content: 'hi' }],
                {
                    tools: [
                        {
                            name: 'add',
                            description: 'Add two numbers',
                            parameters: { type: 'object', properties: {} },
                        },
                    ],
                }
            );

            expect(mocks.create.mock.calls[0][0].tools).toEqual([
                {
                    name: 'add',
                    description: 'Add two numbers',
                    input_schema: { type: 'object', properties: {} },
                },
            ]);
        });

        it('spells a forced tool call as tool_choice any', async () => {
            mocks.create.mockResolvedValue(message('ok'));

            await new AnthropicLM({ apiKey: 'k' }).chatWithTools(
                [{ role: 'user', content: 'hi' }],
                { tools: [{ name: 'add', parameters: {} }], toolChoice: 'required' }
            );

            expect(mocks.create.mock.calls[0][0].tool_choice).toEqual({ type: 'any' });
        });

        it('names a specific tool when the choice is an object', async () => {
            mocks.create.mockResolvedValue(message('ok'));

            await new AnthropicLM({ apiKey: 'k' }).chatWithTools(
                [{ role: 'user', content: 'hi' }],
                { tools: [{ name: 'add', parameters: {} }], toolChoice: { name: 'add' } }
            );

            expect(mocks.create.mock.calls[0][0].tool_choice).toEqual({
                type: 'tool',
                name: 'add',
            });
        });

        it('surfaces tool_use blocks alongside the text', async () => {
            mocks.create.mockResolvedValue(
                message('ignored', {
                    content: [
                        { type: 'text', text: 'Let me add those.' },
                        { type: 'tool_use', id: 'toolu_1', name: 'add', input: { a: 1, b: 2 } },
                    ],
                    stop_reason: 'tool_use',
                })
            );

            const result = await new AnthropicLM({ apiKey: 'k' }).chatWithTools([
                { role: 'user', content: 'hi' },
            ]);

            // textOf() keeps only text blocks; the tool_use block used to be
            // silently discarded here with nothing to replace it.
            expect(result.content).toBe('Let me add those.');
            expect(result.finishReason).toBe('tool_calls');
            expect(result.toolCalls).toEqual([
                { id: 'toolu_1', name: 'add', arguments: { a: 1, b: 2 } },
            ]);
        });

        it('omits tool parameters entirely when no tools are offered', async () => {
            mocks.create.mockResolvedValue(message('ok'));
            await new AnthropicLM({ apiKey: 'k' }).chat([{ role: 'user', content: 'hi' }]);

            expect(mocks.create.mock.calls[0][0]).not.toHaveProperty('tools');
            expect(mocks.create.mock.calls[0][0]).not.toHaveProperty('tool_choice');
        });
    });

    describe('toAnthropicMessages', () => {
        it('lifts system messages into the top-level system parameter', () => {
            const { system, messages } = toAnthropicMessages([
                { role: 'system', content: 'be terse' },
                { role: 'user', content: 'hi' },
            ]);

            expect(system).toBe('be terse');
            expect(messages).toEqual([{ role: 'user', content: 'hi' }]);
        });

        it('merges consecutive same-role turns to preserve alternation', () => {
            const { messages } = toAnthropicMessages([
                { role: 'user', content: 'one' },
                { role: 'user', content: 'two' },
                { role: 'assistant', content: 'reply' },
            ]);

            expect(messages).toEqual([
                { role: 'user', content: 'one\n\ntwo' },
                { role: 'assistant', content: 'reply' },
            ]);
        });

        it('does not mutate the caller array', () => {
            const input = [
                { role: 'user' as const, content: 'a' },
                { role: 'assistant' as const, content: 'b' },
            ];
            toAnthropicMessages(input);
            expect(input).toHaveLength(2);
        });

        it('turns an assistant tool call into text and tool_use blocks', () => {
            const { messages } = toAnthropicMessages([
                {
                    role: 'assistant',
                    content: 'Let me add those.',
                    toolCalls: [{ id: 'toolu_1', name: 'add', arguments: { a: 1, b: 2 } }],
                },
            ]);

            expect(messages).toEqual([
                {
                    role: 'assistant',
                    content: [
                        { type: 'text', text: 'Let me add those.' },
                        { type: 'tool_use', id: 'toolu_1', name: 'add', input: { a: 1, b: 2 } },
                    ],
                },
            ]);
        });

        it('gives parallel calls to one tool distinct synthesized ids', () => {
            // A Gemini-sourced turn carries no ids at all; two calls to the same
            // tool must not collapse onto one tool_use_id.
            const { messages } = toAnthropicMessages([
                {
                    role: 'assistant',
                    content: '',
                    toolCalls: [
                        { name: 'lookup', arguments: { id: 1 } },
                        { name: 'lookup', arguments: { id: 2 } },
                    ],
                },
            ]);

            const ids = (messages[0].content as any[]).map((block) => block.id);
            expect(new Set(ids).size).toBe(2);
        });

        it('turns a tool result into a user turn holding a tool_result block', () => {
            const { messages } = toAnthropicMessages([
                { role: 'tool', name: 'add', toolCallId: 'toolu_1', content: '3' },
            ]);

            expect(messages).toEqual([
                {
                    role: 'user',
                    content: [{ type: 'tool_result', tool_use_id: 'toolu_1', content: '3' }],
                },
            ]);
        });

        it('merges a tool result onto a preceding user turn as blocks, not text', () => {
            // String merging is still the behaviour for two plain user turns
            // (see above), but concatenating a tool result into prose would
            // destroy the `tool_use_id` the API correlates on, so a mixed pair
            // is promoted to block form instead.
            const { messages } = toAnthropicMessages([
                { role: 'user', content: 'context' },
                { role: 'tool', name: 'add', toolCallId: 'toolu_1', content: '3' },
            ]);

            expect(messages).toEqual([
                {
                    role: 'user',
                    content: [
                        { type: 'text', text: 'context' },
                        { type: 'tool_result', tool_use_id: 'toolu_1', content: '3' },
                    ],
                },
            ]);
        });

        it('degrades an uncorrelated tool result to plain user text', () => {
            const { messages } = toAnthropicMessages([{ role: 'tool', content: '3' }]);
            expect(messages).toEqual([{ role: 'user', content: '3' }]);
        });

        it('passes the system parameter through on a chat call', async () => {
            mocks.create.mockResolvedValue(message('ok'));
            await new AnthropicLM({ apiKey: 'k' }).chat([
                { role: 'system', content: 'be terse' },
                { role: 'user', content: 'hi' },
            ]);

            expect(mocks.create.mock.calls[0][0].system).toBe('be terse');
        });
    });

    describe('error handling', () => {
        it('wraps SDK errors in LMError preserving the status', async () => {
            mocks.create.mockRejectedValue(new MockAPIError(429, 'rate limited'));
            const lm = new AnthropicLM({ apiKey: 'k' });

            await expect(lm.generate('Hi')).rejects.toThrow(LMError);
            await expect(lm.generate('Hi')).rejects.toMatchObject({
                provider: 'anthropic',
                status: 429,
            });
        });

        it('counts failures', async () => {
            mocks.create.mockRejectedValue(new MockAPIError(500, 'boom'));
            const lm = new AnthropicLM({ apiKey: 'k' });

            await expect(lm.generate('Hi')).rejects.toThrow();
            expect(lm.getUsage().errorCount).toBe(1);
        });
    });

    it('advertises full capabilities', () => {
        const capabilities = new AnthropicLM({ apiKey: 'k' }).getCapabilities();

        expect(capabilities.supportsStreaming).toBe(true);
        expect(capabilities.supportsStructuredOutput).toBe(true);
        expect(capabilities.supportsFunctionCalling).toBe(true);
        expect(capabilities.maxContextLength).toBe(1_000_000);
    });
});
