import {
    AuthError,
    ContentFilterError,
    ContextLengthError,
    LMError,
    RateLimitError,
    TimeoutError,
} from '@ts-dspy/core';
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
        // The real APIError lifts `error.type` out of the response body; it is
        // the cleanest discriminator any of the three SDKs offers.
        type: string | null;
        constructor(status: number, message: string, type: string | null = null) {
            super(message);
            this.status = status;
            this.type = type;
        }
    }
    // The real class extends APIConnectionError extends APIError, and carries
    // neither a status nor a type.
    class MockAPIConnectionTimeoutError extends MockAPIError {
        constructor(message = 'Request timed out.') {
            super(undefined as unknown as number, message);
        }
    }
    return {
        create: vi.fn(),
        stream: vi.fn(),
        MockAPIError,
        MockAPIConnectionTimeoutError,
    };
});

const { MockAPIError, MockAPIConnectionTimeoutError } = mocks;

vi.mock('@anthropic-ai/sdk', () => ({
    default: class {
        messages = { create: mocks.create, stream: mocks.stream };
        constructor(public options: unknown) {}
    },
    APIError: mocks.MockAPIError,
    APIConnectionTimeoutError: mocks.MockAPIConnectionTimeoutError,
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

        it('throws the shared ContentFilterError, which AnthropicRefusalError now aliases', async () => {
            mocks.create.mockResolvedValue({
                content: [],
                stop_reason: 'refusal',
                stop_details: { type: 'refusal', category: 'cyber' },
                usage: { input_tokens: 5, output_tokens: 0 },
            });

            const error = await new AnthropicLM({ apiKey: 'k' }).generate('Hi').catch((e) => e);

            expect(error).toBeInstanceOf(ContentFilterError);
            expect(AnthropicRefusalError).toBe(ContentFilterError);
            expect(error.provider).toBe('anthropic');
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

        it('counts a refused stream exactly once', async () => {
            // assertNotRefused records the error itself, and it used to sit
            // inside the try, so the catch counted the same refusal twice.
            mocks.stream.mockReturnValue({
                async *[Symbol.asyncIterator]() {},
                finalMessage: async () =>
                    message('', { stop_reason: 'refusal', stop_details: null }),
            });
            const lm = new AnthropicLM({ apiKey: 'k' });

            const consume = async () => {
                for await (const _chunk of lm.generateStream('Hi')) {
                    // drain
                }
            };

            await expect(consume()).rejects.toBeInstanceOf(ContentFilterError);
            expect(lm.getUsage().errorCount).toBe(1);
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

        it('classifies by the typed error union', async () => {
            const lm = new AnthropicLM({ apiKey: 'k' });

            mocks.create.mockRejectedValue(
                new MockAPIError(429, 'rate limited', 'rate_limit_error')
            );
            const rateLimited = await lm.generate('Hi').catch((e) => e);
            expect(rateLimited).toBeInstanceOf(RateLimitError);
            // The new classes are still LMError, so old catch blocks hold.
            expect(rateLimited).toBeInstanceOf(LMError);

            mocks.create.mockRejectedValue(
                new MockAPIError(401, 'bad key', 'authentication_error')
            );
            await expect(lm.generate('Hi')).rejects.toBeInstanceOf(AuthError);

            mocks.create.mockRejectedValue(
                new MockAPIError(403, 'no access', 'permission_error')
            );
            await expect(lm.generate('Hi')).rejects.toBeInstanceOf(AuthError);

            mocks.create.mockRejectedValue(
                new MockAPIError(408, 'took too long', 'timeout_error')
            );
            await expect(lm.generate('Hi')).rejects.toBeInstanceOf(TimeoutError);
        });

        it('classifies a client-side timeout, which carries no status or type', async () => {
            // `timeout_error` only covers a server-side gateway timeout, so a
            // client timeout reached classify() with nothing to go on and came
            // back as a plain LMError.
            mocks.create.mockRejectedValue(new MockAPIConnectionTimeoutError());

            const error = await new AnthropicLM({ apiKey: 'k' }).generate('Hi').catch((e) => e);

            expect(error).toBeInstanceOf(TimeoutError);
            expect(error.status).toBeUndefined();
        });

        it('recognises an over-long prompt despite there being no context-length type', async () => {
            const lm = new AnthropicLM({ apiKey: 'k' });

            mocks.create.mockRejectedValue(
                new MockAPIError(
                    400,
                    'prompt is too long: 250000 tokens > 200000 maximum',
                    'invalid_request_error'
                )
            );
            await expect(lm.generate('Hi')).rejects.toBeInstanceOf(ContextLengthError);

            mocks.create.mockRejectedValue(
                new MockAPIError(400, 'max_tokens is required', 'invalid_request_error')
            );
            const other = await lm.generate('Hi').catch((e) => e);
            expect(other).toBeInstanceOf(LMError);
            expect(other).not.toBeInstanceOf(ContextLengthError);
        });

        it('leaves an unrecognised failure as a plain LMError', async () => {
            mocks.create.mockRejectedValue(new MockAPIError(500, 'boom', 'api_error'));

            const error = await new AnthropicLM({ apiKey: 'k' }).generate('Hi').catch((e) => e);

            expect(error.constructor.name).toBe('LMError');
            expect(error.status).toBe(500);
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
