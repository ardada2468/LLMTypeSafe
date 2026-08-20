import { LMError } from '@ts-dspy/core';
import { OpenAILM, toOpenAIMessages, DEFAULT_OPENAI_MODEL } from './openai-lm';

// Everything the hoisted vi.mock factory touches must itself be hoisted.
const mocks = vi.hoisted(() => {
    class MockAPIError extends Error {
        status: number;
        constructor(status: number, message: string) {
            super(message);
            this.status = status;
        }
    }
    return { create: vi.fn(), list: vi.fn(), MockAPIError };
});

const { MockAPIError } = mocks;

vi.mock('openai', () => ({
    default: class {
        chat = { completions: { create: mocks.create } };
        models = { list: mocks.list };
        constructor(public options: unknown) {}
    },
    APIError: mocks.MockAPIError,
}));

function completion(content: string, extra: Record<string, unknown> = {}) {
    return {
        choices: [{ message: { content }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 12, completion_tokens: 8 },
        ...extra,
    };
}

beforeEach(() => {
    mocks.create.mockReset();
    mocks.list.mockReset();
});

describe('OpenAILM', () => {
    it('defaults to the current model', () => {
        expect(new OpenAILM({ apiKey: 'k' }).getModelName()).toBe(DEFAULT_OPENAI_MODEL);
        expect(DEFAULT_OPENAI_MODEL).not.toBe('gpt-4');
    });

    it('returns the assistant message content', async () => {
        mocks.create.mockResolvedValue(completion('Hello there'));

        const result = await new OpenAILM({ apiKey: 'k' }).generate('Hi');

        expect(result).toBe('Hello there');
        expect(mocks.create.mock.calls[0][0].messages).toEqual([
            { role: 'user', content: 'Hi' },
        ]);
    });

    it('accumulates usage across calls without inventing a cost', async () => {
        mocks.create.mockResolvedValue(completion('ok'));
        const lm = new OpenAILM({ apiKey: 'k' });

        await lm.generate('one');
        await lm.generate('two');

        const usage = lm.getUsage();
        expect(usage.promptTokens).toBe(24);
        expect(usage.completionTokens).toBe(16);
        expect(usage.totalTokens).toBe(40);
        expect(usage.requestCount).toBe(2);
        // The old implementation multiplied by a hardcoded gpt-3.5 rate here,
        // which was wrong by more than an order of magnitude.
        expect(usage.totalCost).toBeUndefined();
    });

    it('resets usage', async () => {
        mocks.create.mockResolvedValue(completion('ok'));
        const lm = new OpenAILM({ apiKey: 'k' });
        await lm.generate('one');
        lm.resetUsage();

        expect(lm.getUsage().totalTokens).toBe(0);
    });

    describe('sampling parameters', () => {
        it('omits temperature and top_p when the caller did not set them', async () => {
            mocks.create.mockResolvedValue(completion('ok'));
            await new OpenAILM({ apiKey: 'k' }).generate('Hi');

            const body = mocks.create.mock.calls[0][0];
            // Reasoning models reject a non-default temperature, so sending a
            // default of 0.7 (as the old implementation did) breaks them.
            expect(body).not.toHaveProperty('temperature');
            expect(body).not.toHaveProperty('top_p');
        });

        it('sends them when the caller does set them', async () => {
            mocks.create.mockResolvedValue(completion('ok'));
            await new OpenAILM({ apiKey: 'k' }).generate('Hi', {
                temperature: 0.2,
                topP: 0.9,
            });

            const body = mocks.create.mock.calls[0][0];
            expect(body.temperature).toBe(0.2);
            expect(body.top_p).toBe(0.9);
        });

        it('uses max_completion_tokens rather than the deprecated max_tokens', async () => {
            mocks.create.mockResolvedValue(completion('ok'));
            await new OpenAILM({ apiKey: 'k' }).generate('Hi', { maxTokens: 256 });

            const body = mocks.create.mock.calls[0][0];
            expect(body.max_completion_tokens).toBe(256);
            expect(body).not.toHaveProperty('max_tokens');
        });

        it('maps timeout and retries onto the SDK request options', async () => {
            mocks.create.mockResolvedValue(completion('ok'));
            await new OpenAILM({ apiKey: 'k' }).generate('Hi', {
                timeout: 5000,
                retries: 1,
            });

            expect(mocks.create.mock.calls[0][1]).toEqual({ timeout: 5000, maxRetries: 1 });
        });

        it('allows a per-call model override', async () => {
            mocks.create.mockResolvedValue(completion('ok'));
            await new OpenAILM({ apiKey: 'k' }).generate('Hi', { model: 'gpt-4.1' });

            expect(mocks.create.mock.calls[0][0].model).toBe('gpt-4.1');
        });
    });

    describe('generateStructured', () => {
        it('requests a strict json_schema response format', async () => {
            mocks.create.mockResolvedValue(completion('{"answer":"Paris"}'));
            const schema = { type: 'object', properties: { answer: { type: 'string' } } };

            const result = await new OpenAILM({ apiKey: 'k' }).generateStructured(
                'Capital of France?',
                schema
            );

            expect(result).toEqual({ answer: 'Paris' });
            expect(mocks.create.mock.calls[0][0].response_format).toEqual({
                type: 'json_schema',
                json_schema: { name: 'signature_output', strict: true, schema },
            });
        });

        it('raises a clear error when the response is truncated', async () => {
            mocks.create.mockResolvedValue({
                choices: [{ message: { content: '{"a"' }, finish_reason: 'length' }],
                usage: { prompt_tokens: 1, completion_tokens: 1 },
            });

            await expect(
                new OpenAILM({ apiKey: 'k' }).generateStructured('x', {})
            ).rejects.toThrow(/truncated/);
        });

        it('raises a clear error when the response is not JSON', async () => {
            mocks.create.mockResolvedValue(completion('not json'));

            await expect(
                new OpenAILM({ apiKey: 'k' }).generateStructured('x', {})
            ).rejects.toThrow(/not valid JSON/);
        });
    });

    describe('streaming', () => {
        it('yields deltas then a final chunk carrying usage', async () => {
            mocks.create.mockResolvedValue(
                (async function* () {
                    yield { choices: [{ delta: { content: 'Hel' } }] };
                    yield { choices: [{ delta: { content: 'lo' } }] };
                    yield {
                        choices: [{ delta: {} }],
                        usage: { prompt_tokens: 3, completion_tokens: 2 },
                    };
                })()
            );

            const chunks = [];
            for await (const chunk of new OpenAILM({ apiKey: 'k' }).generateStream('Hi')) {
                chunks.push(chunk);
            }

            expect(chunks.filter((c) => !c.done).map((c) => c.content)).toEqual(['Hel', 'lo']);
            expect(chunks.at(-1)).toMatchObject({
                done: true,
                usage: { promptTokens: 3, completionTokens: 2, totalTokens: 5 },
            });
        });
    });

    describe('error handling', () => {
        it('wraps SDK errors in LMError preserving the status', async () => {
            mocks.create.mockRejectedValue(new MockAPIError(429, 'Rate limit reached'));
            const lm = new OpenAILM({ apiKey: 'k' });

            await expect(lm.generate('Hi')).rejects.toThrow(LMError);
            await expect(lm.generate('Hi')).rejects.toMatchObject({
                provider: 'openai',
                status: 429,
            });
        });

        it('counts failures in usage stats', async () => {
            mocks.create.mockRejectedValue(new MockAPIError(500, 'boom'));
            const lm = new OpenAILM({ apiKey: 'k' });

            await expect(lm.generate('Hi')).rejects.toThrow();
            expect(lm.getUsage().errorCount).toBe(1);
        });
    });

    describe('capabilities', () => {
        it('advertises streaming, structured output and tool calling', () => {
            const capabilities = new OpenAILM({ apiKey: 'k' }).getCapabilities();

            expect(capabilities.supportsStreaming).toBe(true);
            expect(capabilities.supportsStructuredOutput).toBe(true);
            expect(capabilities.supportsFunctionCalling).toBe(true);
        });

        it('reports a context length matching the configured model', () => {
            // The old implementation returned 4096 for every model.
            expect(
                new OpenAILM({ apiKey: 'k', model: 'gpt-4o' }).getCapabilities()
                    .maxContextLength
            ).toBe(128_000);
            expect(
                new OpenAILM({ apiKey: 'k', model: 'gpt-4' }).getCapabilities().maxContextLength
            ).toBe(8_192);
        });

        it('tracks setModel', () => {
            const lm = new OpenAILM({ apiKey: 'k', model: 'gpt-4' });
            lm.setModel('gpt-4o');

            expect(lm.getModelName()).toBe('gpt-4o');
            expect(lm.getCapabilities().maxContextLength).toBe(128_000);
        });
    });

    describe('toOpenAIMessages', () => {
        it('preserves system, user and assistant roles', () => {
            expect(
                toOpenAIMessages([
                    { role: 'system', content: 's' },
                    { role: 'user', content: 'u' },
                    { role: 'assistant', content: 'a' },
                ])
            ).toEqual([
                { role: 'system', content: 's' },
                { role: 'user', content: 'u' },
                { role: 'assistant', content: 'a' },
            ]);
        });

        it('does not mutate the caller array', () => {
            const messages = [{ role: 'user' as const, content: 'u' }];
            toOpenAIMessages(messages);
            expect(messages).toHaveLength(1);
        });
    });
});
