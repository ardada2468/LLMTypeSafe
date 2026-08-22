import { BaseLM, LMError } from '@ts-dspy/core';
import {
    OpenAICompatibleLM,
    OPENAI_COMPATIBLE_BASE_URLS,
    PLACEHOLDER_API_KEY,
    DEFAULT_COMPATIBLE_CONTEXT_LENGTH,
} from './openai-compatible-lm';

// Everything the hoisted vi.mock factory touches must itself be hoisted.
const mocks = vi.hoisted(() => {
    class MockAPIError extends Error {
        status: number;
        constructor(status: number, message: string) {
            super(message);
            this.status = status;
        }
    }
    return { create: vi.fn(), list: vi.fn(), MockAPIError, constructed: [] as unknown[] };
});

const { MockAPIError } = mocks;

vi.mock('openai', () => ({
    default: class {
        chat = { completions: { create: mocks.create } };
        models = { list: mocks.list };
        constructor(public options: unknown) {
            mocks.constructed.push(options);
        }
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

/** Options the last `new OpenAI(...)` was handed. */
function lastClientOptions(): Record<string, unknown> {
    return mocks.constructed.at(-1) as Record<string, unknown>;
}

const ollama = { baseURL: OPENAI_COMPATIBLE_BASE_URLS.ollama, model: 'llama3.2' };

beforeEach(() => {
    mocks.create.mockReset();
    mocks.list.mockReset();
    mocks.constructed.length = 0;
});

describe('OpenAICompatibleLM', () => {
    it('is a BaseLM, so configure() accepts it like any other provider', () => {
        expect(new OpenAICompatibleLM(ollama)).toBeInstanceOf(BaseLM);
    });

    it('passes the baseURL through to the SDK client', () => {
        new OpenAICompatibleLM(ollama);

        expect(lastClientOptions().baseURL).toBe('http://localhost:11434/v1');
    });

    it('uses the configured model rather than an OpenAI default', () => {
        expect(new OpenAICompatibleLM(ollama).getModelName()).toBe('llama3.2');
    });

    describe('credentials', () => {
        it('supplies a placeholder key for servers that ignore it', () => {
            new OpenAICompatibleLM(ollama);

            // Without this the SDK throws at construction when OPENAI_API_KEY is
            // unset, long before a local server is ever contacted.
            expect(lastClientOptions().apiKey).toBe(PLACEHOLDER_API_KEY);
        });

        it('does not forward OPENAI_API_KEY to a non-OpenAI endpoint', () => {
            vi.stubEnv('OPENAI_API_KEY', 'sk-real-openai-key');
            try {
                new OpenAICompatibleLM(ollama);
            } finally {
                vi.unstubAllEnvs();
            }

            // baseURL here is by definition not api.openai.com, so inheriting the
            // OpenAI credential would send it to whatever host the caller named.
            expect(lastClientOptions().apiKey).toBe(PLACEHOLDER_API_KEY);
        });

        it('keeps a real key when one is given', () => {
            new OpenAICompatibleLM({
                baseURL: OPENAI_COMPATIBLE_BASE_URLS.groq,
                model: 'llama-3.3-70b-versatile',
                apiKey: 'gsk_real',
            });

            expect(lastClientOptions().apiKey).toBe('gsk_real');
        });
    });

    describe('required configuration', () => {
        it('rejects a missing baseURL', () => {
            expect(() => new OpenAICompatibleLM({ model: 'llama3.2' } as never)).toThrow(
                LMError
            );
            expect(() => new OpenAICompatibleLM({ model: 'llama3.2' } as never)).toThrow(
                /baseURL/
            );
        });

        it('rejects a missing model', () => {
            expect(() => new OpenAICompatibleLM({ baseURL: ollama.baseURL } as never)).toThrow(
                /model/
            );
        });
    });

    describe('capabilities', () => {
        it('defaults to the conservative answer for every optimistic flag', () => {
            const capabilities = new OpenAICompatibleLM(ollama).getCapabilities();

            // OpenAILM hardcodes all three to true; none is safe against a local
            // Llama or a small hosted model.
            expect(capabilities.supportsStructuredOutput).toBe(false);
            expect(capabilities.supportsFunctionCalling).toBe(false);
            expect(capabilities.supportsVision).toBe(false);
            expect(capabilities.supportedFormats).toEqual(['text']);
            // Streaming is part of the compatible surface everywhere.
            expect(capabilities.supportsStreaming).toBe(true);
        });

        it('takes every capability from config when given', () => {
            const capabilities = new OpenAICompatibleLM({
                ...ollama,
                supportsStreaming: false,
                supportsStructuredOutput: true,
                supportsFunctionCalling: true,
                supportsVision: true,
                maxContextLength: 32_768,
            }).getCapabilities();

            expect(capabilities).toEqual({
                supportsStreaming: false,
                supportsStructuredOutput: true,
                supportsFunctionCalling: true,
                supportsVision: true,
                maxContextLength: 32_768,
                supportedFormats: ['text', 'json_object', 'json_schema'],
            });
        });

        it('reports a conservative context length instead of guessing 128k', () => {
            // The gpt-* prefix table OpenAILM uses never matches these ids, so it
            // silently reported 128_000 for a model that may hold 4k.
            expect(
                new OpenAICompatibleLM({ ...ollama, model: 'mixtral-8x7b' }).getCapabilities()
                    .maxContextLength
            ).toBe(DEFAULT_COMPATIBLE_CONTEXT_LENGTH);
        });

        it('does not leak its capabilities object to callers', () => {
            const lm = new OpenAICompatibleLM(ollama);
            const capabilities = lm.getCapabilities();
            capabilities.supportsVision = true;
            capabilities.supportedFormats.push('json_schema');

            expect(lm.getCapabilities().supportsVision).toBe(false);
            expect(lm.getCapabilities().supportedFormats).toEqual(['text']);
        });
    });

    describe('generateStructured', () => {
        it('sends no response_format when structured output is unsupported', async () => {
            mocks.create.mockResolvedValue(completion('{"answer":"Paris"}'));

            const result = await new OpenAICompatibleLM(ollama).generateStructured(
                'Capital of France?',
                { type: 'object', properties: { answer: { type: 'string' } } }
            );

            expect(result).toEqual({ answer: 'Paris' });
            // The whole point of the override: a strict json_schema request is
            // rejected outright by most compatible servers.
            expect(mocks.create.mock.calls[0][0]).not.toHaveProperty('response_format');
        });

        it('puts the schema in the prompt on the fallback path', async () => {
            mocks.create.mockResolvedValue(completion('{"answer":"Paris"}'));

            await new OpenAICompatibleLM(ollama).generateStructured('Capital?', {
                type: 'object',
            });

            const [message] = mocks.create.mock.calls[0][0].messages;
            expect(message.content).toContain('Capital?');
            expect(message.content).toContain('"type": "object"');
        });

        it('tolerates a fenced JSON reply, which small models routinely emit', async () => {
            mocks.create.mockResolvedValue(completion('```json\n{"answer":"Paris"}\n```'));

            const result = await new OpenAICompatibleLM(ollama).generateStructured('q', {});

            expect(result).toEqual({ answer: 'Paris' });
        });

        it('uses the native json_schema mode when the endpoint declares it', async () => {
            mocks.create.mockResolvedValue(completion('{"answer":"Paris"}'));
            const schema = { type: 'object', properties: { answer: { type: 'string' } } };

            await new OpenAICompatibleLM({
                ...ollama,
                supportsStructuredOutput: true,
            }).generateStructured('Capital of France?', schema);

            expect(mocks.create.mock.calls[0][0].response_format).toEqual({
                type: 'json_schema',
                json_schema: { name: 'signature_output', strict: true, schema },
            });
        });
    });

    describe('streaming', () => {
        it('streams deltas when the endpoint supports it', async () => {
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
            for await (const chunk of new OpenAICompatibleLM(ollama).generateStream('Hi')) {
                chunks.push(chunk);
            }

            expect(chunks.filter((c) => !c.done).map((c) => c.content)).toEqual(['Hel', 'lo']);
            expect(chunks.at(-1)).toMatchObject({ done: true });
        });

        it('emits one chunk from a plain call when streaming is unsupported', async () => {
            mocks.create.mockResolvedValue(completion('Hello there'));

            const chunks = [];
            const lm = new OpenAICompatibleLM({ ...ollama, supportsStreaming: false });
            for await (const chunk of lm.generateStream('Hi')) {
                chunks.push(chunk);
            }

            expect(mocks.create.mock.calls[0][0]).not.toHaveProperty('stream');
            expect(chunks.filter((c) => !c.done).map((c) => c.content)).toEqual([
                'Hello there',
            ]);
            expect(chunks.at(-1)).toMatchObject({
                done: true,
                usage: { promptTokens: 12, completionTokens: 8, totalTokens: 20 },
            });
        });
    });

    describe('inherited behaviour', () => {
        it('returns the assistant message content', async () => {
            mocks.create.mockResolvedValue(completion('Hello there'));

            expect(await new OpenAICompatibleLM(ollama).generate('Hi')).toBe('Hello there');
            expect(mocks.create.mock.calls[0][0].model).toBe('llama3.2');
        });

        it('wraps SDK errors in LMError preserving the status', async () => {
            mocks.create.mockRejectedValue(new MockAPIError(404, 'model not found'));
            const lm = new OpenAICompatibleLM(ollama);

            await expect(lm.generate('Hi')).rejects.toThrow(LMError);
            expect(lm.getUsage().errorCount).toBe(1);
        });
    });
});
