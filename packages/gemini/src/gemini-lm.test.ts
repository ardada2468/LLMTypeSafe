import { LMError } from '@ts-dspy/core';
import { GeminiLM, toGeminiContents, DEFAULT_GEMINI_MODEL } from './gemini-lm';

const mocks = vi.hoisted(() => ({
    generateContent: vi.fn(),
    generateContentStream: vi.fn(),
    constructorOptions: vi.fn(),
}));

vi.mock('@google/genai', () => ({
    GoogleGenAI: class {
        models = {
            generateContent: mocks.generateContent,
            generateContentStream: mocks.generateContentStream,
        };
        constructor(options: unknown) {
            mocks.constructorOptions(options);
        }
    },
    HarmCategory: {
        HARM_CATEGORY_HARASSMENT: 'HARM_CATEGORY_HARASSMENT',
        HARM_CATEGORY_HATE_SPEECH: 'HARM_CATEGORY_HATE_SPEECH',
        HARM_CATEGORY_SEXUALLY_EXPLICIT: 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
        HARM_CATEGORY_DANGEROUS_CONTENT: 'HARM_CATEGORY_DANGEROUS_CONTENT',
    },
    HarmBlockThreshold: { BLOCK_MEDIUM_AND_ABOVE: 'BLOCK_MEDIUM_AND_ABOVE' },
}));

function response(text: string, extra: Record<string, unknown> = {}) {
    return {
        text,
        usageMetadata: {
            promptTokenCount: 11,
            candidatesTokenCount: 7,
            totalTokenCount: 18,
        },
        ...extra,
    };
}

beforeEach(() => {
    mocks.generateContent.mockReset();
    mocks.generateContentStream.mockReset();
    mocks.constructorOptions.mockReset();
});

describe('GeminiLM', () => {
    it('defaults to a current model', () => {
        expect(new GeminiLM({ apiKey: 'k' }).getModelName()).toBe(DEFAULT_GEMINI_MODEL);
        // gemini-2.0-flash, the previous default, has reached end of life.
        expect(DEFAULT_GEMINI_MODEL).not.toBe('gemini-2.0-flash');
    });

    it('returns the response text', async () => {
        mocks.generateContent.mockResolvedValue(response('Hello there'));

        expect(await new GeminiLM({ apiKey: 'k' }).generate('Hi')).toBe('Hello there');
    });

    it('reports real token usage', async () => {
        mocks.generateContent.mockResolvedValue(response('ok'));
        const lm = new GeminiLM({ apiKey: 'k' });

        await lm.generate('Hi');

        // The old implementation logged "Gemini does not provide token usage
        // stats yet" and always returned zeros.
        expect(lm.getUsage()).toMatchObject({
            promptTokens: 11,
            completionTokens: 7,
            totalTokens: 18,
        });
    });

    it('does not log to the console', async () => {
        const spy = vi.spyOn(console, 'log').mockImplementation(() => {});
        mocks.generateContent.mockResolvedValue(response('ok'));

        const lm = new GeminiLM({ apiKey: 'k' });
        await lm.generate('Hi');
        lm.getUsage();

        expect(spy).not.toHaveBeenCalled();
        spy.mockRestore();
    });

    describe('message conversion', () => {
        it('does not mutate the caller message array', async () => {
            mocks.generateContent.mockResolvedValue(response('ok'));
            const messages = [
                { role: 'user' as const, content: 'first' },
                { role: 'assistant' as const, content: 'second' },
                { role: 'user' as const, content: 'third' },
            ];

            await new GeminiLM({ apiKey: 'k' }).chat(messages);

            // The old implementation called messages.pop(), destroying the last
            // turn of any array the caller reused.
            expect(messages).toHaveLength(3);
            expect(messages[2].content).toBe('third');
        });

        it('maps assistant to the model role and keeps every turn', () => {
            const { contents } = toGeminiContents([
                { role: 'user', content: 'a' },
                { role: 'assistant', content: 'b' },
                { role: 'user', content: 'c' },
            ]);

            expect(contents).toEqual([
                { role: 'user', parts: [{ text: 'a' }] },
                { role: 'model', parts: [{ text: 'b' }] },
                { role: 'user', parts: [{ text: 'c' }] },
            ]);
        });

        it('lifts system messages into a systemInstruction', () => {
            const { contents, systemInstruction } = toGeminiContents([
                { role: 'system', content: 'be terse' },
                { role: 'user', content: 'hi' },
            ]);

            expect(systemInstruction).toBe('be terse');
            expect(contents).toHaveLength(1);
        });
    });

    describe('safety blocking', () => {
        it('throws when the prompt is blocked, before reading the text', async () => {
            // The old implementation read response.text() first, which threw on
            // blocked responses and made the blockReason branch unreachable.
            mocks.generateContent.mockResolvedValue({
                promptFeedback: { blockReason: 'SAFETY' },
                get text(): string {
                    throw new Error('text accessor should not be reached');
                },
            });

            await expect(new GeminiLM({ apiKey: 'k' }).generate('Hi')).rejects.toThrow(
                /blocked by safety filters: SAFETY/
            );
        });

        it('configures all four harm categories by default', async () => {
            mocks.generateContent.mockResolvedValue(response('ok'));
            await new GeminiLM({ apiKey: 'k' }).generate('Hi');

            const settings = mocks.generateContent.mock.calls[0][0].config.safetySettings;
            expect(settings).toHaveLength(4);
        });
    });

    describe('generateStructured', () => {
        it('requests JSON constrained by the supplied schema', async () => {
            mocks.generateContent.mockResolvedValue(response('{"answer":"Paris"}'));
            const schema = { type: 'object', properties: { answer: { type: 'string' } } };

            const result = await new GeminiLM({ apiKey: 'k' }).generateStructured('Q', schema);

            expect(result).toEqual({ answer: 'Paris' });
            const config = mocks.generateContent.mock.calls[0][0].config;
            expect(config.responseMimeType).toBe('application/json');
            expect(config.responseJsonSchema).toBe(schema);
        });

        it('raises a clear error when the response is not JSON', async () => {
            mocks.generateContent.mockResolvedValue(response('not json'));

            await expect(
                new GeminiLM({ apiKey: 'k' }).generateStructured('Q', {})
            ).rejects.toThrow(/not valid JSON/);
        });
    });

    describe('call options', () => {
        it('forwards sampling parameters', async () => {
            mocks.generateContent.mockResolvedValue(response('ok'));
            await new GeminiLM({ apiKey: 'k' }).generate('Hi', {
                temperature: 0.3,
                maxTokens: 100,
                topP: 0.8,
                stopSequences: ['END'],
            });

            const config = mocks.generateContent.mock.calls[0][0].config;
            expect(config.temperature).toBe(0.3);
            expect(config.maxOutputTokens).toBe(100);
            expect(config.topP).toBe(0.8);
            expect(config.stopSequences).toEqual(['END']);
        });

        it('turns a timeout into an abort signal', async () => {
            mocks.generateContent.mockResolvedValue(response('ok'));
            await new GeminiLM({ apiKey: 'k' }).generate('Hi', { timeout: 5000 });

            expect(mocks.generateContent.mock.calls[0][0].config.abortSignal).toBeInstanceOf(
                AbortSignal
            );
        });

        it('allows a per-call model override', async () => {
            mocks.generateContent.mockResolvedValue(response('ok'));
            await new GeminiLM({ apiKey: 'k' }).generate('Hi', { model: 'gemini-3.1-pro' });

            expect(mocks.generateContent.mock.calls[0][0].model).toBe('gemini-3.1-pro');
        });
    });

    describe('streaming', () => {
        it('yields deltas then a final chunk with usage', async () => {
            mocks.generateContentStream.mockResolvedValue(
                (async function* () {
                    yield { text: 'Hel' };
                    yield { text: 'lo' };
                    yield {
                        text: '',
                        usageMetadata: {
                            promptTokenCount: 3,
                            candidatesTokenCount: 2,
                            totalTokenCount: 5,
                        },
                    };
                })()
            );

            const chunks = [];
            for await (const chunk of new GeminiLM({ apiKey: 'k' }).generateStream('Hi')) {
                chunks.push(chunk);
            }

            expect(chunks.filter((c) => !c.done).map((c) => c.content)).toEqual(['Hel', 'lo']);
            expect(chunks.at(-1)).toMatchObject({
                done: true,
                usage: { promptTokens: 3, completionTokens: 2, totalTokens: 5 },
            });
        });
    });

    describe('configuration', () => {
        it('passes Vertex AI options to the client', () => {
            new GeminiLM({ vertexai: true, project: 'p', location: 'us-central1' });

            expect(mocks.constructorOptions).toHaveBeenCalledWith(
                expect.objectContaining({
                    vertexai: true,
                    project: 'p',
                    location: 'us-central1',
                })
            );
        });

        it('passes a custom base URL through httpOptions', () => {
            new GeminiLM({ apiKey: 'k', baseUrl: 'https://proxy.example.com' });

            expect(mocks.constructorOptions).toHaveBeenCalledWith(
                expect.objectContaining({
                    httpOptions: { baseUrl: 'https://proxy.example.com' },
                })
            );
        });
    });

    describe('capabilities', () => {
        it('advertises streaming, structured output and tool calling', () => {
            const capabilities = new GeminiLM({ apiKey: 'k' }).getCapabilities();

            expect(capabilities.supportsStreaming).toBe(true);
            expect(capabilities.supportsStructuredOutput).toBe(true);
            expect(capabilities.supportsFunctionCalling).toBe(true);
            // The old implementation hardcoded 32768 with a "Gemini 1.0 Pro" comment.
            expect(capabilities.maxContextLength).toBe(1_000_000);
        });
    });

    it('wraps SDK failures in LMError and counts them', async () => {
        mocks.generateContent.mockRejectedValue(new Error('network down'));
        const lm = new GeminiLM({ apiKey: 'k' });

        await expect(lm.generate('Hi')).rejects.toThrow(LMError);
        expect(lm.getUsage().errorCount).toBe(1);
    });
});
