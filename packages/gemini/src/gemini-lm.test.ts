import {
    AuthError,
    ContentFilterError,
    ContextLengthError,
    LMError,
    RateLimitError,
} from '@ts-dspy/core';
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
    FinishReason: {
        STOP: 'STOP',
        MAX_TOKENS: 'MAX_TOKENS',
        SAFETY: 'SAFETY',
        RECITATION: 'RECITATION',
        BLOCKLIST: 'BLOCKLIST',
        PROHIBITED_CONTENT: 'PROHIBITED_CONTENT',
        SPII: 'SPII',
        IMAGE_SAFETY: 'IMAGE_SAFETY',
        IMAGE_PROHIBITED_CONTENT: 'IMAGE_PROHIBITED_CONTENT',
    },
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

        it('reports a blocked prompt as ContentFilterError carrying the block reason', async () => {
            // This used to be a bare LMError with no status and no category, so
            // a caller could not tell a safety block from a network failure.
            mocks.generateContent.mockResolvedValue({
                promptFeedback: { blockReason: 'SAFETY' },
                text: '',
            });

            const error = await new GeminiLM({ apiKey: 'k' }).generate('Hi').catch((e) => e);

            expect(error).toBeInstanceOf(ContentFilterError);
            expect(error).toBeInstanceOf(LMError);
            expect(error.category).toBe('SAFETY');
        });

        it('throws when the candidate itself was blocked, rather than returning empty text', async () => {
            // Only promptFeedback.blockReason was checked before, so a blocked
            // candidate came back as an empty string.
            mocks.generateContent.mockResolvedValue(
                response('', { candidates: [{ finishReason: 'SAFETY' }] })
            );
            const lm = new GeminiLM({ apiKey: 'k' });

            await expect(lm.generate('Hi')).rejects.toBeInstanceOf(ContentFilterError);
            expect(lm.getUsage().errorCount).toBe(1);
        });

        it('covers every finish reason that withholds the candidate', async () => {
            // Checking only SAFETY still handed the caller an empty string for
            // the rest of the withholding reasons.
            const lm = new GeminiLM({ apiKey: 'k' });

            for (const reason of ['BLOCKLIST', 'PROHIBITED_CONTENT', 'SPII', 'RECITATION']) {
                mocks.generateContent.mockResolvedValue(
                    response('', { candidates: [{ finishReason: reason }] })
                );
                const error = await lm.generate('Hi').catch((e) => e);

                expect(error).toBeInstanceOf(ContentFilterError);
                expect(error.category).toBe(reason);
            }
        });

        it('leaves an ordinary STOP alone', async () => {
            mocks.generateContent.mockResolvedValue(
                response('fine', { candidates: [{ finishReason: 'STOP' }] })
            );

            expect(await new GeminiLM({ apiKey: 'k' }).generate('Hi')).toBe('fine');
        });

        it('throws when a stream is cut short by the classifiers', async () => {
            // The safety checks lived only in the one-shot path, so a blocked
            // stream just ended early and looked like a short answer.
            mocks.generateContentStream.mockResolvedValue(
                (async function* () {
                    yield { text: 'Sure, ' };
                    yield { text: '', candidates: [{ finishReason: 'SAFETY' }] };
                })()
            );
            const lm = new GeminiLM({ apiKey: 'k' });

            const consume = async () => {
                for await (const _chunk of lm.generateStream('Hi')) {
                    // drain
                }
            };

            await expect(consume()).rejects.toBeInstanceOf(ContentFilterError);
            expect(lm.getUsage().errorCount).toBe(1);
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

        it('reports truncation rather than blaming the JSON', async () => {
            // MAX_TOKENS was never checked, so a truncated reply fell through to
            // JSON.parse and surfaced as a misleading "not valid JSON" error.
            mocks.generateContent.mockResolvedValue(
                response('{"answer":"Par', { candidates: [{ finishReason: 'MAX_TOKENS' }] })
            );

            await expect(
                new GeminiLM({ apiKey: 'k' }).generateStructured('Q', {})
            ).rejects.toThrow(/truncated; raise maxTokens/);
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

        it('forwards a caller-supplied abort signal', async () => {
            mocks.generateContent.mockResolvedValue(response('ok'));
            const controller = new AbortController();

            await new GeminiLM({ apiKey: 'k' }).generate('Hi', { signal: controller.signal });

            expect(mocks.generateContent.mock.calls[0][0].config.abortSignal).toBe(
                controller.signal
            );
        });

        it('combines a caller signal with the timeout signal', async () => {
            mocks.generateContent.mockResolvedValue(response('ok'));
            const controller = new AbortController();

            await new GeminiLM({ apiKey: 'k' }).generate('Hi', {
                signal: controller.signal,
                timeout: 60_000,
            });

            // Gemini has one abortSignal slot, so both have to be folded into a
            // single signal. `toBeInstanceOf(AbortSignal)` cannot tell a merged
            // signal from a timeout-only one — abort the caller's controller and
            // check the merged signal follows it.
            const combined = mocks.generateContent.mock.calls[0][0].config.abortSignal;
            expect(combined).not.toBe(controller.signal);
            expect(combined.aborted).toBe(false);
            controller.abort();
            expect(combined.aborted).toBe(true);
        });

        it('leaves abortSignal unset when neither a signal nor a timeout is given', async () => {
            mocks.generateContent.mockResolvedValue(response('ok'));
            await new GeminiLM({ apiKey: 'k' }).generate('Hi');

            expect(mocks.generateContent.mock.calls[0][0].config.abortSignal).toBeUndefined();
        });

        it('builds a fresh signal per request rather than sharing one', async () => {
            mocks.generateContent.mockResolvedValue(response('ok'));
            const lm = new GeminiLM({ apiKey: 'k' });

            await lm.generate('Hi', { timeout: 5000 });
            await lm.generate('Hi again', { timeout: 5000 });

            const first = mocks.generateContent.mock.calls[0][0].config.abortSignal;
            const second = mocks.generateContent.mock.calls[1][0].config.abortSignal;
            expect(first).not.toBe(second);
        });

        it('applies the constructor timeout when the call sets none', async () => {
            mocks.generateContent.mockResolvedValue(response('ok'));
            await new GeminiLM({ apiKey: 'k', timeout: 5000 }).generate('Hi');

            expect(mocks.generateContent.mock.calls[0][0].config.abortSignal).toBeInstanceOf(
                AbortSignal
            );
        });

        it('lets a longer per-call timeout override the constructor default', async () => {
            mocks.generateContent.mockResolvedValue(response('ok'));
            await new GeminiLM({ apiKey: 'k', timeout: 5 }).generate('Hi', { timeout: 60_000 });

            // The SDK applies a client-level httpOptions.timeout to every request,
            // so routing the default through it would let a per-call timeout only
            // tighten the deadline, never loosen it.
            const signal = mocks.generateContent.mock.calls[0][0].config.abortSignal;
            await new Promise((resolve) => setTimeout(resolve, 40));
            expect(signal.aborted).toBe(false);
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

        it('wraps and counts a mid-stream failure', async () => {
            mocks.generateContentStream.mockResolvedValue(
                (async function* () {
                    yield { text: 'Hel' };
                    throw Object.assign(new Error('This operation was aborted'), {
                        name: 'AbortError',
                    });
                })()
            );

            const lm = new GeminiLM({ apiKey: 'k' });

            await expect(async () => {
                for await (const chunk of lm.generateStream('Hi')) void chunk;
            }).rejects.toThrow(LMError);
            expect(lm.getUsage().errorCount).toBe(1);
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

        it('omits httpOptions entirely when no base URL is given', () => {
            new GeminiLM({ apiKey: 'k' });

            expect(mocks.constructorOptions.mock.calls[0][0]).not.toHaveProperty('httpOptions');
        });
    });

    describe('retries', () => {
        function transient(status: number) {
            return Object.assign(new Error(`status ${status}`), { status });
        }

        it('retries a transient failure and returns the eventual result', async () => {
            mocks.generateContent
                .mockRejectedValueOnce(transient(503))
                .mockResolvedValue(response('ok'));

            // Before this the Gemini provider ignored `retries` entirely.
            expect(await new GeminiLM({ apiKey: 'k' }).generate('Hi', { retries: 1 })).toBe(
                'ok'
            );
            expect(mocks.generateContent).toHaveBeenCalledTimes(2);
        });

        it('falls back to the constructor maxRetries', async () => {
            mocks.generateContent
                .mockRejectedValueOnce(transient(429))
                .mockResolvedValue(response('ok'));

            const lm = new GeminiLM({ apiKey: 'k', maxRetries: 1 });

            expect(await lm.generate('Hi')).toBe('ok');
            expect(mocks.generateContent).toHaveBeenCalledTimes(2);
        });

        it('defaults to two retries, as the OpenAI and Anthropic SDKs do', async () => {
            mocks.generateContent
                .mockRejectedValueOnce(transient(500))
                .mockRejectedValueOnce(transient(500))
                .mockResolvedValue(response('ok'));

            expect(await new GeminiLM({ apiKey: 'k' }).generate('Hi')).toBe('ok');
            expect(mocks.generateContent).toHaveBeenCalledTimes(3);
        });

        it('retries a transport failure, which carries no status', async () => {
            mocks.generateContent
                .mockRejectedValueOnce(new TypeError('fetch failed'))
                .mockResolvedValue(response('ok'));

            expect(await new GeminiLM({ apiKey: 'k' }).generate('Hi', { retries: 1 })).toBe(
                'ok'
            );
            expect(mocks.generateContent).toHaveBeenCalledTimes(2);
        });

        it('does not retry a status-less error that is not a transport failure', async () => {
            mocks.generateContent.mockRejectedValue(new Error('bad argument'));
            const lm = new GeminiLM({ apiKey: 'k' });

            // Retrying a deterministic fault just makes it three times slower.
            await expect(lm.generate('Hi', { retries: 3 })).rejects.toThrow(LMError);
            expect(mocks.generateContent).toHaveBeenCalledTimes(1);
        });

        it('does not retry a client error', async () => {
            mocks.generateContent.mockRejectedValue(transient(401));
            const lm = new GeminiLM({ apiKey: 'k' });

            await expect(lm.generate('Hi', { retries: 3 })).rejects.toThrow(LMError);
            expect(mocks.generateContent).toHaveBeenCalledTimes(1);
        });

        it('preserves the status code on the LMError', async () => {
            mocks.generateContent.mockRejectedValue(transient(400));

            // The SDK's own retry wrapper replaces API errors with generic ones,
            // losing the status; running the loop here keeps it.
            await expect(
                new GeminiLM({ apiKey: 'k', maxRetries: 2 }).generate('Hi')
            ).rejects.toMatchObject({ status: 400 });
        });

        it('stops retrying once the caller aborts, without waiting out the backoff', async () => {
            const controller = new AbortController();
            mocks.generateContent.mockImplementation(() => {
                controller.abort();
                return Promise.reject(transient(503));
            });

            const lm = new GeminiLM({ apiKey: 'k' });
            const startedAt = Date.now();

            await expect(
                lm.generate('Hi', { retries: 3, signal: controller.signal })
            ).rejects.toThrow(LMError);
            expect(mocks.generateContent).toHaveBeenCalledTimes(1);
            // The signal aborts during the attempt, so the backoff sleep has to
            // notice a signal that was already aborted when it started.
            expect(Date.now() - startedAt).toBeLessThan(100);
        });

        it('rejects without calling the SDK when the signal is already aborted', async () => {
            mocks.generateContent.mockResolvedValue(response('ok'));
            const controller = new AbortController();
            controller.abort();

            const lm = new GeminiLM({ apiKey: 'k' });

            // Gemini's client attaches the signal with an `abort` listener, which
            // never fires for a signal that aborted before the call went out.
            await expect(lm.generate('Hi', { signal: controller.signal })).rejects.toThrow(
                LMError
            );
            expect(mocks.generateContent).not.toHaveBeenCalled();
            expect(lm.getUsage().errorCount).toBe(1);
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

    describe('error classification', () => {
        it('never reports a NaN status', async () => {
            // `Number(error.status)` used to produce status: NaN for any error
            // that carried a non-numeric `status`, e.g. a Node system error.
            const systemError = Object.assign(new Error('getaddrinfo ENOTFOUND'), {
                status: 'ENOTFOUND',
            });
            mocks.generateContent.mockRejectedValue(systemError);

            const error = await new GeminiLM({ apiKey: 'k' }).generate('Hi').catch((e) => e);

            expect(error.status).toBeUndefined();
            expect(Number.isNaN(error.status)).toBe(false);
        });

        it('maps a 429 to RateLimitError and 401/403 to AuthError', async () => {
            const lm = new GeminiLM({ apiKey: 'k' });

            mocks.generateContent.mockRejectedValue(
                Object.assign(new Error('Quota exceeded'), { status: 429 })
            );
            await expect(lm.generate('Hi')).rejects.toBeInstanceOf(RateLimitError);

            mocks.generateContent.mockRejectedValue(
                Object.assign(new Error('API key not valid'), { status: 401 })
            );
            await expect(lm.generate('Hi')).rejects.toBeInstanceOf(AuthError);

            mocks.generateContent.mockRejectedValue(
                Object.assign(new Error('Permission denied'), { status: 403 })
            );
            await expect(lm.generate('Hi')).rejects.toBeInstanceOf(AuthError);
        });

        it('falls back to the message for context length, having nothing better', async () => {
            const lm = new GeminiLM({ apiKey: 'k' });

            mocks.generateContent.mockRejectedValue(
                Object.assign(
                    new Error(
                        'INVALID_ARGUMENT: The input token count exceeds the maximum ' +
                            'number of tokens allowed'
                    ),
                    { status: 400 }
                )
            );
            await expect(lm.generate('Hi')).rejects.toBeInstanceOf(ContextLengthError);

            mocks.generateContent.mockRejectedValue(
                Object.assign(new Error('INVALID_ARGUMENT: unknown field "foo"'), {
                    status: 400,
                })
            );
            const other = await lm.generate('Hi').catch((e) => e);
            expect(other).toBeInstanceOf(LMError);
            expect(other).not.toBeInstanceOf(ContextLengthError);
        });
    });
});
