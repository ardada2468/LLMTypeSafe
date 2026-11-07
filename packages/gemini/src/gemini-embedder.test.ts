import { AuthError, LMError, RateLimitError } from '@ts-dspy/core';
import { GeminiEmbedder, DEFAULT_GEMINI_EMBEDDING_MODEL } from './gemini-embedder';

const mocks = vi.hoisted(() => ({ embedContent: vi.fn(), constructed: vi.fn() }));

vi.mock('@google/genai', () => ({
    GoogleGenAI: class {
        models = { embedContent: mocks.embedContent };
        constructor(options: unknown) {
            mocks.constructed(options);
        }
    },
    HarmCategory: {},
    HarmBlockThreshold: {},
    FinishReason: {},
}));

type Request = { contents: Array<{ parts: Array<{ text: string }> }> };

/** One vector per input, numbered across batches. */
function echoEmbeddings() {
    let seen = 0;
    mocks.embedContent.mockImplementation(async (request: Request) => {
        const embeddings = request.contents.map((_, i) => ({ values: [seen + i] }));
        seen += request.contents.length;
        return { embeddings };
    });
}

function apiError(status: number) {
    return Object.assign(new Error(`status ${status}`), { status });
}

beforeEach(() => {
    mocks.embedContent.mockReset();
    mocks.constructed.mockReset();
});

describe('GeminiEmbedder', () => {
    it('defaults to gemini-embedding-001 and its native dimensions', () => {
        const embedder = new GeminiEmbedder({ apiKey: 'k' });
        expect(embedder.model).toBe(DEFAULT_GEMINI_EMBEDDING_MODEL);
        expect(DEFAULT_GEMINI_EMBEDDING_MODEL).toBe('gemini-embedding-001');
        expect(embedder.dimensions).toBe(3072);
    });

    it('sends each text as its own content with task type and dimensionality', async () => {
        echoEmbeddings();
        const embedder = new GeminiEmbedder({
            apiKey: 'k',
            taskType: 'RETRIEVAL_DOCUMENT',
            outputDimensionality: 768,
        });

        await embedder.embed(['a', 'b']);

        expect(embedder.dimensions).toBe(768);
        const request = mocks.embedContent.mock.calls[0][0];
        expect(request.model).toBe('gemini-embedding-001');
        expect(request.contents).toEqual([
            { role: 'user', parts: [{ text: 'a' }] },
            { role: 'user', parts: [{ text: 'b' }] },
        ]);
        expect(request.config).toEqual({
            taskType: 'RETRIEVAL_DOCUMENT',
            outputDimensionality: 768,
        });
    });

    it('splits inputs into batches and keeps input order', async () => {
        echoEmbeddings();
        const vectors = await new GeminiEmbedder({ apiKey: 'k', batchSize: 2 }).embed([
            'a',
            'b',
            'c',
        ]);

        expect(
            mocks.embedContent.mock.calls.map((call) =>
                (call[0] as Request).contents.map((c) => c.parts[0].text)
            )
        ).toEqual([['a', 'b'], ['c']]);
        expect(vectors).toEqual([[0], [1], [2]]);
    });

    it('caps batches at 100 inputs', async () => {
        echoEmbeddings();
        await new GeminiEmbedder({ apiKey: 'k', batchSize: 500 }).embed(
            Array.from({ length: 101 }, (_, i) => `t${i}`)
        );
        expect(
            mocks.embedContent.mock.calls.map((call) => (call[0] as Request).contents.length)
        ).toEqual([100, 1]);
    });

    it('sends one input per request for single-input Vertex models', async () => {
        echoEmbeddings();
        await new GeminiEmbedder({
            vertexai: true,
            project: 'p',
            location: 'us-central1',
            model: 'gemini-embedding-2',
        }).embed(['a', 'b']);
        expect(mocks.embedContent).toHaveBeenCalledTimes(2);
    });

    it('makes no request for an empty input', async () => {
        expect(await new GeminiEmbedder({ apiKey: 'k' }).embed([])).toEqual([]);
        expect(mocks.embedContent).not.toHaveBeenCalled();
    });

    it('passes the abort signal and stops once it aborts', async () => {
        const controller = new AbortController();
        mocks.embedContent.mockImplementation(async (request: Request) => {
            controller.abort();
            return { embeddings: request.contents.map(() => ({ values: [1] })) };
        });

        await expect(
            new GeminiEmbedder({ apiKey: 'k', batchSize: 1 }).embed(['a', 'b'], {
                signal: controller.signal,
            })
        ).rejects.toBeInstanceOf(LMError);
        expect(mocks.embedContent).toHaveBeenCalledTimes(1);
        expect(mocks.embedContent.mock.calls[0][0].config.abortSignal).toBe(controller.signal);
    });

    it('retries transient failures, then classifies the final error', async () => {
        mocks.embedContent
            .mockRejectedValueOnce(apiError(503))
            .mockResolvedValueOnce({ embeddings: [{ values: [1] }] });
        expect(await new GeminiEmbedder({ apiKey: 'k', maxRetries: 1 }).embed(['a'])).toEqual([
            [1],
        ]);

        mocks.embedContent.mockRejectedValue(apiError(429));
        await expect(
            new GeminiEmbedder({ apiKey: 'k', maxRetries: 0 }).embed(['a'])
        ).rejects.toBeInstanceOf(RateLimitError);

        mocks.embedContent.mockReset();
        mocks.embedContent.mockRejectedValue(apiError(401));
        await expect(new GeminiEmbedder({ apiKey: 'k' }).embed(['a'])).rejects.toBeInstanceOf(
            AuthError
        );
        // Client errors are final.
        expect(mocks.embedContent).toHaveBeenCalledTimes(1);
    });

    it('raises when the response is missing vectors', async () => {
        mocks.embedContent.mockResolvedValue({ embeddings: [{ values: [1] }] });
        await expect(new GeminiEmbedder({ apiKey: 'k' }).embed(['a', 'b'])).rejects.toThrow(
            /Expected 2 embeddings/
        );
    });

    it('uses an injected client', async () => {
        const embedContent = vi.fn().mockResolvedValue({ embeddings: [{ values: [5] }] });
        const embedder = new GeminiEmbedder({ client: { models: { embedContent } } as never });
        expect(await embedder.embed(['a'])).toEqual([[5]]);
        expect(mocks.constructed).not.toHaveBeenCalled();
    });
});
