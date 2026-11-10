import { AuthError, LMError, RateLimitError } from '@ts-dspy/core';
import { OpenAIEmbedder, DEFAULT_OPENAI_EMBEDDING_MODEL } from './openai-embedder';

const mocks = vi.hoisted(() => {
    class MockAPIError extends Error {
        status: number;
        code: string | null = null;
        type: string | undefined;
        constructor(status: number, message: string) {
            super(message);
            this.status = status;
        }
    }
    class MockAPIConnectionTimeoutError extends MockAPIError {}
    return {
        create: vi.fn(),
        constructed: vi.fn(),
        MockAPIError,
        MockAPIConnectionTimeoutError,
        MockAPIUserAbortError: class MockAPIUserAbortError extends Error {
            constructor() {
                super('Request was aborted.');
            }
        },
    };
});

vi.mock('openai', () => ({
    default: class {
        embeddings = { create: mocks.create };
        constructor(options: unknown) {
            mocks.constructed(options);
        }
    },
    APIError: mocks.MockAPIError,
    APIConnectionTimeoutError: mocks.MockAPIConnectionTimeoutError,
    APIUserAbortError: mocks.MockAPIUserAbortError,
}));

/** Answer each request with one vector per input, `[batchOffset + i]`. */
function echoEmbeddings(reverse = false) {
    let seen = 0;
    mocks.create.mockImplementation(async (body: { input: string[] }) => {
        const data = body.input.map((_, index) => ({
            index,
            embedding: [seen + index],
            object: 'embedding',
        }));
        seen += body.input.length;
        return { data: reverse ? data.reverse() : data };
    });
}

beforeEach(() => {
    mocks.create.mockReset();
    mocks.constructed.mockReset();
});

describe('OpenAIEmbedder', () => {
    it('defaults to text-embedding-3-small and reports its native dimensions', () => {
        const embedder = new OpenAIEmbedder({ apiKey: 'k' });
        expect(embedder.model).toBe(DEFAULT_OPENAI_EMBEDDING_MODEL);
        expect(DEFAULT_OPENAI_EMBEDDING_MODEL).toBe('text-embedding-3-small');
        expect(embedder.dimensions).toBe(1536);
    });

    it('sends the model, float encoding and requested dimensions', async () => {
        echoEmbeddings();
        const embedder = new OpenAIEmbedder({ apiKey: 'k', dimensions: 256 });

        await embedder.embed(['a']);

        expect(embedder.dimensions).toBe(256);
        expect(mocks.create.mock.calls[0][0]).toEqual({
            model: 'text-embedding-3-small',
            input: ['a'],
            encoding_format: 'float',
            dimensions: 256,
        });
    });

    it('omits dimensions when not requested', async () => {
        echoEmbeddings();
        await new OpenAIEmbedder({ apiKey: 'k' }).embed(['a']);
        expect(mocks.create.mock.calls[0][0]).not.toHaveProperty('dimensions');
    });

    it('splits inputs into batches and keeps input order', async () => {
        echoEmbeddings();
        const vectors = await new OpenAIEmbedder({ apiKey: 'k', batchSize: 2 }).embed([
            'a',
            'b',
            'c',
            'd',
            'e',
        ]);

        expect(mocks.create.mock.calls.map((call) => call[0].input)).toEqual([
            ['a', 'b'],
            ['c', 'd'],
            ['e'],
        ]);
        expect(vectors).toEqual([[0], [1], [2], [3], [4]]);
    });

    it('orders results by index, not by response position', async () => {
        echoEmbeddings(true);
        expect(await new OpenAIEmbedder({ apiKey: 'k' }).embed(['a', 'b', 'c'])).toEqual([
            [0],
            [1],
            [2],
        ]);
    });

    it('caps the batch size at the API limit', async () => {
        echoEmbeddings();
        await new OpenAIEmbedder({ apiKey: 'k', batchSize: 10_000 }).embed(
            Array.from({ length: 2049 }, (_, i) => `t${i}`)
        );
        expect(mocks.create.mock.calls.map((call) => call[0].input.length)).toEqual([2048, 1]);
    });

    it('rejects a nonsensical batch size', () => {
        expect(() => new OpenAIEmbedder({ apiKey: 'k', batchSize: 0 })).toThrow(RangeError);
    });

    it('makes no request for an empty input', async () => {
        expect(await new OpenAIEmbedder({ apiKey: 'k' }).embed([])).toEqual([]);
        expect(mocks.create).not.toHaveBeenCalled();
    });

    it('forwards the abort signal and stops before the next batch once aborted', async () => {
        const controller = new AbortController();
        mocks.create.mockImplementation(async (body: { input: string[] }) => {
            controller.abort();
            return { data: body.input.map((_, index) => ({ index, embedding: [index] })) };
        });

        await expect(
            new OpenAIEmbedder({ apiKey: 'k', batchSize: 1 }).embed(['a', 'b'], {
                signal: controller.signal,
            })
        ).rejects.toBeInstanceOf(LMError);
        expect(mocks.create).toHaveBeenCalledTimes(1);
        expect(mocks.create.mock.calls[0][1]).toEqual({ signal: controller.signal });
    });

    it('classifies SDK errors like OpenAILM does', async () => {
        mocks.create.mockRejectedValueOnce(new mocks.MockAPIError(429, 'slow down'));
        await expect(new OpenAIEmbedder({ apiKey: 'k' }).embed(['a'])).rejects.toBeInstanceOf(
            RateLimitError
        );
        mocks.create.mockRejectedValueOnce(new mocks.MockAPIError(401, 'bad key'));
        await expect(new OpenAIEmbedder({ apiKey: 'k' }).embed(['a'])).rejects.toBeInstanceOf(
            AuthError
        );
    });

    it('raises when the response is missing vectors', async () => {
        mocks.create.mockResolvedValue({ data: [{ index: 0, embedding: [1] }] });
        await expect(new OpenAIEmbedder({ apiKey: 'k' }).embed(['a', 'b'])).rejects.toThrow(
            /Expected 2 embeddings/
        );
    });

    it('uses an injected client instead of constructing one', async () => {
        const create = vi.fn().mockResolvedValue({ data: [{ index: 0, embedding: [9] }] });
        const embedder = new OpenAIEmbedder({
            client: { embeddings: { create } } as never,
        });

        expect(await embedder.embed(['a'])).toEqual([[9]]);
        expect(mocks.constructed).not.toHaveBeenCalled();
    });
});
