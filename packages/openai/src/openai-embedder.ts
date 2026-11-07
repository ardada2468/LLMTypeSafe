import { LMError, type Embedder } from '@ts-dspy/core';
import OpenAI from 'openai';
import { toLMError } from './openai-lm';

/** Current default embedding model: cheap, 1536 dimensions, shortenable. */
export const DEFAULT_OPENAI_EMBEDDING_MODEL = 'text-embedding-3-small';

/** The API accepts at most 2048 inputs per request. */
export const MAX_OPENAI_EMBEDDING_BATCH = 2048;

/** Native vector lengths, used when `dimensions` is not set. */
const NATIVE_DIMENSIONS: Record<string, number> = {
    'text-embedding-3-small': 1536,
    'text-embedding-3-large': 3072,
    'text-embedding-ada-002': 1536,
};

export interface OpenAIEmbedderConfig {
    /** Defaults to {@link DEFAULT_OPENAI_EMBEDDING_MODEL}. */
    model?: string;
    /**
     * Shorten vectors to this length. Supported by `text-embedding-3-*` and
     * later; earlier models reject it.
     */
    dimensions?: number;
    apiKey?: string;
    organization?: string;
    project?: string;
    /** Override for Azure, a proxy, or an OpenAI-compatible embeddings endpoint. */
    baseURL?: string;
    /** Per-request timeout in milliseconds. */
    timeout?: number;
    /** Retry count for each batch request (SDK default: 2). */
    maxRetries?: number;
    /**
     * A ready-made client, e.g. one shared with an `OpenAILM` or wrapped for
     * Azure. When given, the connection options above are ignored.
     */
    client?: Pick<OpenAI, 'embeddings'>;
    /**
     * Inputs per request, capped at {@link MAX_OPENAI_EMBEDDING_BATCH}.
     * Defaults to 512, which keeps each request well under the per-request
     * token limit for typical passages.
     */
    batchSize?: number;
}

/**
 * {@link Embedder} backed by OpenAI's embeddings endpoint.
 *
 * ```ts
 * const embedder = new OpenAIEmbedder({ dimensions: 512 });
 * const [query, doc] = await embedder.embed(['refund policy', passage]);
 * cosineSimilarity(query, doc);
 * ```
 *
 * Inputs are split into batches of `batchSize`, sent one batch at a time, and
 * returned in input order. Failures are raised as the same `LMError`
 * subclasses `OpenAILM` throws (`RateLimitError`, `AuthError`, …).
 */
export class OpenAIEmbedder implements Embedder {
    readonly model: string;
    readonly dimensions?: number;
    private readonly requestedDimensions?: number;
    private readonly batchSize: number;
    private readonly client: Pick<OpenAI, 'embeddings'>;

    constructor(config: OpenAIEmbedderConfig = {}) {
        this.model = config.model ?? DEFAULT_OPENAI_EMBEDDING_MODEL;
        this.requestedDimensions = config.dimensions;
        this.dimensions = config.dimensions ?? NATIVE_DIMENSIONS[this.model];
        this.batchSize = clampBatchSize(config.batchSize ?? 512, MAX_OPENAI_EMBEDDING_BATCH);
        this.client =
            config.client ??
            new OpenAI({
                apiKey: config.apiKey,
                organization: config.organization,
                project: config.project,
                baseURL: config.baseURL,
                timeout: config.timeout,
                maxRetries: config.maxRetries,
            });
    }

    async embed(texts: string[], options?: { signal?: AbortSignal }): Promise<number[][]> {
        const vectors: number[][] = [];
        for (let start = 0; start < texts.length; start += this.batchSize) {
            const batch = texts.slice(start, start + this.batchSize);
            vectors.push(...(await this.embedBatch(batch, options?.signal)));
        }
        return vectors;
    }

    private async embedBatch(batch: string[], signal?: AbortSignal): Promise<number[][]> {
        let response;
        try {
            signal?.throwIfAborted();
            response = await this.client.embeddings.create(
                {
                    model: this.model,
                    input: batch,
                    encoding_format: 'float',
                    ...(this.requestedDimensions !== undefined
                        ? { dimensions: this.requestedDimensions }
                        : {}),
                },
                signal ? { signal } : undefined
            );
        } catch (error) {
            throw toLMError(error);
        }

        // `index` is authoritative; the order of `data` is not promised.
        const ordered = [...response.data].sort((a, b) => a.index - b.index);
        if (ordered.length !== batch.length) {
            throw new LMError(
                'openai',
                `Expected ${batch.length} embeddings but received ${ordered.length}.`
            );
        }
        return ordered.map((item) => item.embedding);
    }
}

/** Shared by the embedders: a positive integer no larger than `max`. */
function clampBatchSize(size: number, max: number): number {
    if (!Number.isInteger(size) || size < 1) {
        throw new RangeError(`batchSize must be a positive integer, got ${size}.`);
    }
    return Math.min(size, max);
}
