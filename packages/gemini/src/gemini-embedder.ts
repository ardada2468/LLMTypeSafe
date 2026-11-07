import { LMError, retryWithBackoff, type Embedder } from '@ts-dspy/core';
import { GoogleGenAI, type EmbedContentConfig } from '@google/genai';
import { isRetryable, toLMError } from './gemini-lm';

/** Current default embedding model on both the Gemini API and Vertex AI. */
export const DEFAULT_GEMINI_EMBEDDING_MODEL = 'gemini-embedding-001';

/** `batchEmbedContents` accepts at most 100 inputs per request. */
export const MAX_GEMINI_EMBEDDING_BATCH = 100;

/** Native vector lengths, used when `outputDimensionality` is not set. */
const NATIVE_DIMENSIONS: Record<string, number> = {
    'gemini-embedding-001': 3072,
};

export interface GeminiEmbedderConfig {
    /** Defaults to {@link DEFAULT_GEMINI_EMBEDDING_MODEL}. */
    model?: string;
    /**
     * What the vectors are for, e.g. `'RETRIEVAL_QUERY'`, `'RETRIEVAL_DOCUMENT'`,
     * `'SEMANTIC_SIMILARITY'` or `'CLASSIFICATION'`. Embed queries and
     * documents with their matching task types for the best retrieval.
     */
    taskType?: string;
    /** Truncate vectors to this length (e.g. 768 or 1536 for `gemini-embedding-001`). */
    outputDimensionality?: number;
    apiKey?: string;
    /** Route through Vertex AI instead of the Gemini API. */
    vertexai?: boolean;
    project?: string;
    location?: string;
    /** Override the API endpoint, e.g. for a proxy. */
    baseUrl?: string;
    /** A ready-made client, e.g. one shared with a `GeminiLM`. Connection options are then ignored. */
    client?: Pick<GoogleGenAI, 'models'>;
    /** Inputs per request, capped at {@link MAX_GEMINI_EMBEDDING_BATCH}. Defaults to 100. */
    batchSize?: number;
    /** Retries per batch for 408/409/429/5xx and transport failures. Defaults to 2. */
    maxRetries?: number;
    /** Per-request timeout in milliseconds. */
    timeout?: number;
}

/**
 * {@link Embedder} backed by Gemini's `embedContent`.
 *
 * ```ts
 * const docs = new GeminiEmbedder({ taskType: 'RETRIEVAL_DOCUMENT', outputDimensionality: 768 });
 * const vectors = await docs.embed(passages);
 * ```
 *
 * Inputs are split into batches of `batchSize`, sent one batch at a time, and
 * returned in input order. Each input travels as its own content, so models
 * that would otherwise merge a list of strings into one multimodal input still
 * return one vector per text. Failures raise the same `LMError` subclasses
 * `GeminiLM` throws, after the same retry policy.
 */
export class GeminiEmbedder implements Embedder {
    readonly model: string;
    readonly dimensions?: number;
    private readonly taskType?: string;
    private readonly outputDimensionality?: number;
    private readonly batchSize: number;
    private readonly maxRetries: number;
    private readonly timeout?: number;
    private readonly client: Pick<GoogleGenAI, 'models'>;

    constructor(config: GeminiEmbedderConfig = {}) {
        this.model = config.model ?? DEFAULT_GEMINI_EMBEDDING_MODEL;
        this.taskType = config.taskType;
        this.outputDimensionality = config.outputDimensionality;
        this.dimensions = config.outputDimensionality ?? NATIVE_DIMENSIONS[this.model];
        this.maxRetries = config.maxRetries ?? 2;
        this.timeout = config.timeout;

        const requested = config.batchSize ?? MAX_GEMINI_EMBEDDING_BATCH;
        if (!Number.isInteger(requested) || requested < 1) {
            throw new RangeError(`batchSize must be a positive integer, got ${requested}.`);
        }
        // On Vertex AI, Gemini embedding models other than gemini-embedding-001
        // go through an endpoint that takes one input per request.
        const singleInput =
            config.vertexai === true &&
            this.model.includes('gemini') &&
            this.model !== 'gemini-embedding-001';
        this.batchSize = singleInput ? 1 : Math.min(requested, MAX_GEMINI_EMBEDDING_BATCH);

        this.client =
            config.client ??
            new GoogleGenAI({
                apiKey: config.apiKey,
                vertexai: config.vertexai,
                project: config.project,
                location: config.location,
                ...(config.baseUrl ? { httpOptions: { baseUrl: config.baseUrl } } : {}),
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
            response = await retryWithBackoff(
                () =>
                    this.client.models.embedContent({
                        model: this.model,
                        contents: batch.map((text) => ({ role: 'user', parts: [{ text }] })),
                        config: this.requestConfig(signal),
                    }),
                { maxRetries: this.maxRetries, signal, shouldRetry: isRetryable }
            );
        } catch (error) {
            throw toLMError(error);
        }

        const embeddings = response.embeddings ?? [];
        if (embeddings.length !== batch.length) {
            throw new LMError(
                'gemini',
                `Expected ${batch.length} embeddings but received ${embeddings.length}.`
            );
        }
        return embeddings.map((embedding) => embedding.values ?? []);
    }

    /** Built per request, so a timeout never outlives the call it bounds. */
    private requestConfig(signal?: AbortSignal): EmbedContentConfig {
        const config: EmbedContentConfig = {};
        if (this.taskType) config.taskType = this.taskType;
        if (this.outputDimensionality !== undefined) {
            config.outputDimensionality = this.outputDimensionality;
        }
        const signals: AbortSignal[] = [];
        if (signal) signals.push(signal);
        if (this.timeout !== undefined) signals.push(AbortSignal.timeout(this.timeout));
        if (signals.length > 0) {
            config.abortSignal = signals.length === 1 ? signals[0] : AbortSignal.any(signals);
        }
        return config;
    }
}
