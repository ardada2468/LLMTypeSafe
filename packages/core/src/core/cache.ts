import { createHash } from 'node:crypto';
import type { ChatMessage, LLMCallOptions } from '../types/language-model';

/** A value that a {@link Cache} may return synchronously or as a promise. */
export type MaybePromise<T> = T | Promise<T>;

/**
 * Storage backing response caching.
 *
 * Both methods may be async so that a Redis-, file-, or SQLite-backed store
 * fits without a wrapper. `get` resolves to `undefined` for a key that has
 * never been written; `null` counts as a miss too, since that is what a Redis
 * client returns for an absent key. Cached values are model responses, which
 * are never `undefined` or `null`, so a miss cannot be mistaken for a hit.
 */
export interface Cache {
    /** Look up a key, resolving to `undefined` or `null` when it is absent. */
    get(key: string): MaybePromise<unknown>;
    /** Store a value under a key, replacing any previous entry. */
    set(key: string, value: unknown): MaybePromise<void>;
    /** Drop every entry. Optional — not every backend can be emptied cheaply. */
    clear?(): MaybePromise<void>;
}

export interface MemoryCacheOptions {
    /**
     * Maximum number of entries retained. Once exceeded, the least recently
     * used entry is evicted. Defaults to 1000.
     */
    maxSize?: number;
}

const DEFAULT_MAX_SIZE = 1000;

/**
 * In-process LRU cache, and the default backing store for
 * `configure({ cache: true })`.
 *
 * `Map` preserves insertion order, so the oldest key is always the first one
 * the iterator yields; reads re-insert their key to mark it most recent.
 */
export class MemoryCache implements Cache {
    private readonly entries = new Map<string, unknown>();

    /** Entry ceiling this cache was constructed with. */
    readonly maxSize: number;

    constructor(options: MemoryCacheOptions = {}) {
        const maxSize = options.maxSize ?? DEFAULT_MAX_SIZE;
        if (!Number.isInteger(maxSize) || maxSize < 1) {
            throw new RangeError(
                `MemoryCache maxSize must be a positive integer, received ${String(maxSize)}`
            );
        }
        this.maxSize = maxSize;
    }

    /** Number of entries currently held. */
    get size(): number {
        return this.entries.size;
    }

    get(key: string): unknown {
        if (!this.entries.has(key)) return undefined;

        const value = this.entries.get(key);
        this.entries.delete(key);
        this.entries.set(key, value);
        return value;
    }

    set(key: string, value: unknown): void {
        this.entries.delete(key);
        this.entries.set(key, value);

        while (this.entries.size > this.maxSize) {
            const oldest = this.entries.keys().next();
            if (oldest.done) break;
            this.entries.delete(oldest.value);
        }
    }

    clear(): void {
        this.entries.clear();
    }
}

/** Everything that distinguishes one model call from another. */
export interface CacheKeyInput {
    provider: string;
    /** The model actually used, after any per-call `options.model` override. */
    model: string;
    operation: 'chat' | 'generate' | 'generateStructured';
    prompt?: string;
    messages?: ChatMessage[];
    /** JSON schema for structured calls. */
    schema?: unknown;
    options?: LLMCallOptions;
    /**
     * Per-instance provider configuration that shapes the request, from
     * `BaseLM.cacheScope()`. Keeps two differently configured clients on the
     * same model from sharing entries.
     */
    scope?: unknown;
}

/**
 * Sampling parameters that change the response and therefore the key. Options
 * that only affect transport — `timeout`, `retries`, `streaming`, `metadata` —
 * are deliberately excluded so a retry budget change does not miss the cache.
 */
const SAMPLING_PARAMS = [
    'temperature',
    'maxTokens',
    'topP',
    'frequencyPenalty',
    'presencePenalty',
    'stopSequences',
] as const;

/** Bumped whenever the key layout changes, so stale entries cannot be read. */
const KEY_VERSION = 1;

/**
 * Derive a stable, fixed-length cache key. Prompts and schemas can run to tens
 * of kilobytes, so the canonical form is hashed rather than stored.
 */
export function buildCacheKey(input: CacheKeyInput): string {
    const options = input.options ?? {};

    const payload = {
        v: KEY_VERSION,
        provider: input.provider,
        model: input.model,
        operation: input.operation,
        prompt: input.prompt ?? null,
        messages:
            input.messages?.map((message) => ({
                role: message.role,
                content: message.content,
                name: message.name ?? null,
                // The dead `functionCall` field went away with native tool
                // calling; `toolCallId` correlates a tool result to the call it
                // answers, so two otherwise-identical turns are not conflated.
                toolCallId: message.toolCallId ?? null,
                toolCalls: message.toolCalls ?? null,
            })) ?? null,
        schema: input.schema ?? null,
        scope: input.scope ?? null,
        sampling: Object.fromEntries(
            SAMPLING_PARAMS.map((param) => [param, options[param] ?? null])
        ),
    };

    return createHash('sha256').update(stableStringify(payload)).digest('hex');
}

/**
 * JSON with object keys sorted, so two schemas that differ only in property
 * order produce the same key instead of two entries for one request.
 */
function stableStringify(value: unknown): string {
    if (value === null || typeof value !== 'object') {
        return JSON.stringify(value) ?? 'null';
    }
    if (Array.isArray(value)) {
        return `[${value.map(stableStringify).join(',')}]`;
    }

    const record = value as Record<string, unknown>;
    const body = Object.keys(record)
        .sort()
        .map((key) => `${JSON.stringify(key)}:${stableStringify(record[key])}`)
        .join(',');
    return `{${body}}`;
}
