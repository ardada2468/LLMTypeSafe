import type {
    ChatMessage,
    ILanguageModel,
    LLMCallOptions,
    ModelCapabilities,
    UsageStats,
} from '../types/language-model';
import { buildCacheKey, type CacheKeyInput } from './cache';
import { getCache } from './config';
import { LMError } from './errors';

/**
 * Shared implementation for language-model providers.
 *
 * Providers implement {@link chat} and {@link getCapabilities}; everything else —
 * usage accounting, response caching, `generate` delegation, and a prompt-based
 * `generateStructured` fallback for providers without native structured output —
 * lives here so it is not copy-pasted per provider.
 *
 * Caching is installed on the instance by this constructor, which puts two
 * requirements on a subclass: implement `chat`/`generate`/`generateStructured`
 * as methods rather than class fields, and do not reassign them after
 * `super()` — either would replace the cached wrapper. Any constructor option
 * that shapes the request, such as a default `maxTokens` or an alternate
 * `baseURL`, belongs in {@link cacheScope} so two differently configured
 * instances of the same model do not share entries.
 */
export abstract class BaseLM implements ILanguageModel {
    protected model: string;
    protected readonly provider: string;

    private promptTokens = 0;
    private completionTokens = 0;
    private requestCount = 0;
    private errorCount = 0;
    private cacheHits = 0;
    private totalLatencyMs = 0;

    protected constructor(provider: string, model: string) {
        this.provider = provider;
        this.model = model;

        // Caching is installed per instance rather than written into the method
        // bodies: `chat` is abstract and providers routinely override
        // `generateStructured`, so the instance is the only place that can
        // intercept every implementation. `generate` is deliberately left
        // unwrapped -- it is pure delegation to `chat`, which is cached, and
        // binding it here would mean a later override or test spy on `chat`
        // was silently bypassed.
        if (typeof this.chat !== 'function') {
            throw new TypeError(
                'A BaseLM subclass must implement chat() as a method. A class field ' +
                    '(`chat = async () => {}`) is initialised after super() returns, so ' +
                    'there is nothing for the base constructor to wrap.'
            );
        }

        const uncachedChat = this.chat.bind(this);
        const uncachedGenerateStructured = this.generateStructured.bind(this);

        this.chat = (messages, options) =>
            this.throughCache({ operation: 'chat', messages, options }, () =>
                uncachedChat(messages, options)
            );

        this.generateStructured = <T>(
            prompt: string,
            schema: unknown,
            options?: LLMCallOptions
        ) =>
            this.throughCache<T>(
                { operation: 'generateStructured', prompt, schema, options },
                () => uncachedGenerateStructured<T>(prompt, schema, options)
            );
    }

    abstract chat(messages: ChatMessage[], options?: LLMCallOptions): Promise<string>;

    abstract getCapabilities(): ModelCapabilities;

    async generate(prompt: string, options?: LLMCallOptions): Promise<string> {
        return this.chat([{ role: 'user', content: prompt }], options);
    }

    /**
     * Default structured-output implementation: ask for JSON in the prompt and
     * parse the reply. Providers with a native JSON-schema mode should override
     * this — the native path constrains decoding, this one only requests it.
     */
    async generateStructured<T>(
        prompt: string,
        schema: unknown,
        options?: LLMCallOptions
    ): Promise<T> {
        const instruction =
            `${prompt}\n\nRespond with JSON matching this schema. ` +
            `Output only the JSON object, with no surrounding prose or code fences.\n` +
            `${JSON.stringify(schema, null, 2)}`;

        const raw = await this.generate(instruction, options);
        return parseJsonResponse<T>(raw, this.provider);
    }

    getUsage(): UsageStats {
        return {
            promptTokens: this.promptTokens,
            completionTokens: this.completionTokens,
            totalTokens: this.promptTokens + this.completionTokens,
            requestCount: this.requestCount,
            errorCount: this.errorCount,
            cacheHits: this.cacheHits,
            averageLatency: this.requestCount > 0 ? this.totalLatencyMs / this.requestCount : 0,
        };
    }

    resetUsage(): void {
        this.promptTokens = 0;
        this.completionTokens = 0;
        this.requestCount = 0;
        this.errorCount = 0;
        this.cacheHits = 0;
        this.totalLatencyMs = 0;
    }

    getModelName(): string {
        return this.model;
    }

    setModel(model: string): void {
        this.model = model;
    }

    /** Record a completed request's token usage and latency. */
    protected recordUsage(usage: {
        promptTokens?: number;
        completionTokens?: number;
        latencyMs?: number;
    }): void {
        this.promptTokens += usage.promptTokens ?? 0;
        this.completionTokens += usage.completionTokens ?? 0;
        this.totalLatencyMs += usage.latencyMs ?? 0;
        this.requestCount += 1;
    }

    /** Record a failed request. */
    protected recordError(): void {
        this.errorCount += 1;
    }

    /**
     * Per-instance configuration that changes what the model returns, folded
     * into every cache key alongside the provider, model, and call options.
     *
     * Provider and model alone do not identify a request: an instance
     * constructed with a default `maxTokens`, non-default safety settings, or a
     * `baseURL` pointing at a proxy answers the same prompt differently. Return
     * those here — any JSON-serialisable value will do — and the entries stay
     * apart. The default is `undefined`, which keeps the key to the call itself.
     */
    protected cacheScope(): unknown {
        return undefined;
    }

    /**
     * Serve `run` from the configured cache when possible.
     *
     * A hit bumps `cacheHits` and nothing else: counting it as a request, or
     * adding the original call's tokens a second time, would quietly overstate
     * spend for every prompt a program repeats. Rejections propagate uncached,
     * so a transient 429 is not pinned to a prompt for the life of the process.
     */
    private async throughCache<T>(
        input: Omit<CacheKeyInput, 'provider' | 'model'>,
        run: () => Promise<T>
    ): Promise<T> {
        const cache = getCache();
        if (!cache) return run();

        const key = buildCacheKey({
            ...input,
            provider: this.provider,
            model: input.options?.model ?? this.model,
            scope: this.cacheScope(),
        });

        // `null` counts as a miss as well as `undefined`: a cached response is
        // never either, and Redis clients resolve a missing key to `null`, so a
        // custom store that forgets to translate would otherwise report a hit
        // and fail somewhere far away from the cache.
        const cached = await cache.get(key);
        if (cached !== undefined && cached !== null) {
            this.cacheHits += 1;
            return isolate(cached, input.operation) as T;
        }

        const value = await run();
        await cache.set(key, isolate(value, input.operation));
        return value;
    }
}

/**
 * Copy a structured result before it crosses the cache boundary.
 *
 * `generateStructured` resolves to a parsed JSON value, and handing every
 * caller the one stored object would let a single mutation downstream corrupt
 * the entry for every later hit. Text responses are immutable, so they pass
 * through untouched.
 */
function isolate<T>(value: T, operation: CacheKeyInput['operation']): T {
    if (operation !== 'generateStructured') return value;
    if (typeof value !== 'object' || value === null) return value;
    return structuredClone(value);
}

/** Extract a JSON object from a model reply, tolerating code fences. */
export function parseJsonResponse<T>(raw: string, provider: string): T {
    const fenced = raw.match(/```(?:json)?\s*([\s\S]*?)```/i);
    const candidate = (fenced?.[1] ?? raw).trim();

    try {
        return JSON.parse(candidate) as T;
    } catch (cause) {
        // Last resort: grab the outermost brace-delimited span.
        const start = candidate.indexOf('{');
        const end = candidate.lastIndexOf('}');
        if (start !== -1 && end > start) {
            try {
                return JSON.parse(candidate.slice(start, end + 1)) as T;
            } catch {
                // fall through to the error below
            }
        }
        throw new LMError(provider, `Model did not return valid JSON: ${candidate}`, {
            cause,
        });
    }
}
