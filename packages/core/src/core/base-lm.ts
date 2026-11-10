import type {
    ChatMessage,
    ChatResult,
    ILanguageModel,
    LLMCallOptions,
    ModelCapabilities,
    UsageStats,
} from '../types/language-model';
import { buildCacheKey, type CacheKeyInput } from './cache';
import { getCache } from './config';
import { LMError } from './errors';
import { recordLMUsage } from './usage';

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
    private cachedTokens = 0;
    private reasoningTokens = 0;
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
     * Default tool-calling implementation: run the turn as plain chat and report
     * no tool calls. Providers with native tool calling override this; a provider
     * that does must also report `supportsFunctionCalling: true`, since that flag
     * is what callers such as `ReAct` branch on.
     */
    async chatWithTools(
        messages: ChatMessage[],
        options?: LLMCallOptions
    ): Promise<ChatResult> {
        return { content: await this.chat(messages, options), finishReason: 'stop' };
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
        const usage: UsageStats = {
            promptTokens: this.promptTokens,
            completionTokens: this.completionTokens,
            totalTokens: this.promptTokens + this.completionTokens,
            requestCount: this.requestCount,
            errorCount: this.errorCount,
            cacheHits: this.cacheHits,
            averageLatency: this.requestCount > 0 ? this.totalLatencyMs / this.requestCount : 0,
        };
        // Optional, and reported only when a provider ever supplied them, so a
        // provider that cannot see these does not claim a confident zero.
        if (this.cachedTokens > 0) usage.cachedTokens = this.cachedTokens;
        if (this.reasoningTokens > 0) usage.reasoningTokens = this.reasoningTokens;
        return usage;
    }

    resetUsage(): void {
        this.promptTokens = 0;
        this.completionTokens = 0;
        this.cachedTokens = 0;
        this.reasoningTokens = 0;
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

    /**
     * The key this instance's usage is reported under in per-call usage maps,
     * e.g. `openai/gpt-5.2`. Also the first key looked up in
     * `configure({ pricing })`.
     */
    protected usageKey(model: string = this.model): string {
        return `${this.provider}/${model}`;
    }

    /**
     * Record a completed request's token usage and latency.
     *
     * Updates this instance's lifetime counters and, when a module invocation
     * or evaluation is collecting usage in the current async scope, that
     * scope's collector too — which is what makes per-call usage exact under
     * concurrency. Pass `model` when the request used a per-call model
     * override, so the usage is attributed to the model that actually ran.
     */
    protected recordUsage(usage: {
        promptTokens?: number;
        completionTokens?: number;
        /** Prompt tokens served from the provider's prompt cache. */
        cachedTokens?: number;
        /** Hidden reasoning tokens, part of `completionTokens`. */
        reasoningTokens?: number;
        latencyMs?: number;
        model?: string;
    }): void {
        this.promptTokens += usage.promptTokens ?? 0;
        this.completionTokens += usage.completionTokens ?? 0;
        this.cachedTokens += usage.cachedTokens ?? 0;
        this.reasoningTokens += usage.reasoningTokens ?? 0;
        this.totalLatencyMs += usage.latencyMs ?? 0;
        this.requestCount += 1;
        recordLMUsage(this.usageKey(usage.model), usage);
    }

    /** Record a failed request. */
    protected recordError(): void {
        this.errorCount += 1;
        recordLMUsage(this.usageKey(), { error: true });
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
        //
        // A cache is an optimisation, never a dependency: a store that is down
        // reads as a miss, and a failed write must not discard a reply that
        // has already been paid for.
        let cached: unknown;
        try {
            cached = await cache.get(key);
        } catch {
            cached = undefined;
        }
        if (cached !== undefined && cached !== null) {
            this.cacheHits += 1;
            recordLMUsage(this.usageKey(input.options?.model), { cacheHit: true });
            return isolate(cached, input.operation) as T;
        }

        const value = await run();
        try {
            await cache.set(key, isolate(value, input.operation));
        } catch {
            // See above.
        }
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

/**
 * Extract a JSON object from a model reply, tolerating code fences and stray
 * prose around the object.
 *
 * Pass a provider name to fail with an {@link LMError}, or a function to build
 * the error yourself — `Predict` reports an unparseable structured reply as a
 * `ValidationError`, so the repair loop can act on it.
 */
export function parseJsonResponse<T>(raw: string, provider: string): T;
export function parseJsonResponse<T>(
    raw: string,
    onError: (candidate: string, cause: unknown) => Error
): T;
export function parseJsonResponse<T>(
    raw: string,
    providerOrOnError: string | ((candidate: string, cause: unknown) => Error)
): T {
    // A reply that is already JSON is taken as it is. Looking for a code fence
    // first would mistake a fenced snippet *inside* a string value — any
    // field holding code or markdown — for the payload.
    const trimmed = raw.trim();
    try {
        return JSON.parse(trimmed) as T;
    } catch {
        // Not bare JSON; look for a fence or a brace-delimited span below.
    }

    const fenced =
        trimmed.match(/^```(?:json)?[ \t]*\n([\s\S]*)\n[ \t]*```$/i) ??
        raw.match(/```(?:json)?\s*([\s\S]*?)```/i);
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
        if (typeof providerOrOnError === 'function') {
            throw providerOrOnError(candidate, cause);
        }
        throw new LMError(providerOrOnError, `Model did not return valid JSON: ${candidate}`, {
            cause,
        });
    }
}
