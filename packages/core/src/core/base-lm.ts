import type {
    ChatMessage,
    ILanguageModel,
    LLMCallOptions,
    ModelCapabilities,
    UsageStats,
} from '../types/language-model';
import { LMError } from './errors';

/**
 * Shared implementation for language-model providers.
 *
 * Providers implement {@link chat} and {@link getCapabilities}; everything else —
 * usage accounting, `generate` delegation, and a prompt-based
 * `generateStructured` fallback for providers without native structured output —
 * lives here so it is not copy-pasted per provider.
 */
export abstract class BaseLM implements ILanguageModel {
    protected model: string;
    protected readonly provider: string;

    private promptTokens = 0;
    private completionTokens = 0;
    private requestCount = 0;
    private errorCount = 0;
    private totalLatencyMs = 0;

    protected constructor(provider: string, model: string) {
        this.provider = provider;
        this.model = model;
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
            averageLatency: this.requestCount > 0 ? this.totalLatencyMs / this.requestCount : 0,
        };
    }

    resetUsage(): void {
        this.promptTokens = 0;
        this.completionTokens = 0;
        this.requestCount = 0;
        this.errorCount = 0;
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
