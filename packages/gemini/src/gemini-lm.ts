import {
    BaseLM,
    LMError,
    type ChatMessage,
    type LLMCallOptions,
    type ModelCapabilities,
    type StreamChunk,
} from '@ts-dspy/core';
import {
    GoogleGenAI,
    HarmBlockThreshold,
    HarmCategory,
    type Content,
    type GenerateContentConfig,
    type GenerateContentResponse,
    type SafetySetting,
} from '@google/genai';

/** Current default. Gemini 2.x models have reached end of life. */
export const DEFAULT_GEMINI_MODEL = 'gemini-3.5-flash';

export interface GeminiConfig {
    /** Gemini API key. Not required when `vertexai` is true and ADC is configured. */
    apiKey?: string;
    model?: string;
    /** Route through Vertex AI instead of the Gemini API. */
    vertexai?: boolean;
    /** GCP project, for Vertex AI. */
    project?: string;
    /** GCP location, for Vertex AI. */
    location?: string;
    /** Override the API endpoint, e.g. for a proxy. */
    baseUrl?: string;
    /** Default per-request timeout in milliseconds. Overridden by `options.timeout`. */
    timeout?: number;
    /**
     * Default retry count, defaulting to 2 as the OpenAI and Anthropic SDKs do.
     * Overridden by `options.retries`.
     */
    maxRetries?: number;
    /**
     * Safety thresholds. Defaults to `BLOCK_MEDIUM_AND_ABOVE` across all four
     * harm categories — the previous implementation configured only harassment
     * and silently left the rest at their service defaults.
     */
    safetySettings?: SafetySetting[];
}

const DEFAULT_SAFETY_SETTINGS: SafetySetting[] = [
    HarmCategory.HARM_CATEGORY_HARASSMENT,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
].map((category) => ({ category, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE }));

/** Context windows by model family; the 1M default matches current Gemini models. */
function contextLengthFor(model: string): number {
    if (model.includes('flash-lite')) return 1_000_000;
    if (model.includes('flash')) return 1_000_000;
    if (model.includes('pro')) return 1_000_000;
    return 1_000_000;
}

/** Statuses worth another attempt: rate limits, timeouts and transient faults. */
const RETRYABLE_STATUS = new Set([408, 409, 429, 500, 502, 503, 504]);

/** First backoff step; doubles per attempt. */
const RETRY_BASE_DELAY_MS = 250;

export class GeminiLM extends BaseLM {
    private readonly client: GoogleGenAI;
    private readonly safetySettings: SafetySetting[];
    private readonly timeout?: number;
    private readonly maxRetries: number;

    constructor(config: GeminiConfig = {}) {
        super('gemini', config.model ?? DEFAULT_GEMINI_MODEL);

        this.client = new GoogleGenAI({
            apiKey: config.apiKey,
            vertexai: config.vertexai,
            project: config.project,
            location: config.location,
            ...(config.baseUrl ? { httpOptions: { baseUrl: config.baseUrl } } : {}),
        });
        this.safetySettings = config.safetySettings ?? DEFAULT_SAFETY_SETTINGS;
        this.timeout = config.timeout;
        // 2 is what the OpenAI and Anthropic SDKs default to; matching them is
        // the point of running a retry loop here at all.
        this.maxRetries = config.maxRetries ?? 2;
    }

    async chat(messages: ChatMessage[], options?: LLMCallOptions): Promise<string> {
        const { contents, systemInstruction } = toGeminiContents(messages);
        const response = await this.send(contents, systemInstruction, options);
        return response.text ?? '';
    }

    async generateStructured<T>(
        prompt: string,
        schema: unknown,
        options?: LLMCallOptions
    ): Promise<T> {
        const { contents, systemInstruction } = toGeminiContents([
            { role: 'user', content: prompt },
        ]);

        const response = await this.send(contents, systemInstruction, options, {
            responseMimeType: 'application/json',
            responseJsonSchema: schema,
        });

        const text = response.text ?? '';
        try {
            return JSON.parse(text) as T;
        } catch (cause) {
            throw new LMError('gemini', `Structured response was not valid JSON: ${text}`, {
                cause,
            });
        }
    }

    async *generateStream(
        prompt: string,
        options?: LLMCallOptions
    ): AsyncGenerator<StreamChunk, void, unknown> {
        yield* this.chatStream([{ role: 'user', content: prompt }], options);
    }

    async *chatStream(
        messages: ChatMessage[],
        options?: LLMCallOptions
    ): AsyncGenerator<StreamChunk, void, unknown> {
        const { contents, systemInstruction } = toGeminiContents(messages);
        const startedAt = Date.now();

        let last: GenerateContentResponse | undefined;
        try {
            const stream = await this.withRetries(options, () =>
                this.client.models.generateContentStream({
                    model: options?.model ?? this.model,
                    contents,
                    config: this.buildConfig(systemInstruction, options),
                })
            );

            // The loop is inside the try so a mid-stream abort surfaces as an
            // LMError and is counted, rather than escaping as a raw SDK error.
            for await (const chunk of stream) {
                last = chunk;
                const text = chunk.text;
                if (text) {
                    yield { content: text, done: false };
                }
            }
        } catch (error) {
            this.recordError();
            throw toLMError(error);
        }

        this.recordUsageFrom(last, startedAt);
        yield { content: '', done: true, usage: usageFrom(last) };
    }

    getCapabilities(): ModelCapabilities {
        return {
            supportsStreaming: true,
            supportsStructuredOutput: true,
            supportsFunctionCalling: true,
            supportsVision: true,
            maxContextLength: contextLengthFor(this.model),
            supportedFormats: ['text', 'json_object', 'json_schema'],
        };
    }

    /** @deprecated Use {@link getModelName}. */
    getModel(): string {
        return this.model;
    }

    private async send(
        contents: Content[],
        systemInstruction: string | undefined,
        options?: LLMCallOptions,
        extraConfig?: Partial<GenerateContentConfig>
    ): Promise<GenerateContentResponse> {
        const startedAt = Date.now();

        let response: GenerateContentResponse;
        try {
            response = await this.withRetries(options, () =>
                this.client.models.generateContent({
                    model: options?.model ?? this.model,
                    contents,
                    config: { ...this.buildConfig(systemInstruction, options), ...extraConfig },
                })
            );
        } catch (error) {
            this.recordError();
            throw toLMError(error);
        }

        // Check the block reason before touching `text`. The previous
        // implementation read `response.text()` first, which threw on blocked
        // responses and made this branch unreachable.
        const blockReason = response.promptFeedback?.blockReason;
        if (blockReason) {
            this.recordError();
            throw new LMError('gemini', `Request blocked by safety filters: ${blockReason}`);
        }

        this.recordUsageFrom(response, startedAt);
        return response;
    }

    private buildConfig(
        systemInstruction: string | undefined,
        options?: LLMCallOptions
    ): GenerateContentConfig {
        const config: GenerateContentConfig = {
            safetySettings: this.safetySettings,
        };

        if (systemInstruction) config.systemInstruction = systemInstruction;
        if (options?.maxTokens !== undefined) config.maxOutputTokens = options.maxTokens;
        if (options?.temperature !== undefined) config.temperature = options.temperature;
        if (options?.topP !== undefined) config.topP = options.topP;
        if (options?.stopSequences) config.stopSequences = options.stopSequences;
        if (options?.frequencyPenalty !== undefined) {
            config.frequencyPenalty = options.frequencyPenalty;
        }
        if (options?.presencePenalty !== undefined) {
            config.presencePenalty = options.presencePenalty;
        }
        // Gemini exposes a single `abortSignal` slot, so a caller-supplied
        // signal and a timeout have to be folded into one. Built per request:
        // hoisting it to the constructor would make the first timeout abort
        // every later call on the same instance.
        //
        // The timeout is applied here rather than through the client's
        // `httpOptions.timeout`, which the SDK adds to *every* request and which
        // a longer per-call `timeout` therefore could not override.
        const signals: AbortSignal[] = [];
        if (options?.signal) signals.push(options.signal);
        const timeout = options?.timeout ?? this.timeout;
        if (timeout !== undefined) signals.push(AbortSignal.timeout(timeout));
        if (signals.length === 1) {
            config.abortSignal = signals[0];
        } else if (signals.length > 1) {
            config.abortSignal = AbortSignal.any(signals);
        }

        return config;
    }

    /**
     * Run one SDK call, retrying transient failures.
     *
     * The Gemini SDK reads its retry policy from client-level options only, so a
     * per-call `retries` cannot be expressed through it — and its own retry
     * wrapper replaces API errors with generic ones, losing the status code, and
     * keeps retrying after an abort. The loop therefore lives here, which gives
     * `retries` the same meaning it has on the OpenAI and Anthropic providers.
     */
    private async withRetries<T>(
        options: LLMCallOptions | undefined,
        attempt: () => Promise<T>
    ): Promise<T> {
        const retries = options?.retries ?? this.maxRetries;
        const signal = options?.signal;

        for (let n = 0; ; n++) {
            // Unlike the OpenAI and Anthropic SDKs, Gemini's client attaches the
            // caller's signal with an `abort` listener, which never fires for a
            // signal that aborted before the call. Check it ourselves so an
            // already-cancelled request costs nothing.
            signal?.throwIfAborted();
            try {
                return await attempt();
            } catch (error) {
                if (n >= retries || !isRetryable(error)) throw error;
                await delay(RETRY_BASE_DELAY_MS * 2 ** n, signal);
            }
        }
    }

    private recordUsageFrom(
        response: GenerateContentResponse | undefined,
        startedAt: number
    ): void {
        const usage = response?.usageMetadata;
        this.recordUsage({
            promptTokens: usage?.promptTokenCount ?? 0,
            completionTokens: usage?.candidatesTokenCount ?? 0,
            latencyMs: Date.now() - startedAt,
        });
    }
}

function usageFrom(response: GenerateContentResponse | undefined) {
    const usage = response?.usageMetadata;
    return {
        promptTokens: usage?.promptTokenCount ?? 0,
        completionTokens: usage?.candidatesTokenCount ?? 0,
        totalTokens: usage?.totalTokenCount ?? 0,
    };
}

/**
 * Convert ts-dspy messages to Gemini contents.
 *
 * System messages become a separate `systemInstruction`, since Gemini has no
 * system role in `contents`. This does not mutate the caller's array — the
 * previous implementation called `messages.pop()`, destroying the last turn of
 * any array a caller reused.
 */
export function toGeminiContents(messages: ChatMessage[]): {
    contents: Content[];
    systemInstruction?: string;
} {
    const systemParts: string[] = [];
    const contents: Content[] = [];

    for (const message of messages) {
        if (message.role === 'system') {
            systemParts.push(message.content);
            continue;
        }
        contents.push({
            role: message.role === 'assistant' ? 'model' : 'user',
            parts: [{ text: message.content }],
        });
    }

    return {
        contents,
        systemInstruction: systemParts.length > 0 ? systemParts.join('\n\n') : undefined,
    };
}

function statusOf(error: unknown): number | undefined {
    if (typeof error !== 'object' || error === null || !('status' in error)) return undefined;
    const status = Number((error as { status: unknown }).status);
    return Number.isFinite(status) ? status : undefined;
}

/**
 * Aborts and client errors are final; transient faults and transport failures
 * are not. An error carrying no status is only retried when it looks like a
 * transport failure — Node's `fetch` reports those as a `TypeError` — so a
 * deterministic bug is not turned into four slow deterministic bugs.
 */
function isRetryable(error: unknown): boolean {
    if (
        error instanceof Error &&
        (error.name === 'AbortError' || error.name === 'TimeoutError')
    ) {
        return false;
    }
    const status = statusOf(error);
    if (status !== undefined) return RETRYABLE_STATUS.has(status);
    return error instanceof TypeError;
}

/** Sleep, unless the caller cancels first. */
function delay(ms: number, signal: AbortSignal | undefined): Promise<void> {
    // An `abort` listener never fires for a signal that aborted earlier, so the
    // sleep has to be short-circuited here rather than waited out.
    if (signal?.aborted) return Promise.reject(signal.reason);

    return new Promise((resolve, reject) => {
        const onAbort = () => {
            clearTimeout(timer);
            reject(signal?.reason);
        };
        const timer = setTimeout(() => {
            signal?.removeEventListener('abort', onAbort);
            resolve();
        }, ms);
        signal?.addEventListener('abort', onAbort, { once: true });
    });
}

function toLMError(error: unknown): LMError {
    if (error instanceof LMError) return error;
    const message = error instanceof Error ? error.message : String(error);
    return new LMError('gemini', message, { cause: error, status: statusOf(error) });
}
