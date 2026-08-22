import {
    BaseLM,
    ContentFilterError,
    LMError,
    classify,
    type ChatMessage,
    type ChatResult,
    type FinishReason,
    type LLMCallOptions,
    type ModelCapabilities,
    type StreamChunk,
    type ToolCall,
} from '@ts-dspy/core';
import {
    FinishReason as GenAIFinishReason,
    GoogleGenAI,
    HarmBlockThreshold,
    HarmCategory,
    type Content,
    type GenerateContentConfig,
    type GenerateContentResponse,
    type Part,
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

/**
 * Finish reasons that mean the candidate was withheld, not merely stopped.
 *
 * Every one of these leaves the candidate without usable content, so checking
 * only `SAFETY` would still hand the caller an empty string for the rest.
 */
const BLOCKING_FINISH_REASONS: ReadonlySet<GenAIFinishReason> = new Set([
    GenAIFinishReason.SAFETY,
    GenAIFinishReason.PROHIBITED_CONTENT,
    GenAIFinishReason.BLOCKLIST,
    GenAIFinishReason.SPII,
    GenAIFinishReason.RECITATION,
    GenAIFinishReason.IMAGE_SAFETY,
    GenAIFinishReason.IMAGE_PROHIBITED_CONTENT,
]);

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
        return (await this.chatWithTools(messages, options)).content;
    }

    async chatWithTools(
        messages: ChatMessage[],
        options?: LLMCallOptions
    ): Promise<ChatResult> {
        const { contents, systemInstruction } = toGeminiContents(messages);
        const response = await this.send(
            contents,
            systemInstruction,
            options,
            toolConfig(options)
        );

        const toolCalls = toolCallsOf(response);
        return {
            content: response.text ?? '',
            ...(toolCalls.length > 0 ? { toolCalls } : {}),
            finishReason: finishReasonOf(response, toolCalls.length > 0),
        };
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

        // A truncated reply is never valid JSON, and until this check existed it
        // surfaced as a misleading "not valid JSON" error. OpenAI and Anthropic
        // have always reported truncation; Gemini was the odd one out.
        if (response.candidates?.[0]?.finishReason === GenAIFinishReason.MAX_TOKENS) {
            this.recordError();
            throw new LMError('gemini', 'Structured response was truncated; raise maxTokens.');
        }

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
                    config: {
                        ...this.buildConfig(systemInstruction, options),
                        ...toolConfig(options),
                    },
                })
            );

            // The loop is inside the try so a mid-stream abort surfaces as an
            // LMError and is counted, rather than escaping as a raw SDK error.
            for await (const chunk of stream) {
                last = chunk;
                // A stream can be cut short by the classifiers just as a one-shot
                // reply can. Without this the stream simply ended early and
                // looked like a short answer.
                this.assertNotFiltered(chunk);
                const text = chunk.text;
                if (text) {
                    yield { content: text, done: false };
                }
            }
        } catch (error) {
            // assertNotFiltered counts the errors it raises, so only count what
            // arrives here uncounted from the SDK.
            if (!(error instanceof LMError)) this.recordError();
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

        this.assertNotFiltered(response);
        this.recordUsageFrom(response, startedAt);
        return response;
    }

    /**
     * Raise on a filtered reply.
     *
     * Gemini reports filtering on a normal 200 response, in two different
     * places: `promptFeedback.blockReason` when the prompt was rejected, and
     * `candidates[].finishReason` when the reply itself was. Only the first was
     * ever checked, so a dropped candidate simply came back as empty text.
     */
    private assertNotFiltered(response: GenerateContentResponse): void {
        // Check the block reason before touching `text`. The previous
        // implementation read `response.text()` first, which threw on blocked
        // responses and made this branch unreachable.
        const blockReason = response.promptFeedback?.blockReason;
        if (blockReason) {
            this.recordError();
            throw new ContentFilterError(
                'gemini',
                `Request blocked by safety filters: ${blockReason}`,
                { category: String(blockReason) }
            );
        }

        const finishReason = response.candidates?.[0]?.finishReason;
        if (finishReason !== undefined && BLOCKING_FINISH_REASONS.has(finishReason)) {
            this.recordError();
            throw new ContentFilterError(
                'gemini',
                `Response blocked by safety filters: ${finishReason}`,
                { category: String(finishReason) }
            );
        }
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
 *
 * Consecutive `functionResponse` turns are folded into one user content: Gemini
 * requires the replies to a parallel call turn to arrive together, matching the
 * `functionCall` parts one for one, and rejects them spread across separate
 * turns.
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

        const parts = toGeminiParts(message);
        const previous = contents.at(-1);

        if (previous?.role === 'user' && isFunctionResponses(previous.parts)) {
            if (isFunctionResponses(parts)) {
                previous.parts = [...(previous.parts ?? []), ...parts];
                continue;
            }
        }

        contents.push({
            role: message.role === 'assistant' ? 'model' : 'user',
            parts,
        });
    }

    return {
        contents,
        systemInstruction: systemParts.length > 0 ? systemParts.join('\n\n') : undefined,
    };
}

/**
 * Build the parts of one turn.
 *
 * Tool traffic is structural in Gemini too: an assistant turn's tool calls
 * become `functionCall` parts, and a `tool` result turn becomes a
 * `functionResponse` part. Both are keyed by function *name* — Gemini has no
 * tool-call identifier, so results are correlated by name and position.
 */
function toGeminiParts(message: ChatMessage): Part[] {
    if (message.role === 'assistant' && message.toolCalls?.length) {
        const parts: Part[] = [];
        if (message.content) parts.push({ text: message.content });
        for (const call of message.toolCalls) {
            parts.push({
                functionCall: { name: call.name, args: call.arguments ?? {} },
            });
        }
        return parts;
    }

    if (message.role === 'tool' || message.role === 'function') {
        // Without a name there is nothing to correlate against, so the result
        // degrades to ordinary text rather than an unaddressed response part.
        if (!message.name) return [{ text: message.content }];
        return [
            {
                functionResponse: {
                    name: message.name,
                    // `response` must be an object, not a bare string.
                    response: { result: message.content },
                },
            },
        ];
    }

    return [{ text: message.content }];
}

function isFunctionResponses(parts: Part[] | undefined): boolean {
    return Boolean(parts?.length) && parts!.every((part) => Boolean(part.functionResponse));
}

/** Extract the `functionCall` parts of a response. Gemini sends `args` already parsed. */
function toolCallsOf(response: GenerateContentResponse): ToolCall[] {
    const parts = response.candidates?.[0]?.content?.parts ?? [];
    const calls = parts
        .map((part) => part.functionCall)
        .filter((call): call is NonNullable<typeof call> => Boolean(call?.name));

    return calls.map((call) => ({
        // Gemini omits `id` on the Gemini API and populates it on some Vertex
        // configurations; `ToolCall.id` is optional precisely for this.
        ...(call.id ? { id: call.id } : {}),
        name: call.name!,
        arguments: (call.args ?? {}) as Record<string, unknown>,
    }));
}

/** Translate tool declarations into the `generateContent` config shape. */
function toolConfig(options?: LLMCallOptions): Partial<GenerateContentConfig> {
    if (!options?.tools?.length) return {};

    const config: Partial<GenerateContentConfig> = {
        tools: [
            {
                functionDeclarations: options.tools.map((tool) => ({
                    name: tool.name,
                    ...(tool.description ? { description: tool.description } : {}),
                    parametersJsonSchema: tool.parameters,
                })),
            },
        ],
    };

    const choice = options.toolChoice;
    if (choice !== undefined) {
        config.toolConfig =
            typeof choice === 'object'
                ? {
                      functionCallingConfig: {
                          mode: 'ANY' as never,
                          allowedFunctionNames: [choice.name],
                      },
                  }
                : {
                      functionCallingConfig: {
                          mode: (choice === 'required' ? 'ANY' : choice.toUpperCase()) as never,
                      },
                  };
    }

    return config;
}

function finishReasonOf(
    response: GenerateContentResponse,
    hasToolCalls: boolean
): FinishReason {
    if (hasToolCalls) return 'tool_calls';
    switch (response.candidates?.[0]?.finishReason) {
        case 'STOP':
            return 'stop';
        case 'MAX_TOKENS':
            return 'length';
        case 'SAFETY':
        case 'PROHIBITED_CONTENT':
            return 'content_filter';
        case undefined:
            return 'stop';
        default:
            return 'other';
    }
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

/**
 * Wrap a Gemini SDK failure.
 *
 * `ApiError` carries a numeric `status` and nothing else — no code, no typed
 * error union — so classification leans on the status, with message matching as
 * the only route to a context-length verdict.
 */
function toLMError(error: unknown): LMError {
    if (error instanceof LMError) return error;
    const message = error instanceof Error ? error.message : String(error);
    const status = statusOf(error);

    const ErrorClass = classify(status, { message });
    return new ErrorClass('gemini', message, { cause: error, status });
}
