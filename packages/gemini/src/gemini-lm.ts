import {
    BaseLM,
    ContentFilterError,
    LMError,
    TimeoutError,
    classify,
    contentToText,
    mergeRequestBody,
    providerOptionsFor,
    retryWithBackoff,
    imageMediaType,
    normalizeImageSource,
    type ChatMessage,
    type ChatResult,
    type FinishReason,
    type ImageContentPart,
    type MessageContent,
    type LLMCallOptions,
    type ModelCapabilities,
    type StreamChunk,
    type ToolCall,
    type UsageStats,
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
    type ThinkingConfig,
    type ThinkingLevel,
} from '@google/genai';

/** Current default. Gemini 2.x models have reached end of life. */
export const DEFAULT_GEMINI_MODEL = 'gemini-3.5-flash';

/**
 * Thinking controls, sent as `thinkingConfig`. Which of them a model honours
 * is model-dependent: Gemini 3 models steer with `level`, earlier thinking
 * models with `budgetTokens`.
 */
export interface GeminiThinking {
    /** `thinkingBudget`: tokens to spend thinking. `0` disables, `-1` lets the model decide. */
    budgetTokens?: number;
    /**
     * `includeThoughts`: return thought summaries. They arrive as `reasoning`
     * on `chatWithTools` and as `reasoningDelta` metadata on streams, never in
     * the answer text.
     */
    includeThoughts?: boolean;
    /** `thinkingLevel`, e.g. `'LOW'` or `'HIGH'`. */
    level?: ThinkingLevel | `${ThinkingLevel}`;
}

/**
 * Per-call options read from `providerOptions.gemini`.
 *
 * `thinking` means what it does on {@link GeminiConfig} and replaces the
 * constructor default; every other key is merged into the request's
 * `GenerateContentConfig` as-is — e.g. `{ gemini: { seed: 7, mediaResolution: … } }`.
 */
export interface GeminiProviderOptions {
    thinking?: GeminiThinking;
    [key: string]: unknown;
}

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
    /** Default thinking controls; overridden per call by `providerOptions.gemini.thinking`. */
    thinking?: GeminiThinking;
}

const DEFAULT_SAFETY_SETTINGS: SafetySetting[] = [
    HarmCategory.HARM_CATEGORY_HARASSMENT,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
].map((category) => ({ category, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE }));

/** Embedding models take text only; every generative Gemini model reads images. */
function supportsVisionFor(model: string): boolean {
    return !model.includes('embedding');
}

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

/** Every current Gemini model — Flash, Flash-Lite and Pro — has a 1M-token window. */
const GEMINI_CONTEXT_LENGTH = 1_000_000;

/** Statuses worth another attempt: rate limits, timeouts and transient faults. */
const RETRYABLE_STATUS = new Set([408, 409, 429, 500, 502, 503, 504]);

/** First backoff step; doubles per attempt. */
const RETRY_BASE_DELAY_MS = 250;

/** Ceiling on a single backoff step. */
const RETRY_MAX_DELAY_MS = 8_000;

export class GeminiLM extends BaseLM {
    private readonly client: GoogleGenAI;
    private readonly safetySettings: SafetySetting[];
    private readonly timeout?: number;
    private readonly maxRetries: number;
    private readonly thinking?: GeminiThinking;
    /** Constructor settings that change what a given prompt returns. */
    private readonly scope: Record<string, unknown>;

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
        this.thinking = config.thinking;
        this.scope = {
            safetySettings: this.safetySettings,
            vertexai: config.vertexai ?? null,
            project: config.project ?? null,
            location: config.location ?? null,
            baseUrl: config.baseUrl ?? null,
            ...(config.thinking ? { thinking: config.thinking } : {}),
        };
    }

    /**
     * Safety settings decide whether a reply comes back at all, and the Vertex
     * and base-URL settings decide which endpoint answered, so none of them can
     * be left out of the cache key.
     */
    protected cacheScope(): unknown {
        return this.scope;
    }

    async chat(messages: ChatMessage[], options?: LLMCallOptions): Promise<string> {
        return this.textOf(await this.chatWithTools(messages, options));
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
        const reasoning = thoughtsOf(response);
        return {
            content: textOf(response),
            ...(toolCalls.length > 0 ? { toolCalls } : {}),
            finishReason: finishReasonOf(response, toolCalls.length > 0),
            ...(reasoning ? { reasoning } : {}),
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
        const toolCalls: ToolCall[] = [];
        try {
            const stream = await this.withRetries(options, () =>
                this.client.models.generateContentStream({
                    model: options?.model ?? this.model,
                    contents,
                    config: this.buildConfig(systemInstruction, options, toolConfig(options)),
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
                toolCalls.push(...toolCallsOf(chunk));
                const thoughts = thoughtsOf(chunk);
                if (thoughts) {
                    // Thought summaries are not answer text, so they travel in metadata.
                    yield { content: '', done: false, metadata: { reasoningDelta: thoughts } };
                }
                const text = textOf(chunk);
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

        this.recordUsageFrom(last, startedAt, options);
        yield {
            content: '',
            done: true,
            usage: streamUsage(last),
            metadata: {
                ...(last ? { finishReason: finishReasonOf(last, toolCalls.length > 0) } : {}),
                ...(toolCalls.length > 0 ? { toolCalls } : {}),
            },
        };
    }

    getCapabilities(): ModelCapabilities {
        return {
            supportsStreaming: true,
            supportsStructuredOutput: true,
            supportsFunctionCalling: true,
            supportsVision: supportsVisionFor(this.model),
            maxContextLength: GEMINI_CONTEXT_LENGTH,
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
                    config: this.buildConfig(systemInstruction, options, extraConfig),
                })
            );
        } catch (error) {
            this.recordError();
            throw toLMError(error);
        }

        this.assertNotFiltered(response);
        this.recordUsageFrom(response, startedAt, options);
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

    /**
     * Assemble the request config: safety settings, sampling, thinking,
     * cancellation, the operation's own fields, then the raw
     * `providerOptions.gemini` keys on top.
     */
    private buildConfig(
        systemInstruction: string | undefined,
        options?: LLMCallOptions,
        extraConfig: Partial<GenerateContentConfig> = {}
    ): GenerateContentConfig {
        const { thinking: callThinking, ...raw } = providerOptionsFor(
            options,
            'gemini'
        ) as GeminiProviderOptions;

        const config: GenerateContentConfig = {
            safetySettings: this.safetySettings,
        };

        const thinkingConfig = toThinkingConfig(callThinking ?? this.thinking);
        if (thinkingConfig) config.thinkingConfig = thinkingConfig;

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

        // The abort signal is not a plain object, so the merge keeps it intact.
        return mergeRequestBody(
            { ...config, ...extraConfig } as Record<string, unknown>,
            raw
        ) as GenerateContentConfig;
    }

    /**
     * Run one SDK call, retrying transient failures.
     *
     * The Gemini SDK reads its retry policy from client-level options only, so a
     * per-call `retries` cannot be expressed through it — and its own retry
     * wrapper replaces API errors with generic ones, losing the status code, and
     * keeps retrying after an abort. Retrying is therefore done here, with
     * core's `retryWithBackoff` (exponential backoff with jitter), which gives
     * `retries` the same meaning it has on the OpenAI and Anthropic providers.
     * For a policy of your own, set `maxRetries: 0` and wrap the model with
     * `wrapLM(gemini, retry({ ... }))`.
     */
    private async withRetries<T>(
        options: LLMCallOptions | undefined,
        attempt: () => Promise<T>
    ): Promise<T> {
        // Unlike the OpenAI and Anthropic SDKs, Gemini's client attaches the
        // caller's signal with an `abort` listener, which never fires for a
        // signal that aborted before the call. The shared helper checks the
        // signal before every attempt, so an already-cancelled request costs
        // nothing, and cuts a pending backoff short when it aborts.
        return retryWithBackoff(attempt, {
            maxRetries: options?.retries ?? this.maxRetries,
            initialDelayMs: RETRY_BASE_DELAY_MS,
            maxDelayMs: RETRY_MAX_DELAY_MS,
            signal: options?.signal,
            // Raw SDK errors, not yet classified, so the policy is Gemini's own.
            shouldRetry: isRetryable,
        });
    }

    private recordUsageFrom(
        response: GenerateContentResponse | undefined,
        startedAt: number,
        options?: LLMCallOptions
    ): void {
        this.recordUsage({
            ...usageOf(response),
            latencyMs: Date.now() - startedAt,
            model: options?.model,
        });
    }
}

/** Translate the ts-dspy thinking option into `thinkingConfig`. */
function toThinkingConfig(thinking: GeminiThinking | undefined): ThinkingConfig | undefined {
    if (!thinking) return undefined;
    const config: ThinkingConfig = {};
    if (thinking.budgetTokens !== undefined) config.thinkingBudget = thinking.budgetTokens;
    if (thinking.includeThoughts !== undefined)
        config.includeThoughts = thinking.includeThoughts;
    if (thinking.level !== undefined) config.thinkingLevel = thinking.level as ThinkingLevel;
    return Object.keys(config).length > 0 ? config : undefined;
}

interface GeminiUsage {
    promptTokens: number;
    completionTokens: number;
    cachedTokens?: number;
    reasoningTokens?: number;
}

/**
 * Token usage. Gemini reports thought tokens apart from `candidatesTokenCount`;
 * they are billed as output, so they are added to the completion total —
 * `reasoningTokens` is then a subset of `completionTokens`, as elsewhere.
 * `cachedContentTokenCount` is already included in `promptTokenCount`.
 */
function usageOf(response: GenerateContentResponse | undefined): GeminiUsage {
    const usage = response?.usageMetadata;
    const cached = usage?.cachedContentTokenCount;
    const thoughts = usage?.thoughtsTokenCount;
    return {
        promptTokens: usage?.promptTokenCount ?? 0,
        completionTokens: (usage?.candidatesTokenCount ?? 0) + (thoughts ?? 0),
        ...(cached !== undefined ? { cachedTokens: cached } : {}),
        ...(thoughts !== undefined ? { reasoningTokens: thoughts } : {}),
    };
}

function streamUsage(response: GenerateContentResponse | undefined): Partial<UsageStats> {
    const usage = usageOf(response);
    return {
        ...usage,
        totalTokens:
            response?.usageMetadata?.totalTokenCount ??
            usage.promptTokens + usage.completionTokens,
    };
}

/** Text of the thought-summary parts (`thought: true`), returned with `includeThoughts`. */
function thoughtsOf(response: GenerateContentResponse): string {
    const parts = response.candidates?.[0]?.content?.parts ?? [];
    return parts
        .filter((part) => part.thought === true && typeof part.text === 'string')
        .map((part) => part.text)
        .join('');
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
            // `systemInstruction` is text-only, so an image aimed at it is
            // flattened to its placeholder rather than silently dropped.
            systemParts.push(contentToText(message.content));
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
        const text = contentToText(message.content);
        if (text) parts.push({ text });
        for (const call of message.toolCalls) {
            // Gemini 3 rejects a replayed function call that has lost the
            // thought signature it was issued with.
            const signature = call.providerMetadata?.thoughtSignature;
            parts.push({
                functionCall: {
                    ...(call.id ? { id: call.id } : {}),
                    name: call.name,
                    args: call.arguments ?? {},
                },
                ...(typeof signature === 'string' ? { thoughtSignature: signature } : {}),
            });
        }
        return parts;
    }

    if (message.role === 'tool' || message.role === 'function') {
        // Without a name there is nothing to correlate against, so the result
        // degrades to ordinary text rather than an unaddressed response part.
        const text = contentToText(message.content);
        if (!message.name) return [{ text }];
        return [
            {
                functionResponse: {
                    name: message.name,
                    // `response` must be an object, not a bare string.
                    response: { result: text },
                },
            },
        ];
    }

    return toGeminiContentParts(message.content);
}

/** A plain string is one text part; parts map one-to-one. */
function toGeminiContentParts(content: MessageContent): Part[] {
    if (typeof content === 'string') return [{ text: content }];
    return content.map((part) =>
        part.type === 'text' ? { text: part.text } : toGeminiImage(part)
    );
}

/** URI schemes `fileData` can actually dereference. */
const FILE_URI_PATTERN = /^(gs:\/\/|https:\/\/generativelanguage\.googleapis\.com\/)/i;

/**
 * Inline bytes go in `inlineData`; a URI the API can resolve goes in `fileData`.
 * A `data:` URI arriving as a URL is rewritten to inline bytes, since `fileData`
 * cannot dereference one either.
 *
 * Unlike OpenAI and Anthropic, Gemini will not fetch an arbitrary web URL, so
 * one is refused here with an explanation rather than sent on to earn a 400.
 */
function toGeminiImage(part: ImageContentPart): Part {
    const source = normalizeImageSource(part.source);
    if (source.kind === 'url') {
        if (!FILE_URI_PATTERN.test(source.url)) {
            throw new LMError(
                'gemini',
                `Gemini cannot fetch "${source.url}": fileData accepts a Files API or ` +
                    'Cloud Storage URI. Pass the image as base64 bytes instead.'
            );
        }
        const mimeType = imageMediaType(source);
        return { fileData: { fileUri: source.url, ...(mimeType ? { mimeType } : {}) } };
    }
    return { inlineData: { mimeType: source.mediaType, data: source.data } };
}

function isFunctionResponses(parts: Part[] | undefined): boolean {
    return Boolean(parts?.length) && parts!.every((part) => Boolean(part.functionResponse));
}

/** Extract the `functionCall` parts of a response. Gemini sends `args` already parsed. */
function toolCallsOf(response: GenerateContentResponse): ToolCall[] {
    const parts = response.candidates?.[0]?.content?.parts ?? [];
    return parts.flatMap((part) => {
        const call = part.functionCall;
        if (!call?.name) return [];
        return [
            {
                // Gemini omits `id` on the Gemini API and populates it on some Vertex
                // configurations; `ToolCall.id` is optional precisely for this.
                ...(call.id ? { id: call.id } : {}),
                name: call.name,
                arguments: (call.args ?? {}) as Record<string, unknown>,
                ...(part.thoughtSignature
                    ? { providerMetadata: { thoughtSignature: part.thoughtSignature } }
                    : {}),
            },
        ];
    });
}

/**
 * The answer text of a response or stream chunk. The SDK's own `.text` getter
 * returns the same thing but logs a `console.warn` whenever the response also
 * holds non-text parts — that is, on every tool-calling turn.
 */
function textOf(response: GenerateContentResponse): string {
    const parts = response.candidates?.[0]?.content?.parts;
    // No parts to read: the getter has nothing to warn about either.
    if (!parts?.length) return response.text ?? '';
    return parts
        .filter((part) => typeof part.text === 'string' && !part.thought)
        .map((part) => part.text)
        .join('');
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
export function isRetryable(error: unknown): boolean {
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

/**
 * Wrap a Gemini SDK failure.
 *
 * `ApiError` carries a numeric `status` and nothing else — no code, no typed
 * error union — so classification leans on the status, with message matching as
 * the only route to a context-length verdict.
 */
export function toLMError(error: unknown): LMError {
    if (error instanceof LMError) return error;
    const message = error instanceof Error ? error.message : String(error);
    const status = statusOf(error);

    // An elapsed per-request timeout and a cancelled signal both surface as an
    // abort with no status; left as a plain LMError they are indistinguishable
    // from a bug and never retried.
    if (
        status === undefined &&
        error instanceof Error &&
        (error.name === 'AbortError' || error.name === 'TimeoutError')
    ) {
        return new TimeoutError('gemini', message, { cause: error });
    }

    const ErrorClass = classify(status, { message });
    return new ErrorClass('gemini', message, { cause: error, status });
}
