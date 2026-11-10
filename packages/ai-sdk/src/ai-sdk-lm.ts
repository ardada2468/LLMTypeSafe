import {
    BaseLM,
    ContentFilterError,
    LMError,
    RateLimitError,
    TimeoutError,
    classify,
    contentToText,
    type ChatMessage,
    type ChatResult,
    type FinishReason,
    type ImageContentPart,
    type LLMCallOptions,
    type ModelCapabilities,
    type StreamChunk,
    type ToolCall,
    type ToolChoice,
    type ToolSpec,
} from '@ts-dspy/core';
import {
    APICallError,
    NoObjectGeneratedError,
    Output,
    RetryError,
    generateText,
    jsonSchema,
    streamText,
    tool,
    type FinishReason as AIFinishReason,
    type FilePart,
    type LanguageModel,
    type LanguageModelUsage,
    type ModelMessage,
    type TextPart,
    type ToolCallPart,
    type ToolChoice as AIToolChoice,
    type ToolSet,
} from 'ai';

/**
 * Provider-specific options forwarded verbatim to the AI SDK, keyed by
 * provider name — e.g. `{ openai: { reasoningEffort: 'high' } }`.
 */
export type AISDKProviderOptions = Record<string, Record<string, any>>;

export interface AISDKConfig {
    /**
     * Any AI SDK language model — `openai('gpt-5.2')`, `anthropic('claude-opus-5')`,
     * a gateway model id string, or one wrapped in AI SDK middleware.
     */
    model: LanguageModel;
    /**
     * Provider name used for usage keys (`<provider>/<model>`), pricing lookups
     * and traces. Defaults to the model's own `provider` with any `.chat` /
     * `.responses` suffix removed (`openai.responses` → `openai`), or `gateway`
     * for a model id string.
     */
    provider?: string;
    /**
     * Override what the model is reported to support. The AI SDK does not say
     * what a model can do, so the defaults assume a current frontier chat
     * model: streaming, structured output, tool calling and vision, with a
     * 128k-token context. Turn off what yours lacks — `supportsFunctionCalling:
     * false` sends `ReAct` down its text path, and `supportsStructuredOutput:
     * false` has `Predict` parse labelled text instead of asking for JSON.
     */
    capabilities?: Partial<ModelCapabilities>;
    /** Default retry count, used when a call does not set `retries`. The AI SDK defaults to 2. */
    maxRetries?: number;
    /** Default `maxTokens`, used when a call does not set one. */
    maxTokens?: number;
    /** Sent with every request. */
    headers?: Record<string, string>;
    /** Provider-specific options sent with every request. */
    providerOptions?: AISDKProviderOptions;
}

/** What a current frontier chat model supports; see {@link AISDKConfig.capabilities}. */
export const DEFAULT_AI_SDK_CAPABILITIES: ModelCapabilities = {
    supportsStreaming: true,
    supportsStructuredOutput: true,
    supportsFunctionCalling: true,
    supportsVision: true,
    maxContextLength: 128_000,
    supportedFormats: ['text', 'json_schema'],
};

/**
 * Runs ts-dspy programs on any [Vercel AI SDK](https://ai-sdk.dev) language
 * model, so the dozens of providers the AI SDK supports — and its middleware,
 * gateway and provider registry — are available without a dedicated ts-dspy
 * package for each.
 *
 * ```ts
 * import { openai } from '@ai-sdk/openai';
 * import { AISDKLM } from '@ts-dspy/ai-sdk';
 *
 * configure({ lm: new AISDKLM({ model: openai('gpt-5.2') }) });
 * ```
 *
 * Calls go through `generateText` / `streamText`, so the AI SDK's own retries,
 * prompt conversion and provider quirks all apply. Tools are declared to the
 * model without an `execute` function: the AI SDK returns the calls rather
 * than running them, and ts-dspy's `ReAct` loop stays in charge of executing
 * tools and reporting them to callbacks. Structured output uses
 * `Output.object` with the signature's JSON Schema passed through
 * `jsonSchema()`, so providers with a native JSON mode constrain decoding.
 */
export class AISDKLM extends BaseLM {
    private readonly languageModel: LanguageModel;
    private readonly capabilities: ModelCapabilities;
    private readonly maxRetries?: number;
    private readonly maxTokens?: number;
    private readonly headers?: Record<string, string>;
    private readonly providerOptions?: AISDKProviderOptions;

    constructor(config: AISDKConfig) {
        super(config.provider ?? providerNameOf(config.model), modelIdOf(config.model));
        this.languageModel = config.model;
        this.capabilities = { ...DEFAULT_AI_SDK_CAPABILITIES, ...config.capabilities };
        this.maxRetries = config.maxRetries;
        this.maxTokens = config.maxTokens;
        this.headers = config.headers;
        this.providerOptions = config.providerOptions;
    }

    /**
     * Everything configured here shapes the reply, so two instances differing
     * in any of it must not share cache entries.
     */
    protected cacheScope(): unknown {
        return {
            maxTokens: this.maxTokens ?? null,
            providerOptions: this.providerOptions ?? null,
            capabilities: this.capabilities,
        };
    }

    /**
     * A per-call `model` is ignored for a model object (see `modelFor`), so it
     * must not name the usage key either: `BaseLM` reports cache hits under
     * the call's `model`, which would otherwise attribute them to a model
     * that never ran.
     */
    protected usageKey(model?: string): string {
        return super.usageKey(typeof this.languageModel === 'string' ? model : undefined);
    }

    async chat(messages: ChatMessage[], options?: LLMCallOptions): Promise<string> {
        return this.textOf(await this.chatWithTools(messages, options));
    }

    async chatWithTools(
        messages: ChatMessage[],
        options?: LLMCallOptions
    ): Promise<ChatResult> {
        const startedAt = Date.now();
        let result: Awaited<ReturnType<typeof generateText>>;
        try {
            result = await generateText({
                ...this.requestSettings(options),
                ...toPrompt(messages),
                ...toolSettings(options),
            });
        } catch (error) {
            this.recordError();
            throw toLMError(error, this.provider);
        }

        this.recordCompletion(result.totalUsage, startedAt, options);
        this.assertNotFiltered(result.finishReason);

        const toolCalls = result.toolCalls.map(fromAIToolCall);
        return {
            content: result.text,
            ...(toolCalls.length > 0 ? { toolCalls } : {}),
            finishReason: fromAIFinishReason(result.finishReason),
        };
    }

    async generateStructured<T>(
        prompt: string,
        schema: unknown,
        options?: LLMCallOptions
    ): Promise<T> {
        const startedAt = Date.now();
        let result: Awaited<ReturnType<typeof generateText>>;
        try {
            result = await generateText({
                ...this.requestSettings(options),
                prompt,
                output: Output.object({
                    schema: jsonSchema<T>(schema as Parameters<typeof jsonSchema>[0]),
                }),
            });
        } catch (error) {
            this.recordError();
            // A truncated reply is the one schema failure with a clear
            // remedy, so it gets a message saying what it is.
            if (NoObjectGeneratedError.isInstance(error) && error.finishReason === 'length') {
                throw this.truncated(error);
            }
            throw toLMError(error, this.provider);
        }

        this.recordCompletion(result.totalUsage, startedAt, options);
        this.assertNotFiltered(result.finishReason);

        // `output` is a getter that throws `NoOutputGeneratedError` when the
        // AI SDK never parsed one — a reply cut off before any text arrived,
        // or one that ended for a reason other than `stop` with no text. That
        // happens after `generateText` resolved, so it must be caught here
        // too, or a raw AI SDK error escapes core's LMError taxonomy.
        try {
            return result.output as T;
        } catch (error) {
            this.recordError();
            if (result.finishReason === 'length') throw this.truncated(error);
            throw toLMError(error, this.provider);
        }
    }

    private truncated(cause: unknown): LMError {
        return new LMError(
            this.provider,
            'Structured response was truncated; raise maxTokens.',
            {
                cause,
            }
        );
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
        const startedAt = Date.now();
        // Errors arrive as `error` parts on the stream rather than as a thrown
        // exception. Without an `onError` the AI SDK also logs each one to the
        // console, which a library must not do.
        const result = streamText({
            ...this.requestSettings(options),
            ...toPrompt(messages),
            ...toolSettings(options),
            onError: () => {},
        });

        const toolCalls: ToolCall[] = [];
        let finishReason: AIFinishReason = 'other';
        let usage: LanguageModelUsage | undefined;
        try {
            for await (const part of result.fullStream) {
                switch (part.type) {
                    case 'text-delta':
                        if (part.text) yield { content: part.text, done: false };
                        break;
                    case 'tool-input-delta':
                        // Arguments stream as JSON fragments; like the other
                        // providers, they travel in metadata, not `content`.
                        yield {
                            content: '',
                            done: false,
                            metadata: {
                                toolInputDelta: { id: part.id, partialJson: part.delta },
                            },
                        };
                        break;
                    case 'tool-call':
                        toolCalls.push(fromAIToolCall(part));
                        break;
                    case 'finish':
                        finishReason = part.finishReason;
                        usage = part.totalUsage;
                        break;
                    case 'error':
                        throw part.error;
                    case 'abort':
                        throw new TimeoutError(
                            this.provider,
                            `Stream aborted${part.reason ? `: ${part.reason}` : ''}`
                        );
                    default:
                        break;
                }
            }
        } catch (error) {
            this.recordError();
            throw toLMError(error, this.provider);
        }

        this.recordCompletion(usage, startedAt, options);
        this.assertNotFiltered(finishReason);

        const metadata: Record<string, unknown> = {
            finishReason: fromAIFinishReason(finishReason),
        };
        if (toolCalls.length > 0) metadata.toolCalls = toolCalls;
        yield {
            content: '',
            done: true,
            metadata,
            usage: {
                promptTokens: usage?.inputTokens ?? 0,
                completionTokens: usage?.outputTokens ?? 0,
                totalTokens: (usage?.inputTokens ?? 0) + (usage?.outputTokens ?? 0),
            },
        };
    }

    getCapabilities(): ModelCapabilities {
        return { ...this.capabilities };
    }

    /** Call settings shared by every method. */
    private requestSettings(options: LLMCallOptions | undefined) {
        return {
            model: this.modelFor(options),
            ...samplingParams(options, this.maxTokens),
            maxRetries: options?.retries ?? this.maxRetries,
            ...(options?.timeout !== undefined ? { timeout: options.timeout } : {}),
            ...(options?.signal ? { abortSignal: options.signal } : {}),
            ...(this.headers ? { headers: this.headers } : {}),
            ...this.providerOptionsFor(options),
        };
    }

    /** Constructor-level provider options with the call's own merged over them, per provider. */
    private providerOptionsFor(options: LLMCallOptions | undefined) {
        const perCall = options?.providerOptions as
            Record<string, Record<string, unknown>> | undefined;
        if (!perCall)
            return this.providerOptions ? { providerOptions: this.providerOptions } : {};

        const merged: Record<string, Record<string, unknown>> = {
            ...(this.providerOptions as Record<string, Record<string, unknown>> | undefined),
        };
        for (const [provider, value] of Object.entries(perCall)) {
            merged[provider] = { ...merged[provider], ...value };
        }
        return { providerOptions: merged as typeof this.providerOptions };
    }

    /**
     * A per-call `model` can only be honoured when the configured model is a
     * gateway id string: a model object is bound to one model at construction,
     * and quietly routing the override through the gateway instead would send
     * the request somewhere the caller never configured.
     */
    private modelFor(options: LLMCallOptions | undefined): LanguageModel {
        if (options?.model && typeof this.languageModel === 'string') return options.model;
        return this.languageModel;
    }

    private recordCompletion(
        usage: LanguageModelUsage | undefined,
        startedAt: number,
        options: LLMCallOptions | undefined
    ): void {
        this.recordUsage({
            promptTokens: usage?.inputTokens ?? 0,
            completionTokens: usage?.outputTokens ?? 0,
            cachedTokens: usage?.inputTokenDetails?.cacheReadTokens ?? undefined,
            reasoningTokens: usage?.outputTokenDetails?.reasoningTokens ?? undefined,
            latencyMs: Date.now() - startedAt,
            model:
                options?.model && typeof this.languageModel === 'string'
                    ? options.model
                    : undefined,
        });
    }

    /**
     * A filtered reply arrives as a normal response with
     * `finishReason: 'content-filter'`, so it has to be checked for rather
     * than caught.
     */
    private assertNotFiltered(finishReason: AIFinishReason): void {
        if (finishReason !== 'content-filter') return;
        this.recordError();
        throw new ContentFilterError(
            this.provider,
            'Request was declined by the provider’s content filter.'
        );
    }
}

/**
 * Convert ts-dspy chat messages to AI SDK `ModelMessage`s plus `instructions`.
 *
 * Leading system turns become `instructions`, which every provider accepts; a
 * system turn later in the conversation stays in place, with
 * `allowSystemInMessages` set so the AI SDK does not reject it. Assistant
 * tool calls and tool results are correlated by id, and a call without one (a
 * Gemini-origin turn, say) gets a stable synthetic id that its result —
 * matched by tool name — reuses, since the AI SDK requires an id on both.
 */
export function toAISDKPrompt(messages: ChatMessage[]): {
    instructions?: string;
    messages: ModelMessage[];
    allowSystemInMessages?: boolean;
} {
    let start = 0;
    const instructions: string[] = [];
    while (start < messages.length && messages[start].role === 'system') {
        instructions.push(contentToText(messages[start].content));
        start += 1;
    }

    const pendingIds = new Map<string, string[]>();
    let synthetic = 0;
    let systemInMessages = false;
    const converted: ModelMessage[] = [];

    for (const message of messages.slice(start)) {
        switch (message.role) {
            case 'system':
                systemInMessages = true;
                converted.push({ role: 'system', content: contentToText(message.content) });
                break;
            case 'user':
                converted.push({ role: 'user', content: toUserContent(message.content) });
                break;
            case 'assistant': {
                const text = contentToText(message.content);
                if (!message.toolCalls?.length) {
                    converted.push({ role: 'assistant', content: text });
                    break;
                }
                const parts: Array<TextPart | ToolCallPart> = text
                    ? [{ type: 'text', text }]
                    : [];
                for (const call of message.toolCalls) {
                    const id = call.id ?? `call_${call.name}_${synthetic++}`;
                    if (!call.id) {
                        const queue = pendingIds.get(call.name) ?? [];
                        queue.push(id);
                        pendingIds.set(call.name, queue);
                    }
                    parts.push({
                        type: 'tool-call',
                        toolCallId: id,
                        toolName: call.name,
                        input: call.arguments,
                    });
                }
                converted.push({ role: 'assistant', content: parts });
                break;
            }
            case 'tool':
            case 'function': {
                const toolName = message.name ?? 'tool';
                const toolCallId =
                    message.toolCallId ??
                    pendingIds.get(toolName)?.shift() ??
                    `call_${toolName}_${synthetic++}`;
                converted.push({
                    role: 'tool',
                    content: [
                        {
                            type: 'tool-result',
                            toolCallId,
                            toolName,
                            output: { type: 'text', value: contentToText(message.content) },
                        },
                    ],
                });
                break;
            }
        }
    }

    return {
        ...(instructions.length > 0 ? { instructions: instructions.join('\n\n') } : {}),
        messages: converted,
        ...(systemInMessages ? { allowSystemInMessages: true } : {}),
    };
}

function toPrompt(messages: ChatMessage[]) {
    return toAISDKPrompt(messages);
}

function toUserContent(content: ChatMessage['content']): string | Array<TextPart | FilePart> {
    if (typeof content === 'string') return content;
    return content.map((part) =>
        part.type === 'text' ? { type: 'text', text: part.text } : toImagePart(part)
    );
}

/**
 * Images go as `file` parts: AI SDK 7 deprecates the `image` part, and using
 * it makes the SDK print a deprecation warning on every call. A URL whose
 * media type is unknown is sent as the top-level `image` type, which the SDK
 * resolves per provider.
 */
function toImagePart(part: ImageContentPart): FilePart {
    const source = part.source;
    if (source.kind === 'base64') {
        return { type: 'file', data: source.data, mediaType: source.mediaType };
    }
    const dataUri = /^data:([^;,]+)(?:;[^,]*)?;base64,(.*)$/s.exec(source.url);
    if (dataUri) {
        return { type: 'file', data: dataUri[2], mediaType: source.mediaType ?? dataUri[1] };
    }
    return { type: 'file', data: new URL(source.url), mediaType: source.mediaType ?? 'image' };
}

/** Declare tools without `execute`, so the AI SDK returns the calls instead of running them. */
function toolSettings(options: LLMCallOptions | undefined): {
    tools?: ToolSet;
    toolChoice?: AIToolChoice<ToolSet>;
} {
    if (!options?.tools?.length) return {};
    return {
        tools: toAISDKTools(options.tools),
        ...(options.toolChoice ? { toolChoice: toAIToolChoice(options.toolChoice) } : {}),
    };
}

/** Map core tool declarations to AI SDK tools, keyed by name, with no `execute`. */
export function toAISDKTools(specs: ToolSpec[]): ToolSet {
    const tools: ToolSet = {};
    for (const spec of specs) {
        tools[spec.name] = tool({
            ...(spec.description ? { description: spec.description } : {}),
            inputSchema: jsonSchema(spec.parameters as Parameters<typeof jsonSchema>[0]),
        });
    }
    return tools;
}

function toAIToolChoice(choice: ToolChoice): AIToolChoice<ToolSet> {
    return typeof choice === 'string' ? choice : { type: 'tool', toolName: choice.name };
}

function fromAIToolCall(call: {
    toolCallId: string;
    toolName: string;
    input: unknown;
}): ToolCall {
    const input = call.input;
    if (typeof input === 'string') {
        // Only reached when the AI SDK could not parse the arguments itself;
        // keep the raw text so the failure is inspectable.
        try {
            const parsed = JSON.parse(input);
            if (isRecord(parsed)) {
                return { id: call.toolCallId, name: call.toolName, arguments: parsed };
            }
        } catch {
            // fall through
        }
        return { id: call.toolCallId, name: call.toolName, arguments: {}, rawArguments: input };
    }
    return {
        id: call.toolCallId,
        name: call.toolName,
        arguments: isRecord(input) ? input : {},
    };
}

function fromAIFinishReason(reason: AIFinishReason): FinishReason {
    switch (reason) {
        case 'stop':
            return 'stop';
        case 'length':
            return 'length';
        case 'tool-calls':
            return 'tool_calls';
        case 'content-filter':
            return 'content_filter';
        default:
            return 'other';
    }
}

function samplingParams(options: LLMCallOptions | undefined, defaultMaxTokens?: number) {
    const maxOutputTokens = options?.maxTokens ?? defaultMaxTokens;
    return {
        ...(options?.temperature !== undefined ? { temperature: options.temperature } : {}),
        ...(maxOutputTokens !== undefined ? { maxOutputTokens } : {}),
        ...(options?.topP !== undefined ? { topP: options.topP } : {}),
        ...(options?.stopSequences?.length ? { stopSequences: options.stopSequences } : {}),
        ...(options?.frequencyPenalty !== undefined
            ? { frequencyPenalty: options.frequencyPenalty }
            : {}),
        ...(options?.presencePenalty !== undefined
            ? { presencePenalty: options.presencePenalty }
            : {}),
    };
}

/**
 * Map an AI SDK failure onto core's error taxonomy.
 *
 * After its retries run out the AI SDK throws a `RetryError` wrapping the
 * last attempt's error, so that is unwrapped first: the HTTP status that
 * decides the class lives on the inner `APICallError`.
 */
export function toLMError(error: unknown, provider: string): LMError {
    if (error instanceof LMError) return error;

    const inner = RetryError.isInstance(error) ? error.lastError : error;
    if (isAbort(inner) || isAbort(error)) {
        const message = inner instanceof Error ? inner.message : 'Request aborted';
        return new TimeoutError(provider, message, { cause: error });
    }
    if (APICallError.isInstance(inner)) {
        const ErrorClass = classify(inner.statusCode, {
            type: errorField(inner.data, 'type'),
            code: errorField(inner.data, 'code'),
            message: inner.message,
        });
        if (ErrorClass === RateLimitError) {
            return new RateLimitError(provider, inner.message, {
                cause: error,
                status: inner.statusCode,
                retryAfterMs: retryAfterMs(inner.responseHeaders),
            });
        }
        return new ErrorClass(provider, inner.message, {
            cause: error,
            status: inner.statusCode,
        });
    }
    const message = error instanceof Error ? error.message : String(error);
    return new LMError(provider, message, { cause: error });
}

function isAbort(error: unknown): boolean {
    return (
        error instanceof Error && (error.name === 'AbortError' || error.name === 'TimeoutError')
    );
}

/**
 * Provider error bodies nest their discriminators differently — OpenAI's
 * `{ error: { code, type } }`, Anthropic's `{ error: { type } }` — so look in
 * both places.
 */
function errorField(data: unknown, field: 'type' | 'code'): string | undefined {
    if (!isRecord(data)) return undefined;
    const direct = data[field];
    if (typeof direct === 'string') return direct;
    const nested = data.error;
    if (isRecord(nested) && typeof nested[field] === 'string') return nested[field] as string;
    return undefined;
}

function retryAfterMs(headers: Record<string, string> | undefined): number | undefined {
    if (!headers) return undefined;
    const ms = headers['retry-after-ms'];
    if (ms !== undefined && Number.isFinite(Number(ms))) return Number(ms);
    const seconds = headers['retry-after'];
    if (seconds === undefined) return undefined;
    if (Number.isFinite(Number(seconds))) return Number(seconds) * 1000;
    const date = Date.parse(seconds);
    return Number.isNaN(date) ? undefined : Math.max(0, date - Date.now());
}

function providerNameOf(model: LanguageModel): string {
    if (typeof model === 'string') return 'gateway';
    // AI SDK providers report e.g. `openai.chat` or `openai.responses`; the
    // part before the dot is the provider a price table or a trace expects.
    return model.provider.split('.')[0] || 'ai-sdk';
}

function modelIdOf(model: LanguageModel): string {
    return typeof model === 'string' ? model : model.modelId;
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}
