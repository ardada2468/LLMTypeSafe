export interface LLMCallOptions {
    temperature?: number;
    maxTokens?: number;
    stopSequences?: string[];
    topP?: number;
    frequencyPenalty?: number;
    presencePenalty?: number;
    streaming?: boolean;
    /** Override the model for this call only. */
    model?: string;
    /** Per-request timeout in milliseconds, passed through to the provider SDK. */
    timeout?: number;
    /**
     * Maximum retries for this request, passed through to the provider SDK.
     * Retries are owned by the official SDKs, which honour `retry-after` headers;
     * ts-dspy does not add a second retry layer on top.
     */
    retries?: number;
    /**
     * Tools the model may call on this turn. Providers advertising
     * `supportsFunctionCalling` translate these into their own request shape;
     * providers without native tool calling ignore them.
     */
    tools?: ToolSpec[];
    /** How hard to push the model towards calling a tool. Defaults to the provider's own default. */
    toolChoice?: ToolChoice;
    /**
     * Extra round-trips to spend repairing a response that fails validation.
     *
     * On a `ValidationError` the model is re-prompted with the offending field
     * names, their declared types and the values it actually sent, then the
     * result is re-validated. Defaults to `0` — validation failures rethrow
     * immediately, so self-repair is strictly opt-in. Capped at 10, and cut
     * short when an attempt reproduces the previous failure exactly.
     */
    repairAttempts?: number;
    /**
     * Cancellation signal for this request. Aborting it rejects the call — use
     * it to drop work a React unmount or a cancelled server request no longer
     * needs. Combined with `timeout` when both are supplied, so whichever fires
     * first wins.
     */
    signal?: AbortSignal;
    metadata?: Record<string, any>;
}

/**
 * A tool offered to the model.
 *
 * `parameters` is a JSON Schema object describing the arguments. Every provider
 * accepts JSON Schema here, so this is the one representation that survives the
 * trip through all three SDKs unchanged.
 */
export interface ToolSpec {
    name: string;
    description?: string;
    parameters: Record<string, unknown>;
}

export type ToolChoice = 'auto' | 'none' | 'required' | { name: string };

/**
 * A tool call requested by the model.
 *
 * The shape is deliberately provider-neutral rather than a copy of any one SDK:
 *
 * - `arguments` is always a **parsed object**. OpenAI sends a JSON string, which
 *   is parsed on the way in; Anthropic (`input`) and Gemini (`args`) already
 *   send objects.
 * - `id` is optional because Gemini's function calls have no identifier &mdash;
 *   results there are correlated by function name.
 * - `rawArguments` keeps the provider's original encoding when there was one, so
 *   an assistant turn can be replayed byte-for-byte and so a call whose arguments
 *   failed to parse is still inspectable.
 */
export interface ToolCall {
    /** Provider-assigned identifier. Absent on Gemini. */
    id?: string;
    name: string;
    arguments: Record<string, unknown>;
    /** The unparsed argument payload, when the provider sent one (OpenAI only). */
    rawArguments?: string;
}

export interface ChatMessage {
    role: 'system' | 'user' | 'assistant' | 'function' | 'tool';
    content: string;
    /** On a `tool`/`function` turn, the name of the tool that produced the result. */
    name?: string;
    /** On a `tool`/`function` turn, the id of the call being answered. */
    toolCallId?: string;
    /** On an `assistant` turn, the tool calls the model requested. */
    toolCalls?: ToolCall[];
}

/** Why the model stopped, normalised across providers. */
export type FinishReason = 'stop' | 'tool_calls' | 'length' | 'content_filter' | 'other';

/** A full chat turn, including any tool calls the model asked for. */
export interface ChatResult {
    content: string;
    toolCalls?: ToolCall[];
    finishReason?: FinishReason;
}

export interface UsageStats {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
    /**
     * Estimated spend, when a provider reports it. ts-dspy no longer computes
     * this from a built-in price table — those go stale and were producing
     * numbers off by more than an order of magnitude. Compute it from
     * `promptTokens`/`completionTokens` and current published pricing instead.
     */
    totalCost?: number;
    requestCount?: number;
    errorCount?: number;
    /**
     * Responses served from the configured cache. Cache hits are counted here
     * and deliberately excluded from `requestCount` and the token totals, so
     * usage keeps reflecting real provider traffic and cost stays accurate.
     */
    cacheHits?: number;
    /** Mean round-trip latency in milliseconds across recorded requests. */
    averageLatency?: number;
}

export interface StreamChunk {
    content: string;
    done: boolean;
    usage?: Partial<UsageStats>;
    metadata?: Record<string, any>;
}

export interface ModelCapabilities {
    supportsStreaming: boolean;
    supportsStructuredOutput: boolean;
    supportsFunctionCalling: boolean;
    supportsVision: boolean;
    maxContextLength: number;
    supportedFormats: string[];
}

export interface ILanguageModel {
    generate(prompt: string, options?: LLMCallOptions): Promise<string>;
    /**
     * Generate a value conforming to `schema` (a JSON Schema object, as produced
     * by `buildOutputJsonSchema`). Providers advertising
     * `supportsStructuredOutput` constrain decoding natively; others fall back to
     * requesting JSON in the prompt.
     */
    generateStructured<T>(
        prompt: string,
        schema: unknown,
        options?: LLMCallOptions
    ): Promise<T>;
    chat(messages: ChatMessage[], options?: LLMCallOptions): Promise<string>;
    /**
     * Run one chat turn and return the tool calls alongside the text.
     *
     * Providers advertising `supportsFunctionCalling` implement this natively;
     * {@link BaseLM} supplies a text-only default for everyone else, so callers
     * can rely on the capability flag rather than feature-detecting the method.
     */
    chatWithTools?(messages: ChatMessage[], options?: LLMCallOptions): Promise<ChatResult>;
    generateStream?(
        prompt: string,
        options?: LLMCallOptions
    ): AsyncGenerator<StreamChunk, void, unknown>;
    chatStream?(
        messages: ChatMessage[],
        options?: LLMCallOptions
    ): AsyncGenerator<StreamChunk, void, unknown>;
    getUsage(): UsageStats;
    resetUsage(): void;
    getCapabilities(): ModelCapabilities;
    getModelName(): string;
    setModel?(model: string): void;
    listModels?(): Promise<string[]>;
    isHealthy?(): Promise<boolean>;
    getCostEstimate?(prompt: string, options?: LLMCallOptions): Promise<number>;
}
