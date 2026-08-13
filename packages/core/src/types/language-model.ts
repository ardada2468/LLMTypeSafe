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
    metadata?: Record<string, any>;
}

export interface ChatMessage {
    role: 'system' | 'user' | 'assistant' | 'function' | 'tool';
    content: string;
    name?: string;
    functionCall?: {
        name: string;
        arguments: string;
    };
    toolCalls?: ToolCall[];
}

export interface ToolCall {
    id: string;
    type: 'function';
    function: {
        name: string;
        arguments: string;
    };
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
