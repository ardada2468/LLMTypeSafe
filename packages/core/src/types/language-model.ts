export interface LLMCallOptions {
    temperature?: number;
    maxTokens?: number;
    stopSequences?: string[];
    topP?: number;
    frequencyPenalty?: number;
    presencePenalty?: number;
    streaming?: boolean;
    model?: string;
    timeout?: number;
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
    totalCost?: number;
    requestCount?: number;
    errorCount?: number;
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
    generateStructured<T>(prompt: string, schema: any, options?: LLMCallOptions): Promise<T>;
    chat(messages: ChatMessage[], options?: LLMCallOptions): Promise<string>;
    generateStream?(prompt: string, options?: LLMCallOptions): AsyncGenerator<StreamChunk, void, unknown>;
    chatStream?(messages: ChatMessage[], options?: LLMCallOptions): AsyncGenerator<StreamChunk, void, unknown>;
    getUsage(): UsageStats;
    resetUsage(): void;
    getCapabilities(): ModelCapabilities;
    getModelName(): string;
    setModel?(model: string): void;
    listModels?(): Promise<string[]>;
    isHealthy?(): Promise<boolean>;
    getCostEstimate?(prompt: string, options?: LLMCallOptions): Promise<number>;
} 