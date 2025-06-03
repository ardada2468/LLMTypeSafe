export interface LLMCallOptions {
    temperature?: number;
    maxTokens?: number;
    stopSequences?: string[];
    topP?: number;
    frequencyPenalty?: number;
    presencePenalty?: number;
}

export interface ChatMessage {
    role: 'system' | 'user' | 'assistant';
    content: string;
}

export interface UsageStats {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
    totalCost?: number;
}

export interface ILanguageModel {
    generate(prompt: string, options?: LLMCallOptions): Promise<string>;
    generateStructured<T>(prompt: string, schema: any, options?: LLMCallOptions): Promise<T>;
    chat(messages: ChatMessage[], options?: LLMCallOptions): Promise<string>;
    getUsage(): UsageStats;
    resetUsage(): void;
} 