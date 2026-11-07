import { BaseLM } from './base-lm';
import { contentToText } from '../utils/content';
import type { ChatMessage, LLMCallOptions, ModelCapabilities } from '../types/language-model';

/** What {@link EchoLM} does for one question. */
export interface EchoBehaviour {
    /** Milliseconds to wait before replying. */
    delayMs?: number;
    promptTokens?: number;
    completionTokens?: number;
    cachedTokens?: number;
}

/**
 * Test model for concurrency: replies `answer: <question>` after a per-question
 * delay, recording per-question token counts. Lets a test make calls finish in
 * the opposite order to the one they started in, and tell their usage apart.
 *
 * Test-only; not exported from the package.
 */
export class EchoLM extends BaseLM {
    readonly prompts: string[] = [];

    constructor(
        private readonly behaviour: Record<string, EchoBehaviour> = {},
        model = 'echo'
    ) {
        super('test', model);
    }

    async chat(messages: ChatMessage[], _options?: LLMCallOptions): Promise<string> {
        const prompt = messages.map((message) => contentToText(message.content)).join('\n');
        this.prompts.push(prompt);
        // Read the question from the last (user) turn: the system message's
        // structure template mentions the field too, as a placeholder.
        const turn = contentToText(messages.at(-1)?.content ?? '');
        const question = /(?:question: |\[\[ ## question ## \]\]\n)(\S+)/.exec(turn)?.[1] ?? '';
        const behaviour = this.behaviour[question] ?? {};
        await new Promise((resolve) => setTimeout(resolve, behaviour.delayMs ?? 0));
        this.recordUsage({
            promptTokens: behaviour.promptTokens ?? 1,
            completionTokens: behaviour.completionTokens ?? 1,
            cachedTokens: behaviour.cachedTokens,
            latencyMs: behaviour.delayMs ?? 0,
        });
        return `answer: ${question}`;
    }

    getCapabilities(): ModelCapabilities {
        return {
            supportsStreaming: false,
            supportsStructuredOutput: false,
            supportsFunctionCalling: false,
            supportsVision: false,
            maxContextLength: 8192,
            supportedFormats: ['text'],
        };
    }
}
