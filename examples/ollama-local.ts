/**
 * TS-DSPy against a local, OpenAI-compatible server — no cloud key required.
 *
 *   ollama serve
 *   ollama pull llama3.2
 *   npm run example:ollama
 *
 * Point it elsewhere without editing the file:
 *
 *   OPENAI_COMPATIBLE_BASE_URL=http://localhost:1234/v1 \
 *   OPENAI_COMPATIBLE_MODEL=qwen2.5-7b-instruct \
 *   npm run example:ollama
 */
import { Signature, InputField, OutputField, Predict, configure } from '@ts-dspy/core';
import { OpenAICompatibleLM, OPENAI_COMPATIBLE_BASE_URLS } from '@ts-dspy/openai';
import { section } from './utils';

class AnswerQuestion extends Signature {
    static description = 'Answer a factual question as concisely as possible.';

    @InputField({ description: 'the question to answer' })
    question!: string;

    @OutputField({ description: 'a concise answer' })
    answer!: string;

    @OutputField({ description: 'confidence between 0 and 1', type: 'number' })
    confidence!: number;
}

async function main(): Promise<void> {
    const baseURL =
        process.env.OPENAI_COMPATIBLE_BASE_URL ?? OPENAI_COMPATIBLE_BASE_URLS.ollama;
    const model = process.env.OPENAI_COMPATIBLE_MODEL ?? 'llama3.2';

    // No apiKey: local servers want the header present but ignore its value, so
    // the provider supplies a placeholder rather than failing on a missing
    // OPENAI_API_KEY. A hosted endpoint (Groq, Together, OpenRouter) needs a
    // real one — pass `apiKey` there.
    const lm = new OpenAICompatibleLM({
        baseURL,
        model,
        // Left at their conservative defaults: a local Llama has no strict
        // JSON-schema mode, so Predict falls back to labelled-text parsing
        // instead of sending a response_format the server would reject.
        // supportsStructuredOutput: true,
        // maxContextLength: 128_000,
        timeout: 120_000,
    });
    configure({ lm });

    console.log(`Endpoint: ${baseURL}`);
    console.log(`Model:    ${lm.getModelName()}`);
    console.log(`Native structured output: ${lm.getCapabilities().supportsStructuredOutput}`);

    section('Predict');

    const result = await new Predict(AnswerQuestion).forward({
        question: 'What is the capital of France?',
    });

    console.log(`answer:     ${result.answer}`);
    // Still a real number: the fallback path parses labelled text and validates
    // it against the signature exactly as the native path does.
    console.log(`confidence: ${result.confidence} (${typeof result.confidence})`);

    section('Streaming');

    for await (const chunk of lm.generateStream('Name three primary colours.')) {
        if (!chunk.done) process.stdout.write(chunk.content);
    }
    console.log();

    section('Usage');
    const usage = lm.getUsage();
    console.log(`requests: ${usage.requestCount}`);
    console.log(
        `tokens:   ${usage.totalTokens} (${usage.promptTokens} in, ${usage.completionTokens} out)`
    );
}

main().catch((error) => {
    console.error(error);
    console.error(
        '\nIs the server running? For Ollama: `ollama serve` and `ollama pull llama3.2`.'
    );
    process.exit(1);
});
