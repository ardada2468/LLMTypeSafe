/**
 * TS-DSPy with Google Gemini.
 *
 *   export GEMINI_API_KEY="..."
 *   npm run example:gemini
 */
import {
    Signature,
    InputField,
    OutputField,
    Predict,
    ChainOfThought,
    configure,
} from '@ts-dspy/core';
import { GeminiLM } from '@ts-dspy/gemini';
import { requireEnv, section } from './utils';

class SummarizeArticle extends Signature {
    static description = 'Summarize an article and extract its key points.';

    @InputField({ description: 'the article text' })
    article!: string;

    @OutputField({ description: 'a one-sentence summary' })
    summary!: string;

    @OutputField({ description: 'the main points', type: 'string[]' })
    keyPoints!: string[];

    @OutputField({ description: 'reading difficulty from 1 to 10', type: 'int' })
    difficulty!: number;
}

const ARTICLE = `
TypeScript's structural type system checks shapes rather than names, so any two
types with the same members are interchangeable. This makes it a good fit for
describing data that crosses a boundary — an HTTP response, a config file, or a
language model's output — but the checking happens only at compile time. Values
arriving at runtime have to be validated separately; the type annotation alone
is an assertion, not a guarantee.
`.trim();

async function main(): Promise<void> {
    const lm = new GeminiLM({ apiKey: requireEnv('GEMINI_API_KEY') });
    configure({ lm });

    console.log(`Model: ${lm.getModelName()}`);

    // --- Structured output --------------------------------------------------
    section('Predict');

    const result = await new Predict(SummarizeArticle).forward({ article: ARTICLE });

    console.log(`summary:    ${result.summary}`);
    console.log(`difficulty: ${result.difficulty}/10`);
    console.log('keyPoints:');
    for (const point of result.keyPoints) {
        console.log(`  - ${point}`);
    }

    // --- Reasoning ----------------------------------------------------------
    section('ChainOfThought');

    class MathProblem extends Signature {
        @InputField({ description: 'the problem' })
        problem!: string;

        @OutputField({ description: 'the numeric answer', type: 'number' })
        answer!: number;
    }

    const solved = await new ChainOfThought(MathProblem).forward({
        problem: 'A shirt costs $40 after a 20% discount. What was the original price?',
    });

    console.log(`reasoning: ${solved.reasoning.slice(0, 200)}...`);
    console.log(`answer:    $${solved.answer}`);

    // --- Streaming ----------------------------------------------------------
    section('Streaming');

    process.stdout.write('response: ');
    for await (const chunk of lm.generateStream('Name three uses for TypeScript, briefly.')) {
        if (!chunk.done) {
            process.stdout.write(chunk.content);
        }
    }
    console.log();

    // --- Usage --------------------------------------------------------------
    section('Usage');
    const usage = lm.getUsage();
    console.log(`requests: ${usage.requestCount}`);
    console.log(
        `tokens:   ${usage.totalTokens} (${usage.promptTokens} in, ${usage.completionTokens} out)`
    );
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});
