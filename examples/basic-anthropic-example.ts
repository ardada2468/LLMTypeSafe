/**
 * TS-DSPy with Anthropic Claude.
 *
 *   export ANTHROPIC_API_KEY="sk-ant-..."
 *   npm run example:anthropic
 */
import {
    Signature,
    InputField,
    OutputField,
    Predict,
    ChainOfThought,
    configure,
} from '@ts-dspy/core';
import { AnthropicLM, AnthropicRefusalError } from '@ts-dspy/anthropic';
import { requireEnv, section } from './utils';

class ReviewCode extends Signature {
    static description = 'Review a code snippet for correctness and clarity.';

    @InputField({ description: 'the code to review' })
    code!: string;

    @OutputField({ description: 'overall verdict: approve, or request changes' })
    verdict!: string;

    @OutputField({ description: 'specific issues found', type: 'string[]' })
    issues!: string[];

    @OutputField({ description: 'severity from 1 (nit) to 5 (blocking)', type: 'int' })
    severity!: number;
}

const SNIPPET = `
function average(values) {
    let total = 0;
    for (let i = 0; i <= values.length; i++) {
        total += values[i];
    }
    return total / values.length;
}
`.trim();

async function main(): Promise<void> {
    const lm = new AnthropicLM({ apiKey: requireEnv('ANTHROPIC_API_KEY') });
    configure({ lm });

    console.log(`Model: ${lm.getModelName()}`);

    // --- Structured output --------------------------------------------------
    section('Predict');

    const review = await new Predict(ReviewCode).forward({ code: SNIPPET });

    console.log(`verdict:  ${review.verdict}`);
    console.log(`severity: ${review.severity}/5`);
    console.log('issues:');
    for (const issue of review.issues) {
        console.log(`  - ${issue}`);
    }

    // --- Reasoning ----------------------------------------------------------
    section('ChainOfThought');

    class Explain extends Signature {
        @InputField({ description: 'the topic' })
        topic!: string;

        @OutputField({ description: 'a plain-language explanation' })
        explanation!: string;
    }

    const explained = await new ChainOfThought(Explain).forward({
        topic: 'why structural typing does not validate runtime data',
    });
    console.log(explained.explanation);

    // --- Streaming ----------------------------------------------------------
    section('Streaming');

    process.stdout.write('response: ');
    for await (const chunk of lm.generateStream(
        'Describe TypeScript generics in two sentences.'
    )) {
        if (!chunk.done) {
            process.stdout.write(chunk.content);
        }
    }
    console.log();

    // --- Refusals -----------------------------------------------------------
    // Claude's safety classifiers return a normal 200 response with
    // stop_reason "refusal" rather than an HTTP error, so the provider surfaces
    // it as a typed error instead of an empty string.
    section('Refusal handling');
    try {
        await lm.generate('Summarize the plot of a well-known novel in one sentence.');
        console.log('Request completed normally.');
    } catch (error) {
        if (error instanceof AnthropicRefusalError) {
            console.log(`Declined (category: ${error.category ?? 'unspecified'})`);
        } else {
            throw error;
        }
    }

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
