/**
 * Core TS-DSPy usage with OpenAI.
 *
 *   export OPENAI_API_KEY="sk-..."
 *   npm run example:openai
 */
import {
    Signature,
    InputField,
    OutputField,
    Predict,
    ChainOfThought,
    configure,
    ValidationError,
} from '@ts-dspy/core';
import { OpenAILM } from '@ts-dspy/openai';
import { requireEnv, section } from './utils';

// --- Signatures -------------------------------------------------------------

class AnswerQuestion extends Signature {
    static description = 'Answer a factual question as concisely as possible.';

    @InputField({ description: 'the question to answer' })
    question!: string;

    @OutputField({ description: 'a concise answer' })
    answer!: string;

    @OutputField({ description: 'confidence between 0 and 1', type: 'number' })
    confidence!: number;
}

class AnalyzeReview extends Signature {
    static description = 'Analyze a product review.';

    @InputField({ description: 'the review text' })
    review!: string;

    @OutputField({ description: 'positive, negative, or neutral' })
    sentiment!: string;

    @OutputField({ description: 'rating from 1 to 5', type: 'int' })
    rating!: number;

    @OutputField({ description: 'key themes mentioned', type: 'string[]' })
    themes!: string[];

    @OutputField({ description: 'a follow-up question, if one is warranted', required: false })
    followUp?: string;
}

async function main(): Promise<void> {
    const lm = new OpenAILM({ apiKey: requireEnv('OPENAI_API_KEY') });
    configure({ lm });

    console.log(`Model: ${lm.getModelName()}`);

    // --- Predict ------------------------------------------------------------
    section('Predict');

    const qa = new Predict(AnswerQuestion);
    const answer = await qa.forward({ question: 'What is the capital of France?' });

    console.log(`answer:     ${answer.answer}`);
    // confidence is a real number, not a string that looks like one — the
    // library validates the model's output against the declared field types.
    console.log(`confidence: ${answer.confidence} (${typeof answer.confidence})`);

    // --- Typed fields, including arrays and optionals ------------------------
    section('Structured output');

    // Decorators record fields at runtime, so TypeScript cannot infer per-field
    // types from the class. Naming the output shape gets you precise types on
    // the result; validation comes from the signature either way.
    type ReviewAnalysis = {
        sentiment: string;
        rating: number;
        themes: string[];
        followUp?: string;
    };

    const analysis = await new Predict<typeof AnalyzeReview, ReviewAnalysis>(
        AnalyzeReview
    ).forward({
        review: 'The battery lasts all day and the screen is gorgeous, but it is heavy.',
    });

    console.log(`sentiment: ${analysis.sentiment}`);
    console.log(`rating:    ${analysis.rating} (${typeof analysis.rating})`);
    console.log(`themes:    ${analysis.themes.join(', ')}`);
    console.log(`followUp:  ${analysis.followUp ?? '(none)'}`);

    // --- ChainOfThought -----------------------------------------------------
    section('ChainOfThought');

    const reasoned = await new ChainOfThought(AnswerQuestion).forward({
        question: 'If a train travels 60 km in 45 minutes, what is its average speed in km/h?',
    });

    console.log(`reasoning: ${reasoned.reasoning.slice(0, 200)}...`);
    console.log(`answer:    ${reasoned.answer}`);

    // --- Per-call options ---------------------------------------------------
    section('Call options');

    const precise = await qa.forward(
        { question: 'What is the boiling point of water at sea level in Celsius?' },
        { temperature: 0, timeout: 30_000, retries: 2 }
    );
    console.log(`answer: ${precise.answer}`);

    // --- Validation failures are loud ---------------------------------------
    section('Validation');

    class ImpossibleShape extends Signature {
        @InputField({ description: 'anything' })
        input!: string;

        @OutputField({ description: 'a field the model will not produce', type: 'number' })
        nonexistentField!: number;
    }

    try {
        await new Predict(ImpossibleShape).forward({ input: 'hello' });
        console.log('Model produced the field after all.');
    } catch (error) {
        if (error instanceof ValidationError) {
            // Previously this returned null or a raw string while TypeScript
            // claimed the field was a number.
            console.log('Caught ValidationError, as expected:');
            for (const issue of error.issues) {
                console.log(`  - ${issue.field} (${issue.expected}): ${issue.message}`);
            }
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
