/**
 * Zod-native signatures: inferred output types, no decorators, no type argument.
 *
 *   export OPENAI_API_KEY="sk-..."
 *   npm run example:zod
 */
import { z } from 'zod';
import {
    signature,
    Predict,
    ChainOfThought,
    buildOutputJsonSchema,
    configure,
    ValidationError,
} from '@ts-dspy/core';
import { OpenAILM } from '@ts-dspy/openai';
import { requireEnv, section } from './utils';

// --- Signatures -------------------------------------------------------------
//
// A zod signature is a plain object, so nothing here needs
// `experimentalDecorators`, and the schema carries the field types into the type
// system rather than losing them at runtime.

const AnalyzeReview = signature({
    description: 'Analyze a product review.',
    input: z.object({
        review: z.string().describe('the review text'),
    }),
    output: z.object({
        sentiment: z.enum(['positive', 'negative', 'neutral']),
        rating: z.number().int().min(1).max(5).describe('rating from 1 to 5'),
        themes: z.array(z.string()).describe('key themes mentioned'),
        followUp: z.string().optional().describe('a follow-up question, if one is warranted'),
    }),
});

// Nested objects and unions are expressible too — neither has any spelling in
// the flat decorator field-type list.
const TriageTicket = signature({
    description: 'Triage a support ticket.',
    input: z.object({ ticket: z.string() }),
    output: z.object({
        priority: z.enum(['p0', 'p1', 'p2', 'p3']),
        owner: z.object({
            team: z.enum(['billing', 'platform', 'support']),
            escalate: z.boolean(),
        }),
        estimateHours: z.union([z.number(), z.literal('unknown')]),
    }),
});

async function main(): Promise<void> {
    const lm = new OpenAILM({ apiKey: requireEnv('OPENAI_API_KEY') });
    configure({ lm });

    console.log(`Model: ${lm.getModelName()}`);

    // --- Inference without a type argument ----------------------------------
    section('Inferred output types');

    const analysis = await new Predict(AnalyzeReview).forward({
        review: 'The battery lasts all day and the screen is gorgeous, but it is heavy.',
    });

    // Every one of these is precisely typed, with no `TOutput` written by hand.
    const sentiment: 'positive' | 'negative' | 'neutral' = analysis.sentiment;
    const themes: string[] = analysis.themes;

    console.log(`sentiment: ${sentiment}`);
    console.log(`rating:    ${analysis.rating} (${typeof analysis.rating})`);
    console.log(`themes:    ${themes.join(', ')}`);
    console.log(`followUp:  ${analysis.followUp ?? '(none)'}`);

    // --- Nested objects and unions ------------------------------------------
    section('Nested output');

    const triage = await new Predict(TriageTicket).forward({
        ticket: 'Card was charged twice for the October invoice.',
    });

    console.log(`priority:  ${triage.priority}`);
    console.log(`team:      ${triage.owner.team} (escalate: ${triage.owner.escalate})`);
    console.log(`estimate:  ${triage.estimateHours}`);

    // --- The schema sent to the provider ------------------------------------
    section('Structured-output schema');

    // OpenAI strict mode: every property in `required`, `additionalProperties`
    // false at every level, optional fields expressed as nullable.
    console.log(JSON.stringify(buildOutputJsonSchema(TriageTicket), null, 2));

    // --- ChainOfThought keeps the same inference ----------------------------
    section('ChainOfThought');

    const reasoned = await new ChainOfThought(AnalyzeReview).forward({
        review: 'Arrived cracked and support never replied.',
    });

    console.log(`reasoning: ${reasoned.reasoning.slice(0, 200)}...`);
    console.log(`sentiment: ${reasoned.sentiment}`);

    // --- Constraints are enforced, not merely suggested ---------------------
    section('Validation');

    const Impossible = signature({
        input: z.object({ input: z.string() }),
        output: z.object({
            // No model will answer with a value in an empty range.
            impossible: z.number().min(10).max(1),
        }),
    });

    try {
        await new Predict(Impossible).forward({ input: 'hello' });
        console.log('Model produced the field after all.');
    } catch (error) {
        if (error instanceof ValidationError) {
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
