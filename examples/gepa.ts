/**
 * GEPA: reflective prompt evolution, driven by metric feedback.
 *
 * A metric that returns `{ score, feedback }` tells the optimizer *why* an
 * output fell short. GEPA hands that feedback, with the run that earned it, to
 * a reflection model that rewrites the predictor's instructions; candidates
 * that win on any validation example stay on a Pareto front and keep evolving.
 *
 *   export OPENAI_API_KEY="sk-..."
 *   npm run example:gepa
 */
import {
    Example,
    GEPA,
    Predict,
    configure,
    evaluate,
    signature,
    type Metric,
} from '@ts-dspy/core';
import { OpenAILM } from '@ts-dspy/openai';
import { requireEnv, section } from './utils';

// Extract a structured, normalised order line from a free-text message.
const ExtractOrder = signature(
    'message -> sku: string, quantity: int, express: bool',
    'Extract the order from the customer message.'
);

const labelled = [
    ['Send 3 of the ACME-42 widgets, no rush.', 'ACME-42', 3, false],
    ['I need a dozen ZX-9 cables overnight please!', 'ZX-9', 12, true],
    ['Can I get one LT-100 lamp, standard shipping.', 'LT-100', 1, false],
    ['Two boxes of PP-7 paper, ASAP — express delivery.', 'PP-7', 2, true],
    ['half a dozen acme-42 widgets, whenever', 'ACME-42', 6, false],
    ['Rush order: 4x zx-9 cables', 'ZX-9', 4, true],
    ['Please ship ten lt-100 lamps by regular post.', 'LT-100', 10, false],
    ['need 1 pp-7 box tomorrow morning, express', 'PP-7', 1, true],
].map(([message, sku, quantity, express]) =>
    new Example({ message, sku, quantity, express }).withInputs('message')
);

const trainset = labelled.slice(0, 4);
const valset = labelled.slice(4);

/**
 * Partial credit per field, plus feedback naming each mistake. GEPA may also
 * pass `trace`, `predName` and `predTrace`, to aim feedback at one predictor
 * of a multi-step program; this program has only one, so they go unused.
 */
const orderMetric: Metric = (example, prediction) => {
    const problems: string[] = [];
    const sku = String(prediction.get('sku') ?? '');
    if (sku !== example.get('sku')) {
        problems.push(`sku should be "${example.get('sku')}" (upper case), got "${sku}"`);
    }
    if (prediction.get('quantity') !== example.get('quantity')) {
        problems.push(
            `quantity should be ${example.get('quantity')} (spell out words like "dozen" as numbers), got ${prediction.get('quantity')}`
        );
    }
    if (prediction.get('express') !== example.get('express')) {
        problems.push(
            `express should be ${example.get('express')} — words like "rush", "ASAP" or "overnight" mean express`
        );
    }
    return {
        score: (3 - problems.length) / 3,
        feedback: problems.length === 0 ? 'All fields correct.' : problems.join('; '),
    };
};

async function main(): Promise<void> {
    const apiKey = requireEnv('OPENAI_API_KEY');
    configure({ lm: new OpenAILM({ apiKey, model: 'gpt-4.1-mini' }) });

    const student = new Predict(ExtractOrder);

    section('Baseline');
    const before = await evaluate(student, valset, orderMetric, { concurrency: 4 });
    console.log(`score: ${before.score.toFixed(2)}`);
    for (const result of before.results) {
        if (result.score < 1) console.log(`  - ${result.feedback}`);
    }

    section('GEPA');
    const gepa = new GEPA({
        metric: orderMetric,
        // Use a strong model for reflection: it reads runs and feedback and
        // writes the new instructions.
        reflectionLM: new OpenAILM({ apiKey, model: 'gpt-4.1' }),
        // Exactly one budget: `auto`, `maxMetricCalls` or `maxFullEvals`.
        maxMetricCalls: 60,
        reflectionMinibatchSize: 2,
        trackStats: true,
        seed: 1,
        onProgress: (event) => console.log(`  [${event.phase}] ${event.message ?? ''}`),
    });

    const optimized = await gepa.compile(student, { trainset, valset });
    const details = optimized.detailedResults;

    console.log(
        `\n${details.candidates.length} candidates, ${details.totalMetricCalls}/${details.budget} metric calls`
    );
    const best = details.candidates[details.bestIndex];
    console.log(`best candidate #${best.id} (${best.origin}), score ${best.score.toFixed(2)}`);
    // The seed candidate keeps the signature's own instructions (no override).
    console.log(
        `instructions:\n  ${optimized.getInstructions() ?? "(unchanged: the signature's own)"}`
    );

    section('After');
    const after = await evaluate(optimized, valset, orderMetric, { concurrency: 4 });
    console.log(`score: ${after.score.toFixed(2)}`);
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});
