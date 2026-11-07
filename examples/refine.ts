/**
 * BestOfN and Refine: sample until a reward function is satisfied.
 *
 * `BestOfN` runs a module up to `n` times at a sampling temperature and returns
 * the first prediction whose reward reaches `threshold` (or the best one).
 * `Refine` does the same, but after each miss asks the model for advice and
 * feeds it to the next attempt as a hint.
 *
 *   export OPENAI_API_KEY="sk-..."
 *   npm run example:refine
 */
import {
    BestOfN,
    ChainOfThought,
    Predict,
    Refine,
    configure,
    type RewardFn,
} from '@ts-dspy/core';
import { OpenAILM } from '@ts-dspy/openai';
import { requireEnv, section } from './utils';

// The constraint is easy to state and easy for a model to miss: a headline of
// at most eight words that names the product.
const MAX_WORDS = 8;

const headlineReward: RewardFn = (inputs, prediction) => {
    const headline = String(prediction.get('headline') ?? '');
    const words = headline.split(/\s+/).filter(Boolean).length;
    const namesProduct = headline.toLowerCase().includes(String(inputs.product).toLowerCase());
    return (words <= MAX_WORDS ? 0.5 : 0) + (namesProduct ? 0.5 : 0);
};

async function main(): Promise<void> {
    configure({ lm: new OpenAILM({ apiKey: requireEnv('OPENAI_API_KEY') }) });

    const inputs = {
        product: 'Lumen',
        notes: 'Lumen is a desk lamp that adjusts colour temperature to the time of day, has a USB-C port, and dims automatically when the room is bright.',
    };

    section('BestOfN');
    const bestOf = new BestOfN(new Predict('product, notes -> headline'), {
        n: 4,
        rewardFn: headlineReward,
        threshold: 1,
    });
    const sampled = await bestOf.forward(inputs);
    console.log(`headline: ${sampled.get('headline')}`);
    console.log(`reward:   ${await headlineReward(inputs, sampled)}`);

    section('Refine');
    const refine = new Refine(new ChainOfThought('product, notes -> headline'), {
        n: 3,
        rewardFn: headlineReward,
        threshold: 1,
    });
    const refined = await refine.forward(inputs);
    console.log(`headline: ${refined.get('headline')}`);
    console.log(`reward:   ${await headlineReward(inputs, refined)}`);
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});
