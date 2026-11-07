/**
 * MIPROv2: search instructions and demos together.
 *
 * MIPROv2 bootstraps several demo sets, asks a prompt model to propose
 * candidate instructions grounded in the data and the program, then runs a
 * seeded Bayesian (TPE) search over the combinations, scoring each on a
 * minibatch of the validation set. The student is never modified; you get a
 * compiled copy back, plus a record of every trial.
 *
 *   export OPENAI_API_KEY="sk-..."
 *   npm run example:mipro
 */
import {
    ChainOfThought,
    Example,
    MIPROv2,
    configure,
    evaluate,
    signature,
    type Metric,
} from '@ts-dspy/core';
import { OpenAILM } from '@ts-dspy/openai';
import { requireEnv, section } from './utils';

const TEAMS = ['billing', 'bug', 'account', 'feature'] as const;

const labelled = [
    ['My card was charged twice this month.', 'billing'],
    ['The export button does nothing on Safari.', 'bug'],
    ['I cannot reset my password, the email never arrives.', 'account'],
    ['Could you add dark mode to the dashboard?', 'feature'],
    ['Why is my invoice higher than the quoted plan price?', 'billing'],
    ['The app crashes when I upload a file over 10 MB.', 'bug'],
    ['Please remove my colleague from the workspace.', 'account'],
    ['It would help if reports could be scheduled weekly.', 'feature'],
    ['I was billed after cancelling my subscription.', 'billing'],
    ['Search returns no results even for exact titles.', 'bug'],
    ['Can I merge two accounts into one?', 'account'],
    ['A Slack integration would save us a lot of copying.', 'feature'],
    ['We were charged in USD but our contract is in EUR.', 'billing'],
    ['Two-factor codes are rejected as expired immediately.', 'account'],
    ['Charts render blank after the latest update.', 'bug'],
    ['Please support exporting to Parquet.', 'feature'],
].map(([ticket, team]) => new Example({ ticket, team }).withInputs('ticket'));

// Disjoint splits: candidates are scored on rows they did not learn from.
const trainset = labelled.slice(0, 8);
const valset = labelled.slice(8);

const routedCorrectly: Metric = (example, prediction) =>
    String(prediction.get('team') ?? '')
        .trim()
        .toLowerCase() === example.get('team');

async function main(): Promise<void> {
    const apiKey = requireEnv('OPENAI_API_KEY');
    configure({ lm: new OpenAILM({ apiKey, model: 'gpt-4.1-mini' }) });

    const student = new ChainOfThought(
        signature(
            `ticket -> team: enum(${TEAMS.join('|')})`,
            'Route the support ticket to the team that owns it.'
        )
    );

    section('Baseline');
    const before = await evaluate(student, valset, routedCorrectly, { concurrency: 4 });
    console.log(`accuracy: ${(before.score * 100).toFixed(0)}%`);

    section('MIPROv2');
    const optimizer = new MIPROv2({
        metric: routedCorrectly,
        // A stronger model writes the candidate instructions; the configured
        // model runs the program during the search.
        promptModel: new OpenAILM({ apiKey, model: 'gpt-4.1' }),
        // A small explicit budget for the example. In real use, start with
        // `auto: 'light'` (the default) and move to 'medium' / 'heavy'.
        auto: null,
        numCandidates: 3,
        numTrials: 6,
        minibatchSize: 4,
        maxBootstrappedDemos: 2,
        maxLabeledDemos: 2,
        seed: 7,
        onProgress: (event) => {
            const score = event.score === undefined ? '' : ` score=${event.score.toFixed(2)}`;
            console.log(`  [${event.phase}] ${event.message ?? ''}${score}`);
        },
    });

    const compiled = await optimizer.compile(student, { trainset, valset });
    const results = compiled.miproResults;

    console.log(
        `\nbest score: ${results.bestScore.toFixed(2)} after ${results.numTrials} trials`
    );
    for (const [path, choice] of Object.entries(results.best)) {
        console.log(
            `  ${path}: instruction #${choice.instructionIndex}, demo set #${choice.demoSetIndex}`
        );
        console.log(`    "${choice.instruction}"`);
    }

    section('After');
    const after = await evaluate(compiled, valset, routedCorrectly, { concurrency: 4 });
    console.log(`accuracy: ${(after.score * 100).toFixed(0)}%`);

    // The compiled program is an ordinary module: save its learned state with
    // `await compiled.save('router.json')` and load it into a fresh
    // ChainOfThought of the same signature later.
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});
