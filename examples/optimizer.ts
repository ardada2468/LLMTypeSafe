/**
 * Few-shot demos and optimizers.
 *
 * A program that improves itself from data: run a labelled trainset, keep the
 * runs a metric approved of, and put those runs in the prompt as worked
 * examples. Optionally let a stronger model generate the demos that a cheaper
 * one then imitates.
 *
 *   export OPENAI_API_KEY="sk-..."
 *   npm run example:optimizer
 */
import {
    Signature,
    InputField,
    OutputField,
    Predict,
    Example,
    LabeledFewShot,
    BootstrapFewShot,
    configure,
    type Metric,
} from '@ts-dspy/core';
import { OpenAILM } from '@ts-dspy/openai';
import { requireEnv, section } from './utils';

// --- Signature --------------------------------------------------------------

class RouteTicket extends Signature {
    static description =
        'Route a support ticket to the team that owns it. ' +
        'Answer with exactly one of: billing, bug, account, feature.';

    @InputField({ description: 'the ticket text' })
    ticket!: string;

    @OutputField({ description: 'billing, bug, account, or feature' })
    team!: string;
}

type TicketRouting = { team: string };

// --- Trainset ---------------------------------------------------------------

// Labelled data. `withInputs` marks which fields are the question, so the rest
// is understood to be the answer — an optimizer must never feed the label back
// in as input, or every run would be trivially correct.
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
].map(([ticket, team]) => new Example({ ticket, team }).withInputs('ticket'));

// Disjoint splits. Scoring on rows the optimizer compiled from would let a demo
// for a ticket appear in the prompt used to classify that same ticket, and the
// comparison against the baseline would mean nothing.
const trainset = labelled.slice(0, 8);
const devset = labelled.slice(8);

// --- Metric -----------------------------------------------------------------

// The metric is the whole specification of "good" — everything the optimizer
// does is downstream of it. This one is exact match on a normalised label.
const routedCorrectly: Metric = (example, prediction) => {
    const expected = String(example.get('team')).trim().toLowerCase();
    const actual = String(prediction.get('team') ?? '')
        .trim()
        .toLowerCase();
    return actual === expected;
};

async function accuracy(module: Predict<typeof RouteTicket, TicketRouting>): Promise<number> {
    let correct = 0;
    for (const example of devset) {
        try {
            const prediction = await module.forward(example.getInputs());
            if (routedCorrectly(example, prediction)) {
                correct += 1;
            }
        } catch {
            // A run that fails is a run that scored nothing.
        }
    }
    return correct / devset.length;
}

async function main(): Promise<void> {
    const apiKey = requireEnv('OPENAI_API_KEY');

    // A cheap student, and a stronger teacher used only at compile time.
    const student = new OpenAILM({ apiKey, model: 'gpt-4.1-mini' });
    const teacher = new OpenAILM({ apiKey, model: 'gpt-4.1' });
    configure({ lm: student });

    // --- Baseline -----------------------------------------------------------
    section('Baseline (no demos)');

    const baseline = new Predict<typeof RouteTicket, TicketRouting>(RouteTicket);
    console.log(`accuracy: ${(await accuracy(baseline)) * 100}%`);

    // --- LabeledFewShot -----------------------------------------------------
    section('LabeledFewShot');

    // No model calls at all: it just puts k of your labels in the prompt. Seeded,
    // so the same seed always picks the same demos.
    const labeled = new LabeledFewShot({ k: 3, seed: 42 }).compile(baseline, { trainset });

    for (const demo of labeled.getDemos()) {
        console.log(`  demo: ${demo.get('ticket')} -> ${demo.get('team')}`);
    }
    console.log(`accuracy: ${(await accuracy(labeled)) * 100}%`);

    // --- BootstrapFewShot ---------------------------------------------------
    section('BootstrapFewShot');

    // Run the trainset through the teacher, keep the runs the metric approved
    // of, and attach them to the student. The student ends up imitating work the
    // stronger model did, without paying for the stronger model at run time.
    const optimizer = new BootstrapFewShot({
        metric: routedCorrectly,
        maxBootstrappedDemos: 4,
        concurrency: 4,
        seed: 42,
        teacher,
        callOptions: { temperature: 0 },
        // Library code never prints; progress arrives through this callback, and
        // the example is what decides to put it on the terminal.
        onProgress: (event) => {
            const detail = event.status === 'error' ? ` (${String(event.error)})` : '';
            console.log(`  [${event.index + 1}/${event.total}] ${event.status}${detail}`);
        },
    });

    const compiled = await optimizer.compile(baseline, { trainset });

    console.log(`\nbootstrapped ${compiled.getDemos().length} demos:`);
    for (const demo of compiled.getDemos()) {
        console.log(`  ${demo.get('ticket')} -> ${demo.get('team')}`);
    }

    console.log(`\naccuracy: ${(await accuracy(compiled)) * 100}%`);

    // --- Using the compiled program -----------------------------------------
    section('Compiled program');

    // Demos are not magic: they are text placed in front of the real input, in
    // whichever shape the reply is expected to take — labelled `field: value`
    // for a text provider, JSON for one with native structured output.
    const routed = await compiled.forward({ ticket: 'Two invoices arrived this month.' });
    console.log(`team: ${routed.team}`);

    // --- Usage --------------------------------------------------------------
    section('Usage');
    const teacherUsage = teacher.getUsage();
    const studentUsage = student.getUsage();
    console.log(`teacher: ${teacherUsage.requestCount} requests (compile time only)`);
    console.log(`student: ${studentUsage.requestCount} requests`);
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});
