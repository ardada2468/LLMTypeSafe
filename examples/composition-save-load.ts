/**
 * Composition, scoped settings, and saving a program's learned state.
 *
 * A program is an ordinary class that holds modules as fields. ts-dspy finds
 * them by walking those fields, so every predictor inside has a stable path —
 * which is what optimizers tune and what `save()`/`load()` persist.
 *
 *   export OPENAI_API_KEY="sk-..."
 *   npm run example:save-load
 */
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
    ChainOfThought,
    Example,
    Module,
    Predict,
    configure,
    settings,
    type Prediction,
} from '@ts-dspy/core';
import { OpenAILM } from '@ts-dspy/openai';
import { requireEnv, section } from './utils';

/** Draft an answer, then tighten it. Two predictors, one program. */
class DraftAndPolish extends Module {
    draft = new ChainOfThought('question -> answer');
    polish = new Predict('question, draft -> answer: string, confidence: float');

    async forward(inputs: { question: string }): Promise<Prediction & Record<string, any>> {
        const { answer } = await this.draft.forward(inputs);
        return this.polish.forward({ question: inputs.question, draft: answer });
    }
}

async function main(): Promise<void> {
    const apiKey = requireEnv('OPENAI_API_KEY');

    // Modules resolve their model when they run, not when they are built, so
    // constructing the program before configure() is fine.
    const program = new DraftAndPolish();
    configure({ lm: new OpenAILM({ apiKey }), trackUsage: true });

    section('Predictor paths');
    for (const [path, predictor] of program.namedPredictors()) {
        console.log(`  ${path.padEnd(8)} ${predictor.moduleId}`);
    }

    // --- Teach it something, by hand ----------------------------------------
    // An optimizer would normally do this; here we set an instruction and a
    // demo directly so there is learned state to save.
    program.polish.setInstructions(
        'Rewrite the draft as one short, direct sentence. Give a calibrated confidence.'
    );
    program.polish.setDemos([
        new Example({
            question: 'What is the tallest mountain on Earth?',
            draft: 'The tallest mountain above sea level is Mount Everest, in the Himalayas.',
            answer: 'Mount Everest.',
            confidence: 0.98,
        }).withInputs('question', 'draft'),
    ]);

    section('Run');
    const result = await program.forward({ question: 'Which planet has the most moons?' });
    console.log(`answer:     ${result.answer}`);
    console.log(`confidence: ${result.confidence}`);
    // Exact per-invocation usage, keyed by provider/model.
    console.log('usage:', result.getLMUsage());

    // --- Save and load ------------------------------------------------------
    section('Save and load');
    const dir = await mkdtemp(join(tmpdir(), 'ts-dspy-'));
    const file = join(dir, 'program.json');
    try {
        await program.save(file);
        const saved = JSON.parse(await readFile(file, 'utf8'));
        console.log(`saved predictors: ${Object.keys(saved).join(', ')}`);

        // A freshly built program of the same shape picks up the same state.
        const restored = await new DraftAndPolish().load(file);
        console.log(`restored instructions: ${restored.polish.getInstructions()}`);
        console.log(`restored demos:        ${restored.polish.getDemos().length}`);
    } finally {
        await rm(dir, { recursive: true, force: true });
    }

    // --- Scoped settings ----------------------------------------------------
    section('settings.context');
    // Override the model (or adapter, callbacks, cache...) for one async scope
    // only; concurrent requests outside it are unaffected.
    const cheap = new OpenAILM({ apiKey, model: 'gpt-4.1-mini' });
    const scoped = await settings.context({ lm: cheap }, () =>
        program.forward({ question: 'What is the chemical symbol for gold?' })
    );
    console.log(`answer (gpt-4.1-mini): ${scoped.answer}`);
    console.log(`models used:           ${Object.keys(scoped.getLMUsage()).join(', ')}`);
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});
