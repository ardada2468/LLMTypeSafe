/**
 * Streaming a whole program with streamify().
 *
 * `predict.stream()` streams one predictor. `streamify(program)` streams any
 * program — including hand-written multi-step ones — as a sequence of events:
 * `status` messages as modules and tools start and finish, `field` events with
 * each predictor's output as it is written, and a final `prediction`.
 *
 *   export OPENAI_API_KEY="sk-..."
 *   npm run example:streamify
 */
import {
    ChainOfThought,
    Module,
    Predict,
    configure,
    streamify,
    type Prediction,
} from '@ts-dspy/core';
import { OpenAILM } from '@ts-dspy/openai';
import { requireEnv, section } from './utils';

/** Outline an article, then write its opening paragraph. */
class Writer extends Module {
    outline = new ChainOfThought('topic -> outline: string[]');
    intro = new Predict('topic, outline: string[] -> paragraph');

    async forward(inputs: { topic: string }): Promise<Prediction & Record<string, any>> {
        const { outline } = await this.outline.forward(inputs);
        return this.intro.forward({ topic: inputs.topic, outline });
    }
}

async function main(): Promise<void> {
    configure({ lm: new OpenAILM({ apiKey: requireEnv('OPENAI_API_KEY') }) });

    const stream = streamify(new Writer(), {
        // Customise or silence the status messages; returning undefined skips one.
        statusMessageProvider: {
            moduleStart: (event) => `starting ${event.moduleId}`,
            lmEnd: (event) => `${event.model} replied in ${event.durationMs} ms`,
        },
    });

    section('Events');
    let lastPath = '';
    for await (const event of stream({ topic: 'Why type-safe LLM output matters' })) {
        switch (event.type) {
            case 'status':
                console.log(`\n[status] ${event.message}`);
                lastPath = '';
                break;
            case 'field':
                // Text fields grow by appending; print just the new part.
                if (event.delta !== undefined) {
                    const label = `${event.path}.${event.field}`;
                    if (label !== lastPath) process.stdout.write(`\n${label}: `);
                    process.stdout.write(event.delta);
                    lastPath = label;
                }
                break;
            case 'prediction':
                console.log(`\n\n[prediction] ${String(event.prediction.get('paragraph'))}`);
                break;
        }
    }
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});
