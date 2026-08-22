import { type Example } from '../core/example';
import { createRng, shuffled } from './random';
import { type DemoModule } from './types';

export interface LabeledFewShotOptions {
    /** How many demos to select. Default 4. */
    k?: number;
    /** Seed for the selection shuffle. The same seed always selects the same demos. */
    seed?: number;
}

/** What a compile call needs: the labelled data to select from. */
export interface LabeledFewShotConfig {
    trainset: Example[];
}

/**
 * The simplest optimizer: put *k* of your labelled examples in the prompt.
 *
 * No model calls, so compiling is free and instant. It is often a surprisingly
 * strong baseline, and it is the thing to try before reaching for
 * {@link BootstrapFewShot} — if hand-labelled demos already get you where you
 * need to be, there is nothing to bootstrap.
 *
 * ```ts
 * const trainset = [
 *     new Example({ question: 'Capital of France?', answer: 'Paris' }).withInputs('question'),
 *     new Example({ question: 'Capital of Japan?', answer: 'Tokyo' }).withInputs('question'),
 * ];
 *
 * const compiled = new LabeledFewShot({ k: 2, seed: 7 }).compile(new Predict(QA), { trainset });
 * ```
 */
export class LabeledFewShot {
    private readonly k: number;
    private readonly seed: number;

    constructor(options: LabeledFewShotOptions = {}) {
        this.k = options.k ?? 4;
        this.seed = options.seed ?? 0;
    }

    /**
     * Select demos and return a configured copy of `student`.
     *
     * Synchronous: selection touches no model, and making callers `await` a
     * pure array shuffle would only obscure that.
     */
    compile<M extends DemoModule>(student: M, config: LabeledFewShotConfig): M {
        return student.withDemos(this.select(config.trainset));
    }

    /**
     * The demos this optimizer would select, without compiling anything.
     * Exposed so a caller can inspect or diff a selection.
     */
    select(trainset: Example[]): Example[] {
        if (this.k <= 0 || trainset.length === 0) {
            return [];
        }
        return shuffled(trainset, createRng(this.seed)).slice(0, this.k);
    }
}
