import { Example } from '../core/example';
import { type Prediction } from '../core/prediction';
import type { ILanguageModel, LLMCallOptions } from '../types/language-model';
import { createRng, mapWithConcurrency, shuffled } from './random';
import { type DemoModule, type Metric, type MetricResult, resolveExampleInputs } from './types';

/** One trainset example's outcome, reported as the run proceeds. */
export interface BootstrapProgressEvent {
    /** Position in the shuffled trainset. */
    index: number;
    /** Size of the trainset. */
    total: number;
    /** `passed` and `failed` mean the metric ran; `error` means the attempt threw. */
    status: 'passed' | 'failed' | 'error';
    /** The metric's verdict, when the metric ran. */
    score?: MetricResult;
    /** Whatever the attempt threw, when it threw. */
    error?: unknown;
}

export interface BootstrapFewShotOptions {
    /** Judges each attempt. Only the attempts that pass become demos. */
    metric: Metric;
    /**
     * Cap on demos promoted from successful runs. Default 4. The run stops once
     * this many are collected, so a large trainset costs no more than it has to.
     */
    maxBootstrappedDemos?: number;
    /**
     * Plain labelled examples added alongside the bootstrapped ones, capped at
     * this many. Default 0 — self-generated demos are the point, so padding with
     * labels is opt-in. Set `maxBootstrappedDemos: 0` alongside it for a
     * labels-only compile that makes no model calls at all.
     */
    maxLabeledDemos?: number;
    /**
     * Trainset examples attempted at once, and the size of the batch the run
     * stops between. Default 4.
     */
    concurrency?: number;
    /** Seed for the trainset shuffle. The same seed always yields the same demos. */
    seed?: number;
    /**
     * Score at which a numeric metric counts as a pass. Default 0.5, which
     * treats an exact-match 0/1 metric the obvious way while still giving a
     * partial-credit metric a sensible midpoint. Booleans ignore this.
     */
    threshold?: number;
    /**
     * A stronger model used to generate the demos. The compiled module is still
     * the student, on the student's own model — it just imitates work the
     * teacher did. This is the technique that makes bootstrapping worth the
     * trouble: you pay for the strong model once, at compile time.
     */
    teacher?: ILanguageModel;
    /** Input field names, when the trainset examples do not declare their own. */
    inputKeys?: string[];
    /**
     * Called once per attempted trainset example. Events arrive in completion
     * order, which with concurrency above 1 is not trainset order; the demos
     * themselves are deterministic regardless. Fewer than `total` events arrive
     * when the run collects enough demos and stops early. Anything this callback
     * throws is swallowed — reporting progress must not lose a compile.
     */
    onProgress?: (event: BootstrapProgressEvent) => void;
    /** Call options for the trainset runs, e.g. a temperature of 0. */
    callOptions?: LLMCallOptions;
}

export interface BootstrapFewShotConfig {
    trainset: Example[];
}

/** A successful attempt, kept with its position so ordering stays deterministic. */
interface Candidate {
    index: number;
    demo: Example;
}

/**
 * Teach a module from its own successes.
 *
 * Runs the student over a labelled trainset, scores each attempt with a metric,
 * and promotes the runs that passed into demos on the returned module. The
 * program improves from data rather than from prompt edits, which is the whole
 * idea it inherits from DSPy.
 *
 * ```ts
 * const compiled = await new BootstrapFewShot({
 *     metric: (example, prediction) => example.get('answer') === prediction.get('answer'),
 *     maxBootstrappedDemos: 3,
 *     teacher: strongLM,
 *     seed: 42,
 * }).compile(new Predict(QA, cheapLM), { trainset });
 * ```
 *
 * A trainset example whose attempt throws — a provider error, a reply that fails
 * validation — is skipped, not fatal: one bad row must not throw away every demo
 * already paid for.
 */
export class BootstrapFewShot {
    private readonly metric: Metric;
    private readonly maxBootstrappedDemos: number;
    private readonly maxLabeledDemos: number;
    private readonly concurrency: number;
    private readonly seed: number;
    private readonly threshold: number;
    private readonly teacher?: ILanguageModel;
    private readonly inputKeys?: string[];
    private readonly onProgress?: (event: BootstrapProgressEvent) => void;
    private readonly callOptions?: LLMCallOptions;

    constructor(options: BootstrapFewShotOptions) {
        if (typeof options?.metric !== 'function') {
            throw new Error('BootstrapFewShot requires a metric function.');
        }

        this.metric = options.metric;
        this.maxBootstrappedDemos = options.maxBootstrappedDemos ?? 4;
        this.maxLabeledDemos = options.maxLabeledDemos ?? 0;
        this.concurrency = options.concurrency ?? 4;
        this.seed = options.seed ?? 0;
        this.threshold = options.threshold ?? 0.5;
        this.teacher = options.teacher;
        this.inputKeys = options.inputKeys;
        this.onProgress = options.onProgress;
        this.callOptions = options.callOptions;
    }

    /** Run the trainset and return a copy of `student` carrying the demos it earned. */
    async compile<M extends DemoModule>(
        student: M,
        config: BootstrapFewShotConfig
    ): Promise<M> {
        const ordered = shuffled(config.trainset, createRng(this.seed));

        // Resolve every input split up front. A trainset that never declared its
        // inputs is a configuration mistake, and it should say so once rather
        // than look like every single example happened to fail.
        const inputsByIndex = ordered.map((example, index) =>
            resolveExampleInputs(example, this.inputKeys, index)
        );

        const runner = this.teacherModule(student);
        const total = ordered.length;
        const batchSize = Math.max(1, Math.trunc(this.concurrency) || 1);

        // Run in batches and stop once enough demos are in hand. A 500-row
        // trainset should not cost 500 teacher calls to keep four demos. The
        // batch boundary is what keeps this deterministic: every index below the
        // boundary has finished, so the successes collected so far really are
        // the first ones in trainset order.
        const bootstrapped: Candidate[] = [];
        for (let start = 0; start < ordered.length; start += batchSize) {
            if (bootstrapped.length >= this.maxBootstrappedDemos) {
                break;
            }

            const batch = ordered.slice(start, start + batchSize);
            const results = await mapWithConcurrency<Example, Candidate | null>(
                batch,
                this.concurrency,
                async (example, offset) => {
                    const index = start + offset;
                    try {
                        const inputs = inputsByIndex[index];
                        const prediction = await runner.forward(inputs, this.callOptions);
                        const score = await this.metric(example, prediction);

                        if (!this.passes(score)) {
                            this.report({ index, total, status: 'failed', score });
                            return null;
                        }

                        this.report({ index, total, status: 'passed', score });
                        return { index, demo: this.toDemo(inputs, prediction) };
                    } catch (error) {
                        this.report({ index, total, status: 'error', error });
                        return null;
                    }
                }
            );

            for (const result of results) {
                if (result && bootstrapped.length < this.maxBootstrappedDemos) {
                    bootstrapped.push(result);
                }
            }
        }

        return student.withDemos([
            ...bootstrapped.map((c) => c.demo),
            ...this.labeledDemos(ordered, bootstrapped),
        ]);
    }

    /**
     * Plain labelled examples to show alongside the bootstrapped ones. Examples
     * already used as a bootstrapped demo are excluded, so the model does not
     * see the same item twice.
     */
    private labeledDemos(ordered: Example[], bootstrapped: Candidate[]): Example[] {
        if (this.maxLabeledDemos <= 0) {
            return [];
        }

        const used = new Set(bootstrapped.map((c) => c.index));
        return ordered.filter((_, index) => !used.has(index)).slice(0, this.maxLabeledDemos);
    }

    /** The module that generates the demos: the teacher's, when one was given. */
    private teacherModule<M extends DemoModule>(student: M): M {
        if (!this.teacher) {
            return student;
        }
        if (typeof student.withLM !== 'function') {
            throw new Error(
                'A teacher model was supplied, but this module has no withLM() method. ' +
                    'Implement withLM(lm) on the module, or drop the teacher option.'
            );
        }
        return student.withLM(this.teacher);
    }

    private passes(score: MetricResult): boolean {
        if (typeof score === 'boolean') {
            return score;
        }
        // NaN fails this comparison, which is the right answer for a metric that
        // could not produce a number.
        return score >= this.threshold;
    }

    /**
     * Turn a successful run into a demo: the example's inputs, paired with what
     * the model actually produced. The prediction is used rather than the label
     * so the demo shows a complete, self-consistent piece of work in the model's
     * own voice — including any reasoning field the label never had.
     */
    private toDemo(inputs: Record<string, any>, prediction: Prediction): Example {
        const inputKeys = Object.keys(inputs);
        return new Example({ ...inputs, ...prediction.toObject() }).withInputs(...inputKeys);
    }

    /**
     * A progress callback is an observer, not a participant. If it throws, the
     * throw is dropped: letting it escape would be caught as an attempt failure,
     * and a callback that throws every time would abandon the whole compile.
     */
    private report(event: BootstrapProgressEvent): void {
        if (!this.onProgress) {
            return;
        }
        try {
            this.onProgress(event);
        } catch {
            // Deliberately ignored.
        }
    }
}
