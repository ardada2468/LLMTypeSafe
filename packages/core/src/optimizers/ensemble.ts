import { Module } from '../core/module';
import { Prediction } from '../core/prediction';
import type { LLMCallOptions } from '../types/language-model';
import { createRng, type Rng } from './random';
import { sampleMinibatch } from './utils';

/** Combines the predictions of an ensemble's members into one. */
export type ReduceFn = (predictions: Prediction[]) => Prediction | Promise<Prediction>;

export interface MajorityOptions {
    /**
     * Vote on this one field. By default the vote is over every output field
     * together (except `reasoning`, which is never worded the same way twice),
     * so the winner is a whole, self-consistent answer.
     */
    field?: string;
    /**
     * Normalise values before comparing them: trim, collapse whitespace,
     * lower-case. Default `true`, so `"Paris"` and `" paris"` vote together.
     */
    normalize?: boolean;
}

function normalizeValue(value: unknown, normalize: boolean): string {
    const text = typeof value === 'string' ? value : JSON.stringify(value ?? null);
    return normalize ? text.trim().replace(/\s+/g, ' ').toLowerCase() : text;
}

/**
 * Majority vote: the prediction whose answer the most members agree on.
 *
 * Returns the *first* member prediction carrying the winning answer, so every
 * field of the result — reasoning included — comes from one coherent
 * completion. Ties go to the answer seen first, which keeps the result
 * deterministic for a fixed member order.
 *
 * @throws when given no predictions.
 */
export function majority(predictions: Prediction[], options: MajorityOptions = {}): Prediction {
    if (predictions.length === 0) {
        throw new Error('majority() needs at least one prediction.');
    }
    const normalize = options.normalize ?? true;
    const keyOf = (prediction: Prediction): string => {
        const data = prediction.toObject() as Record<string, unknown>;
        if (options.field !== undefined) return normalizeValue(data[options.field], normalize);
        const fields = Object.keys(data)
            .filter((name) => name !== 'reasoning')
            .sort();
        return JSON.stringify(
            fields.map((name) => [name, normalizeValue(data[name], normalize)])
        );
    };

    const counts = new Map<string, { count: number; first: Prediction }>();
    for (const prediction of predictions) {
        const key = keyOf(prediction);
        const entry = counts.get(key);
        if (entry) entry.count += 1;
        else counts.set(key, { count: 1, first: prediction });
    }
    let best: { count: number; first: Prediction } | undefined;
    for (const entry of counts.values()) {
        if (!best || entry.count > best.count) best = entry;
    }
    return best!.first;
}

export interface EnsembleOptions {
    /** Combines the members' predictions. Default: {@link majority}. */
    reduceFn?: ReduceFn;
    /**
     * Run a random sample of this many members per call instead of all of
     * them — cheaper, and still a vote. Sampled with the seeded RNG.
     */
    size?: number;
    /** Seed for sampling members. Default 0. */
    seed?: number;
    /** Members run at once per call. Default: all of them. */
    concurrency?: number;
}

/**
 * A program that runs several programs and reduces their predictions.
 *
 * Members live in the `programs` array, so `namedPredictors()` reports them as
 * `programs[0].…`, `programs[1].…`, and state saves and loads like any
 * composite. A member that throws is left out of the vote; the call fails only
 * if every member does.
 */
export class EnsembleProgram extends Module {
    readonly programs: Module[];
    private readonly reduceFn: ReduceFn;
    private readonly size?: number;
    private readonly concurrency?: number;
    private rng: Rng;

    constructor(programs: Module[], options: EnsembleOptions = {}) {
        super();
        this.programs = [...programs];
        this.reduceFn = options.reduceFn ?? ((predictions) => majority(predictions));
        this.size = options.size;
        this.concurrency = options.concurrency;
        this.rng = createRng(options.seed ?? 0);
        for (const key of ['reduceFn', 'size', 'concurrency', 'rng']) {
            Object.defineProperty(this, key, { enumerable: false });
        }
    }

    async forward(inputs: Record<string, any>, options?: LLMCallOptions): Promise<Prediction> {
        const members =
            this.size !== undefined && this.size < this.programs.length
                ? sampleMinibatch(this.programs, this.size, this.rng)
                : this.programs;

        const lanes = Math.max(1, Math.trunc(this.concurrency ?? members.length));
        const outcomes: Array<{ ok: true; value: Prediction } | { ok: false; error: unknown }> =
            [];
        for (let start = 0; start < members.length; start += lanes) {
            const batch = members.slice(start, start + lanes);
            const settled = await Promise.allSettled(
                batch.map((member) => member.forward(inputs, options))
            );
            for (const result of settled) {
                outcomes.push(
                    result.status === 'fulfilled'
                        ? { ok: true, value: result.value }
                        : { ok: false, error: result.reason }
                );
            }
        }

        const predictions = outcomes.flatMap((o) => (o.ok ? [o.value] : []));
        if (predictions.length === 0) {
            const first = outcomes.find((o) => !o.ok) as { error: unknown } | undefined;
            throw first?.error ?? new Error('Ensemble has no members to run.');
        }
        const reduced = await this.reduceFn(predictions);
        return reduced instanceof Prediction ? reduced : new Prediction(reduced);
    }
}

/**
 * DSPy's `Ensemble` optimizer: combine several programs — typically the
 * candidates of another optimizer — into one that votes.
 *
 * ```ts
 * const ensemble = new Ensemble({ size: 3 }).compile([programA, programB, programC]);
 * const prediction = await ensemble.forward({ question });
 * ```
 *
 * The members are deep copies, so the programs passed in are never modified.
 */
export class Ensemble {
    constructor(private readonly options: EnsembleOptions = {}) {
        if (
            options.size !== undefined &&
            (!Number.isInteger(options.size) || options.size < 1)
        ) {
            throw new RangeError(
                `Ensemble size must be a positive integer, got ${options.size}.`
            );
        }
    }

    compile(programs: Module[]): EnsembleProgram {
        if (programs.length === 0) throw new Error('Ensemble needs at least one program.');
        return new EnsembleProgram(
            programs.map((program) => program.deepCopy()),
            this.options
        );
    }
}
