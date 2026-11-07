import { type Example } from '../core/example';
import { Module } from '../core/module';
import { type Prediction } from '../core/prediction';
import { cosineSimilarity, type Embedder } from '../core/embedder';
import type { LLMCallOptions } from '../types/language-model';
import { BootstrapFewShot, type BootstrapFewShotOptions } from './bootstrap-few-shot';
import { resolveExampleInputs } from './types';
import { withDemosByPath } from './utils';

export interface KNNFewShotOptions {
    /** Demos per call: the `k` training examples nearest the call's inputs. */
    k: number;
    /** Embeds the training examples' inputs and each call's inputs. */
    embedder: Embedder;
    /** The pool demos are drawn from. Each example must declare its inputs. */
    trainset: Example[];
    /**
     * Turn inputs into the text that gets embedded. Defaults to one
     * `field: value` line per input, in key order.
     */
    vectorizeInputs?: (inputs: Record<string, any>) => string;
    /**
     * Run {@link BootstrapFewShot} over the `k` neighbours at call time, as
     * DSPy's `KNNFewShot` does, so the demos show the program's own successful
     * work rather than raw labels. Costs extra model calls on every call;
     * omit it to use the neighbours as labelled demos directly.
     */
    bootstrap?: Omit<BootstrapFewShotOptions, 'onProgress'>;
    /** Input field names, when the trainset examples do not declare their own. */
    inputKeys?: string[];
}

/** One neighbour picked for a call. */
export interface KNNNeighbour {
    example: Example;
    similarity: number;
}

/** Default text for embedding a set of inputs: `field: value` lines, keys sorted. */
export function defaultVectorizeInputs(inputs: Record<string, any>): string {
    return Object.keys(inputs)
        .sort()
        .map((key) => {
            const value = inputs[key];
            return `${key}: ${typeof value === 'string' ? value : JSON.stringify(value)}`;
        })
        .join('\n');
}

/**
 * A program that picks its demos per call: before each `forward`, the `k`
 * training examples nearest the inputs become every predictor's demos.
 *
 * The wrapped program is held as the `program` field, so `namedPredictors()`
 * reports its predictors under `program.…` and saving/loading state works as
 * for any composite. The training vectors are computed once, on first use,
 * and kept off the enumerable fields.
 */
export class KNNProgram<M extends Module = Module> extends Module {
    readonly program: M;

    constructor(
        program: M,
        private readonly knn: KNNFewShotOptions
    ) {
        super();
        this.program = program;
        Object.defineProperty(this, 'knn', { enumerable: false });
        Object.defineProperty(this, 'vectors', {
            value: undefined,
            writable: true,
            enumerable: false,
        });
    }

    private vectors?: Promise<number[][]>;

    /** The `k` training examples nearest `inputs`, most similar first. */
    async neighbours(inputs: Record<string, any>): Promise<KNNNeighbour[]> {
        const { trainset, k, embedder } = this.knn;
        if (k <= 0 || trainset.length === 0) return [];

        const vectorize = this.knn.vectorizeInputs ?? defaultVectorizeInputs;
        this.vectors ??= embedder.embed(
            trainset.map((example, index) =>
                vectorize(resolveExampleInputs(example, this.knn.inputKeys, index))
            )
        );
        let train: number[][];
        try {
            train = await this.vectors;
        } catch (error) {
            // Let the next call retry rather than caching a failure forever.
            this.vectors = undefined;
            throw error;
        }
        const [query] = await embedder.embed([vectorize(inputs)]);

        return train
            .map((vector, index) => ({
                example: trainset[index],
                similarity: cosineSimilarity(query, vector),
                index,
            }))
            .sort((a, b) => b.similarity - a.similarity || a.index - b.index)
            .slice(0, k)
            .map(({ example, similarity }) => ({ example, similarity }));
    }

    async forward(inputs: Record<string, any>, options?: LLMCallOptions): Promise<Prediction> {
        const examples = (await this.neighbours(inputs)).map((n) => n.example);
        let configured: M;
        if (this.knn.bootstrap) {
            configured = (await new BootstrapFewShot({
                inputKeys: this.knn.inputKeys,
                ...this.knn.bootstrap,
            }).compile(this.program, { trainset: examples })) as M;
        } else {
            configured = withDemosByPath(this.program, {}, examples);
        }
        return configured.forward(inputs, options);
    }
}

/**
 * DSPy's `KNNFewShot`: demos chosen per input, by similarity.
 *
 * Instead of one fixed demo set, each call embeds its inputs and shows the
 * model the `k` training examples nearest to them — the most relevant worked
 * examples for this particular question. Optionally bootstraps those
 * neighbours first (see {@link KNNFewShotOptions.bootstrap}).
 *
 * ```ts
 * const knn = new KNNFewShot({ k: 3, embedder, trainset });
 * const program = knn.compile(new Predict(QA));
 * await program.forward({ question }); // demos: the 3 nearest training questions
 * ```
 *
 * `compile` returns a {@link KNNProgram} wrapping a copy of the student; the
 * student is not modified.
 */
export class KNNFewShot {
    private readonly options: KNNFewShotOptions;

    constructor(options: KNNFewShotOptions) {
        if (!options?.embedder || typeof options.embedder.embed !== 'function') {
            throw new Error('KNNFewShot requires an embedder.');
        }
        if (!Number.isInteger(options.k) || options.k < 0) {
            throw new RangeError(
                `KNNFewShot k must be a non-negative integer, got ${options.k}.`
            );
        }
        this.options = { ...options, trainset: [...(options.trainset ?? [])] };
    }

    /** Wrap a copy of `student` so each call picks its own demos. */
    compile<M extends Module>(student: M): KNNProgram<M> {
        return new KNNProgram(student.deepCopy(), this.options);
    }
}
