import { Module } from '../core/module';
import { type Prediction } from '../core/prediction';
import { TsDspyError } from '../core/errors';
import { getRetriever } from '../core/config';
import { type Passage, type Retriever } from '../core/retrieve';
import type { LLMCallOptions } from '../types/language-model';

export interface RetrieveOptions {
    /** Passages to return. Defaults to 3, as in DSPy. */
    k?: number;
    /**
     * The retriever to search. Defaults to the one configured with
     * `configure({ retriever })` or `settings.context({ retriever }, …)`,
     * resolved at call time.
     */
    retriever?: Retriever;
}

/** What {@link Retrieve} produces. */
export interface RetrieveOutput {
    /** The passages found, most relevant first. */
    passages: Passage[];
}

/**
 * Retrieval as a module, so it composes with predictors in a program and
 * shows up in traces and callbacks like any other step — DSPy's `Retrieve`.
 *
 * ```ts
 * class RAG extends Module {
 *     retrieve = new Retrieve({ k: 3 });
 *     answer = new ChainOfThought('context, question -> answer');
 *
 *     async forward({ question }: { question: string }) {
 *         const { passages } = await this.retrieve.forward({ query: question });
 *         return this.answer.forward({
 *             context: passages.map((p) => p.text).join('\n'),
 *             question,
 *         });
 *     }
 * }
 * ```
 *
 * Unlike DSPy, which returns bare strings, each passage keeps its `score` and
 * `metadata`, so a citation or source link survives the trip. `k` may be
 * overridden per call with `inputs.k`.
 */
export class Retrieve extends Module {
    readonly k: number;
    private readonly retriever?: Retriever;

    constructor(options: RetrieveOptions = {}) {
        super();
        const k = options.k ?? 3;
        if (!Number.isInteger(k) || k < 1) {
            throw new RangeError(`k must be a positive integer, received ${String(k)}`);
        }
        this.k = k;
        this.retriever = options.retriever;
    }

    async forward(
        inputs: { query: string; k?: number },
        options?: LLMCallOptions
    ): Promise<Prediction<RetrieveOutput> & RetrieveOutput> {
        const retriever = this.retriever ?? getRetriever();
        if (!retriever) {
            throw new TsDspyError(
                'No retriever configured. Pass one to Retrieve, call configure({ retriever }), ' +
                    'or run inside settings.context({ retriever }, ...).'
            );
        }
        if (typeof inputs?.query !== 'string') {
            throw new TsDspyError('Retrieve expects a string `query` input.');
        }
        const k = inputs.k ?? this.k;
        const prediction = await this.traced<RetrieveOutput>(inputs, async () => ({
            passages: await retriever.retrieve(inputs.query, k, { signal: options?.signal }),
        }));
        return prediction as Prediction<RetrieveOutput> & RetrieveOutput;
    }
}
