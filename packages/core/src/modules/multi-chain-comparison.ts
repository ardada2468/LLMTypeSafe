import { Module } from '../core/module';
import { Prediction } from '../core/prediction';
import { TsDspyError } from '../core/errors';
import { extendSignature, type SignatureLike } from '../core/signature';
import { getOutputFieldConfigs } from '../utils/schema';
import { Predict, type PredictOptions } from './predict';
import type { ILanguageModel, LLMCallOptions } from '../types/language-model';

export interface MultiChainComparisonOptions extends PredictOptions {
    /** How many attempts are compared. Defaults to 3, as in DSPy. */
    m?: number;
    /**
     * Sampling temperature for the comparison call, beneath anything the call
     * itself passes. Defaults to 0.7, as in DSPy.
     */
    temperature?: number;
}

/** One attempt to compare: a prediction, or its fields as a plain object. */
export type Completion = Prediction | Record<string, any>;

/**
 * Compare several reasoning attempts and produce a final, corrected answer —
 * DSPy's `MultiChainComparison`.
 *
 * The signature gains `reasoning_attempt_1` … `reasoning_attempt_m` inputs and
 * a leading `rationale` output. Pass the `m` attempts — typically
 * `ChainOfThought` predictions sampled at a temperature — as `completions`,
 * alongside the signature's own inputs:
 *
 * ```ts
 * const cot = new ChainOfThought('question -> answer');
 * const attempts = await Promise.all(
 *     [0, 1, 2].map(() => cot.forward({ question }, { temperature: 1 }))
 * );
 * const final = await new MultiChainComparison('question -> answer').forward({
 *     question,
 *     completions: attempts,
 * });
 * final.rationale; final.answer;
 * ```
 *
 * Each attempt is summarised the way DSPy does it: the first line of its
 * `rationale` (or `reasoning`) and the first line of its last output field,
 * as «I'm trying to … I'm not sure but my prediction is …».
 */
export class MultiChainComparison extends Module {
    /** The comparison predictor, over the extended signature. */
    readonly predict: Predict;
    readonly m: number;
    private readonly lastKey: string;
    private readonly temperature: number;

    constructor(
        signature: SignatureLike,
        options: MultiChainComparisonOptions | ILanguageModel = {}
    ) {
        super();
        const {
            m = 3,
            temperature = 0.7,
            ...predictOptions
        } = isLM(options) ? { lm: options } : options;
        if (!Number.isInteger(m) || m < 1) {
            throw new RangeError(`m must be a positive integer, received ${String(m)}`);
        }

        const outputs = Object.keys(getOutputFieldConfigs(signature));
        const lastKey = outputs.at(-1);
        if (lastKey === undefined) {
            throw new TsDspyError('MultiChainComparison needs a signature with an output.');
        }

        const attempts = Object.fromEntries(
            Array.from({ length: m }, (_, index) => [
                `reasoning_attempt_${index + 1}`,
                {
                    description: `Student Attempt #${index + 1}`,
                    prefix: `Student Attempt #${index + 1}:`,
                },
            ])
        );
        const extended = extendSignature(signature, {
            appendInputs: attempts,
            prependOutputs: {
                rationale: {
                    description:
                        "Accurate reasoning: thank you everyone. Let's step back and holistically " +
                        'arrive at the corrected reasoning',
                    prefix: "Accurate Reasoning: Thank you everyone. Let's step back and holistically",
                },
            },
        });

        this.m = m;
        this.lastKey = lastKey;
        this.temperature = temperature;
        this.predict = new Predict(extended, predictOptions);
    }

    async forward(
        inputs: Record<string, any> & { completions: Completion[] },
        options?: LLMCallOptions
    ): Promise<Prediction & Record<string, any>> {
        const { completions, ...rest } = inputs;
        if (!Array.isArray(completions)) {
            throw new TsDspyError(
                'MultiChainComparison expects `completions`: an array of attempts.'
            );
        }
        if (completions.length !== this.m) {
            throw new TsDspyError(
                `MultiChainComparison was built to compare ${this.m} attempts but received ` +
                    `${completions.length}.`
            );
        }

        const attempts = Object.fromEntries(
            completions.map((completion, index) => [
                `reasoning_attempt_${index + 1}`,
                this.summarise(completion),
            ])
        );
        return this.predict.forward(
            { ...attempts, ...rest },
            { ...options, temperature: options?.temperature ?? this.temperature }
        );
    }

    /** One attempt as DSPy phrases it for the comparison prompt. */
    private summarise(completion: Completion): string {
        const fields =
            completion instanceof Prediction
                ? (completion.toObject() as Record<string, unknown>)
                : completion;
        const rationale = firstLine(fields.rationale ?? fields.reasoning ?? '');
        const answer = firstLine(fields[this.lastKey] ?? '');
        return `«I'm trying to ${rationale} I'm not sure but my prediction is ${answer}»`;
    }
}

function firstLine(value: unknown): string {
    const text = typeof value === 'string' ? value : (JSON.stringify(value) ?? String(value));
    return text.trim().split('\n')[0]!.trim();
}

function isLM(value: unknown): value is ILanguageModel {
    return (
        typeof value === 'object' &&
        value !== null &&
        typeof (value as ILanguageModel).chat === 'function' &&
        typeof (value as ILanguageModel).generate === 'function'
    );
}
