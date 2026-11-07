import { type Example } from '../core/example';
import { type Prediction } from '../core/prediction';
import type { ILanguageModel, LLMCallOptions } from '../types/language-model';

// One metric type serves both `evaluate` and the optimizers; it lives with
// `evaluate` and is re-exported here for the optimizers' convenience.
export type { Metric, MetricResult } from '../evaluate/types';

/**
 * The slice of a module an optimizer needs: run it, and produce a configured
 * copy of it.
 *
 * Deliberately structural rather than `Predict`-typed, so a hand-written module
 * can be optimized as long as it can run and can accept demos. `withLM` is
 * optional and only required when a teacher model is supplied.
 */
export interface DemoModule {
    forward(inputs: Record<string, any>, options?: LLMCallOptions): Promise<Prediction>;
    withDemos(demos: Example[]): this;
    withLM?(lm: ILanguageModel): this;
}

/**
 * Resolve the input half of a labelled example.
 *
 * Optimizers have to call the module with inputs alone — feeding the labels back
 * in would make every run trivially correct — so the split has to be known.
 * `withInputs()` on the example is the usual way to declare it; `inputKeys` on
 * the optimizer covers a trainset built without it.
 */
export function resolveExampleInputs(
    example: Example,
    inputKeys: string[] | undefined,
    position: number
): Record<string, any> {
    if (inputKeys && inputKeys.length > 0) {
        const data = example.toObject();
        const picked: Record<string, any> = {};
        for (const key of inputKeys) {
            // A typo here would call the module with no input at all: the prompt
            // builder skips undefined values, the model would answer noise, the
            // metric would reject every row, and the whole trainset would be
            // paid for to produce nothing. Say so instead.
            if (!Object.prototype.hasOwnProperty.call(data, key)) {
                throw new Error(
                    `Trainset example at index ${position} has no field "${key}". ` +
                        `It has: ${Object.keys(data).join(', ') || '(nothing)'}.`
                );
            }
            picked[key] = data[key];
        }
        return picked;
    }

    try {
        return example.getInputs();
    } catch {
        throw new Error(
            `Trainset example at index ${position} does not declare its input fields. ` +
                'Call example.withInputs(...keys) when building the trainset, or pass ' +
                'inputKeys to the optimizer.'
        );
    }
}
