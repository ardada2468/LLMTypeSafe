import { z } from 'zod';
import { Predict } from './predict';
import { BestOfN, rethrowIfAborted, selectBest, type BestOfNOptions } from './best-of-n';
import { type Module } from '../core/module';
import { type Prediction } from '../core/prediction';
import { signature, getSignatureInstructions, type SignatureLike } from '../core/signature';
import { context, getCallbacks } from '../core/config';
import { type Callback } from '../core/callbacks';
import { runWithPredictorHints } from '../core/hints';
import { runRollout } from '../core/rollout';
import { getInputFieldConfigs, getOutputFieldConfigs } from '../utils/schema';
import type { ILanguageModel, LLMCallOptions } from '../types/language-model';

/** Options for {@link Refine}: those of {@link BestOfN}. */
export type RefineOptions = BestOfNOptions;

/**
 * The signature `Refine` asks a model to fill after a failed attempt, after
 * DSPy's `OfferFeedback`. Inputs are pre-rendered as text (JSON where the
 * value is structured), as DSPy does.
 *
 * `advice` is text holding a JSON object rather than an object field: strict
 * structured-output modes cannot express a free-form record, and an
 * `object` field would be pinned to `{}` on those providers. It is parsed
 * leniently in {@link parseAdvice}.
 */
export const OfferFeedback = signature({
    description:
        'In the discussion, assign blame to each module that contributed to the final reward ' +
        'being below the threshold, if any. Then, prescribe concrete advice of how the module ' +
        'should act on its future input when we retry the process, if it were to receive the ' +
        'same or similar inputs. If a module is not to blame, the advice should be N/A.\n' +
        'The module will not see its own history, so it needs to rely on entirely concrete and ' +
        'actionable advice from you to avoid the same mistake on the same or similar inputs.',
    input: z.object({
        program_code: z.string().describe('The code of the program that we are analyzing'),
        modules_defn: z
            .string()
            .describe('The definition of each module in the program, including its I/O'),
        program_inputs: z.string().describe('The inputs to the program that we are analyzing'),
        program_trajectory: z
            .string()
            .describe("The trajectory of the program's execution, showing each module's I/O"),
        program_outputs: z
            .string()
            .describe('The outputs of the program that we are analyzing'),
        reward_code: z
            .string()
            .describe('The code of the reward function that we are analyzing'),
        target_threshold: z.number().describe('The target threshold for the reward function'),
        reward_value: z.number().describe("The reward value assigned to the program's outputs"),
        module_names: z
            .string()
            .describe('The names of the modules in the program, for which we seek advice'),
    }),
    output: z.object({
        discussion: z
            .string()
            .describe('Discussing blame of where each module went wrong, if it did'),
        advice: z
            .string()
            .describe(
                'A JSON object mapping each module name to advice. For each module, describe ' +
                    'very concretely, in this order: the specific scenarios in which it has made ' +
                    'mistakes in the past and what each mistake was, followed by what it should ' +
                    'do differently in that kind of scenario in the future. If the module is not ' +
                    'to blame, write N/A.'
            ),
    }),
});

/** One predictor call in an attempt, as the feedback model is shown it. */
interface TrajectoryStep {
    module_name: string;
    inputs: Record<string, any>;
    outputs?: Record<string, any>;
}

/**
 * Like {@link BestOfN}, but each failed attempt is followed by a request for
 * advice, which the next attempt's predictors receive as a hint — DSPy's
 * `Refine`.
 *
 * After an attempt whose reward falls short of `threshold` (and that is not
 * the last), a {@link Predict} over {@link OfferFeedback} is shown the
 * program's source, its predictors' signatures, the inputs, every predictor
 * call the attempt made with its inputs and outputs, the outputs, the reward
 * function's source, the threshold and the reward. It answers with advice per
 * predictor path (`namedPredictors()` names). On the next attempt each
 * predictor gains a trailing `hint_` input — "A hint to the module from an
 * earlier run" — carrying its advice, or `N/A`.
 *
 * Mirrors DSPy in what it does not do too: advice replaces, rather than
 * accumulates across, earlier advice; and the best-scoring attempt is
 * returned when none reaches the threshold, exactly as in `BestOfN`.
 *
 * The hint travels in an async-local scope keyed by predictor, not in call
 * options, so it reaches predictors inside hand-written `forward` methods,
 * concurrent runs never see each other's hints, and nothing about it is ever
 * written by `dumpState()`. The feedback predictor is likewise not part of the
 * program: it is invisible to `namedPredictors()`, so optimizers never tune it
 * and saved state never includes it.
 *
 * The feedback call uses `options.lm`, else the first model found on the
 * program's predictors, else the configured one. A failed feedback call
 * counts against `failCount` like a failed attempt.
 */
export class Refine extends BestOfN {
    declare private readonly feedback: Predict;

    constructor(module: Module, options: RefineOptions) {
        super(module, options);
        // Non-enumerable, so the composition walk — and with it
        // `namedPredictors`, `dumpState` and every optimizer — skips it.
        Object.defineProperty(this, 'feedback', {
            value: new Predict(OfferFeedback),
            enumerable: false,
            writable: true,
            configurable: true,
        });
    }

    async forward(
        inputs: Record<string, any>,
        options?: LLMCallOptions
    ): Promise<Prediction & Record<string, any>> {
        const outcomes: Array<{ prediction: Prediction; reward: number }> = [];
        let advice: Record<string, string> | undefined;
        let failures = 0;
        let lastError: unknown;

        for (let rolloutId = 0; rolloutId < this.n; rolloutId++) {
            options?.signal?.throwIfAborted();
            try {
                const names = new Map<object, string>();
                const hints = new Map<object, string>();
                const trajectory: TrajectoryStep[] = [];
                const currentAdvice = advice;

                const prediction = await runWithPredictorHints(hints, () =>
                    context(
                        { callbacks: [...getCallbacks(), recordTrajectory(names, trajectory)] },
                        () =>
                            runRollout(
                                this.module,
                                inputs,
                                options,
                                { rolloutId, temperature: this.temperature },
                                (copy) => {
                                    for (const [name, predictor] of copy.namedPredictors()) {
                                        names.set(predictor, name);
                                        if (currentAdvice) {
                                            hints.set(predictor, currentAdvice[name] ?? 'N/A');
                                        }
                                    }
                                }
                            )
                    )
                );

                const reward = await this.rewardFn(inputs, prediction);
                outcomes.push({ prediction, reward });
                if (reward >= this.threshold || rolloutId === this.n - 1) break;

                advice = await this.offerFeedback(
                    inputs,
                    prediction,
                    reward,
                    trajectory,
                    options
                );
            } catch (error) {
                rethrowIfAborted(options?.signal, error);
                failures += 1;
                lastError = error;
                if (failures > this.failCount) throw error;
            }
        }

        return selectBest(outcomes, lastError);
    }

    /** Ask the feedback model for per-predictor advice on a failed attempt. */
    private async offerFeedback(
        inputs: Record<string, any>,
        prediction: Prediction,
        reward: number,
        trajectory: TrajectoryStep[],
        options?: LLMCallOptions
    ): Promise<Record<string, string>> {
        const predictors = this.module.namedPredictors();
        const names = predictors.map(([name]) => name);

        const lm = options?.lm ?? firstModel(predictors.map(([, predictor]) => predictor));
        const feedback = await this.feedback.forward(
            {
                program_code: sourceOf(this.module),
                modules_defn: predictors
                    .map(([name, predictor]) => describePredictor(name, predictor))
                    .join('\n\n'),
                program_inputs: toJson(inputs),
                program_trajectory: toJson(trajectory),
                program_outputs: toJson(prediction.toObject()),
                reward_code: safeToString(this.rewardFn),
                target_threshold: this.threshold,
                reward_value: reward,
                module_names: toJson(names),
            },
            { ...(lm ? { lm } : {}), ...(options?.signal ? { signal: options.signal } : {}) }
        );
        return parseAdvice(String(feedback.advice ?? ''), names);
    }
}

/**
 * A callback recording every call of a predictor in `names` — its inputs and
 * outputs — in the order the calls started.
 */
function recordTrajectory(names: Map<object, string>, trajectory: TrajectoryStep[]): Callback {
    const open = new Map<string, TrajectoryStep>();
    return {
        onModuleStart(event) {
            const name = names.get(event.instance);
            if (name === undefined) return;
            const step: TrajectoryStep = { module_name: name, inputs: { ...event.inputs } };
            open.set(event.callId, step);
            trajectory.push(step);
        },
        onModuleEnd(event) {
            const step = open.get(event.callId);
            if (!step) return;
            open.delete(event.callId);
            if (event.outputs) step.outputs = { ...event.outputs };
        },
    };
}

/**
 * Read the feedback model's advice. A JSON object (optionally fenced) maps
 * predictor names to advice; anything else is taken as advice for every
 * predictor, rather than discarded — the model said something, and an
 * unparseable hint is still better than none.
 */
export function parseAdvice(text: string, names: readonly string[]): Record<string, string> {
    const trimmed = text
        .trim()
        .replace(/^```(?:json)?\s*/i, '')
        .replace(/\s*```$/, '');
    try {
        const parsed: unknown = JSON.parse(trimmed);
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
            return Object.fromEntries(
                Object.entries(parsed as Record<string, unknown>).map(([key, value]) => [
                    key,
                    typeof value === 'string' ? value : toJson(value),
                ])
            );
        }
    } catch {
        // Not JSON; fall through.
    }
    return Object.fromEntries(names.map((name) => [name, text.trim() || 'N/A']));
}

/** The predictor's I/O and instructions, as DSPy's `inspect_modules` lays them out. */
function describePredictor(name: string, predictor: Predict<any, any>): string {
    const signature = (predictor as unknown as { signature?: SignatureLike }).signature;
    if (!signature) return `Module ${name}`;
    const instructions = predictor.getInstructions() ?? getSignatureInstructions(signature);
    const lines = [
        `Module ${name}`,
        `\tInput Fields: ${Object.keys(getInputFieldConfigs(signature)).join(', ')}`,
        `\tOutput Fields: ${Object.keys(getOutputFieldConfigs(signature)).join(', ')}`,
    ];
    if (instructions) lines.push(`\tOriginal Instructions: ${instructions}`);
    return lines.join('\n');
}

/** The first model a predictor holds of its own. */
function firstModel(predictors: Array<Predict<any, any>>): ILanguageModel | undefined {
    for (const predictor of predictors) {
        const lm = (predictor as unknown as { lm?: ILanguageModel }).lm;
        if (lm) return lm;
    }
    return undefined;
}

/** The program's `forward` source — the nearest thing to DSPy's `inspect.getsource`. */
function sourceOf(module: Module): string {
    const forward = (Object.getPrototypeOf(module) as { forward?: unknown }).forward;
    return `class ${module.constructor.name} {\n${safeToString(forward)}\n}`;
}

function safeToString(value: unknown): string {
    try {
        return typeof value === 'function' ? value.toString() : String(value);
    } catch {
        return '(source unavailable)';
    }
}

/**
 * JSON for the feedback prompt. Never throws: values JSON cannot carry —
 * BigInts, cycles, functions — become placeholders, as DSPy's
 * `recursive_mask` does.
 */
function toJson(value: unknown): string {
    const seen = new WeakSet<object>();
    try {
        return JSON.stringify(
            value,
            (_key, item: unknown) => {
                if (typeof item === 'bigint') return item.toString();
                if (typeof item === 'function') return '<function>';
                if (item && typeof item === 'object') {
                    if (seen.has(item)) return '<circular>';
                    seen.add(item);
                }
                return item;
            },
            2
        );
    } catch {
        return String(value);
    }
}
