/**
 * Building blocks shared by the optimizers: scoring a program on a dataset,
 * recording which predictor did what during a run, bootstrapping per-predictor
 * demos from successful runs, proposing instructions with a model, and seeded
 * minibatch sampling.
 *
 * Every optimizer works on a *copy* of the student, addressed by predictor path
 * (see `Module.namedPredictors`). Paths are the common currency: a trace
 * recorded on a teacher copy, a demo set chosen by a search, and a saved
 * `dumpState()` all name predictors the same way, so state moves between
 * copies of one program without anyone holding a reference into another copy.
 */
import { Example } from '../core/example';
import { LMError, TsDspyError } from '../core/errors';
import { type Prediction } from '../core/prediction';
import { type Module } from '../core/module';
import { context, getCallbacks } from '../core/config';
import { type Callback } from '../core/callbacks';
import { signature } from '../core/signature';
import { signatureLayout } from '../adapters/base';
import { Predict } from '../modules/predict';
import { evaluate } from '../evaluate/evaluate';
import {
    metricFeedback,
    metricScore,
    type EvaluationResult,
    type Metric,
    type MetricResult,
    type MetricTraceStep,
} from '../evaluate/types';
import { mapWithConcurrency } from '../utils/pool';
import type { ILanguageModel, LLMCallOptions } from '../types/language-model';
import { createRng, shuffled, type Rng } from './random';
import { resolveExampleInputs } from './types';

export { metricScore, metricFeedback };

// ------------------------------------------------------------------ options

/**
 * A progress report from an optimizer. Every optimizer reports through the
 * same shape so one logger works for all of them; `phase` says what stage the
 * run is in, and the numeric fields are filled in where they mean something.
 */
export interface OptimizerProgressEvent {
    /** The reporting optimizer, e.g. `'MIPROv2'`. */
    optimizer: string;
    /** Stage of the run, e.g. `'bootstrap'`, `'propose'`, `'trial'`, `'evaluate'`. */
    phase: string;
    /** Position within the phase, 1-based. */
    step?: number;
    /** Steps in the phase, when known. */
    total?: number;
    /** The score just measured, if this event reports one. */
    score?: number;
    /** The best score seen so far. */
    bestScore?: number;
    /** Human-readable detail. */
    message?: string;
}

/** Options every optimizer beyond the few-shot basics accepts. */
export interface BaseOptimizerOptions {
    /** Seed for every random choice the optimizer makes. Default 0. */
    seed?: number;
    /** Examples evaluated (or bootstrapped) at once. Default 4. */
    concurrency?: number;
    /**
     * Called as the run proceeds. An observer only: anything it throws is
     * swallowed, because reporting progress must never lose a compile.
     */
    onProgress?: (event: OptimizerProgressEvent) => void;
    /** Input field names, when the dataset's examples do not declare their own. */
    inputKeys?: string[];
    /** Call options for the student's runs, e.g. `{ temperature: 0 }`. */
    callOptions?: LLMCallOptions;
}

/** What `compile` takes: training data and, optionally, separate validation data. */
export interface CompileConfig {
    trainset: Example[];
    /**
     * Data to score candidates on. Defaults to the trainset, which is fine
     * for a quick run but overfits: a candidate that memorised its demos looks
     * better than it is.
     */
    valset?: Example[];
}

/** Call `onProgress` with `event`, dropping anything it throws. */
export function reportProgress(
    onProgress: ((event: OptimizerProgressEvent) => void) | undefined,
    event: OptimizerProgressEvent
): void {
    if (!onProgress) return;
    try {
        onProgress(event);
    } catch {
        // Deliberately ignored: an observer must not be able to fail the run.
    }
}

/** A concurrency setting turned into a usable lane count. */
export function lanes(concurrency: number | undefined, fallback = 4): number {
    const value = Math.trunc(concurrency ?? fallback);
    return Number.isFinite(value) && value >= 1 ? value : 1;
}

/** Whether a metric result counts as a pass at `threshold`. Booleans ignore it. */
export function metricPasses(result: MetricResult, threshold: number): boolean {
    if (typeof result === 'boolean') return result;
    // NaN fails the comparison, which is right for a metric that could not
    // produce a number.
    const value = typeof result === 'number' ? result : Number(result?.score);
    return value >= threshold;
}

/**
 * Attach `value` to `program` as a non-enumerable property.
 *
 * Non-enumerable on purpose: the composition walk (`namedSubModules`) only
 * follows enumerable fields, so results that mention other programs can never
 * be mistaken for sub-modules, and they stay out of spreads and JSON.
 */
export function attachResults<M extends object, V>(program: M, key: string, value: V): M {
    Object.defineProperty(program, key, {
        value,
        writable: true,
        configurable: true,
        enumerable: false,
    });
    return program;
}

/** Arithmetic mean, or `0` for an empty list. */
export function mean(values: readonly number[]): number {
    return values.length === 0 ? 0 : values.reduce((sum, v) => sum + v, 0) / values.length;
}

// --------------------------------------------------------------- evaluation

/** Per-example scores and their mean, as the optimizers consume them. */
export interface ProgramEvaluation {
    /** Mean score, failures counted as `0`. */
    score: number;
    /** One score per example, in dataset order. */
    scores: number[];
    /** The metric's feedback per example, when it returned any. */
    feedback: Array<string | undefined>;
    /** The full per-example results from `evaluate`. */
    results: EvaluationResult[];
}

export interface EvaluateProgramOptions {
    concurrency?: number;
    inputKeys?: string[];
    callOptions?: LLMCallOptions;
    /** Score for an example whose program or metric threw. Default `0`. */
    failureScore?: number;
}

/**
 * The part of the student's call options that is about transport rather than
 * sampling. Proposer and reflection calls get only these: a student tuned with
 * `{ maxTokens: 8, stopSequences: ['\n'] }` must not have every proposed
 * instruction cut to eight tokens.
 */
export function transportOptions(options: LLMCallOptions | undefined): LLMCallOptions {
    const { timeout, retries, signal } = options ?? {};
    return {
        ...(timeout !== undefined ? { timeout } : {}),
        ...(retries !== undefined ? { retries } : {}),
        ...(signal !== undefined ? { signal } : {}),
    };
}

/**
 * Score `program` on `dataset` through `evaluate`, so optimizers measure
 * candidates exactly as a user's own evaluation would — same failure handling,
 * same score normalisation. An empty dataset scores `0`.
 */
export async function evaluateProgram(
    program: Module,
    dataset: readonly Example[],
    metric: Metric,
    options: EvaluateProgramOptions = {}
): Promise<ProgramEvaluation> {
    if (dataset.length === 0) {
        return { score: 0, scores: [], feedback: [], results: [] };
    }
    const report = await evaluate(program, dataset, metric, {
        concurrency: lanes(options.concurrency),
        inputKeys: options.inputKeys,
        callOptions: options.callOptions,
        failureScore: options.failureScore,
    });
    // Every example failing at the provider is an outage or a bad key, not a
    // weak candidate. Scoring it zero would let a whole compile "succeed"
    // after issuing every scheduled call and learning nothing.
    if (
        report.errorCount === report.count &&
        report.results.every((result) => result.error instanceof LMError)
    ) {
        const first = report.results[0]!.error as LMError;
        throw new TsDspyError(
            `Every one of ${report.count} examples failed with a provider error, so there ` +
                `is nothing to optimise against. First failure: ${first.message}`,
            { cause: first }
        );
    }
    return {
        score: report.score,
        scores: report.results.map((result) => result.score),
        feedback: report.results.map((result) => result.feedback),
        results: report.results,
    };
}

// ------------------------------------------------------------------ traces

/** What one traced run produced. */
export interface TracedRun {
    /** Absent when the program threw. */
    prediction?: Prediction;
    /** Every predictor call the run made, in completion order. */
    trace: MetricTraceStep[];
    /** Whatever the run threw. */
    error?: unknown;
}

/**
 * Run `program` once and record each of its predictors' calls, by path.
 *
 * A callback is layered over the active ones for this call's async scope
 * only, so concurrent traced runs never see each other's steps and the user's
 * own callbacks keep firing. A predictor that runs several times — ReAct's
 * step predictor — contributes one step per call.
 */
export async function runWithTrace(
    program: Module,
    inputs: Record<string, any>,
    callOptions?: LLMCallOptions
): Promise<TracedRun> {
    const paths = new Map<object, string>();
    for (const [path, predictor] of program.namedPredictors()) paths.set(predictor, path);

    const open = new Map<string, { path: string; inputs: Record<string, any> }>();
    const trace: MetricTraceStep[] = [];
    const recorder: Callback = {
        onModuleStart(event) {
            const path = paths.get(event.instance);
            if (path !== undefined)
                open.set(event.callId, { path, inputs: { ...event.inputs } });
        },
        onModuleEnd(event) {
            const started = open.get(event.callId);
            open.delete(event.callId);
            if (started && event.error === undefined && event.outputs) {
                trace.push({ ...started, outputs: { ...event.outputs } });
            }
        },
    };

    try {
        const prediction = await context({ callbacks: [...getCallbacks(), recorder] }, () =>
            program.forward(inputs, callOptions)
        );
        return { prediction, trace };
    } catch (error) {
        return { trace, error };
    }
}

/** Turn one recorded predictor call into a demo: its inputs plus what it produced. */
export function stepToDemo(step: MetricTraceStep): Example {
    return new Example({ ...step.inputs, ...step.outputs }).withInputs(
        ...Object.keys(step.inputs)
    );
}

/**
 * A copy of `program` with every predictor pinned to `lm`.
 *
 * Used for teacher runs: setting a scoped default model would not reach a
 * predictor that was built with its own model, and a teacher must replace
 * those too.
 */
export function programOnLM<M extends Module>(program: M, lm: ILanguageModel): M {
    const copy = program.deepCopy();
    for (const predictor of copy.predictors()) {
        (predictor as unknown as { lm?: ILanguageModel }).lm = lm;
    }
    return copy;
}

// ---------------------------------------------------------------- bootstrap

/** One bootstrap attempt's outcome. */
export interface BootstrapAttempt {
    /** Position in the shuffled trainset. */
    index: number;
    total: number;
    status: 'passed' | 'failed' | 'error';
    score?: MetricResult;
    error?: unknown;
}

export interface BootstrapTracesOptions {
    /** Stop once this many runs have passed. */
    maxBootstrappedDemos: number;
    /** Seed for the trainset shuffle. */
    seed?: number;
    /** Attempt the trainset in its given order instead of shuffling it. */
    preserveOrder?: boolean;
    /** Attempts in flight at once, and the batch the run stops between. */
    concurrency?: number;
    /** Score at which a numeric metric passes. Default 0.5. */
    threshold?: number;
    /** Generate the traces with this model instead of the student's. */
    teacher?: ILanguageModel;
    inputKeys?: string[];
    callOptions?: LLMCallOptions;
    /** Called once per attempt, in completion order. Throws are swallowed. */
    onAttempt?: (attempt: BootstrapAttempt) => void;
}

/** The demos a bootstrap earned, per predictor path, and what it tried. */
export interface BootstrapTracesResult {
    /** Demos per predictor path, in trainset order. Paths with none are absent. */
    demos: Record<string, Example[]>;
    /** The trainset in the order it was attempted. */
    ordered: Example[];
    /** Positions in `ordered` whose run passed and supplied demos. */
    used: Set<number>;
    /** How many attempts ran. */
    attempted: number;
}

/**
 * Run `program` over a shuffled trainset, keep the runs the metric passes,
 * and turn each passing run's trace into one demo per predictor it called.
 *
 * This is what makes bootstrapping work for multi-step programs: a pipeline's
 * retrieval step and its answer step each get demos of *their own* inputs and
 * outputs, recorded from a run whose final answer was right. For a predictor
 * called several times in one run, the last call is kept — the one closest to
 * the answer the metric approved.
 *
 * The metric receives the run's trace as its third argument, as in DSPy.
 * Attempts run in batches and stop once enough runs have passed, and the batch
 * boundary keeps the result deterministic for a seed: every attempt below it
 * has finished, so the passes kept are the first ones in trainset order.
 */
export async function bootstrapTraces(
    program: Module,
    trainset: readonly Example[],
    metric: Metric,
    options: BootstrapTracesOptions
): Promise<BootstrapTracesResult> {
    const ordered = options.preserveOrder
        ? [...trainset]
        : shuffled(trainset, createRng(options.seed ?? 0));
    const inputsByIndex = ordered.map((example, index) =>
        resolveExampleInputs(example, options.inputKeys, index)
    );
    const runner = options.teacher ? programOnLM(program, options.teacher) : program;
    const threshold = options.threshold ?? 0.5;
    const batchSize = lanes(options.concurrency);
    const total = ordered.length;
    const report = (attempt: BootstrapAttempt) => {
        try {
            options.onAttempt?.(attempt);
        } catch {
            // An observer must not be able to fail the run.
        }
    };

    const passes: Array<{ index: number; trace: MetricTraceStep[] }> = [];
    let attempted = 0;
    for (let start = 0; start < total; start += batchSize) {
        if (passes.length >= options.maxBootstrappedDemos) break;

        const batch = ordered.slice(start, start + batchSize);
        attempted += batch.length;
        const settled = await mapWithConcurrency(
            batch,
            async (example, offset) => {
                const index = start + offset;
                const run = await runWithTrace(
                    runner,
                    inputsByIndex[index],
                    options.callOptions
                );
                if (!run.prediction) {
                    report({ index, total, status: 'error', error: run.error });
                    return null;
                }
                try {
                    const score = await metric(example, run.prediction, run.trace);
                    const status = metricPasses(score, threshold) ? 'passed' : 'failed';
                    report({ index, total, status, score });
                    return status === 'passed' ? { index, trace: run.trace } : null;
                } catch (error) {
                    report({ index, total, status: 'error', error });
                    return null;
                }
            },
            { concurrency: batchSize }
        );

        for (const result of settled) {
            const value = result.status === 'fulfilled' ? result.value : null;
            if (value && passes.length < options.maxBootstrappedDemos) passes.push(value);
        }
    }

    const demos: Record<string, Example[]> = {};
    for (const pass of passes) {
        const lastByPath = new Map<string, MetricTraceStep>();
        for (const step of pass.trace) lastByPath.set(step.path, step);
        for (const [path, step] of lastByPath) (demos[path] ??= []).push(stepToDemo(step));
    }

    return { demos, ordered, used: new Set(passes.map((p) => p.index)), attempted };
}

/**
 * Up to `max` labelled examples from `ordered` that were not already used as
 * a bootstrapped demo, so the model never sees one item twice.
 */
export function labeledDemos(
    ordered: readonly Example[],
    used: Set<number>,
    max: number
): Example[] {
    if (max <= 0) return [];
    return ordered.filter((_, index) => !used.has(index)).slice(0, max);
}

/**
 * A copy of `program` whose predictors carry the given demos, by path. A
 * path absent from `demosByPath` gets `extra` alone; `extra` (labelled demos)
 * follows the bootstrapped ones on every predictor, as in DSPy.
 */
export function withDemosByPath<M extends Module>(
    program: M,
    demosByPath: Record<string, Example[]>,
    extra: Example[] = []
): M {
    const copy = program.deepCopy();
    for (const [path, predictor] of copy.namedPredictors()) {
        predictor.setDemos([...(demosByPath[path] ?? []), ...extra]);
    }
    return copy;
}

/**
 * A copy of `program` with every predictor's demos cleared and everything
 * else — instructions included — kept.
 *
 * This is what DSPy's `reset_copy()` means for a program whose instructions
 * live in its signatures: a fresh demo slate, not a fresh prompt. (`resetCopy`
 * here also drops instruction overrides, which would silently undo an earlier
 * instruction optimizer such as COPRO when a demo optimizer runs after it.)
 */
export function withoutDemos<M extends Module>(program: M): M {
    const copy = program.deepCopy();
    for (const predictor of copy.predictors()) predictor.setDemos([]);
    return copy;
}

/** How a candidate demo set is built, by DSPy's seed convention. */
export interface DemoSetPlan {
    kind: 'zero-shot' | 'labeled' | 'bootstrap' | 'shuffled-bootstrap';
    /** `-3` zero-shot, `-2` labelled, `-1` unshuffled bootstrap, then 0, 1, … */
    seed: number;
}

/**
 * The plans for `count` candidate demo sets: zero-shot, labelled-only,
 * unshuffled bootstrap, then shuffled bootstraps with seeds `seed`,
 * `seed + 1`, … — as DSPy's `create_n_fewshot_demo_sets`. A count below 3
 * keeps the first `count` fixed plans.
 */
export function demoSetPlans(count: number, seed: number): DemoSetPlan[] {
    const fixed: DemoSetPlan[] = [
        { kind: 'zero-shot', seed: -3 },
        { kind: 'labeled', seed: -2 },
        { kind: 'bootstrap', seed: -1 },
    ];
    const shuffledPlans = Array.from({ length: Math.max(0, count - 3) }, (_, i) => ({
        kind: 'shuffled-bootstrap' as const,
        seed: seed + i,
    }));
    return [...fixed, ...shuffledPlans].slice(0, Math.max(0, count));
}

export interface DemoSetOptions {
    maxBootstrappedDemos: number;
    maxLabeledDemos: number;
    /** Seed for the labelled-only selection. */
    seed?: number;
    /** Draws each shuffled bootstrap's demo budget, between 1 and the maximum. */
    sizeRng: Rng;
    concurrency?: number;
    threshold?: number;
    teacher?: ILanguageModel;
    inputKeys?: string[];
    callOptions?: LLMCallOptions;
}

/**
 * Build one candidate demo set per predictor path from a plan. Bootstrapped
 * plans trace the program over the trainset; labelled demos fill whatever
 * room the bootstrapped ones left, as in DSPy.
 */
export async function buildDemoSet(
    program: Module,
    trainset: readonly Example[],
    metric: Metric,
    plan: DemoSetPlan,
    options: DemoSetOptions
): Promise<Record<string, Example[]>> {
    const paths = program.namedPredictors().map(([path]) => path);
    const forAll = (demos: Example[]) => Object.fromEntries(paths.map((p) => [p, [...demos]]));
    if (plan.kind === 'zero-shot') return forAll([]);
    if (plan.kind === 'labeled') {
        return forAll(
            shuffled(trainset, createRng(options.seed ?? 0)).slice(
                0,
                Math.max(0, options.maxLabeledDemos)
            )
        );
    }

    const maxBoot = options.maxBootstrappedDemos;
    const budget =
        plan.kind === 'bootstrap'
            ? maxBoot
            : // Zero means zero: a shuffled candidate must not bootstrap one
              // demo (and pay the teacher for it) when none were asked for.
              maxBoot <= 0
              ? 0
              : 1 + Math.floor(options.sizeRng() * maxBoot);
    const result = await bootstrapTraces(program, trainset, metric, {
        maxBootstrappedDemos: budget,
        seed: plan.seed,
        // The unshuffled candidate keeps the trainset order, which is what seed
        // -1 means in DSPy.
        preserveOrder: plan.kind === 'bootstrap',
        concurrency: options.concurrency,
        threshold: options.threshold,
        teacher: options.teacher,
        inputKeys: options.inputKeys,
        callOptions: options.callOptions,
    });
    const labeled = labeledDemos(
        result.ordered,
        result.used,
        options.maxLabeledDemos - result.used.size
    );
    return Object.fromEntries(
        paths.map((path) => [path, [...(result.demos[path] ?? []), ...labeled]])
    );
}

// ------------------------------------------------------------- description

/**
 * The instructions a predictor currently sends: its override, or the
 * signature's description, or the default naming its fields.
 */
export function predictorInstruction(predictor: Predict<any, any>): string {
    return (
        predictor.getInstructions() ?? signatureLayout(predictor.getSignature()).instructions
    );
}

/** `question, context -> answer` for a predictor's signature. */
export function predictorShape(predictor: Predict<any, any>): string {
    const layout = signatureLayout(predictor.getSignature());
    const names = (fields: Array<{ name: string }>) => fields.map((f) => f.name).join(', ');
    return `${names(layout.inputs)} -> ${names(layout.outputs)}`;
}

/**
 * A plain-text description of a program built from its structure: each
 * predictor's path, fields and current instructions. Instruction proposers
 * read it to understand where a step sits in the whole.
 */
export function describeProgram(program: Module): string {
    const predictors = program.namedPredictors();
    const lines = [
        `A program of class ${program.constructor.name} with ${predictors.length} step(s):`,
    ];
    predictors.forEach(([path, predictor], index) => {
        lines.push(
            `${index + 1}. ${path} (${predictor.constructor.name}): ${predictorShape(predictor)}`,
            `   Instructions: ${predictorInstruction(predictor)}`
        );
    });
    return lines.join('\n');
}

/** Examples rendered as numbered blocks of `field: value` lines, for prompts. */
export function formatExamples(examples: readonly Example[], limit = 5): string {
    if (examples.length === 0) return 'No examples available.';
    return examples
        .slice(0, limit)
        .map((example, index) => {
            const fields = Object.entries(example.toObject())
                .map(([key, value]) => `${key}: ${stringify(value)}`)
                .join('\n');
            return `Example ${index + 1}:\n${fields}`;
        })
        .join('\n\n');
}

function stringify(value: unknown): string {
    if (typeof value === 'string') return value;
    try {
        return JSON.stringify(value);
    } catch {
        return String(value);
    }
}

// ----------------------------------------------------------------- proposer

const ProposeInstruction = signature(
    [
        'basic_instruction "the task\'s current instructions"',
        'program_description "what the whole program does and where this step sits in it"',
        'dataset_summary "observations about the data the program runs on"',
        'task_demos "worked examples of this step\'s inputs and outputs"',
        'previous_instructions "instructions tried before, with their scores (higher is better)"',
        'tip "a suggestion for how to write the instruction"',
    ].join(', ') + ' -> proposed_instruction "the improved instruction for this step"',
    'Use the information below to write a new, improved instruction for a step of a ' +
        'language-model program. The instruction should tell the model clearly and ' +
        "specifically how to turn the step's inputs into its outputs. Learn from the " +
        'previous instructions and their scores: keep what worked, fix what did not.'
);

/** What {@link proposeInstructions} needs to know about one predictor. */
export interface ProposeInstructionsOptions {
    /** The proposer ("prompt model"). Defaults to the configured model. */
    lm?: ILanguageModel;
    /** How many instructions to ask for. */
    n: number;
    /** The predictor's current instruction. */
    basicInstruction: string;
    programDescription?: string;
    datasetSummary?: string;
    /** Rendered demos of this predictor. */
    demos?: string;
    /** Earlier attempts and their scores, best last. */
    previous?: Array<{ instruction: string; score: number }>;
    /** A tip per proposal, cycled. */
    tips?: string[];
    /**
     * Position of the first proposal in a longer series, from 0. Default 0.
     * A caller asking for one proposal at a time passes its running count, so
     * every request in the series is distinct (see {@link proposeInstructions}).
     */
    index?: number;
    /** Length of the whole series. Default `index + n`. */
    total?: number;
    /** Sampling temperature for the proposer. Default 1. */
    temperature?: number;
    callOptions?: LLMCallOptions;
}

/**
 * Ask a model for `n` candidate instructions for one predictor, grounded in
 * whatever context is supplied — program description, dataset summary, demos,
 * previous attempts with scores, and a tip.
 *
 * Proposals are requested one at a time, in order, so a seeded run asks the
 * same questions in the same order. Each proposal names its position in the
 * series — even when it also carries a tip, since tips repeat — so with the
 * cache on, two slots never send the identical request and get one cached
 * reply back for both. Blank and duplicate replies
 * are dropped, so fewer than `n` may come back; a proposer that fails is
 * skipped rather than failing the compile.
 */
export async function proposeInstructions(
    options: ProposeInstructionsOptions
): Promise<string[]> {
    const proposer = new Predict(ProposeInstruction);
    const previous =
        options.previous && options.previous.length > 0
            ? options.previous
                  .map((p) => `Instruction: ${p.instruction}\nScore: ${p.score.toFixed(3)}`)
                  .join('\n\n')
            : 'None yet.';
    const results: string[] = [];
    const seen = new Set<string>();
    const offset = Math.max(0, Math.trunc(options.index ?? 0));
    const total = options.total ?? offset + options.n;
    for (let i = 0; i < options.n; i++) {
        const tip =
            options.tips && options.tips.length > 0
                ? options.tips[i % options.tips.length]
                : '';
        const position = `Proposal ${offset + i + 1} of ${total}.`;
        try {
            const prediction = await proposer.forward(
                {
                    basic_instruction: options.basicInstruction,
                    program_description: options.programDescription ?? 'Not available.',
                    dataset_summary: options.datasetSummary ?? 'Not available.',
                    task_demos: options.demos ?? 'Not available.',
                    previous_instructions: previous,
                    tip: tip ? `${tip}\n(${position})` : position,
                },
                {
                    ...transportOptions(options.callOptions),
                    ...(options.lm ? { lm: options.lm } : {}),
                    temperature: options.temperature ?? 1,
                }
            );
            const text = cleanInstruction(String(prediction.get('proposed_instruction') ?? ''));
            if (text && !seen.has(text)) {
                seen.add(text);
                results.push(text);
            }
        } catch {
            // A failed proposal costs one candidate, not the compile.
        }
    }
    return results;
}

/** Trim whitespace and a wrapping pair of quotes or backticks. */
export function cleanInstruction(text: string): string {
    let result = text.trim();
    const match = /^(["'`])([\s\S]*)\1$/.exec(result);
    if (match) result = match[2].trim();
    return result;
}

const SummarizeDataset = signature(
    'examples "a sample of the dataset" -> observations "a concise summary of the task, ' +
        'the inputs, the expected outputs and any patterns worth knowing"',
    'Look at the examples of a dataset and write a few sentences describing what the ' +
        'task is, what the inputs look like and what makes a good output.'
);

/**
 * Ask a model to summarise a dataset from a sample of it — the "dataset
 * summary" that grounds MIPROv2's instruction proposals. Returns `undefined`
 * if the model fails: a missing summary weakens proposals, it does not stop
 * the compile.
 */
export async function summarizeDataset(
    trainset: readonly Example[],
    options: {
        lm?: ILanguageModel;
        sampleSize?: number;
        rng?: Rng;
        callOptions?: LLMCallOptions;
    } = {}
): Promise<string | undefined> {
    if (trainset.length === 0) return undefined;
    const sample = options.rng
        ? sampleMinibatch(trainset, options.sampleSize ?? 10, options.rng)
        : trainset.slice(0, options.sampleSize ?? 10);
    try {
        const prediction = await new Predict(SummarizeDataset).forward(
            { examples: formatExamples(sample, sample.length) },
            { ...options.callOptions, ...(options.lm ? { lm: options.lm } : {}) }
        );
        const text = String(prediction.get('observations') ?? '').trim();
        return text || undefined;
    } catch {
        return undefined;
    }
}

// ---------------------------------------------------------------- sampling

/**
 * `size` distinct items drawn with `rng`, in the order drawn. The whole list
 * (shuffled) when `size` is at least its length.
 */
export function sampleMinibatch<T>(items: readonly T[], size: number, rng: Rng): T[] {
    return shuffled(items, rng).slice(0, Math.max(0, Math.min(size, items.length)));
}

/** An integer in `[0, n)`. */
export function randomInt(rng: Rng, n: number): number {
    return Math.min(n - 1, Math.floor(rng() * n));
}
