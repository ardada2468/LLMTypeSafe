import { type SignatureLike } from './signature';
import { Prediction } from './prediction';
import type { ILanguageModel, LLMCallOptions } from '../types/language-model';
import { getDefaultLM } from './config';
import { beginTrace, nextModuleId, type TraceSpan } from './trace';

export abstract class Module {
    protected lm: ILanguageModel;
    protected signature?: SignatureLike;

    /** Identifies this instance in trace entries, e.g. `Predict#1`. */
    readonly moduleId: string;

    constructor(signature?: SignatureLike, lm?: ILanguageModel) {
        this.signature = signature;
        this.lm = lm || getDefaultLM();
        this.moduleId = nextModuleId(this.constructor.name);
    }

    abstract forward(
        inputs: Record<string, any>,
        options?: LLMCallOptions
    ): Promise<Prediction>;

    /** Alias for {@link forward}, so modules read like function calls. */
    async call(inputs: Record<string, any>, options?: LLMCallOptions): Promise<Prediction> {
        return this.forward(inputs, options);
    }

    /** Alias for {@link forward}, mirroring DSPy's Python naming. */
    async __call__(inputs: Record<string, any>, options?: LLMCallOptions): Promise<Prediction> {
        return this.forward(inputs, options);
    }

    /**
     * Run one module invocation and wrap its output in a `Prediction`.
     *
     * With `configure({ tracing: true })` the callback receives a span to report
     * each language-model call to, and the resulting entry is attached to the
     * prediction as `trace` and appended to the history behind
     * `inspectHistory()`. Failures are recorded too — the prompt behind a
     * `ValidationError` is exactly what you want to read — and then rethrown.
     *
     * With tracing off the span is `undefined` and nothing is timed, copied, or
     * stored.
     */
    protected async traced<TOutput extends Record<string, any>>(
        inputs: Record<string, any>,
        run: (span: TraceSpan | undefined) => Promise<TOutput>
    ): Promise<Prediction<TOutput>> {
        const span = beginTrace(this.moduleId, this.lm, inputs);
        if (!span) {
            return new Prediction(await run(undefined));
        }

        let output: TOutput;
        try {
            output = await run(span);
        } catch (error) {
            span.fail(error);
            throw error;
        }

        return new Prediction(output, span.finish(output));
    }
}
