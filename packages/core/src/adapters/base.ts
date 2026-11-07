import type {
    ChatMessage,
    ContentPart,
    ILanguageModel,
    ImageInput,
    LLMCallOptions,
    MessageContent,
} from '../types/language-model';
import {
    getSignatureInstructions,
    isHistoryFieldType,
    isImageFieldType,
    type SignatureLike,
} from '../core/signature';
import { type Example } from '../core/example';
import { type History, toHistory } from '../core/history';
import { type TraceSpan } from '../core/trace';
import { invokeLM, providerOptions } from '../core/lm-call';
import { emit, hasCallbacks, newCallFrame } from '../core/callbacks';
import { ValidationError } from '../core/errors';
import {
    buildRepairInstruction,
    isRepeatedFailure,
    normaliseRepairAttempts,
} from '../core/repair';
import {
    getInputFieldConfigs,
    getOutputFieldConfigs,
    isPlaceholderDescription,
    validateOutputs,
    zodFieldHintFor,
} from '../utils/schema';
import { contentToText, imagePart, textPart } from '../utils/content';
import type { FieldConfig } from '../types/signature';

/** What an adapter needs to turn one call into chat messages. */
export interface AdapterFormatInput {
    signature: SignatureLike;
    /** Worked examples, rendered as user/assistant pairs before the real turn. */
    demos?: Example[];
    /** The call's input field values. */
    inputs: Record<string, unknown>;
    /**
     * Earlier turns, rendered as user/assistant pairs after the demos. A
     * signature input declared `history` supplies this from `inputs` instead;
     * when both are present, this one comes first.
     */
    history?: History;
    /** Instructions replacing the signature's description. */
    instructions?: string;
}

/** {@link AdapterFormatInput} plus what the language-model call needs. */
export interface AdapterCallInput extends AdapterFormatInput {
    /** Call options, passed to the provider once ts-dspy's own are stripped. */
    options?: LLMCallOptions;
    /** Open when tracing is on; every round trip is reported to it. */
    span?: TraceSpan;
    /** The module making the call, for trace entries and callback events. */
    moduleId?: string;
}

/** One field of a signature as an adapter describes it. */
export interface AdapterField {
    name: string;
    /** The declared type: `string`, `number`, `enum`, `image`, `history`, … */
    type: string;
    /** What the author wrote about the field; absent when it is only a placeholder. */
    description?: string;
    required: boolean;
    /** Members of an enum field. */
    values?: string[];
    /**
     * A phrase completing "the value you produce must be …", or `undefined`
     * when the field is free text and there is nothing to say.
     */
    hint?: string;
}

/** A signature laid out for formatting: instructions and fields, in order. */
export interface SignatureLayout {
    instructions: string;
    /** Inputs rendered as fields of a turn — history inputs excluded. */
    inputs: AdapterField[];
    outputs: AdapterField[];
    /** The name of the input declared `history`, if any. */
    historyField?: string;
}

/**
 * Turns a signature, its demos and one call's inputs into chat messages, and a
 * model's reply back into output field values.
 *
 * DSPy's adapter layer: the prompt format is a strategy, separate from the
 * module that uses it, so the same `Predict` can speak `[[ ## field ## ]]`
 * markers ({@link ChatAdapter}), JSON ({@link JSONAdapter}), XML tags
 * ({@link XMLAdapter}) or a two-model handoff ({@link TwoStepAdapter}).
 *
 * Subclasses implement {@link format} and {@link parse}; {@link call} runs
 * the whole round trip — format, call the model through the shared
 * instrumented LM helper (so tracing, usage and callbacks see it), parse,
 * validate against the signature, and repair on request.
 */
export abstract class Adapter {
    /** Reported as `adapter` in `onAdapterFormat`/`onAdapterParse` events. */
    abstract readonly name: string;

    /**
     * Render one call as chat messages: a system message describing the
     * fields and the task, demos and history as alternating user/assistant
     * pairs, then the user turn carrying the inputs.
     */
    abstract format(input: AdapterFormatInput): ChatMessage[];

    /**
     * Read a reply's raw output field values. No coercion or validation —
     * that happens once, against the signature's schema, in
     * {@link parseOutputs}. Fields that cannot be located are left out.
     */
    abstract parse(signature: SignatureLike, completion: string): Record<string, unknown>;

    /**
     * Read whatever output fields a reply has produced so far, for streaming.
     * Values are uncoerced text; never throws. Adapters that cannot stream
     * leave this undefined, and `Predict.stream` then makes one ordinary call.
     */
    parsePartial?(signature: SignatureLike, text: string): Record<string, unknown>;

    /**
     * Run one prediction: format, call `lm`, parse and validate — repairing up
     * to `options.repairAttempts` times when the reply fails validation.
     *
     * @throws {ValidationError} when the reply does not satisfy the signature
     *   and no repair attempt fixed it.
     */
    async call(lm: ILanguageModel, input: AdapterCallInput): Promise<Record<string, any>> {
        const messages = this.formatMessages(input);
        return this.complete(lm, messages, input);
    }

    /**
     * Finish a prediction whose first reply already arrived some other way —
     * streamed, say: parse and validate `completion`, then repair through
     * ordinary calls exactly as {@link call} would. `messages` are the ones
     * `completion` answered.
     */
    completeFrom(
        lm: ILanguageModel,
        messages: ChatMessage[],
        completion: string,
        input: AdapterCallInput
    ): Promise<Record<string, any>> {
        return this.complete(lm, messages, input, completion);
    }

    /** {@link format}, reported to `onAdapterFormat` callbacks. */
    formatMessages(input: AdapterFormatInput): ChatMessage[] {
        return observe('onAdapterFormat', this.name, input.inputs, () => this.format(input));
    }

    /**
     * {@link parse} then validate against the signature, reported to
     * `onAdapterParse` callbacks.
     *
     * @throws {ValidationError} when a required field is missing or a value
     *   cannot be coerced to its declared type.
     */
    parseOutputs(signature: SignatureLike, completion: string): Record<string, any> {
        return observe('onAdapterParse', this.name, completion, () =>
            validateOutputs(signature, this.parse(signature, completion), completion)
        );
    }

    /**
     * Call the model with `messages` and validate the reply, repairing on
     * request. A repair appends the rejected reply as an assistant turn and the
     * validation issues as a further user turn — always to the *original*
     * messages, so successive attempts do not stack up earlier corrections.
     * `firstCompletion`, when given, stands in for the first request.
     */
    protected async complete(
        lm: ILanguageModel,
        messages: ChatMessage[],
        input: AdapterCallInput,
        firstCompletion?: string
    ): Promise<Record<string, any>> {
        const repairAttempts = normaliseRepairAttempts(input.options?.repairAttempts);
        let attemptMessages = messages;
        let previousError: ValidationError | undefined;

        for (let attempt = 0; ; attempt++) {
            const completion =
                attempt === 0 && firstCompletion !== undefined
                    ? firstCompletion
                    : await this.request(lm, attemptMessages, input);
            try {
                return this.parseOutputs(input.signature, completion);
            } catch (error) {
                if (!(error instanceof ValidationError) || attempt >= repairAttempts) {
                    throw error;
                }
                // An identical failure would produce an identical repair turn,
                // so every remaining attempt is guaranteed to fail the same way
                // against a deterministic model. Stop rather than bill for it.
                if (previousError && isRepeatedFailure(previousError, error)) {
                    throw error;
                }
                previousError = error;
                attemptMessages = [
                    ...messages,
                    // Providers reject an empty assistant turn outright.
                    { role: 'assistant', content: completion.trim() || '(empty response)' },
                    {
                        role: 'user',
                        content: buildRepairInstruction(
                            error,
                            this.repairClosing(input.signature)
                        ),
                    },
                ];
            }
        }
    }

    /** One instrumented chat call, returning the reply text. */
    protected request(
        lm: ILanguageModel,
        messages: ChatMessage[],
        input: AdapterCallInput
    ): Promise<string> {
        const options = providerOptions(input.options);
        return invokeLM(
            {
                lm,
                method: 'chat',
                traceText: renderTranscript(messages),
                messages,
                options,
                span: input.span,
                moduleId: input.moduleId,
            },
            () => lm.chat(messages, options),
            (text) => text
        );
    }

    /** How a repair turn asks for the corrected reply, in this adapter's format. */
    protected repairClosing(_signature: SignatureLike): string {
        return 'Reply again with every required field, using the declared type for each.';
    }
}

// ------------------------------------------------------------------ events

function observe<T>(
    hook: 'onAdapterFormat' | 'onAdapterParse',
    adapter: string,
    inputs: any,
    run: () => T
): T {
    if (!hasCallbacks()) return run();

    const frame = newCallFrame();
    const startTime = Date.now();
    const base = { callId: frame.callId, parentCallId: frame.parentCallId, startTime, adapter };
    const finish = (result: { outputs?: any; error?: unknown }) => {
        const endTime = Date.now();
        emit(hook, {
            ...base,
            inputs,
            endTime,
            durationMs: endTime - startTime,
            ...result,
        } as never);
    };

    let outputs: T;
    try {
        outputs = run();
    } catch (error) {
        finish({ error });
        throw error;
    }
    finish({ outputs });
    return outputs;
}

// ------------------------------------------------------------------ layout

/**
 * A phrase completing "the value you produce must be …" for a field declared
 * with a field config.
 */
function configHint(config: FieldConfig): string | undefined {
    switch ((config.type ?? 'string').toLowerCase()) {
        case 'number':
        case 'float':
            return 'a number';
        case 'int':
        case 'integer':
            return 'an integer';
        case 'boolean':
        case 'bool':
            return 'true or false';
        case 'string[]':
            return 'a JSON array of strings';
        case 'number[]':
            return 'a JSON array of numbers';
        case 'array':
        case 'list':
            return 'a JSON array';
        case 'object':
        case 'json':
            return 'a JSON object';
        case 'enum':
            return config.values?.length ? `one of: ${config.values.join('; ')}` : undefined;
        default:
            return undefined;
    }
}

function toAdapterField(
    signature: SignatureLike,
    side: 'input' | 'output',
    name: string,
    config: FieldConfig
): AdapterField {
    const zodHint = zodFieldHintFor(signature, side, name);
    // A zod hint is a compact type description (`int, between 1 and 5`); a
    // bare `string` says nothing the model does not already assume.
    const hint =
        zodHint !== undefined
            ? zodHint === 'string'
                ? undefined
                : zodHint.startsWith('one of:')
                  ? zodHint
                  : `of type ${zodHint}`
            : configHint(config);
    return {
        name,
        type: config.type ?? 'string',
        description: isPlaceholderDescription(name, config.description)
            ? undefined
            : config.description,
        required: config.required !== false,
        values: config.values,
        hint,
    };
}

/**
 * Lay a signature out for formatting: instructions (the override, the
 * description, or a default naming the fields) and its fields in declaration
 * order, with the history input pulled out.
 */
export function signatureLayout(
    signature: SignatureLike,
    instructions?: string
): SignatureLayout {
    const inputs: AdapterField[] = [];
    let historyField: string | undefined;
    for (const [name, config] of Object.entries(getInputFieldConfigs(signature))) {
        if (isHistoryFieldType(config.type)) {
            historyField ??= name;
            continue;
        }
        inputs.push(toAdapterField(signature, 'input', name, config));
    }
    const outputs = Object.entries(getOutputFieldConfigs(signature)).map(([name, config]) =>
        toAdapterField(signature, 'output', name, config)
    );

    const list = (fields: AdapterField[]) =>
        fields.map((field) => `\`${field.name}\``).join(', ');
    const resolved =
        instructions ??
        getSignatureInstructions(signature) ??
        `Given the fields ${list(inputs)}, produce the fields ${list(outputs)}.`;

    return { instructions: resolved, inputs, outputs, historyField };
}

/** `1. \`answer\` (number): the answer`, one line per field. */
export function describeFieldList(fields: AdapterField[]): string {
    return fields
        .map(
            (field, index) =>
                `${index + 1}. \`${field.name}\` (${field.type})` +
                (field.description ? `: ${field.description}` : '')
        )
        .join('\n');
}

/** "Your input fields are: … Your output fields are: …" */
export function describeFields(layout: SignatureLayout): string {
    const parts: string[] = [];
    if (layout.inputs.length > 0) {
        parts.push(`Your input fields are:\n${describeFieldList(layout.inputs)}`);
    }
    parts.push(`Your output fields are:\n${describeFieldList(layout.outputs)}`);
    return parts.join('\n');
}

// ------------------------------------------------------------------ values

/** Render one field value as prompt text: strings verbatim, the rest as JSON. */
export function formatFieldValue(value: unknown): string {
    if (typeof value === 'string') return value;
    if (value instanceof Date) return value.toISOString();
    if (value === null || typeof value !== 'object') return String(value);
    try {
        return JSON.stringify(value);
    } catch {
        return String(value);
    }
}

/**
 * Accumulates message content as text and images, collapsing to a plain string
 * when no image was added — so a text-only prompt stays a string, exactly as
 * providers have always received it.
 */
export class ContentBuilder {
    private readonly parts: ContentPart[] = [];

    text(text: string): this {
        if (text === '') return this;
        const last = this.parts.at(-1);
        if (last?.type === 'text') last.text += text;
        else this.parts.push(textPart(text));
        return this;
    }

    image(value: ImageInput): this {
        this.parts.push(imagePart(value));
        return this;
    }

    get empty(): boolean {
        return this.parts.length === 0;
    }

    build(): MessageContent {
        if (this.parts.length === 0) return '';
        if (this.parts.length === 1 && this.parts[0].type === 'text') return this.parts[0].text;
        return [...this.parts];
    }
}

/**
 * Render the input fields of one turn through `block`, which writes a single
 * field. Image inputs become real image parts after their marker; undefined
 * inputs are skipped. Returns whether anything was written.
 */
export function writeInputFields(
    builder: ContentBuilder,
    fields: AdapterField[],
    values: Record<string, unknown>,
    block: (name: string, value: string) => string,
    separator = '\n\n'
): boolean {
    let wrote = false;
    for (const field of fields) {
        const value = values[field.name];
        if (value === undefined) continue;
        if (wrote) builder.text(separator);
        if (isImageFieldType(field.type)) {
            // The block is written around a placeholder and split, so a
            // format with a closing marker (`</picture>`) keeps it after the image.
            const [before, after] = block(field.name, IMAGE_SLOT).split(IMAGE_SLOT);
            builder.text(before);
            builder.image(value as ImageInput);
            builder.text(after ?? '');
        } else {
            builder.text(block(field.name, formatFieldValue(value)));
        }
        wrote = true;
    }
    return wrote;
}

// NUL never appears in a legitimate prompt, so the slot cannot collide.
const IMAGE_SLOT = '\u0000ts-dspy:image\u0000';

// ------------------------------------------------------------------ demos

/** One demo or history turn, split into its input half and its output half. */
export interface Turn {
    inputs: Record<string, unknown>;
    outputs: Record<string, unknown>;
}

/**
 * Split one demo into its input half and its output half.
 *
 * An `Example` that has been through `withInputs()` already knows its own
 * split, so honour it. One that has not is split by the signature instead,
 * which is why `new Example({ question, answer })` works as a demo without
 * extra ceremony. Only fields the signature declares render — an extra key
 * such as a bootstrapped `reasoning` renders once the signature declares it,
 * which `ChainOfThought`'s does.
 *
 * A demo may leave fields out — a `ChainOfThought` demo without `reasoning`
 * is fine — and renders with what it has.
 */
export function splitDemo(demo: Example, layout: SignatureLayout): Turn {
    let inputSource: Record<string, unknown>;
    let outputSource: Record<string, unknown>;
    try {
        inputSource = demo.getInputs();
        outputSource = demo.getOutputs();
    } catch {
        // No explicit input keys: let the signature decide which side is which.
        const data = demo.toObject();
        inputSource = data;
        outputSource = data;
    }
    return {
        inputs: pickPresent(
            inputSource,
            layout.inputs.map((field) => field.name)
        ),
        outputs: pickPresent(
            outputSource,
            layout.outputs.map((field) => field.name)
        ),
    };
}

/** The demos that share at least one field with the signature, split. */
export function demoTurns(demos: Example[] | undefined, layout: SignatureLayout): Turn[] {
    return (demos ?? [])
        .map((demo) => splitDemo(demo, layout))
        .filter(
            (turn) => Object.keys(turn.inputs).length + Object.keys(turn.outputs).length > 0
        );
}

/**
 * The prior turns of a call: an explicit `history`, then the value of the
 * signature's `history` input, each entry split by the signature.
 */
export function historyTurns(input: AdapterFormatInput, layout: SignatureLayout): Turn[] {
    const histories: History[] = [];
    if (input.history) histories.push(input.history);
    if (layout.historyField) {
        const fromInputs = toHistory(input.inputs[layout.historyField]);
        if (fromInputs) histories.push(fromInputs);
    }
    const inputNames = layout.inputs.map((field) => field.name);
    const outputNames = layout.outputs.map((field) => field.name);
    return histories
        .flatMap((history) => history.messages)
        .map((message) => ({
            inputs: pickPresent(message, inputNames),
            outputs: pickPresent(message, outputNames),
        }))
        .filter(
            (turn) => Object.keys(turn.inputs).length + Object.keys(turn.outputs).length > 0
        );
}

function pickPresent(
    source: Record<string, unknown>,
    names: string[]
): Record<string, unknown> {
    const picked: Record<string, unknown> = {};
    for (const name of names) {
        const value = source[name];
        if (value !== undefined && value !== null) picked[name] = value;
    }
    return picked;
}

/** Only the named output fields of `values`, matched case-insensitively. */
export function pickOutputs(
    values: Record<string, unknown>,
    outputNames: string[]
): Record<string, unknown> {
    const byLowerName = new Map(Object.keys(values).map((key) => [key.toLowerCase(), key]));
    const picked: Record<string, unknown> = {};
    for (const name of outputNames) {
        const key = name in values ? name : byLowerName.get(name.toLowerCase());
        if (key !== undefined && values[key] !== undefined) picked[name] = values[key];
    }
    return picked;
}

// ------------------------------------------------------------------ text

/**
 * Messages as one block of text, for trace entries: `role: content` per
 * message, images as their placeholders.
 */
export function renderTranscript(messages: ChatMessage[]): string {
    return messages
        .map((message) => `${message.role}: ${contentToText(message.content)}`)
        .join('\n\n');
}

/** Output field names of a signature, in declaration order. */
export function outputNames(signature: SignatureLike): string[] {
    return Object.keys(getOutputFieldConfigs(signature));
}
