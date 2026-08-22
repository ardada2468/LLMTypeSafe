import { Signature } from '../core/signature';
import { type Example } from '../core/example';
import { ValidationError, type FieldValidationIssue } from '../core/errors';
import { buildOutputSchema, getOutputFieldConfigs } from './schema';

/**
 * How demos are written into the prompt.
 *
 * `labelled` mirrors the `field: value` text {@link parseOutput} reads back, and
 * suits a provider answering in plain text. `json` suits a provider whose
 * decoding is constrained to a JSON schema, where labelled examples would be
 * demonstrating a shape the model is not allowed to emit.
 */
export type DemoFormat = 'labelled' | 'json';

export interface RenderDemosOptions {
    /** Defaults to `labelled`. */
    format?: DemoFormat;
}

/**
 * Render a prompt for one call.
 *
 * `demos` are worked examples shown before the real input, so the model can see
 * the task performed correctly before attempting it. They render in the shape
 * the reply is expected to take, which is what makes them teach the output
 * format rather than merely illustrate the task. With no demos the output is
 * byte-for-byte what it was before few-shot support existed.
 */
export function buildPrompt(
    signature: typeof Signature | string,
    inputs: Record<string, any>,
    demos: Example[] = [],
    options: RenderDemosOptions = {}
): string {
    const demoBlock = renderDemos(signature, demos, options);

    if (typeof signature === 'string') {
        return buildPromptFromString(signature, inputs, demoBlock);
    }
    return buildPromptFromClass(signature, inputs, demoBlock);
}

function buildPromptFromString(
    signatureStr: string,
    inputs: Record<string, any>,
    demoBlock = ''
): string {
    const parsed = Signature.parseStringSignature(signatureStr);

    let prompt = demoBlock;

    for (const inputKey of parsed.inputs) {
        if (inputs[inputKey] !== undefined) {
            prompt += `${inputKey}: ${inputs[inputKey]}\n`;
        }
    }

    if (parsed.outputs.length === 1) {
        const outputKey = parsed.outputs[0];
        prompt += `\nProvide the ${outputKey} in this format:\n${outputKey}: [your response]`;
    } else {
        prompt += '\nProvide the following fields:\n';
        for (const outputKey of parsed.outputs) {
            const typeInfo = parsed.types[outputKey] ? ` (${parsed.types[outputKey]})` : '';
            prompt += `${outputKey}${typeInfo}: [your response]\n`;
        }
    }

    return prompt.trim();
}

function buildPromptFromClass(
    signatureClass: typeof Signature,
    inputs: Record<string, any>,
    demoBlock = ''
): string {
    const inputFields = signatureClass.getInputFields();
    const outputFields = signatureClass.getOutputFields();

    let prompt = '';

    if (signatureClass.description) {
        prompt += `${signatureClass.description}\n\n`;
    }

    // After the task description, before the real input: the model reads what
    // the task is, then sees it done, then does it.
    prompt += demoBlock;

    Object.entries(inputFields).forEach(([key, config]) => {
        if (inputs[key] !== undefined) {
            const prefix = config.prefix || `${key}:`;
            prompt += `${prefix} ${inputs[key]}\n`;
        }
    });

    prompt += '\nProvide:\n';
    Object.entries(outputFields).forEach(([key, config]) => {
        const desc = config.description ? ` (${config.description})` : '';
        prompt += `${key}${desc}:\n`;
    });

    return prompt.trim();
}

/**
 * Render worked examples as a prompt preamble.
 *
 * Exported so a caller can inspect exactly what few-shot text a set of demos
 * produces — useful when tuning a prompt by hand. Returns an empty string when
 * there is nothing to show, so callers can concatenate unconditionally.
 */
export function renderDemos(
    signature: typeof Signature | string,
    demos: Example[] = [],
    options: RenderDemosOptions = {}
): string {
    if (demos.length === 0) {
        return '';
    }

    const format = options.format ?? 'labelled';
    const { inputs: inputNames, outputs: outputNames } = signatureFieldNames(signature);
    const inputFields = typeof signature === 'string' ? {} : signature.getInputFields();

    const blocks: string[] = [];
    for (const demo of demos) {
        const { inputs, outputs } = splitDemo(demo, inputNames, outputNames);

        // A demo sharing no fields with the signature teaches nothing, so skip
        // it rather than emitting an empty numbered block.
        if (Object.keys(inputs).length === 0 && Object.keys(outputs).length === 0) {
            continue;
        }

        const body =
            format === 'json'
                ? renderJsonDemo(inputs, outputs)
                : renderLabelledDemo(inputs, outputs, inputFields);
        blocks.push(`Example ${blocks.length + 1}:\n${body}`);
    }

    if (blocks.length === 0) {
        return '';
    }

    const verb = blocks.length === 1 ? 'is' : 'are';
    const noun = blocks.length === 1 ? 'example' : 'examples';
    // The labelled form is the only one that can promise "the same format": on
    // the JSON path the schema instruction, not the demo, dictates the shape.
    const trailer =
        format === 'json'
            ? 'Now complete the next one.'
            : 'Now complete the next one in the same format.';

    return (
        `Here ${verb} ${blocks.length} worked ${noun} of this task:\n\n` +
        `${blocks.join('\n\n')}\n\n` +
        `${trailer}\n\n`
    );
}

function renderLabelledDemo(
    inputs: Record<string, any>,
    outputs: Record<string, any>,
    inputFields: Record<string, { prefix?: string }>
): string {
    const lines: string[] = [];

    for (const [key, value] of Object.entries(inputs)) {
        const prefix = inputFields[key]?.prefix || `${key}:`;
        lines.push(`${prefix} ${formatDemoValue(value)}`);
    }
    // Output labels stay plain `key: value` even when the input side uses a
    // custom prefix: that is the shape parseOutput reads back, and a demo
    // teaching any other shape would teach the model to break the parser.
    for (const [key, value] of Object.entries(outputs)) {
        lines.push(`${key}: ${formatDemoValue(value)}`);
    }

    return lines.join('\n');
}

function renderJsonDemo(inputs: Record<string, any>, outputs: Record<string, any>): string {
    return `input: ${JSON.stringify(inputs)}\noutput: ${JSON.stringify(outputs)}`;
}

/** A signature's declared field names, in declaration order. */
function signatureFieldNames(signature: typeof Signature | string): {
    inputs: string[];
    outputs: string[];
} {
    if (typeof signature === 'string') {
        const parsed = Signature.parseStringSignature(signature);
        return { inputs: parsed.inputs, outputs: parsed.outputs };
    }
    return {
        inputs: Object.keys(signature.getInputFields()),
        outputs: Object.keys(signature.getOutputFields()),
    };
}

/**
 * Split one demo into its input half and its output half.
 *
 * An `Example` that has been through `withInputs()` already knows its own split,
 * so honour it. One that has not is split by the signature instead, which is why
 * `new Example({ question, answer })` works as a demo without extra ceremony.
 *
 * Declared fields lead, in signature order, so demos stay stable and match the
 * shape of the real call. An example that declared its own split may also carry
 * output fields the signature never declared, and those follow — `reasoning` on
 * a bootstrapped `ChainOfThought` demo is exactly that, and dropping it would
 * throw away the most valuable part of the trace. Where no split was declared
 * there is no way to tell a stray key from an input, so only declared fields
 * render.
 */
function splitDemo(
    demo: Example,
    inputNames: string[],
    outputNames: string[]
): { inputs: Record<string, any>; outputs: Record<string, any> } {
    let inputSource: Record<string, any>;
    let outputSource: Record<string, any>;
    let declaredOwnSplit: boolean;

    try {
        inputSource = demo.getInputs();
        outputSource = demo.getOutputs();
        declaredOwnSplit = true;
    } catch {
        // No explicit input keys: let the signature decide which side is which.
        const data = demo.toObject();
        inputSource = data;
        outputSource = data;
        declaredOwnSplit = false;
    }

    const extras = declaredOwnSplit
        ? Object.keys(outputSource).filter((key) => !outputNames.includes(key))
        : [];

    return {
        inputs: pickInOrder(inputSource, inputNames),
        outputs: pickInOrder(outputSource, [...outputNames, ...extras]),
    };
}

function pickInOrder(source: Record<string, any>, names: string[]): Record<string, any> {
    const picked: Record<string, any> = {};
    for (const name of names) {
        const value = source[name];
        if (value !== undefined && value !== null) {
            picked[name] = value;
        }
    }
    return picked;
}

/** Render a demo value the way {@link parseOutput} would read it back. */
function formatDemoValue(value: unknown): string {
    if (typeof value === 'string') {
        return value;
    }
    if (value instanceof Date) {
        return value.toISOString();
    }
    if (typeof value === 'object') {
        return JSON.stringify(value);
    }
    return String(value);
}

/**
 * Parse and validate a model's raw text output against a signature.
 *
 * Fields are extracted heuristically from the text, then validated against the
 * signature's declared types.
 *
 * @throws {ValidationError} when a required field is missing or a field's value
 * cannot be coerced to its declared type.
 */
export function parseOutput(
    signature: typeof Signature | string,
    rawOutput: string
): Record<string, any> {
    const fields = getOutputFieldConfigs(signature);
    const fieldNames = Object.keys(fields);
    const text = typeof rawOutput === 'string' ? rawOutput : String(rawOutput);

    const extracted: Record<string, unknown> = {};
    for (const name of fieldNames) {
        const value = extractFieldValue(text, name, fieldNames);
        // Absent rather than null: an optional field should pass validation when
        // missing, and a required one should fail with a clear message.
        if (value !== null) {
            extracted[name] = value;
        }
    }

    const result = buildOutputSchema(signature).safeParse(extracted);
    if (result.success) {
        return result.data as Record<string, any>;
    }

    const issues: FieldValidationIssue[] = result.error.issues.map((issue) => {
        const field = String(issue.path[0] ?? '(root)');
        const declaredType = fields[field]?.type ?? 'string';
        const received = extracted[field];
        const message =
            received === undefined
                ? 'field not found in model output'
                : `${issue.message} (received ${JSON.stringify(received)})`;
        return { field, expected: declaredType, received, message };
    });

    throw new ValidationError(issues, text);
}

/** Escape a field name so it can be safely interpolated into a RegExp. */
function escapeRegExp(value: string): string {
    return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function cleanValue(value: string): string {
    return value
        .replace(/^\*+|\*+$/g, '')
        .replace(/^["']|["']$/g, '')
        .trim();
}

/**
 * Pull one field's raw text value out of a model response.
 *
 * Returns `null` when the field cannot be located, leaving the decision about
 * whether that is an error to the validation step.
 */
function extractFieldValue(
    text: string,
    fieldName: string,
    allFieldNames: string[]
): string | null {
    const escaped = escapeRegExp(fieldName);
    // Stop at the next known field label so multi-field responses, and values
    // that legitimately span several lines, don't bleed into one another.
    const nextLabel = allFieldNames.map(escapeRegExp).join('|');

    const patterns = [
        // "fieldName: value", running to the next known label or end of input.
        // No `m` flag: `$` must mean end of input, not end of line, or a
        // multi-line value would be truncated at its first newline.
        new RegExp(
            `(?:^|\\n)[ \\t]*${escaped}[ \\t]*[:=][ \\t]*([\\s\\S]*?)(?=\\n[ \\t]*(?:${nextLabel})[ \\t]*[:=]|$)`,
            'i'
        ),
        // Field name appearing mid-line, e.g. "**answer:** 42".
        new RegExp(`${escaped}\\s*[:=]\\s*(.+)`, 'i'),
    ];

    for (const pattern of patterns) {
        const match = text.match(pattern);
        if (match?.[1]) {
            const value = cleanValue(match[1]);
            if (value !== '') {
                return value;
            }
        }
    }

    // A single-output signature answered with a bare value carries no `field:`
    // marker to match on, so fall back to treating the whole response as the
    // value. Only after labelled extraction has failed — a field name containing
    // regex metacharacters still labels its value, and should win.
    if (allFieldNames.length === 1) {
        const value = cleanValue(text);
        return value === '' ? null : value;
    }

    return null;
}
