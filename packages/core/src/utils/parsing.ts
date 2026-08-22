import { Signature, isImageFieldType } from '../core/signature';
import { ValidationError, type FieldValidationIssue } from '../core/errors';
import type { ContentPart, ImageInput, MessageContent } from '../types/language-model';
import { contentToText, imagePart, textPart } from './content';
import { buildOutputSchema, getOutputFieldConfigs } from './schema';

/**
 * Render a signature and its inputs as a plain-text prompt.
 *
 * Image inputs are flattened to a `[image: …]` placeholder, because a string
 * cannot carry pixels. Use {@link buildPromptContent} to send them for real.
 */
export function buildPrompt(
    signature: typeof Signature | string,
    inputs: Record<string, any>
): string {
    return contentToText(buildPromptContent(signature, inputs));
}

/**
 * Render a signature and its inputs as chat message content.
 *
 * Returns a plain `string` when every input is text — identical to what
 * {@link buildPrompt} produces — and an array of {@link ContentPart}s when any
 * input field is declared `image`, so the image travels as an image:
 *
 * ```ts
 * const content = buildPromptContent(DescribeReceipt, { receipt: dataUri });
 * await lm.chat([{ role: 'user', content }]);
 * ```
 */
export function buildPromptContent(
    signature: typeof Signature | string,
    inputs: Record<string, any>
): MessageContent {
    const parts =
        typeof signature === 'string'
            ? buildPartsFromString(signature, inputs)
            : buildPartsFromClass(signature, inputs);
    return collapse(parts);
}

/**
 * Accumulates prompt text, breaking it into parts wherever an image lands.
 *
 * Text is buffered so that a prompt without images ends up as exactly one part
 * holding exactly the string the old string-only builder produced.
 */
function partBuilder() {
    const parts: ContentPart[] = [];
    let buffer = '';

    const flush = () => {
        if (buffer !== '') {
            parts.push(textPart(buffer));
            buffer = '';
        }
    };

    return {
        text(chunk: string): void {
            buffer += chunk;
        },
        image(value: ImageInput): void {
            flush();
            parts.push(imagePart(value));
        },
        done(): ContentPart[] {
            flush();
            return trimEnds(parts);
        },
    };
}

/** Mirror the trailing `prompt.trim()` the string builders used to end with. */
function trimEnds(parts: ContentPart[]): ContentPart[] {
    const trimmed = [...parts];
    const first = trimmed[0];
    if (first?.type === 'text') {
        trimmed[0] = textPart(first.text.replace(/^\s+/, ''));
    }
    const last = trimmed[trimmed.length - 1];
    if (last?.type === 'text') {
        trimmed[trimmed.length - 1] = textPart(last.text.replace(/\s+$/, ''));
    }
    return trimmed.filter((part) => part.type !== 'text' || part.text !== '');
}

/** One text part is just a string; anything else stays a part array. */
function collapse(parts: ContentPart[]): MessageContent {
    if (parts.length === 0) return '';
    if (parts.length === 1 && parts[0].type === 'text') return parts[0].text;
    return parts;
}

function buildPartsFromString(
    signatureStr: string,
    inputs: Record<string, any>
): ContentPart[] {
    const parsed = Signature.parseStringSignature(signatureStr);
    const prompt = partBuilder();

    for (const inputKey of parsed.inputs) {
        if (inputs[inputKey] !== undefined) {
            prompt.text(`${inputKey}: `);
            if (isImageFieldType(parsed.types[inputKey])) {
                prompt.image(inputs[inputKey]);
            } else {
                prompt.text(`${inputs[inputKey]}`);
            }
            prompt.text('\n');
        }
    }

    if (parsed.outputs.length === 1) {
        const outputKey = parsed.outputs[0];
        prompt.text(
            `\nProvide the ${outputKey} in this format:\n${outputKey}: [your response]`
        );
    } else {
        prompt.text('\nProvide the following fields:\n');
        for (const outputKey of parsed.outputs) {
            const typeInfo = parsed.types[outputKey] ? ` (${parsed.types[outputKey]})` : '';
            prompt.text(`${outputKey}${typeInfo}: [your response]\n`);
        }
    }

    return prompt.done();
}

function buildPartsFromClass(
    signatureClass: typeof Signature,
    inputs: Record<string, any>
): ContentPart[] {
    const inputFields = signatureClass.getInputFields();
    const outputFields = signatureClass.getOutputFields();
    const prompt = partBuilder();

    if (signatureClass.description) {
        prompt.text(`${signatureClass.description}\n\n`);
    }

    Object.entries(inputFields).forEach(([key, config]) => {
        if (inputs[key] !== undefined) {
            const prefix = config.prefix || `${key}:`;
            prompt.text(`${prefix} `);
            if (isImageFieldType(config.type)) {
                prompt.image(inputs[key]);
            } else {
                prompt.text(`${inputs[key]}`);
            }
            prompt.text('\n');
        }
    });

    prompt.text('\nProvide:\n');
    Object.entries(outputFields).forEach(([key, config]) => {
        const desc = config.description ? ` (${config.description})` : '';
        prompt.text(`${key}${desc}:\n`);
    });

    return prompt.done();
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
