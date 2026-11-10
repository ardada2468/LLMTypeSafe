import { type SignatureLike } from '../core/signature';
import { getOutputFieldConfigs, validateOutputs } from './schema';

/**
 * Tolerant `field: value` parsing.
 *
 * The adapters own prompt formatting and parsing; what is left here is the
 * heuristic reader `ChatAdapter` falls back on when a model ignores its
 * `[[ ## field ## ]]` markers and answers `answer: Paris` instead — common on
 * small local models, and not worth a second round trip to recover.
 */

/** Render one input value for a prompt line, keeping structured values readable. */
export function formatInputValue(value: unknown): string {
    if (value === null || typeof value !== 'object') return String(value);
    return JSON.stringify(value);
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
export function parseOutput(signature: SignatureLike, rawOutput: string): Record<string, any> {
    const text = typeof rawOutput === 'string' ? rawOutput : String(rawOutput);
    const fieldNames = Object.keys(getOutputFieldConfigs(signature));
    return validateOutputs(signature, extractLabelledFields(text, fieldNames), text);
}

/**
 * Pull `field: value` lines for each named field out of free text, without
 * validating them. Fields that cannot be located are absent from the result.
 *
 * A single-field reply with no label at all is taken as that field's value,
 * which is how a model asked for one thing usually answers.
 */
export function extractLabelledFields(
    text: string,
    fieldNames: string[]
): Record<string, string> {
    const extracted: Record<string, string> = {};
    for (const name of fieldNames) {
        const value = extractFieldValue(text, name, fieldNames);
        // Absent rather than null: an optional field should pass validation when
        // missing, and a required one should fail with a clear message.
        if (value !== null) {
            extracted[name] = value;
        }
    }
    return extracted;
}

/** Escape a field name so it can be safely interpolated into a RegExp. */
function escapeRegExp(value: string): string {
    return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function cleanValue(value: string): string {
    const bare = value.replace(/^\*+|\*+$/g, '').trim();
    // Only a quote that wraps the whole value is decoration; stripping each
    // end independently would eat the closing quote of `She said "maybe"`.
    const quote = bare[0];
    if (bare.length >= 2 && (quote === '"' || quote === "'") && bare.endsWith(quote)) {
        return bare.slice(1, -1).trim();
    }
    return bare;
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
        // The lookbehind keeps `type` from matching inside `prototype:`.
        new RegExp(`(?<![\\w])${escaped}\\s*[:=]\\s*(.+)`, 'i'),
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
