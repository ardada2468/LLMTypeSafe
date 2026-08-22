import { type ValidationError, type FieldValidationIssue } from './errors';

/**
 * Prompt fragments for asking a model to fix a response that failed validation.
 *
 * A model that gets a field's type wrong has usually understood the task and
 * merely fumbled the shape, so one more round-trip that names the offending
 * fields tends to recover it. `Predict`, `ChainOfThought` and `RespAct` all
 * build their corrective prompt here so there is a single wording to tune.
 */

/** Which output shape the model is being asked to correct. */
export type RepairFormat = 'text' | 'structured';

/** Comma-separated names of the fields that failed, de-duplicated, in order. */
export function listFailingFields(issues: readonly FieldValidationIssue[]): string {
    return [...new Set(issues.map((issue) => issue.field))].join(', ');
}

/**
 * One indented line per failing field: what the signature declared, and what the
 * model actually sent.
 *
 * Deliberately does *not* quote `issue.message`. Those messages come from the
 * coercion layer and describe the post-coercion value, so a model that wrote
 * `very high` into a `number` field is told it sent `NaN` — which it did not,
 * and which is the one thing it must not repeat back. Developers still get the
 * full message from `error.issues`.
 */
export function describeValidationIssues(issues: readonly FieldValidationIssue[]): string {
    return issues
        .map((issue) => {
            const received =
                issue.received === undefined ? 'nothing' : JSON.stringify(issue.received);
            return `  - ${issue.field}: expected ${issue.expected}, received ${received}`;
        })
        .join('\n');
}

/**
 * Build the follow-up prompt for a failed prediction.
 *
 * Takes the *original* prompt rather than the previous repair prompt, so a
 * multi-attempt repair loop does not accumulate every earlier correction.
 */
export function buildRepairPrompt(
    basePrompt: string,
    error: ValidationError,
    format: RepairFormat = 'text'
): string {
    const closing =
        format === 'structured'
            ? 'Return the corrected object with every required field, using the declared type for each.'
            : 'Reply again with every required field on its own "field: value" line, using the declared type for each.';

    return [
        basePrompt,
        '',
        `Your previous response failed validation for: ${listFailingFields(error.issues)}.`,
        describeValidationIssues(error.issues),
        '',
        'Previous response:',
        error.rawOutput,
        '',
        closing,
    ].join('\n');
}

/**
 * Build the corrective `Observation:` body RespAct appends to its transcript
 * when a Final Answer fails validation.
 */
export function buildRepairObservation(error: ValidationError): string {
    return [
        `Your Final Answer was missing or malformed for: ${listFailingFields(error.issues)}.`,
        describeValidationIssues(error.issues),
        'Provide a Final Answer with every required field on its own "field: value" line.',
    ].join('\n');
}

/**
 * Upper bound on `repairAttempts`.
 *
 * Every attempt is a real, billed completion, so a typo in a config file should
 * not be able to fire off thousands of them. A model that has not found the
 * shape in ten tries is not going to.
 */
export const MAX_REPAIR_ATTEMPTS = 10;

/**
 * Normalise a caller-supplied `repairAttempts` into an integer in
 * `[0, MAX_REPAIR_ATTEMPTS]`.
 *
 * `Infinity` reads as "keep trying", so it clamps to the maximum rather than
 * falling back to zero. `NaN` is a mistake, not an intent, and yields zero.
 */
export function normaliseRepairAttempts(attempts: number | undefined): number {
    if (typeof attempts !== 'number' || Number.isNaN(attempts)) {
        return 0;
    }
    if (attempts === Number.POSITIVE_INFINITY) {
        return MAX_REPAIR_ATTEMPTS;
    }
    return Math.min(MAX_REPAIR_ATTEMPTS, Math.max(0, Math.trunc(attempts)));
}

/**
 * Whether two validation failures are indistinguishable.
 *
 * A repair prompt is a function of the original prompt and the latest error, so
 * an error identical to its predecessor produces a byte-identical prompt. Against
 * a deterministic model every remaining attempt is then guaranteed to fail the
 * same way, and stopping early saves the calls.
 */
export function isRepeatedFailure(a: ValidationError, b: ValidationError): boolean {
    return (
        a.rawOutput === b.rawOutput &&
        describeValidationIssues(a.issues) === describeValidationIssues(b.issues)
    );
}
