/** Base class for every error thrown by ts-dspy. */
export class TsDspyError extends Error {
    constructor(message: string, options?: { cause?: unknown }) {
        super(message, options);
        this.name = new.target.name;
    }
}

/** One field's worth of detail about why validation failed. */
export interface FieldValidationIssue {
    /** Name of the output field that failed. */
    field: string;
    /** The declared field type from the signature, e.g. `number`. */
    expected: string;
    /** The raw value extracted from the model output, before coercion. */
    received: unknown;
    /** Human-readable explanation. */
    message: string;
}

/**
 * Thrown when a model's output does not satisfy the signature's output fields.
 *
 * Previously ts-dspy silently returned the raw string when coercion failed, so a
 * field declared `number` could arrive as a string and TypeScript would never know.
 * Validation failures are now loud.
 */
export class ValidationError extends TsDspyError {
    readonly issues: FieldValidationIssue[];
    readonly rawOutput: string;

    constructor(issues: FieldValidationIssue[], rawOutput: string) {
        const summary = issues
            .map((issue) => `  - ${issue.field} (${issue.expected}): ${issue.message}`)
            .join('\n');
        super(`Model output failed validation:\n${summary}`);
        this.issues = issues;
        this.rawOutput = rawOutput;
    }
}

/** Thrown when a provider call fails, wrapping the underlying SDK error. */
export class LMError extends TsDspyError {
    /** Provider name, e.g. `openai`. */
    readonly provider: string;
    /** HTTP status, when the underlying SDK reported one. */
    readonly status?: number;

    constructor(
        provider: string,
        message: string,
        options?: { cause?: unknown; status?: number }
    ) {
        super(`[${provider}] ${message}`, { cause: options?.cause });
        this.provider = provider;
        this.status = options?.status;
    }
}
