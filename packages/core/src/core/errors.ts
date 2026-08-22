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

/**
 * Thrown when the provider rejected the call for exceeding a rate or quota
 * limit. Almost always worth retrying after a backoff.
 */
export class RateLimitError extends LMError {}

/**
 * Thrown when the credentials were missing, wrong, or not entitled to the
 * model. Retrying is pointless until the key or the entitlement changes.
 */
export class AuthError extends LMError {}

/**
 * Thrown when the prompt did not fit the model's context window. The remedy is
 * a shorter prompt or a larger model, never a retry.
 */
export class ContextLengthError extends LMError {}

/**
 * Thrown when the provider's safety classifiers declined the request.
 *
 * All three providers report this on a 200 response rather than as a thrown SDK
 * error — Anthropic sets `stop_reason: 'refusal'`, Gemini sets
 * `promptFeedback.blockReason` or `finishReason: 'SAFETY'`, and OpenAI sets
 * `finish_reason: 'content_filter'` — so it comes from inspecting the response,
 * not from {@link classify}.
 */
export class ContentFilterError extends LMError {
    /**
     * Provider-specific category, when one is reported: Anthropic's refusal
     * category, or Gemini's block/finish reason.
     */
    readonly category?: string;

    constructor(
        provider: string,
        message: string,
        options?: { cause?: unknown; status?: number; category?: string }
    ) {
        super(provider, message, options);
        this.category = options?.category;
    }
}

/** Thrown when the request timed out or was aborted before a reply arrived. */
export class TimeoutError extends LMError {}

/** The shape every {@link LMError} subclass shares, as returned by {@link classify}. */
export type LMErrorClass = new (
    provider: string,
    message: string,
    options?: { cause?: unknown; status?: number }
) => LMError;

/** Provider-reported discriminators, as far as each SDK supplies them. */
export interface ErrorDiscriminators {
    /**
     * The provider's own error type. Anthropic's `error.type` union is the
     * cleanest of the three; OpenAI's is a loose string; Gemini has none.
     */
    type?: string | null;
    /**
     * The provider's own error code. Only OpenAI reports one, and it is the
     * reliable discriminator for context-length overflow.
     */
    code?: string | null;
    /**
     * The error message. Consulted only as a last resort, for providers that
     * report nothing more structured than an HTTP status.
     */
    message?: string;
}

/**
 * Message shapes that mean "the prompt did not fit".
 *
 * Text matching is fragile, so it is reached only when a provider gives us
 * nothing better: Gemini reports a bare 400, and Anthropic has no dedicated
 * context-length type — an over-long prompt is `invalid_request_error`.
 */
const CONTEXT_OVERFLOW_PATTERN =
    /context (?:length|window)|maximum context|too many tokens|(?:prompt|input|request) is too long|exceeds? the (?:maximum|context)/i;

/**
 * Pick the {@link LMError} subclass for a provider failure.
 *
 * Codes and types are consulted before the HTTP status, because they are what
 * actually distinguishes (say) a context-length overflow from any other 400.
 * An unrecognised failure stays a plain `LMError`, so every provider error
 * remains catchable as one.
 */
export function classify(
    status?: number,
    discriminators: ErrorDiscriminators = {}
): LMErrorClass {
    const { type, code, message } = discriminators;

    // OpenAI's `code` is the dependable context-length signal; matching on the
    // message text instead breaks whenever the wording is tweaked. Note that
    // `string_above_max_length` is deliberately absent: it means one string
    // parameter was too long, which shortening the prompt would never fix.
    if (code === 'context_length_exceeded') {
        return ContextLengthError;
    }
    if (type === 'timeout_error' || status === 408) return TimeoutError;
    if (type === 'rate_limit_error' || status === 429) return RateLimitError;
    if (
        type === 'authentication_error' ||
        type === 'permission_error' ||
        code === 'invalid_api_key' ||
        status === 401 ||
        status === 403
    ) {
        return AuthError;
    }
    if (status === 400 && message !== undefined && CONTEXT_OVERFLOW_PATTERN.test(message)) {
        return ContextLengthError;
    }

    return LMError;
}
