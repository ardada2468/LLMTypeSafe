import {
    AuthError,
    ContentFilterError,
    ContextLengthError,
    LMError,
    RateLimitError,
    TimeoutError,
    TsDspyError,
    classify,
} from './errors';

describe('LMError subclasses', () => {
    it('keeps every subclass catchable as LMError and TsDspyError', () => {
        for (const ErrorClass of [
            RateLimitError,
            AuthError,
            ContextLengthError,
            ContentFilterError,
            TimeoutError,
        ]) {
            const error = new ErrorClass('openai', 'nope');

            expect(error).toBeInstanceOf(LMError);
            expect(error).toBeInstanceOf(TsDspyError);
            expect(error).toBeInstanceOf(Error);
        }
    });

    it('names itself after the concrete subclass', () => {
        expect(new RateLimitError('openai', 'slow down').name).toBe('RateLimitError');
    });

    it('prefixes the message with the provider and keeps the cause', () => {
        const cause = new Error('underlying');
        const error = new AuthError('gemini', 'bad key', { cause, status: 401 });

        expect(error.message).toBe('[gemini] bad key');
        expect(error.provider).toBe('gemini');
        expect(error.status).toBe(401);
        expect(error.cause).toBe(cause);
    });

    it('carries a category on ContentFilterError', () => {
        const error = new ContentFilterError('anthropic', 'declined', { category: 'cyber' });

        expect(error.category).toBe('cyber');
    });
});

describe('classify', () => {
    it('maps rate limiting by status or by type', () => {
        expect(classify(429)).toBe(RateLimitError);
        expect(classify(undefined, { type: 'rate_limit_error' })).toBe(RateLimitError);
    });

    it('maps authentication and permission failures to AuthError', () => {
        expect(classify(401)).toBe(AuthError);
        expect(classify(403)).toBe(AuthError);
        expect(classify(undefined, { type: 'authentication_error' })).toBe(AuthError);
        expect(classify(undefined, { type: 'permission_error' })).toBe(AuthError);
        expect(classify(400, { code: 'invalid_api_key' })).toBe(AuthError);
    });

    it('maps timeouts to TimeoutError', () => {
        expect(classify(408)).toBe(TimeoutError);
        expect(classify(undefined, { type: 'timeout_error' })).toBe(TimeoutError);
    });

    it('prefers the code over the status for context length', () => {
        // OpenAI reports context overflow as a plain 400; only the code says so.
        expect(classify(400, { code: 'context_length_exceeded' })).toBe(ContextLengthError);
        expect(classify(400, { code: 'invalid_value' })).toBe(LMError);
        // `string_above_max_length` means one parameter was too long, which
        // shortening the prompt would never fix.
        expect(classify(400, { code: 'string_above_max_length' })).toBe(LMError);
    });

    it('falls back to the message only when a provider offers nothing better', () => {
        expect(classify(400, { message: 'prompt is too long: 250000 tokens' })).toBe(
            ContextLengthError
        );
        expect(classify(400, { message: 'The input token count exceeds the maximum' })).toBe(
            ContextLengthError
        );
        expect(classify(400, { message: 'unknown field "foo"' })).toBe(LMError);
        // A matching message on a status that is not 400 proves nothing.
        expect(classify(500, { message: 'context length' })).toBe(LMError);
    });

    it('leaves anything unrecognised as a plain LMError', () => {
        expect(classify(500)).toBe(LMError);
        expect(classify(undefined)).toBe(LMError);
        expect(classify(undefined, { type: 'api_error' })).toBe(LMError);
    });

    it('never returns ContentFilterError, which is a 200-response condition', () => {
        for (const status of [400, 401, 403, 408, 429, 500]) {
            expect(classify(status)).not.toBe(ContentFilterError);
        }
    });
});
