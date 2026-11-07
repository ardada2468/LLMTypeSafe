import type { LLMCallOptions } from '../types/language-model';

/**
 * The `providerOptions` entry addressed to `provider`, or an empty object.
 * A non-object entry is ignored rather than spread, so a typo cannot turn into
 * a request body made of string indices.
 */
export function providerOptionsFor(
    options: LLMCallOptions | undefined,
    provider: string
): Record<string, unknown> {
    const entry = options?.providerOptions?.[provider];
    return isPlainObject(entry) ? { ...entry } : {};
}

/**
 * Merge `extra` into a provider request body. Plain objects merge
 * recursively, so `{ output_config: { effort: 'low' } }` keeps an
 * `output_config.format` already in the body; anything else — arrays,
 * primitives — replaces the existing value. Neither input is mutated.
 */
export function mergeRequestBody<T extends Record<string, unknown>>(
    base: T,
    extra: Record<string, unknown> | undefined
): T {
    if (!extra) return base;
    const merged: Record<string, unknown> = { ...base };
    for (const [key, value] of Object.entries(extra)) {
        if (value === undefined) continue;
        const current = merged[key];
        merged[key] =
            isPlainObject(current) && isPlainObject(value)
                ? mergeRequestBody(current, value)
                : value;
    }
    return merged as T;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
    if (value === null || typeof value !== 'object' || Array.isArray(value)) return false;
    const proto = Object.getPrototypeOf(value);
    return proto === Object.prototype || proto === null;
}
