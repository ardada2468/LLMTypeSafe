import { buildCacheKey } from './cache';
import { mergeRequestBody, providerOptionsFor } from './provider-options';

describe('providerOptionsFor', () => {
    it('returns a copy of the entry for the provider', () => {
        const entry = { reasoningEffort: 'low' };
        const result = providerOptionsFor({ providerOptions: { openai: entry } }, 'openai');
        expect(result).toEqual(entry);
        expect(result).not.toBe(entry);
    });

    it('ignores other providers and non-object entries', () => {
        expect(providerOptionsFor({ providerOptions: { gemini: { a: 1 } } }, 'openai')).toEqual(
            {}
        );
        expect(providerOptionsFor({ providerOptions: { openai: 'x' } }, 'openai')).toEqual({});
        expect(providerOptionsFor(undefined, 'openai')).toEqual({});
    });
});

describe('mergeRequestBody', () => {
    it('merges nested objects and replaces everything else', () => {
        const base = { a: 1, nested: { keep: true, swap: 1 }, list: [1, 2] };
        const merged = mergeRequestBody(base, {
            nested: { swap: 2, added: 'x' },
            list: [3],
            b: 'new',
            skipped: undefined,
        });
        expect(merged).toEqual({
            a: 1,
            nested: { keep: true, swap: 2, added: 'x' },
            list: [3],
            b: 'new',
        });
        expect(base.nested).toEqual({ keep: true, swap: 1 });
    });

    it('returns the base unchanged without extras', () => {
        const base = { a: 1 };
        expect(mergeRequestBody(base, undefined)).toBe(base);
    });
});

describe('providerOptions in the cache key', () => {
    const input = { provider: 'openai', model: 'm', operation: 'chat' as const, prompt: 'p' };

    it('separates calls with different provider options', () => {
        const low = buildCacheKey({
            ...input,
            options: { providerOptions: { openai: { reasoningEffort: 'low' } } },
        });
        const high = buildCacheKey({
            ...input,
            options: { providerOptions: { openai: { reasoningEffort: 'high' } } },
        });
        expect(low).not.toBe(high);
    });

    it('leaves keys without provider options unchanged', () => {
        expect(buildCacheKey({ ...input, options: {} })).toBe(buildCacheKey(input));
    });
});
