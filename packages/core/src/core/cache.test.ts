import { MemoryCache, buildCacheKey, type Cache, type CacheKeyInput } from './cache';
import { clearCache, configure, getCache, isCacheEnabled } from './config';
import { BaseLM } from './base-lm';
import { MockLM } from '../test-utils';
import type { ChatMessage, LLMCallOptions, ModelCapabilities } from '../types/language-model';

const baseKeyInput: CacheKeyInput = {
    provider: 'mock',
    model: 'mock-model',
    operation: 'chat',
    messages: [{ role: 'user', content: 'Capital of France?' }],
};

describe('MemoryCache', () => {
    it('returns a stored value and undefined for an unknown key', () => {
        const cache = new MemoryCache();

        cache.set('a', 'value');

        expect(cache.get('a')).toBe('value');
        expect(cache.get('missing')).toBeUndefined();
    });

    it('evicts the least recently used entry once maxSize is exceeded', () => {
        const cache = new MemoryCache({ maxSize: 2 });

        cache.set('a', 1);
        cache.set('b', 2);
        cache.set('c', 3);

        expect(cache.get('a')).toBeUndefined();
        expect(cache.get('b')).toBe(2);
        expect(cache.get('c')).toBe(3);
        expect(cache.size).toBe(2);
    });

    it('counts a read as a use, sparing the entry from the next eviction', () => {
        const cache = new MemoryCache({ maxSize: 2 });
        cache.set('a', 1);
        cache.set('b', 2);

        cache.get('a');
        cache.set('c', 3);

        expect(cache.get('a')).toBe(1);
        expect(cache.get('b')).toBeUndefined();
    });

    it('overwrites an existing key without growing', () => {
        const cache = new MemoryCache({ maxSize: 2 });

        cache.set('a', 1);
        cache.set('a', 2);

        expect(cache.get('a')).toBe(2);
        expect(cache.size).toBe(1);
    });

    it('empties on clear', () => {
        const cache = new MemoryCache();
        cache.set('a', 1);

        cache.clear();

        expect(cache.size).toBe(0);
        expect(cache.get('a')).toBeUndefined();
    });

    it('rejects a maxSize that cannot hold an entry', () => {
        expect(() => new MemoryCache({ maxSize: 0 })).toThrow(RangeError);
        expect(() => new MemoryCache({ maxSize: 1.5 })).toThrow(RangeError);
    });
});

describe('buildCacheKey', () => {
    it('produces the same fixed-length key for identical input', () => {
        const first = buildCacheKey(baseKeyInput);
        const second = buildCacheKey({ ...baseKeyInput });

        expect(first).toBe(second);
        expect(first).toHaveLength(64);
    });

    it('separates calls that differ in provider, model, or operation', () => {
        const key = buildCacheKey(baseKeyInput);

        expect(buildCacheKey({ ...baseKeyInput, provider: 'other' })).not.toBe(key);
        expect(buildCacheKey({ ...baseKeyInput, model: 'other-model' })).not.toBe(key);
        expect(buildCacheKey({ ...baseKeyInput, operation: 'generate' })).not.toBe(key);
    });

    it('separates calls that differ in message content or role', () => {
        const key = buildCacheKey(baseKeyInput);

        expect(
            buildCacheKey({ ...baseKeyInput, messages: [{ role: 'user', content: 'Other?' }] })
        ).not.toBe(key);
        expect(
            buildCacheKey({
                ...baseKeyInput,
                messages: [{ role: 'system', content: 'Capital of France?' }],
            })
        ).not.toBe(key);
    });

    it('separates every sampling parameter that changes the response', () => {
        const key = buildCacheKey(baseKeyInput);
        const variants: CacheKeyInput['options'][] = [
            { temperature: 0.7 },
            { topP: 0.5 },
            { maxTokens: 128 },
            { stopSequences: ['\n'] },
            { frequencyPenalty: 0.2 },
            { presencePenalty: 0.2 },
        ];

        const keys = variants.map((options) => buildCacheKey({ ...baseKeyInput, options }));

        expect(new Set([key, ...keys]).size).toBe(variants.length + 1);
    });

    it('separates structured calls that differ only in schema', () => {
        const input: CacheKeyInput = {
            ...baseKeyInput,
            operation: 'generateStructured',
            prompt: 'Extract',
            messages: undefined,
        };

        const withString = buildCacheKey({
            ...input,
            schema: { type: 'object', properties: { a: { type: 'string' } } },
        });
        const withNumber = buildCacheKey({
            ...input,
            schema: { type: 'object', properties: { a: { type: 'number' } } },
        });

        expect(withString).not.toBe(withNumber);
    });

    it('ignores schema property order so one request keeps one entry', () => {
        const input: CacheKeyInput = { ...baseKeyInput, operation: 'generateStructured' };

        const first = buildCacheKey({ ...input, schema: { type: 'object', title: 'Q' } });
        const second = buildCacheKey({ ...input, schema: { title: 'Q', type: 'object' } });

        expect(first).toBe(second);
    });

    it('separates instances whose provider configuration differs', () => {
        const key = buildCacheKey(baseKeyInput);

        expect(buildCacheKey({ ...baseKeyInput, scope: { maxTokens: 64 } })).not.toBe(key);
        expect(buildCacheKey({ ...baseKeyInput, scope: { maxTokens: 64 } })).not.toBe(
            buildCacheKey({ ...baseKeyInput, scope: { maxTokens: 8192 } })
        );
    });

    it('ignores transport options that cannot change the response', () => {
        const key = buildCacheKey(baseKeyInput);

        expect(buildCacheKey({ ...baseKeyInput, options: { timeout: 5000, retries: 3 } })).toBe(
            key
        );
    });
});

describe('configure({ cache })', () => {
    beforeEach(async () => {
        // The memory cache is a process-wide singleton, so empty it between cases.
        await clearCache();
        configure({ cache: false });
    });

    it('is off until it is asked for', () => {
        expect(isCacheEnabled()).toBe(false);
        expect(getCache()).toBeUndefined();
    });

    it('hands back a memory cache when enabled with a boolean', () => {
        configure({ cache: true });

        expect(isCacheEnabled()).toBe(true);
        expect(getCache()).toBeInstanceOf(MemoryCache);
    });

    it('hands back the same instance across calls so entries survive', () => {
        configure({ cache: true });

        expect(getCache()).toBe(getCache());
    });

    it('empties the memory cache even while caching is disabled', async () => {
        configure({ cache: true });
        const cache = getCache() as MemoryCache;
        cache.set('a', 1);
        configure({ cache: false });

        await clearCache();
        configure({ cache: true });

        expect((getCache() as MemoryCache).size).toBe(0);
    });

    it('accepts a custom implementation in place of a boolean', () => {
        const custom: Cache = { get: () => undefined, set: () => {} };

        configure({ cache: custom });

        expect(isCacheEnabled()).toBe(true);
        expect(getCache()).toBe(custom);
    });
});

describe('cached language-model calls', () => {
    beforeEach(async () => {
        await clearCache();
        configure({ cache: true });
    });

    afterEach(() => {
        configure({ cache: false });
    });

    it('serves a repeated chat call without reaching the provider', async () => {
        const lm = new MockLM({ responses: ['first', 'second'] });
        const messages = [{ role: 'user' as const, content: 'Capital of France?' }];

        const first = await lm.chat(messages);
        const second = await lm.chat(messages);

        expect(first).toBe('first');
        expect(second).toBe('first');
        expect(lm.calls).toHaveLength(1);
    });

    it('reaches the provider again when the per-call model override differs', async () => {
        const lm = new MockLM({ responses: ['from default', 'from override'] });

        const base = await lm.generate('Same prompt');
        const overridden = await lm.generate('Same prompt', { model: 'other-model' });

        expect(base).toBe('from default');
        expect(overridden).toBe('from override');
        expect(lm.calls).toHaveLength(2);
    });

    it('reaches the provider again when a sampling parameter differs', async () => {
        const lm = new MockLM({ responses: ['cold', 'warm'] });

        const cold = await lm.generate('Same prompt', { temperature: 0 });
        const warm = await lm.generate('Same prompt', { temperature: 0.9 });

        expect(cold).toBe('cold');
        expect(warm).toBe('warm');
        expect(lm.calls).toHaveLength(2);
    });

    it('keeps generate and chat entries apart from another prompt', async () => {
        const lm = new MockLM({ responses: ['a', 'b'] });

        await lm.generate('prompt one');
        await lm.generate('prompt two');

        expect(lm.calls).toHaveLength(2);
    });

    it('serves a repeated structured call without reaching the provider', async () => {
        const lm = new MockLM({
            structuredResponses: [{ answer: 'Paris' }, { answer: 'Rome' }],
        });
        const schema = { type: 'object', properties: { answer: { type: 'string' } } };

        const first = await lm.generateStructured('Capital of France?', schema);
        const second = await lm.generateStructured('Capital of France?', schema);

        expect(first).toEqual({ answer: 'Paris' });
        expect(second).toEqual({ answer: 'Paris' });
        expect(lm.structuredCalls).toHaveLength(1);
    });

    it('reaches the provider again when the schema differs', async () => {
        const lm = new MockLM({
            structuredResponses: [{ answer: 'Paris' }, { city: 'Paris' }],
        });

        const first = await lm.generateStructured('Capital of France?', {
            properties: { answer: { type: 'string' } },
        });
        const second = await lm.generateStructured('Capital of France?', {
            properties: { city: { type: 'string' } },
        });

        expect(first).toEqual({ answer: 'Paris' });
        expect(second).toEqual({ city: 'Paris' });
        expect(lm.structuredCalls).toHaveLength(2);
    });

    it('leaves request and token counts untouched on a hit, counting cacheHits instead', async () => {
        const lm = new MockLM({ responses: ['only'] });

        await lm.generate('Capital of France?');
        const afterMiss = lm.getUsage();
        await lm.generate('Capital of France?');
        const afterHit = lm.getUsage();

        expect(afterMiss.cacheHits).toBe(0);
        expect(afterHit.cacheHits).toBe(1);
        expect(afterHit.requestCount).toBe(afterMiss.requestCount);
        expect(afterHit.totalTokens).toBe(afterMiss.totalTokens);
    });

    it('writes one entry per public call rather than nesting a second', async () => {
        const cache = getCache() as MemoryCache;
        const lm = new MockLM({ responses: ['only'] });

        await lm.generate('Capital of France?');

        expect(cache.size).toBe(1);
    });

    it('clears cacheHits along with the rest of the usage counters', async () => {
        const lm = new MockLM({ responses: ['only'] });
        await lm.generate('Capital of France?');
        await lm.generate('Capital of France?');

        lm.resetUsage();

        expect(lm.getUsage().cacheHits).toBe(0);
    });

    it('does not cache a failure, so the next call retries the provider', async () => {
        const lm = new MockLM({ responses: [] });

        await expect(lm.generate('Capital of France?')).rejects.toThrow('no more scripted');
        lm.setResponses(['recovered']);

        await expect(lm.generate('Capital of France?')).resolves.toBe('recovered');
    });

    it('reaches the provider on every call once caching is turned off', async () => {
        configure({ cache: false });
        const lm = new MockLM({ responses: ['first', 'second'] });

        const first = await lm.generate('Capital of France?');
        const second = await lm.generate('Capital of France?');

        expect(first).toBe('first');
        expect(second).toBe('second');
        expect(lm.getUsage().cacheHits).toBe(0);
    });

    it('routes lookups through a custom asynchronous implementation', async () => {
        const store = new Map<string, unknown>();
        const custom: Cache = {
            get: async (key) => store.get(key),
            set: async (key, value) => {
                store.set(key, value);
            },
        };
        configure({ cache: custom });
        const lm = new MockLM({ responses: ['first', 'second'] });

        await lm.generate('Capital of France?');
        const second = await lm.generate('Capital of France?');

        expect(store.size).toBe(1);
        expect(second).toBe('first');
        expect(lm.calls).toHaveLength(1);
    });

    it('treats a null from a Redis-style store as a miss, not a hit', async () => {
        const custom: Cache = { get: () => null, set: () => {} };
        configure({ cache: custom });
        const lm = new MockLM({ responses: ['first', 'second'] });

        const first = await lm.generate('Capital of France?');
        const second = await lm.generate('Capital of France?');

        expect(first).toBe('first');
        expect(second).toBe('second');
        expect(lm.getUsage().cacheHits).toBe(0);
    });

    it('hands each structured caller its own object, so a mutation cannot spread', async () => {
        const lm = new MockLM({ structuredResponses: [{ answer: 'Paris' }] });
        const schema = { type: 'object' };

        const first = await lm.generateStructured<{ answer: string }>('Capital?', schema);
        first.answer = 'mutated';
        const second = await lm.generateStructured<{ answer: string }>('Capital?', schema);

        expect(second.answer).toBe('Paris');
    });

    it('keeps two instances apart when their cacheScope differs', async () => {
        const short = new ScopedLM(64, ['short answer']);
        const long = new ScopedLM(8192, ['long answer']);

        const first = await short.generate('Capital of France?');
        const second = await long.generate('Capital of France?');

        expect(first).toBe('short answer');
        expect(second).toBe('long answer');
    });

    it('shares entries between two instances configured identically', async () => {
        const first = new ScopedLM(64, ['first']);
        const second = new ScopedLM(64, ['second']);

        await first.generate('Capital of France?');
        const replayed = await second.generate('Capital of France?');

        expect(replayed).toBe('first');
        expect(second.getUsage().cacheHits).toBe(1);
    });
});

/** Stands in for a provider whose constructor options shape the request. */
class ScopedLM extends BaseLM {
    constructor(
        private readonly defaultMaxTokens: number,
        private readonly replies: string[]
    ) {
        super('scoped', 'scoped-1');
    }

    async chat(_messages: ChatMessage[], _options?: LLMCallOptions): Promise<string> {
        const reply = this.replies.shift();
        if (reply === undefined) throw new Error('ScopedLM: no more scripted replies');
        this.recordUsage({ promptTokens: 1, completionTokens: 1, latencyMs: 1 });
        return reply;
    }

    protected cacheScope(): unknown {
        return { defaultMaxTokens: this.defaultMaxTokens };
    }

    getCapabilities(): ModelCapabilities {
        return {
            supportsStreaming: false,
            supportsStructuredOutput: false,
            supportsFunctionCalling: false,
            supportsVision: false,
            maxContextLength: 8192,
            supportedFormats: ['text'],
        };
    }
}
