import { AsyncLocalStorage } from 'node:async_hooks';
import type { UsageStats } from '../types/language-model';
import { getPricing, type ModelPricing } from './config';

/**
 * Usage keyed by `provider/model`, e.g. `{ 'openai/gpt-5.2': { ... } }`. One
 * invocation can touch several models — a fallback, a teacher, a judge — and
 * each is priced differently, so they are never summed together implicitly.
 */
export type LMUsage = Record<string, UsageStats>;

/** One provider round trip's worth of usage, as a provider reports it. */
export interface UsageDelta {
    promptTokens?: number;
    completionTokens?: number;
    /** Prompt tokens served from the provider's prompt cache (a subset of `promptTokens`). */
    cachedTokens?: number;
    /** Hidden reasoning tokens (a subset of `completionTokens`). */
    reasoningTokens?: number;
    latencyMs?: number;
    /** Counts towards `requestCount`. Defaults to true. */
    request?: boolean;
    /** Counts towards `errorCount`. */
    error?: boolean;
    /** Counts towards `cacheHits` (ts-dspy's response cache, not the provider's). */
    cacheHit?: boolean;
}

/** Running totals for one model, with latency kept as a sum so means combine. */
interface Tally {
    promptTokens: number;
    completionTokens: number;
    cachedTokens: number;
    reasoningTokens: number;
    requestCount: number;
    errorCount: number;
    cacheHits: number;
    latencySum: number;
    cost: number;
    priced: boolean;
}

/**
 * Accumulates usage for one scope — a module invocation, one LM call, an
 * evaluation run — and forwards every delta to the enclosing scope's
 * collector as well, so a nested module's usage is part of its parent's.
 *
 * Each scope gets its own collector via `AsyncLocalStorage`, so two
 * invocations running concurrently against one shared model are counted
 * separately and exactly. Differencing the model's lifetime counters, which is
 * what this replaces, attributed whatever happened to finish in between.
 */
export class UsageCollector {
    private readonly tallies = new Map<string, Tally>();

    constructor(private readonly parent?: UsageCollector) {}

    add(key: string, delta: UsageDelta, pricing?: ModelPricing): void {
        let tally = this.tallies.get(key);
        if (!tally) {
            tally = {
                promptTokens: 0,
                completionTokens: 0,
                cachedTokens: 0,
                reasoningTokens: 0,
                requestCount: 0,
                errorCount: 0,
                cacheHits: 0,
                latencySum: 0,
                cost: 0,
                priced: false,
            };
            this.tallies.set(key, tally);
        }

        const prompt = delta.promptTokens ?? 0;
        const completion = delta.completionTokens ?? 0;
        const cached = delta.cachedTokens ?? 0;
        tally.promptTokens += prompt;
        tally.completionTokens += completion;
        tally.cachedTokens += cached;
        tally.reasoningTokens += delta.reasoningTokens ?? 0;
        tally.latencySum += delta.latencyMs ?? 0;
        if (delta.cacheHit) tally.cacheHits += 1;
        else if (delta.error) tally.errorCount += 1;
        else if (delta.request !== false) tally.requestCount += 1;

        if (pricing) {
            tally.priced = true;
            tally.cost += costOf(prompt, completion, cached, pricing);
        }

        this.parent?.add(key, delta, pricing);
    }

    /** Whether anything at all was recorded. */
    isEmpty(): boolean {
        return this.tallies.size === 0;
    }

    /** A snapshot of the totals, keyed by `provider/model`. */
    snapshot(): LMUsage {
        const out: LMUsage = {};
        for (const [key, tally] of this.tallies) {
            out[key] = toStats(tally);
        }
        return out;
    }

    /** Everything recorded, summed across models. */
    total(): UsageStats {
        return sumUsage(Object.values(this.snapshot()));
    }
}

function toStats(tally: Tally): UsageStats {
    const stats: UsageStats = {
        promptTokens: tally.promptTokens,
        completionTokens: tally.completionTokens,
        totalTokens: tally.promptTokens + tally.completionTokens,
        requestCount: tally.requestCount,
        errorCount: tally.errorCount,
        cacheHits: tally.cacheHits,
        averageLatency: tally.requestCount > 0 ? tally.latencySum / tally.requestCount : 0,
    };
    if (tally.cachedTokens > 0) stats.cachedTokens = tally.cachedTokens;
    if (tally.reasoningTokens > 0) stats.reasoningTokens = tally.reasoningTokens;
    if (tally.priced) stats.totalCost = tally.cost;
    return stats;
}

function costOf(
    prompt: number,
    completion: number,
    cached: number,
    pricing: ModelPricing
): number {
    const uncached = Math.max(0, prompt - cached);
    return (
        (uncached * pricing.input +
            cached * (pricing.cachedInput ?? pricing.input) +
            completion * pricing.output) /
        1_000_000
    );
}

/**
 * Sum usage figures. `averageLatency` is re-weighted by request count, and
 * `totalCost` is present only if at least one input carried one.
 */
export function sumUsage(items: readonly UsageStats[]): UsageStats {
    const total: UsageStats = {
        promptTokens: 0,
        completionTokens: 0,
        totalTokens: 0,
        requestCount: 0,
        errorCount: 0,
    };
    let latencySum = 0;
    let cacheHits = 0;
    let cached = 0;
    let reasoning = 0;

    for (const item of items) {
        total.promptTokens += item.promptTokens;
        total.completionTokens += item.completionTokens;
        total.totalTokens += item.totalTokens;
        total.requestCount! += item.requestCount ?? 0;
        total.errorCount! += item.errorCount ?? 0;
        latencySum += (item.averageLatency ?? 0) * (item.requestCount ?? 0);
        cacheHits += item.cacheHits ?? 0;
        cached += item.cachedTokens ?? 0;
        reasoning += item.reasoningTokens ?? 0;
        if (item.totalCost !== undefined) {
            total.totalCost = (total.totalCost ?? 0) + item.totalCost;
        }
    }

    if (cacheHits > 0) total.cacheHits = cacheHits;
    if (cached > 0) total.cachedTokens = cached;
    if (reasoning > 0) total.reasoningTokens = reasoning;
    if (latencySum > 0 && total.requestCount! > 0) {
        total.averageLatency = latencySum / total.requestCount!;
    }
    return total;
}

const store = new AsyncLocalStorage<UsageCollector>();

/** The collector for the current async scope, if one is open. */
export function currentUsageCollector(): UsageCollector | undefined {
    return store.getStore();
}

/**
 * Run `fn` with a fresh collector nested inside the current one, and return
 * both its result and the collector. The collector holds only what `fn` (and
 * everything it awaited) recorded; the enclosing collector receives it too.
 */
export async function collectUsage<T>(
    fn: () => Promise<T>
): Promise<{ result: T; usage: UsageCollector }> {
    const collector = new UsageCollector(store.getStore());
    const result = await store.run(collector, fn);
    return { result, usage: collector };
}

/**
 * Run `fn` inside `collector`. Lower-level than {@link collectUsage}, for
 * callers that need the collector even when `fn` throws.
 */
export function runWithCollector<T>(collector: UsageCollector, fn: () => T): T {
    return store.run(collector, fn);
}

/**
 * Record usage against the current scope's collector, if any.
 *
 * `BaseLM.recordUsage` calls this for you; call it directly from a
 * hand-written `ILanguageModel` so that its usage shows up in
 * `prediction.getLMUsage()` and evaluation reports. A no-op when nothing is
 * collecting, so it is free on the default path.
 */
export function recordLMUsage(key: string, delta: UsageDelta): void {
    const collector = store.getStore();
    if (!collector) return;
    collector.add(key, delta, priceFor(key));
}

/** Look a model up in the configured pricing table: `provider/model`, then `model`. */
function priceFor(key: string): ModelPricing | undefined {
    const table = getPricing();
    if (!table) return undefined;
    if (table[key]) return table[key];
    const slash = key.indexOf('/');
    return slash === -1 ? undefined : table[key.slice(slash + 1)];
}
