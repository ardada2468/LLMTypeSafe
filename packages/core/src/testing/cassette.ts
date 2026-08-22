import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import type { ChatMessage, LLMCallOptions } from '../types/language-model';

/**
 * Sampling options that change what a model replies, and therefore take part in
 * the cassette key. Transport options — `timeout`, `retries`, `streaming`,
 * `metadata` — deliberately do not: re-running a test with a longer timeout
 * should still hit the recording.
 */
const KEYED_OPTIONS = [
    'frequencyPenalty',
    'maxTokens',
    'presencePenalty',
    'stopSequences',
    'temperature',
    'topP',
] as const;

/** A recorded `chat` (or `generate`) call. */
export interface CassetteChatRequest {
    kind: 'chat';
    model: string;
    messages: ChatMessage[];
    options?: Record<string, unknown>;
}

/** A recorded `generateStructured` call. */
export interface CassetteStructuredRequest {
    kind: 'structured';
    model: string;
    prompt: string;
    schema: unknown;
    options?: Record<string, unknown>;
}

export type CassetteRequest = CassetteChatRequest | CassetteStructuredRequest;

/**
 * One recorded exchange. `key` is derived from `request`, so it is regenerated
 * whenever a cassette is loaded or saved — it exists to make duplicates and
 * changed prompts obvious in a diff, not as a second source of truth.
 */
export interface CassetteEntry {
    key: string;
    request: CassetteRequest;
    /** A string for `chat` entries; the decoded value for `structured` ones. */
    response: unknown;
}

/** Keep only the options that affect the reply, in a stable order. */
export function keyedOptions(options?: LLMCallOptions): Record<string, unknown> | undefined {
    if (!options) {
        return undefined;
    }

    const kept: Record<string, unknown> = {};
    for (const name of KEYED_OPTIONS) {
        const value = options[name];
        if (value !== undefined) {
            kept[name] = value;
        }
    }

    return Object.keys(kept).length > 0 ? kept : undefined;
}

/**
 * Deep copy of a value a caller owns.
 *
 * A cassette is re-serialised on every save, so holding on to, say, the
 * `messages` array of a chat loop would let a later turn rewrite the exchanges
 * already recorded — and change their keys with them.
 */
export function snapshot<T>(value: T): T {
    try {
        return structuredClone(value);
    } catch {
        // Not structured-cloneable (a function inside a schema, say). Keeping the
        // reference is better than failing the call.
        return value;
    }
}

/** Recursively sort object keys so equal requests serialise identically. */
function canonicalise(value: unknown): unknown {
    if (Array.isArray(value)) {
        return value.map(canonicalise);
    }
    if (value !== null && typeof value === 'object') {
        const sorted: Record<string, unknown> = {};
        for (const name of Object.keys(value as Record<string, unknown>).sort()) {
            const child = (value as Record<string, unknown>)[name];
            if (child !== undefined) {
                sorted[name] = canonicalise(child);
            }
        }
        return sorted;
    }
    return value;
}

/** Stable short hash of a request: the cassette's lookup key. */
export function hashRequest(request: CassetteRequest): string {
    return createHash('sha256')
        .update(JSON.stringify(canonicalise(request)))
        .digest('hex')
        .slice(0, 16);
}

function isRequest(value: unknown): value is CassetteRequest {
    if (value === null || typeof value !== 'object') {
        return false;
    }
    const kind = (value as { kind?: unknown }).kind;
    return kind === 'chat' || kind === 'structured';
}

/**
 * Read a cassette file. Keys are recomputed from the requests, so a hand-edited
 * prompt takes effect without anyone having to fix up a hash by hand.
 */
export function readCassette(file: string): CassetteEntry[] {
    // Read outside the try: a missing file should not be reported as bad JSON.
    const text = readFileSync(file, 'utf8');

    let parsed: unknown;
    try {
        parsed = JSON.parse(text);
    } catch (cause) {
        throw new Error(`Cassette ${file} is not valid JSON`, { cause });
    }

    if (!Array.isArray(parsed)) {
        throw new Error(
            `Cassette ${file} must contain an array of { request, response } entries`
        );
    }

    return parsed.map((entry, index) => {
        if (entry === null || typeof entry !== 'object' || !isRequest((entry as any).request)) {
            throw new Error(`Cassette ${file} entry ${index} has no valid \`request\``);
        }
        const request = (entry as any).request as CassetteRequest;
        return { key: hashRequest(request), request, response: (entry as any).response };
    });
}

/** Write a cassette file, creating its directory if needed. */
export function writeCassette(file: string, entries: readonly CassetteEntry[]): void {
    const directory = dirname(file);
    if (!existsSync(directory)) {
        mkdirSync(directory, { recursive: true });
    }

    const serialisable = entries.map((entry) => ({
        key: hashRequest(entry.request),
        request: entry.request,
        response: entry.response,
    }));

    writeFileSync(file, `${JSON.stringify(serialisable, null, 2)}\n`, 'utf8');
}
