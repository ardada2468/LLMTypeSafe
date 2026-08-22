import { type Example } from '../core/example';
import { type Prediction } from '../core/prediction';
import { type Metric } from './types';

export interface MetricOptions {
    /**
     * Output fields to grade. Defaults to every field the example expects,
     * which for a signature-shaped example is every output field.
     */
    fields?: string[];
    /**
     * Compare text case-insensitively and with runs of whitespace collapsed.
     * Defaults to `false`.
     */
    normalize?: boolean;
    /**
     * Absolute tolerance for numeric fields. When set, values that parse as
     * finite numbers are compared with `|expected - actual| <= tolerance`
     * instead of as text.
     */
    tolerance?: number;
}

/** Lowercase, collapse whitespace runs, trim. */
function normalizeText(value: unknown): string {
    return String(value).toLowerCase().replace(/\s+/g, ' ').trim();
}

/** Parse a value as a finite number, or return `undefined`. */
function toNumber(value: unknown): number | undefined {
    if (typeof value === 'number') {
        return Number.isFinite(value) ? value : undefined;
    }
    if (typeof value === 'string' && value.trim() !== '') {
        const parsed = Number(value.trim());
        return Number.isFinite(parsed) ? parsed : undefined;
    }
    return undefined;
}

function isPlainObject(value: unknown): value is Record<string, any> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** Deep, order-sensitive comparison honouring `normalize` and `tolerance`. */
function valuesMatch(expected: unknown, actual: unknown, options: MetricOptions): boolean {
    if (
        expected === null ||
        expected === undefined ||
        actual === null ||
        actual === undefined
    ) {
        return expected === actual;
    }

    if (options.tolerance !== undefined) {
        const left = toNumber(expected);
        const right = toNumber(actual);
        if (left !== undefined && right !== undefined) {
            return Math.abs(left - right) <= options.tolerance;
        }
    }

    if (Array.isArray(expected) || Array.isArray(actual)) {
        if (!Array.isArray(expected) || !Array.isArray(actual)) return false;
        if (expected.length !== actual.length) return false;
        return expected.every((item, index) => valuesMatch(item, actual[index], options));
    }

    if (isPlainObject(expected) || isPlainObject(actual)) {
        if (!isPlainObject(expected) || !isPlainObject(actual)) return false;
        const keys = Object.keys(expected);
        if (keys.length !== Object.keys(actual).length) return false;
        return keys.every(
            (key) =>
                Object.prototype.hasOwnProperty.call(actual, key) &&
                valuesMatch(expected[key], actual[key], options)
        );
    }

    return options.normalize
        ? normalizeText(expected) === normalizeText(actual)
        : String(expected) === String(actual);
}

/**
 * The fields to grade, paired with the expected and predicted values.
 *
 * `Prediction` defines its data with `Object.defineProperty`, so this reads
 * through `toObject()` rather than assuming enumerable own keys.
 */
function fieldPairs(
    example: Example,
    prediction: Prediction,
    fields?: string[]
): Array<{ field: string; expected: unknown; actual: unknown }> {
    const expected = example.getOutputs();
    const actual = prediction.toObject() as Record<string, any>;
    const names = fields ?? Object.keys(expected);

    return names.map((field) => ({
        field,
        expected: expected[field],
        actual: actual[field],
    }));
}

/**
 * Grade one field pair.
 *
 * A field neither side has — a misspelled entry in `fields`, a column missing
 * from the dataset — counts as a miss. Reading `undefined === undefined` as a
 * match would let a typo score a perfect run.
 */
function pairMatches(
    pair: { expected: unknown; actual: unknown },
    options: MetricOptions
): boolean {
    if (pair.expected === undefined && pair.actual === undefined) return false;
    return valuesMatch(pair.expected, pair.actual, options);
}

/**
 * All-or-nothing field comparison: `1` when every graded field matches, `0`
 * otherwise. An example with no graded fields scores `0` rather than passing
 * vacuously.
 */
export function matchMetric(options: MetricOptions = {}): Metric {
    return (example, prediction) => {
        const pairs = fieldPairs(example, prediction, options.fields);
        if (pairs.length === 0) return 0;
        return pairs.every((pair) => pairMatches(pair, options));
    };
}

/** Every expected field must match exactly, as text. */
export const exactMatch: Metric = matchMetric();

/** Every expected field must match ignoring case and whitespace runs. */
export const normalizedMatch: Metric = matchMetric({ normalize: true });

/**
 * Every expected field must match within `tolerance`.
 *
 * Parseability decides, not type: `'42'` is graded as a number, and so is a
 * zero-padded id like `'007'`. A pair where either side does not parse as a
 * finite number falls back to a case- and whitespace-insensitive text
 * comparison. Pass `fields` when the dataset mixes numbers with values that
 * merely look numeric.
 */
export function numericMatch(tolerance = 1e-6, fields?: string[]): Metric {
    return matchMetric({ tolerance, fields, normalize: true });
}

/**
 * Per-field accuracy: the fraction of graded fields that match, so a
 * multi-output signature earns partial credit instead of failing whole.
 */
export function fieldAccuracyMetric(options: MetricOptions = {}): Metric {
    return (example, prediction) => {
        const pairs = fieldPairs(example, prediction, options.fields);
        if (pairs.length === 0) return 0;
        const hits = pairs.filter((pair) => pairMatches(pair, options));
        return hits.length / pairs.length;
    };
}

/** Per-field accuracy, comparing text case- and whitespace-insensitively. */
export const fieldAccuracy: Metric = fieldAccuracyMetric({ normalize: true });

/**
 * Split into lowercase letter-and-digit tokens.
 *
 * The class is Unicode-aware on purpose: splitting on `[^a-z0-9]` would reduce
 * every CJK or Cyrillic answer to no tokens at all, and two empty token lists
 * compare as a perfect match — an eval set in those scripts would score 1.000
 * however wrong the model was.
 */
function tokenize(value: unknown): string[] {
    if (value === null || value === undefined) return [];
    const text = Array.isArray(value) ? value.join(' ') : String(value);
    return text
        .toLowerCase()
        .split(/[^\p{L}\p{N}]+/u)
        .filter((token) => token.length > 0);
}

/** Multiset token overlap between two token lists. */
function overlapCount(expected: string[], actual: string[]): number {
    const remaining = new Map<string, number>();
    for (const token of expected) {
        remaining.set(token, (remaining.get(token) ?? 0) + 1);
    }

    let overlap = 0;
    for (const token of actual) {
        const left = remaining.get(token) ?? 0;
        if (left > 0) {
            remaining.set(token, left - 1);
            overlap += 1;
        }
    }
    return overlap;
}

/** Token-level F1 for one field. */
function tokenF1Field(expected: unknown, actual: unknown): number {
    const expectedTokens = tokenize(expected);
    const actualTokens = tokenize(actual);

    if (expectedTokens.length === 0 && actualTokens.length === 0) {
        // Nothing tokenizable on either side — punctuation, an empty cell. Fall
        // back to comparing the raw text rather than calling it a match.
        return normalizeText(expected ?? '') === normalizeText(actual ?? '') ? 1 : 0;
    }
    if (expectedTokens.length === 0 || actualTokens.length === 0) return 0;

    const overlap = overlapCount(expectedTokens, actualTokens);
    if (overlap === 0) return 0;

    const precision = overlap / actualTokens.length;
    const recall = overlap / expectedTokens.length;
    return (2 * precision * recall) / (precision + recall);
}

/**
 * Mean token-level F1 across the graded fields — the right shape for free-text
 * answers, where an exact match is too strict to be informative.
 */
export function tokenF1Metric(options: Pick<MetricOptions, 'fields'> = {}): Metric {
    return (example, prediction) => {
        const pairs = fieldPairs(example, prediction, options.fields);
        if (pairs.length === 0) return 0;
        const sum = pairs.reduce(
            (total, pair) => total + tokenF1Field(pair.expected, pair.actual),
            0
        );
        return sum / pairs.length;
    };
}

/** Mean token-level F1 across every expected field. */
export const tokenF1: Metric = tokenF1Metric();
