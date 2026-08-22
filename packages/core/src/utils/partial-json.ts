/**
 * Incremental JSON parsing for streamed model output.
 *
 * A streaming response arrives as a JSON document that is truncated at an
 * arbitrary byte: mid-string, mid-key, after a comma, halfway through an escape
 * sequence. `JSON.parse` rejects every one of those, which makes it useless for
 * showing a field filling in as tokens arrive. This parser reads as much of the
 * document as is unambiguously present and discards the rest, without throwing.
 *
 * It is dependency-free and pure: same text in, same value out.
 */

/** Nothing usable could be read at this position. */
const NOTHING = Symbol('nothing');

interface Parsed {
    value: unknown;
    /** Whether the value was closed off in the source rather than cut short. */
    terminated: boolean;
}

type Outcome = Parsed | typeof NOTHING;

const ESCAPES: Record<string, string> = {
    '"': '"',
    '\\': '\\',
    '/': '/',
    b: '\b',
    f: '\f',
    n: '\n',
    r: '\r',
    t: '\t',
};

const LITERALS: Array<{ text: string; value: unknown }> = [
    { text: 'true', value: true },
    { text: 'false', value: false },
    { text: 'null', value: null },
];

const NUMBER_CHARS = /[-+0-9.eE]/;
const WHITESPACE = /\s/;
const HEX = /^[0-9a-fA-F]{4}$/;

export interface PartialJsonOptions {
    /**
     * Include the characters of a string value that is still being written.
     *
     * On by default — a half-written string is exactly what makes a field
     * fill in progressively. Turn it off when a partially-correct value is worse
     * than no value, and the key will be omitted until its string closes.
     */
    partialStrings?: boolean;
}

/**
 * Parse as much of a possibly-truncated JSON document as is unambiguously
 * present.
 *
 * A leading code fence or conversational preamble is skipped: parsing starts at
 * the bracket that opens the document, which is the first one that runs to the
 * end of the text, so a stray `[1]` in the preamble is not mistaken for it.
 * Anything after the top-level value ends is ignored. What survives truncation:
 *
 * - a half-written string keeps the characters written so far (see
 *   {@link PartialJsonOptions.partialStrings}), with escapes already decoded;
 * - an unclosed object or array yields the members completed before the cut;
 * - a trailing comma, or a key with no value yet, is dropped;
 * - a number or keyword cut off at the end is dropped, because `1` may still
 *   become `12` and `tru` is not yet `true`.
 *
 * Returns `undefined` when the text holds nothing usable yet.
 */
export function parsePartialJson<T = unknown>(
    text: string,
    options: PartialJsonOptions = {}
): T | undefined {
    if (typeof text !== 'string') {
        return undefined;
    }

    const partialStrings = options.partialStrings !== false;
    let fallback: Outcome = NOTHING;

    for (const start of candidateStarts(text)) {
        const reader = new PartialJsonReader(text, start, partialStrings);
        const result = reader.readValue();
        if (result === NOTHING) {
            continue;
        }

        // A streamed document runs to the end of the buffer, so a candidate that
        // does is the document; one that stops short of real text is a bracket in
        // the preamble — `Sure [1] here: {"answer": 2}` starts at the brace.
        if (TRAILING_NOISE.test(text.slice(reader.position))) {
            return result.value as T;
        }
        if (fallback === NOTHING && hasContent(result.value)) {
            fallback = result;
        }
    }

    return fallback === NOTHING ? undefined : (fallback.value as T);
}

/** Backticks and space are all a finished document may be followed by. */
const TRAILING_NOISE = /^[\s`]*$/;

/**
 * Every position the top-level value could start at, in order.
 *
 * Models prepend prose and code fences often enough that anchoring on the first
 * structural character is more robust than trusting the reply to begin with one.
 * A reply with no brackets at all is read as a bare scalar.
 */
function candidateStarts(text: string): number[] {
    const starts: number[] = [];
    for (let i = 0; i < text.length; i++) {
        const char = text[i];
        if (char === '{' || char === '[') {
            starts.push(i);
        }
    }

    if (starts.length === 0) {
        const first = text.search(/\S/);
        if (first !== -1) {
            starts.push(first);
        }
    }
    return starts;
}

/** Whether a parsed value carries anything, so an empty `{}` loses to a real one. */
function hasContent(value: unknown): boolean {
    if (Array.isArray(value)) {
        return value.length > 0;
    }
    if (value !== null && typeof value === 'object') {
        return Object.keys(value).length > 0;
    }
    return true;
}

class PartialJsonReader {
    private index: number;

    constructor(
        private readonly source: string,
        start: number,
        private readonly partialStrings: boolean
    ) {
        this.index = start;
    }

    /** How far reading got, so the caller can see what is left over. */
    get position(): number {
        return this.index;
    }

    readValue(): Outcome {
        this.skipWhitespace();
        if (this.index >= this.source.length) {
            return NOTHING;
        }

        const char = this.source[this.index];
        if (char === '{') return this.readObject();
        if (char === '[') return this.readArray();
        if (char === '"') return this.readString();
        if (char === 't' || char === 'f' || char === 'n') return this.readLiteral();
        return this.readNumber();
    }

    private readObject(): Parsed {
        this.index += 1; // past '{'
        const result: Record<string, unknown> = {};

        for (;;) {
            this.skipWhitespace();
            if (this.index >= this.source.length) {
                return { value: result, terminated: false };
            }

            const char = this.source[this.index];
            if (char === '}') {
                this.index += 1;
                return { value: result, terminated: true };
            }
            if (char === ',') {
                // Also absorbs a trailing comma: the next pass sees '}' or EOF.
                this.index += 1;
                continue;
            }
            if (char !== '"') {
                // A key that has not started with a quote yet, or junk.
                return { value: result, terminated: false };
            }

            const key = this.readString();
            if (key === NOTHING || !key.terminated) {
                return { value: result, terminated: false };
            }

            this.skipWhitespace();
            if (this.source[this.index] !== ':') {
                return { value: result, terminated: false };
            }
            this.index += 1;

            const value = this.readValue();
            if (value === NOTHING) {
                return { value: result, terminated: false };
            }

            // Assigned rather than set directly so a `__proto__` key in model
            // output cannot reassign the result's prototype.
            Object.defineProperty(result, key.value as string, {
                value: value.value,
                enumerable: true,
                writable: true,
                configurable: true,
            });

            if (!value.terminated) {
                return { value: result, terminated: false };
            }
        }
    }

    private readArray(): Parsed {
        this.index += 1; // past '['
        const result: unknown[] = [];

        for (;;) {
            this.skipWhitespace();
            if (this.index >= this.source.length) {
                return { value: result, terminated: false };
            }

            const char = this.source[this.index];
            if (char === ']') {
                this.index += 1;
                return { value: result, terminated: true };
            }
            if (char === ',') {
                this.index += 1;
                continue;
            }

            const value = this.readValue();
            if (value === NOTHING) {
                return { value: result, terminated: false };
            }

            result.push(value.value);
            if (!value.terminated) {
                return { value: result, terminated: false };
            }
        }
    }

    private readString(): Outcome {
        this.index += 1; // past the opening quote
        let value = '';

        while (this.index < this.source.length) {
            const char = this.source[this.index];

            if (char === '"') {
                this.index += 1;
                return { value, terminated: true };
            }

            if (char !== '\\') {
                value += char;
                this.index += 1;
                continue;
            }

            const escape = this.source[this.index + 1];
            if (escape === undefined) {
                // The backslash is the last character written so far; its meaning
                // arrives with the next chunk, so drop it rather than guess.
                break;
            }
            if (escape === 'u') {
                const digits = this.source.slice(this.index + 2, this.index + 6);
                if (!HEX.test(digits)) {
                    break;
                }
                value += String.fromCharCode(parseInt(digits, 16));
                this.index += 6;
                continue;
            }
            if (escape in ESCAPES) {
                value += ESCAPES[escape];
                this.index += 2;
                continue;
            }

            // Not a JSON escape at all. Keep the character it introduced, which is
            // friendlier than dropping the rest of a value over one stray slash.
            value += escape;
            this.index += 2;
        }

        this.index = this.source.length;
        return this.partialStrings ? { value, terminated: false } : NOTHING;
    }

    private readNumber(): Outcome {
        const start = this.index;
        while (this.index < this.source.length && NUMBER_CHARS.test(this.source[this.index])) {
            this.index += 1;
        }

        // Running to the end of the buffer means the number may still grow: `1`
        // becomes `12` with the next chunk, so emitting 1 would be a lie.
        if (start === this.index || this.index >= this.source.length) {
            return NOTHING;
        }

        const parsed = Number(this.source.slice(start, this.index));
        return Number.isNaN(parsed) ? NOTHING : { value: parsed, terminated: true };
    }

    private readLiteral(): Outcome {
        for (const literal of LITERALS) {
            if (this.source.startsWith(literal.text, this.index)) {
                this.index += literal.text.length;
                return { value: literal.value, terminated: true };
            }
        }

        // `tru` at the end of the buffer is a keyword in progress, not a value.
        return NOTHING;
    }

    private skipWhitespace(): void {
        while (this.index < this.source.length && WHITESPACE.test(this.source[this.index])) {
            this.index += 1;
        }
    }
}
