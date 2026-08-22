import { parsePartialJson } from './partial-json';

describe('parsePartialJson', () => {
    it.each([
        ['an empty buffer', ''],
        ['whitespace only', '   \n  '],
        ['an opening brace only', '{'],
        ['a key still being written', '{"ans'],
        ['a finished key with no colon', '{"answer"'],
        ['a colon with no value', '{"answer":'],
        ['a keyword in progress', '{"ok": tru'],
        ['a number that may still grow', '{"score": 0.9'],
        ['a leading minus sign', '{"score": -'],
    ])('reads no fields from %s', (_label, text) => {
        const value = parsePartialJson<Record<string, unknown>>(text);

        expect(value === undefined || Object.keys(value).length === 0).toBe(true);
    });

    it.each([
        ['a half-written string', '{"answer": "Par', { answer: 'Par' }],
        ['a trailing comma', '{"a": 1, "b": 2,', { a: 1, b: 2 }],
        ['an unclosed object', '{"a": 1, "b": 2', { a: 1 }],
        ['an escaped quote mid-value', '{"a": "he said \\"hi', { a: 'he said "hi' }],
        ['a dangling backslash', '{"a": "line\\', { a: 'line' }],
        ['a half-written unicode escape', '{"a": "snow\\u26', { a: 'snow' }],
        ['a complete unicode escape', '{"a": "snow\\u2603"}', { a: 'snow☃' }],
        ['a decoded newline escape', '{"a": "one\\ntwo"}', { a: 'one\ntwo' }],
        ['a closed object', '{"a": 1, "b": "two"}', { a: 1, b: 'two' }],
        ['whitespace between tokens', '{ "a" : 1 , "b" : true }', { a: 1, b: true }],
        ['a null value', '{"a": null}', { a: null }],
        ['a nested unclosed object', '{"a": {"b": 1}, "c": {"d": 2', { a: { b: 1 }, c: {} }],
        ['a partial array', '{"tags": ["one", "tw', { tags: ['one', 'tw'] }],
        ['a closed array', '{"tags": ["one", "two"], "n": 2}', { tags: ['one', 'two'], n: 2 }],
        ['trailing prose after the object', '{"a": 1} and that is all', { a: 1 }],
        ['a leading code fence', '```json\n{"a": 1}\n```', { a: 1 }],
        ['a conversational preamble', 'Sure! Here you go:\n{"a": 1}', { a: 1 }],
        ['a preamble containing a bracket', 'Sure [1] here:\n{"a": 1}', { a: 1 }],
        ['a preamble containing a brace', 'The { below:\n{"a": "b"}', { a: 'b' }],
        ['a fenced document cut short', '```json\n{"a": "hal', { a: 'hal' }],
    ])('parses %s', (_label, text, expected) => {
        expect(parsePartialJson(text)).toEqual(expected);
    });

    it('drops the pair whose string is unfinished when partialStrings is off', () => {
        const value = parsePartialJson('{"a": "done", "b": "half', { partialStrings: false });

        expect(value).toEqual({ a: 'done' });
    });

    it('matches JSON.parse on a complete document', () => {
        const text = '{"a": [1, 2, {"b": "c"}], "d": {"e": null}, "f": -1.5e2, "g": false}';

        expect(parsePartialJson(text)).toEqual(JSON.parse(text));
    });

    it('grows monotonically as a document arrives one character at a time', () => {
        const text = '{"answer": "Paris", "confidence": 0.92, "sources": ["atlas"]}';
        const complete = JSON.parse(text);

        const snapshots: Array<Record<string, unknown>> = [];
        for (let end = 1; end <= text.length; end++) {
            const value = parsePartialJson<Record<string, unknown>>(text.slice(0, end));
            if (value) {
                snapshots.push(value);
            }
        }

        expect(snapshots.at(-1)).toEqual(complete);
        for (let i = 1; i < snapshots.length; i++) {
            const previous = Object.keys(snapshots[i - 1]);
            const current = Object.keys(snapshots[i]);
            expect(current.slice(0, previous.length)).toEqual(previous);
        }
    });

    it('parses a top-level array', () => {
        expect(parsePartialJson('[1, 2, 3')).toEqual([1, 2]);
    });

    it('never lets a __proto__ key in the document reach the prototype', () => {
        const value = parsePartialJson<Record<string, any>>('{"__proto__": {"polluted": 1}}');

        expect(({} as any).polluted).toBeUndefined();
        expect(Object.getPrototypeOf(value)).toBe(Object.prototype);
        expect(value!['__proto__']).toEqual({ polluted: 1 });
    });

    it('returns undefined for text holding no JSON at all', () => {
        expect(parsePartialJson('I cannot answer that.')).toBeUndefined();
    });

    it('is pure: the same text always yields the same value', () => {
        const text = '{"a": "half';

        expect(parsePartialJson(text)).toEqual(parsePartialJson(text));
    });
});
