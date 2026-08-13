import { fieldConfigToZod, buildOutputSchema, buildOutputJsonSchema } from './schema';
import { Signature, OutputField } from '../core/signature';

describe('fieldConfigToZod', () => {
    const parse = (type: string | undefined, value: unknown) =>
        fieldConfigToZod({ description: '', type }).safeParse(value);

    it.each([
        ['number', '42', 42],
        ['float', '3.14', 3.14],
        ['int', '7', 7],
        ['integer', '7', 7],
        ['number', '1,234', 1234],
        ['number', '85%', 85],
    ])('coerces %s field from %s', (type, input, expected) => {
        const result = parse(type, input);
        expect(result.success && result.data).toBe(expected);
    });

    it('rejects an empty string for a number field rather than reading it as 0', () => {
        expect(parse('number', '   ').success).toBe(false);
    });

    it('rejects a fractional value for an int field', () => {
        expect(parse('int', '4.5').success).toBe(false);
    });

    it.each([
        ['true', true],
        ['FALSE', false],
        ['1', true],
        ['0', false],
        ['yes', true],
        ['off', false],
    ])('coerces boolean from %s', (input, expected) => {
        const result = parse('bool', input);
        expect(result.success && result.data).toBe(expected);
    });

    it('rejects a non-boolean-like string', () => {
        expect(parse('boolean', 'perhaps').success).toBe(false);
    });

    it('parses JSON arrays', () => {
        const result = parse('array', '["a","b"]');
        expect(result.success && result.data).toEqual(['a', 'b']);
    });

    it('splits delimited lists when the value is not JSON', () => {
        const result = parse('string[]', 'a, b; c');
        expect(result.success && result.data).toEqual(['a', 'b', 'c']);
    });

    it('coerces elements of a number array', () => {
        const result = parse('number[]', '1, 2, 3');
        expect(result.success && result.data).toEqual([1, 2, 3]);
    });

    it('parses JSON objects', () => {
        const result = parse('object', '{"a":1}');
        expect(result.success && result.data).toEqual({ a: 1 });
    });

    it('rejects a malformed object', () => {
        expect(parse('object', '{not json').success).toBe(false);
    });

    it('defaults unknown types to string', () => {
        const result = parse('mystery', 'hello');
        expect(result.success && result.data).toBe('hello');
    });

    it('makes non-required fields optional', () => {
        const schema = fieldConfigToZod({ description: '', type: 'number', required: false });
        expect(schema.safeParse(undefined).success).toBe(true);
    });
});

describe('buildOutputSchema', () => {
    it('validates every output field of a class signature', () => {
        class S extends Signature {
            @OutputField({ description: 'a' })
            a!: string;

            @OutputField({ description: 'b', type: 'number' })
            b!: number;
        }

        const schema = buildOutputSchema(S);
        expect(schema.safeParse({ a: 'x', b: '2' }).success).toBe(true);
        expect(schema.safeParse({ a: 'x' }).success).toBe(false);
    });

    it('reads output fields out of a string signature', () => {
        const schema = buildOutputSchema('q -> answer, score: float');
        const result = schema.safeParse({ answer: 'x', score: '0.5' });
        expect(result.success && result.data).toEqual({ answer: 'x', score: 0.5 });
    });

    it('passes through unexpected extra keys rather than dropping them', () => {
        const schema = buildOutputSchema('q -> answer');
        const result = schema.safeParse({ answer: 'x', extra: 'kept' });
        expect(result.success && (result.data as Record<string, unknown>).extra).toBe('kept');
    });
});

describe('buildOutputJsonSchema', () => {
    it('emits a strict-mode-compatible schema', () => {
        const schema = buildOutputJsonSchema('q -> answer, score: float');

        expect(schema.type).toBe('object');
        expect(schema.additionalProperties).toBe(false);
        // OpenAI strict mode requires every property to appear in `required`.
        expect(schema.required).toEqual(['answer', 'score']);
        expect(schema.properties.answer).toMatchObject({ type: 'string' });
        expect(schema.properties.score).toMatchObject({ type: 'number' });
    });

    it('expresses optional fields as nullable instead of omitting them', () => {
        class S extends Signature {
            @OutputField({ description: 'required one' })
            a!: string;

            @OutputField({ description: 'optional one', required: false })
            b?: string;
        }

        const schema = buildOutputJsonSchema(S);
        expect(schema.required).toEqual(['a', 'b']);
        expect(schema.properties.b.type).toEqual(['string', 'null']);
    });

    it('carries field descriptions through to the schema', () => {
        class S extends Signature {
            @OutputField({ description: 'how confident, 0 to 1', type: 'number' })
            confidence!: number;
        }

        expect(buildOutputJsonSchema(S).properties.confidence.description).toBe(
            'how confident, 0 to 1'
        );
    });

    it('maps array and object field types', () => {
        const schema = buildOutputJsonSchema('q -> tags: string[], meta: object');
        expect(schema.properties.tags).toMatchObject({
            type: 'array',
            items: { type: 'string' },
        });
        expect(schema.properties.meta).toMatchObject({ type: 'object' });
    });
});
