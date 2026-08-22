import {
    fieldConfigToZod,
    buildOutputSchema,
    buildOutputJsonSchema,
    getOutputFieldConfigs,
} from './schema';
import { Signature, OutputField } from '../core/signature';

/**
 * Walk a JSON Schema and collect everything OpenAI strict mode refuses.
 *
 * Strict mode is stricter than JSON Schema: every object in the document has to
 * carry `additionalProperties: false`, and `items` cannot be the empty "any
 * element" schema.
 */
function strictModeViolations(node: unknown, path = '$'): string[] {
    if (Array.isArray(node)) {
        return node.flatMap((child, index) => strictModeViolations(child, `${path}[${index}]`));
    }
    if (node === null || typeof node !== 'object') return [];

    const schema = node as Record<string, unknown>;
    const violations: string[] = [];

    const declaresObject =
        schema.type === 'object' ||
        (Array.isArray(schema.type) && schema.type.includes('object'));
    if (declaresObject && schema.additionalProperties !== false) {
        violations.push(`${path}: object without additionalProperties: false`);
    }

    if ('items' in schema) {
        const items = schema.items;
        if (
            items === null ||
            typeof items !== 'object' ||
            Object.keys(items as object).length === 0
        ) {
            violations.push(`${path}.items: empty or non-schema items`);
        }
    }

    for (const [key, value] of Object.entries(schema)) {
        violations.push(...strictModeViolations(value, `${path}.${key}`));
    }
    return violations;
}

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

    const parseEnum = (value: unknown, type = 'enum', values?: string[]) =>
        fieldConfigToZod({ description: '', type, values }).safeParse(value);

    it.each([
        ['positive', 'positive'],
        ['Positive', 'positive'],
        ['  NEUTRAL  ', 'neutral'],
        ['negative', 'negative'],
    ])('coerces enum field from %s', (input, expected) => {
        const result = parseEnum(input, 'enum', ['positive', 'negative', 'neutral']);

        expect(result.success && result.data).toBe(expected);
    });

    it('rejects an enum value outside the declared set', () => {
        const result = parseEnum('ecstatic', 'enum', ['positive', 'negative', 'neutral']);

        expect(result.success).toBe(false);
    });

    it('reads enum members from the inline string-signature form', () => {
        const result = parseEnum('b', 'enum(a|b|c)');

        expect(result.success && result.data).toBe('b');
    });

    it('rejects a value outside an inline enum set', () => {
        expect(parseEnum('d', 'enum(a|b|c)').success).toBe(false);
    });

    it('refuses to build an enum with no declared members', () => {
        expect(() => fieldConfigToZod({ description: '', type: 'enum' })).toThrow(/non-empty/);
    });

    it.each([['enum(a|b'], ['enum a|b'], ['enum()x']])(
        'refuses a malformed enum declaration %s',
        (type) => {
            // Falling through to the `string` default here would leave the field
            // accepting anything at all — silently, and only because of a typo.
            expect(() => fieldConfigToZod({ description: '', type })).toThrow(/Unrecognised/);
        }
    );

    it('leaves a type that merely starts with the letters "enum" alone', () => {
        const result = parse('enumerable', 'anything');

        expect(result.success && result.data).toBe('anything');
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

    it('emits nested objects and arrays in a shape OpenAI strict mode accepts', () => {
        // The old implementation emitted `{ type: 'object', additionalProperties: true }`
        // and `{ type: 'array', items: {} }` here. OpenAI's API refuses both — strict
        // mode requires `additionalProperties: false` on every object in the document
        // and will not take an empty `items` — so any signature with an `object` or a
        // bare `array` output field was rejected outright on the provider path.
        const schema = buildOutputJsonSchema(
            'q -> meta: object, blob: json, items: array, more: list, tags: string[]'
        );

        expect(strictModeViolations(schema)).toEqual([]);
        expect(schema.properties.meta).toEqual({
            type: 'object',
            properties: {},
            required: [],
            additionalProperties: false,
            description: 'Output field: meta',
        });
        expect(schema.properties.items).toMatchObject({
            type: 'array',
            items: { type: 'string' },
        });
    });

    it('emits an enum field as a constrained string', () => {
        class S extends Signature {
            @OutputField({
                description: 'overall sentiment',
                type: 'enum',
                values: ['positive', 'negative', 'neutral'],
            })
            sentiment!: string;
        }

        const schema = buildOutputJsonSchema(S);

        expect(schema.properties.sentiment).toMatchObject({
            type: 'string',
            enum: ['positive', 'negative', 'neutral'],
        });
        expect(strictModeViolations(schema)).toEqual([]);
    });

    it('reads enum members declared inline in a string signature', () => {
        const schema = buildOutputJsonSchema('review -> sentiment: enum(positive|negative)');

        expect(schema.properties.sentiment).toMatchObject({
            type: 'string',
            enum: ['positive', 'negative'],
        });
    });

    it('admits null into the member list of an optional enum', () => {
        class S extends Signature {
            @OutputField({
                description: 'optional grade',
                type: 'enum',
                values: ['pass', 'fail'],
                required: false,
            })
            grade?: string;
        }

        const schema = buildOutputJsonSchema(S);

        // Without this, `type: ['string', 'null']` and `enum: ['pass', 'fail']`
        // contradict each other and the field can never actually be null.
        expect(schema.properties.grade.type).toEqual(['string', 'null']);
        expect(schema.properties.grade.enum).toEqual(['pass', 'fail', null]);
    });
});

describe('getOutputFieldConfigs', () => {
    it('normalises an inline enum declaration into type and values', () => {
        const schema = buildOutputSchema('review -> sentiment: enum(Positive|Negative)');
        const result = schema.safeParse({ sentiment: 'positive' });

        // The declared spelling wins, so callers can compare against their own list.
        expect(result.success && (result.data as Record<string, unknown>).sentiment).toBe(
            'Positive'
        );
        expect(schema.safeParse({ sentiment: 'unsure' }).success).toBe(false);
    });

    it('carries enum members through to the field config', () => {
        expect(getOutputFieldConfigs('review -> sentiment: enum(a|b)').sentiment).toMatchObject(
            {
                type: 'enum',
                values: ['a', 'b'],
            }
        );
    });
});
