import { z } from 'zod';
import { isZodSignature, signature, Signature, OutputField } from './signature';
import { MockLM } from '../test-utils';
import { Predict } from '../modules/predict';
import { ChainOfThought } from '../modules/chain-of-thought';
import { ValidationError } from './errors';
import { buildPrompt, parseOutput } from '../utils/parsing';
import { buildOutputJsonSchema, getOutputFieldConfigs } from '../utils/schema';

const AnalyzeReview = signature({
    description: 'Analyze a product review.',
    input: z.object({ review: z.string() }),
    output: z.object({
        sentiment: z.enum(['positive', 'negative', 'neutral']),
        rating: z.number().int().min(1).max(5),
        themes: z.array(z.string()),
        followUp: z.string().optional(),
    }),
});

describe('signature()', () => {
    it('brands the result so every consumer can branch on it', () => {
        expect(isZodSignature(AnalyzeReview)).toBe(true);
    });

    it('does not mistake a Signature subclass or a string for a zod signature', () => {
        class Decorated extends Signature {
            @OutputField({ description: 'an answer' })
            answer!: string;
        }

        expect(isZodSignature(Decorated)).toBe(false);
        expect(isZodSignature('question -> answer')).toBe(false);
        expect(isZodSignature(null)).toBe(false);
    });

    it('rejects an output that is not a zod object', () => {
        expect(() =>
            signature({ input: z.object({ a: z.string() }), output: z.string() as any })
        ).toThrow(/"output" must be a z\.object/);
    });
});

describe('getOutputFieldConfigs with a zod signature', () => {
    it('labels each field with the type it declares', () => {
        const fields = getOutputFieldConfigs(AnalyzeReview);

        expect(fields.sentiment.type).toBe('enum');
        expect(fields.rating.type).toBe('int');
        expect(fields.themes.type).toBe('string[]');
        expect(fields.followUp.type).toBe('string');
    });

    it('marks an optional field as not required', () => {
        const fields = getOutputFieldConfigs(AnalyzeReview);

        expect(fields.sentiment.required).toBe(true);
        expect(fields.followUp.required).toBe(false);
    });

    it('carries a field description written with .describe()', () => {
        const described = signature({
            input: z.object({ q: z.string() }),
            output: z.object({ answer: z.string().describe('a concise answer') }),
        });

        expect(getOutputFieldConfigs(described).answer.description).toBe('a concise answer');
    });
});

describe('buildPrompt with a zod signature', () => {
    it('opens with the description and lists the supplied inputs', () => {
        const prompt = buildPrompt(AnalyzeReview, { review: 'Great mug.' });

        expect(prompt).toContain('Analyze a product review.');
        expect(prompt).toContain('review: Great mug.');
    });

    it('spells out the accepted enum options', () => {
        const prompt = buildPrompt(AnalyzeReview, { review: 'Great mug.' });

        expect(prompt).toContain('one of: positive, negative, neutral');
    });

    it('states numeric bounds and which fields may be omitted', () => {
        const prompt = buildPrompt(AnalyzeReview, { review: 'Great mug.' });

        expect(prompt).toContain('rating (int, between 1 and 5)');
        expect(prompt).toContain('followUp (string, optional)');
    });

    it('puts an input field description beside its value', () => {
        const described = signature({
            input: z.object({ review: z.string().describe('the review text') }),
            output: z.object({ sentiment: z.string() }),
        });

        expect(buildPrompt(described, { review: 'Great mug.' })).toContain(
            'review (the review text): Great mug.'
        );
    });

    it('serialises a structured input value as JSON', () => {
        const nested = signature({
            input: z.object({ order: z.object({ id: z.string() }) }),
            output: z.object({ status: z.string() }),
        });

        expect(buildPrompt(nested, { order: { id: 'A1' } })).toContain('order: {"id":"A1"}');
    });
});

describe('parseOutput with a zod signature', () => {
    it('coerces labelled text into the declared types', () => {
        const result = parseOutput(
            AnalyzeReview,
            'sentiment: positive\nrating: 5\nthemes: durability, price'
        );

        expect(result).toEqual({
            sentiment: 'positive',
            rating: 5,
            themes: ['durability', 'price'],
        });
    });

    it('enforces a numeric bound the flat field types cannot express', () => {
        expect(() =>
            parseOutput(AnalyzeReview, 'sentiment: positive\nrating: 9\nthemes: price')
        ).toThrow(ValidationError);
    });

    it('rejects a value outside the enum', () => {
        let caught: ValidationError | undefined;

        try {
            parseOutput(AnalyzeReview, 'sentiment: ecstatic\nrating: 5\nthemes: price');
        } catch (error) {
            caught = error as ValidationError;
        }

        expect(caught?.issues.map((issue) => issue.field)).toEqual(['sentiment']);
        expect(caught?.issues[0].expected).toBe('enum');
    });

    it('accepts output that omits an optional field', () => {
        const result = parseOutput(
            AnalyzeReview,
            'sentiment: neutral\nrating: 3\nthemes: shipping'
        );

        expect(result.followUp).toBeUndefined();
    });

    it('parses a nested object field from JSON', () => {
        const nested = signature({
            input: z.object({ text: z.string() }),
            output: z.object({
                shipping: z.object({ carrier: z.string(), days: z.number().int() }),
            }),
        });

        const result = parseOutput(nested, 'shipping: {"carrier":"UPS","days":3}');

        expect(result.shipping).toEqual({ carrier: 'UPS', days: 3 });
    });

    it('coerces towards whichever union branch the value satisfies', () => {
        const estimate = signature({
            input: z.object({ ticket: z.string() }),
            output: z.object({
                hours: z.union([z.number(), z.literal('unknown')]),
            }),
        });

        expect(parseOutput(estimate, 'hours: 7').hours).toBe(7);
        expect(parseOutput(estimate, 'hours: unknown').hours).toBe('unknown');
    });

    it('honours a refinement written on the output object', () => {
        const bounded = signature({
            input: z.object({ text: z.string() }),
            output: z
                .object({ low: z.number(), high: z.number() })
                .refine((value) => value.low <= value.high, { message: 'low exceeds high' }),
        });

        expect(() => parseOutput(bounded, 'low: 9\nhigh: 2')).toThrow(ValidationError);
    });
});

describe('buildOutputJsonSchema with a zod signature', () => {
    it('lists every property in required, including the optional one', () => {
        const schema = buildOutputJsonSchema(AnalyzeReview);

        expect(schema.required).toEqual(['sentiment', 'rating', 'themes', 'followUp']);
        expect(schema.additionalProperties).toBe(false);
    });

    it('expresses an optional field as nullable rather than omitting it', () => {
        const schema = buildOutputJsonSchema(AnalyzeReview);

        expect(schema.properties.followUp).toMatchObject({ type: ['string', 'null'] });
    });

    it('preserves enum options and numeric bounds', () => {
        const schema = buildOutputJsonSchema(AnalyzeReview);

        expect(schema.properties.sentiment).toMatchObject({
            enum: ['positive', 'negative', 'neutral'],
        });
        expect(schema.properties.rating).toMatchObject({
            type: 'integer',
            minimum: 1,
            maximum: 5,
        });
    });

    it('applies strict-mode rules to nested objects too', () => {
        const nested = signature({
            input: z.object({ text: z.string() }),
            output: z.object({
                shipping: z.object({ carrier: z.string(), days: z.number().optional() }),
            }),
        });

        const shipping = buildOutputJsonSchema(nested).properties.shipping as any;

        expect(shipping.additionalProperties).toBe(false);
        expect(shipping.required).toEqual(['carrier', 'days']);
        expect(shipping.properties.days).toMatchObject({ type: ['number', 'null'] });
    });

    it('adds null to a nullable enum so the type and the options agree', () => {
        const optionalEnum = signature({
            input: z.object({ text: z.string() }),
            output: z.object({ tone: z.enum(['warm', 'cold']).optional() }),
        });

        expect(buildOutputJsonSchema(optionalEnum).properties.tone).toMatchObject({
            type: ['string', 'null'],
            enum: ['warm', 'cold', null],
        });
    });

    it('turns an optional literal into a union rather than a contradiction', () => {
        const optionalLiteral = signature({
            input: z.object({ text: z.string() }),
            output: z.object({ status: z.literal('done').optional() }),
        });

        expect(buildOutputJsonSchema(optionalLiteral).properties.status).toEqual({
            anyOf: [{ type: 'string', const: 'done' }, { type: 'null' }],
        });
    });

    it('drops the $schema key providers do not accept', () => {
        expect(buildOutputJsonSchema(AnalyzeReview)).not.toHaveProperty('$schema');
    });

    it('explains an output type that has no JSON Schema spelling', () => {
        const dated = signature({
            input: z.object({ text: z.string() }),
            output: z.object({ due: z.date() }),
        });

        expect(() => buildOutputJsonSchema(dated)).toThrow(
            /cannot be converted to JSON Schema/
        );
    });
});

describe('Predict with a zod signature', () => {
    it('infers the output type with no type argument', async () => {
        const lm = new MockLM({
            responses: ['sentiment: positive\nrating: 5\nthemes: durability, price'],
        });

        const result = await new Predict(AnalyzeReview, lm).forward({ review: 'Great mug.' });

        // Compile-time assertions: these fail `npm run typecheck` if inference regresses.
        const sentiment: 'positive' | 'negative' | 'neutral' = result.sentiment;
        const rating: number = result.rating;
        const themes: string[] = result.themes;
        const followUp: string | undefined = result.followUp;

        expect(sentiment).toBe('positive');
        expect(rating).toBe(5);
        expect(themes.join(', ')).toBe('durability, price');
        expect(followUp).toBeUndefined();
    });

    it('sends the strict JSON Schema on the native structured-output path', async () => {
        const lm = new MockLM({
            structuredResponses: [
                {
                    sentiment: 'negative',
                    rating: 2,
                    themes: ['leaks'],
                    followUp: null,
                },
            ],
            capabilities: { supportsStructuredOutput: true },
        });

        const result = await new Predict(AnalyzeReview, lm).forward({ review: 'It leaks.' });

        expect(lm.structuredCalls[0].schema).toMatchObject({
            type: 'object',
            additionalProperties: false,
            required: ['sentiment', 'rating', 'themes', 'followUp'],
        });
        expect(result.sentiment).toBe('negative');
        expect(result.followUp).toBeUndefined();
    });

    it('keeps a null the caller declared nullable rather than treating it as absent', async () => {
        const withNullable = signature({
            input: z.object({ text: z.string() }),
            output: z.object({ error: z.string().nullable() }),
        });
        const lm = new MockLM({
            structuredResponses: [{ error: null }],
            capabilities: { supportsStructuredOutput: true },
        });

        const result = await new Predict(withNullable, lm).forward({ text: 'fine' });

        expect(result.error).toBeNull();
    });

    it('drops a null standing in for an absent field inside a nested object', async () => {
        const nested = signature({
            input: z.object({ text: z.string() }),
            output: z.object({
                address: z.object({ city: z.string(), zip: z.string().optional() }),
            }),
        });
        const lm = new MockLM({
            structuredResponses: [{ address: { city: 'Leeds', zip: null } }],
            capabilities: { supportsStructuredOutput: true },
        });

        const result = await new Predict(nested, lm).forward({ text: 'ship it' });

        expect(result.address).toEqual({ city: 'Leeds' });
    });

    it('rejects a structured response that violates the zod constraints', async () => {
        const lm = new MockLM({
            structuredResponses: [
                { sentiment: 'positive', rating: 11, themes: [], followUp: null },
            ],
            capabilities: { supportsStructuredOutput: true },
        });

        await expect(
            new Predict(AnalyzeReview, lm).forward({ review: 'Great mug.' })
        ).rejects.toThrow(ValidationError);
    });

    it('still accepts an explicit output type argument that overrides inference', async () => {
        type LooseReview = { sentiment: string; rating: number; themes: string[] };
        const lm = new MockLM({ responses: ['sentiment: neutral\nrating: 3\nthemes: price'] });

        const result = await new Predict<typeof AnalyzeReview, LooseReview>(
            AnalyzeReview,
            lm
        ).forward({ review: 'It is fine.' });

        const sentiment: string = result.sentiment;

        expect(sentiment).toBe('neutral');
    });
});

describe('ChainOfThought with a zod signature', () => {
    it('infers the output type and adds reasoning to it', async () => {
        const lm = new MockLM({
            responses: [
                'The reviewer praises the mug.',
                'sentiment: positive\nrating: 4\nthemes: build quality',
            ],
        });

        const result = await new ChainOfThought(AnalyzeReview, lm).forward({
            review: 'Great mug.',
        });

        const sentiment: 'positive' | 'negative' | 'neutral' = result.sentiment;
        const reasoning: string = result.reasoning;

        expect(sentiment).toBe('positive');
        expect(reasoning).toBe('The reviewer praises the mug.');
    });
});
