import { buildPrompt, buildPromptContent, parseOutput } from './parsing';
import { Signature, ImageField, InputField, OutputField } from '../core/signature';
import { ValidationError } from '../core/errors';

describe('Parsing Utils', () => {
    describe('buildPrompt', () => {
        it('should build prompt from string signature', () => {
            const prompt = buildPrompt('question -> answer', {
                question: 'What is the capital of France?',
            });

            expect(prompt).toContain('question: What is the capital of France?');
            expect(prompt).toContain('Provide the answer in this format:');
            expect(prompt).toContain('answer: [your response]');
        });

        it('should handle multiple inputs and outputs', () => {
            const prompt = buildPrompt('question, context -> answer, confidence: float', {
                question: 'What is the capital?',
                context: 'France is a country in Europe.',
            });

            expect(prompt).toContain('question: What is the capital?');
            expect(prompt).toContain('context: France is a country in Europe.');
            expect(prompt).toContain('Provide the following fields:');
            expect(prompt).toContain('answer (string): [your response]');
            expect(prompt).toContain('confidence (float): [your response]');
        });

        it('should skip undefined inputs', () => {
            const prompt = buildPrompt('question, context -> answer', {
                question: 'What is the capital?',
            });

            expect(prompt).toContain('question: What is the capital?');
            expect(prompt).not.toContain('context:');
        });

        it('should build prompt from a class signature with descriptions', () => {
            class Rate extends Signature {
                static description = 'Rate a review';

                @InputField({ description: 'the review' })
                review!: string;

                @OutputField({ description: 'sentiment label' })
                sentiment!: string;
            }

            const prompt = buildPrompt(Rate, { review: 'Great product' });

            expect(prompt).toContain('Rate a review');
            expect(prompt).toContain('review: Great product');
            expect(prompt).toContain('sentiment (sentiment label):');
        });
    });

    describe('parseOutput', () => {
        it('should parse simple output from string signature', () => {
            const parsed = parseOutput('question -> answer', 'answer: Paris is the capital.');
            expect(parsed.answer).toBe('Paris is the capital.');
        });

        it('should parse multiple outputs', () => {
            const parsed = parseOutput(
                'question -> answer, confidence: float',
                'answer: Paris\nconfidence: 0.95'
            );

            expect(parsed.answer).toBe('Paris');
            expect(parsed.confidence).toBe(0.95);
        });

        it('should coerce declared types', () => {
            const parsed = parseOutput(
                'input -> number_val: int, float_val: float, bool_val: bool, array_val: array',
                'number_val: 42\nfloat_val: 3.14\nbool_val: true\narray_val: ["a", "b", "c"]'
            );

            expect(parsed.number_val).toBe(42);
            expect(parsed.float_val).toBe(3.14);
            expect(parsed.bool_val).toBe(true);
            expect(parsed.array_val).toEqual(['a', 'b', 'c']);
        });

        it('should split delimited lists when the value is not JSON', () => {
            const parsed = parseOutput(
                'input -> array_val: array',
                'array_val: item1, item2, item3'
            );
            expect(parsed.array_val).toEqual(['item1', 'item2', 'item3']);
        });

        it('should treat a bare response as the value of a lone output field', () => {
            const parsed = parseOutput('question -> answer', 'Paris');
            expect(parsed.answer).toBe('Paris');
        });

        it('should not bleed one field value into the next', () => {
            const parsed = parseOutput(
                'input -> summary, followup',
                'summary: The first part\nof a multi-line summary\nfollowup: Ask about pricing'
            );

            expect(parsed.summary).toBe('The first part\nof a multi-line summary');
            expect(parsed.followup).toBe('Ask about pricing');
        });

        it('should handle field names containing regex metacharacters', () => {
            class Money extends Signature {
                @OutputField({ description: 'total in dollars' })
                'cost($)'!: string;
            }

            const parsed = parseOutput(Money, 'cost($): 42.50');
            expect(parsed['cost($)']).toBe('42.50');
        });

        it('should parse object fields', () => {
            const parsed = parseOutput(
                'input -> obj_val: object',
                'obj_val: {"key": "value", "number": 42}'
            );
            expect(parsed.obj_val).toEqual({ key: 'value', number: 42 });
        });

        describe('boolean coercion', () => {
            const sig = 'input -> bool_val: bool';

            it.each([
                ['true', true],
                ['false', false],
                ['1', true],
                ['0', false],
                ['TRUE', true],
                ['yes', true],
                ['no', false],
            ])('coerces %s to %s', (raw, expected) => {
                expect(parseOutput(sig, `bool_val: ${raw}`).bool_val).toBe(expected);
            });

            it('rejects a value that is not boolean-like', () => {
                expect(() => parseOutput(sig, 'bool_val: maybe')).toThrow(ValidationError);
            });
        });
    });

    describe('validation failures', () => {
        it('throws when a required field is missing', () => {
            expect(() => parseOutput('input -> a, b', 'a: present')).toThrow(ValidationError);
        });

        it('throws rather than returning a string for a numeric field', () => {
            // The old behaviour returned 'not_a_number' here, so a field typed
            // `int` could hold a string while TypeScript claimed otherwise.
            expect(() => parseOutput('input -> num_val: int', 'num_val: not_a_number')).toThrow(
                ValidationError
            );
        });

        it('reports the offending field, its declared type and the raw output', () => {
            try {
                parseOutput('input -> score: float', 'score: high');
                expect.unreachable('should have thrown');
            } catch (error) {
                expect(error).toBeInstanceOf(ValidationError);
                const validationError = error as ValidationError;
                expect(validationError.issues).toHaveLength(1);
                expect(validationError.issues[0].field).toBe('score');
                expect(validationError.issues[0].expected).toBe('float');
                expect(validationError.rawOutput).toBe('score: high');
            }
        });

        it('allows optional fields to be absent', () => {
            class Partial extends Signature {
                @OutputField({ description: 'always present' })
                answer!: string;

                @OutputField({ description: 'sometimes present', required: false })
                note?: string;
            }

            const parsed = parseOutput(Partial, 'answer: done');
            expect(parsed.answer).toBe('done');
            expect(parsed.note).toBeUndefined();
        });

        it('rejects an integer field given a fractional value', () => {
            expect(() => parseOutput('input -> count: int', 'count: 4.5')).toThrow(
                ValidationError
            );
        });
    });

    describe('buildPromptContent', () => {
        const PNG = 'iVBORw0KGgo=';
        const DATA_URI = `data:image/png;base64,${PNG}`;

        class ReadSign extends Signature {
            static description = 'Read the sign in the photo.';

            @ImageField({ description: 'photo of the sign' })
            photo!: string;

            @InputField({ description: 'the language to answer in' })
            language!: string;

            @OutputField({ description: 'the words on the sign' })
            words!: string;
        }

        it('returns a plain string when every input is text', () => {
            const content = buildPromptContent('question -> answer', { question: 'why?' });

            expect(content).toBe(buildPrompt('question -> answer', { question: 'why?' }));
            expect(typeof content).toBe('string');
        });

        it('emits content parts when a class input field is an image', () => {
            const content = buildPromptContent(ReadSign, {
                photo: DATA_URI,
                language: 'French',
            });

            expect(content).toEqual([
                { type: 'text', text: 'Read the sign in the photo.\n\nphoto: ' },
                {
                    type: 'image',
                    source: { kind: 'base64', mediaType: 'image/png', data: PNG },
                },
                {
                    type: 'text',
                    text: '\nlanguage: French\n\nProvide:\nwords (the words on the sign):',
                },
            ]);
        });

        it('emits content parts for a string signature declaring an image input', () => {
            const content = buildPromptContent('photo: image, question -> answer', {
                photo: 'https://example.com/sign.png',
                question: 'what does it say?',
            });

            expect(Array.isArray(content)).toBe(true);
            expect(content[1]).toEqual({
                type: 'image',
                source: { kind: 'url', url: 'https://example.com/sign.png' },
            });
        });

        it('skips an image input that was not supplied', () => {
            const content = buildPromptContent(ReadSign, { language: 'French' });

            expect(content).toBe(
                'Read the sign in the photo.\n\nlanguage: French\n\n' +
                    'Provide:\nwords (the words on the sign):'
            );
        });
    });

    describe('buildPrompt with images', () => {
        it('flattens an image input to a placeholder, since a string has no pixels', () => {
            const prompt = buildPrompt('photo: image -> caption', {
                photo: 'data:image/png;base64,iVBORw0KGgo=',
            });

            expect(prompt).toContain('photo: [image: image/png]');
        });
    });
});

describe('enum fields', () => {
    it('names the allowed members in the prompt of a class signature', () => {
        class Classify extends Signature {
            @InputField({ description: 'the review' })
            review!: string;

            @OutputField({
                description: 'overall sentiment',
                type: 'enum',
                values: ['positive', 'negative'],
            })
            sentiment!: string;
        }

        const prompt = buildPrompt(Classify, { review: 'lovely' });

        // The structured path gets the members as a JSON Schema `enum`; on the text
        // path the prompt is the only place they can appear, and without them the
        // model is being marked against a rubric it was never shown.
        expect(prompt).toContain('[one of: positive, negative]');
    });

    it('reports the members, not the bare word enum, when a value is refused', () => {
        expect(() => parseOutput('review -> sentiment: enum(a|b)', 'sentiment: c')).toThrow(
            ValidationError
        );

        try {
            parseOutput('review -> sentiment: enum(a|b)', 'sentiment: c');
        } catch (error) {
            expect((error as ValidationError).issues[0].expected).toBe('enum(a|b)');
        }
    });
});
