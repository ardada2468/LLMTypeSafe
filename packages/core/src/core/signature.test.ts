import 'reflect-metadata';
import { Signature, InputField, OutputField } from './signature';

describe('Signature', () => {
    describe('parseStringSignature', () => {
        it('should parse simple string signature', () => {
            const result = Signature.parseStringSignature('question -> answer');
            expect(result.inputs).toEqual(['question']);
            expect(result.outputs).toEqual(['answer']);
            expect(result.types.question).toBe('string');
            expect(result.types.answer).toBe('string');
        });

        it('should parse typed string signature', () => {
            const result = Signature.parseStringSignature('question, context -> answer: string, confidence: float');
            expect(result.inputs).toEqual(['question', 'context']);
            expect(result.outputs).toEqual(['answer', 'confidence']);
            expect(result.types.question).toBe('string');
            expect(result.types.context).toBe('string');
            expect(result.types.answer).toBe('string');
            expect(result.types.confidence).toBe('float');
        });

        it('should handle whitespace in string signature', () => {
            const result = Signature.parseStringSignature('  input1 , input2  ->  output1: int , output2  ');
            expect(result.inputs).toEqual(['input1', 'input2']);
            expect(result.outputs).toEqual(['output1', 'output2']);
            expect(result.types.output1).toBe('int');
            expect(result.types.output2).toBe('string');
        });
    });

    describe('Decorators', () => {
        it('should register input and output fields via decorators', () => {
            class TestSignature extends Signature {
                @InputField({ description: 'Test input' })
                input!: string;

                @OutputField({ description: 'Test output', type: 'number' })
                output!: number;
            }

            const inputFields = TestSignature.getInputFields();
            const outputFields = TestSignature.getOutputFields();

            expect(inputFields.input).toBeDefined();
            expect(inputFields.input.description).toBe('Test input');
            expect(inputFields.input.type).toBe('string');
            expect(inputFields.input.required).toBe(true);

            expect(outputFields.output).toBeDefined();
            expect(outputFields.output.description).toBe('Test output');
            expect(outputFields.output.type).toBe('number');
            expect(outputFields.output.required).toBe(true);
        });

        it('should support optional fields', () => {
            class TestSignature extends Signature {
                @InputField({ description: 'Optional input', required: false })
                optionalInput?: string;
            }

            const inputFields = TestSignature.getInputFields();
            expect(inputFields.optionalInput.required).toBe(false);
        });

        it('should generate prompt format', () => {
            class TestSignature extends Signature {
                @InputField()
                question!: string;

                @OutputField()
                answer!: string;

                @OutputField()
                confidence!: number;
            }

            const format = TestSignature.getPromptFormat();
            expect(format).toBe('question -> answer, confidence');
        });
    });

    describe('Class-based signatures', () => {
        it('should handle empty fields gracefully', () => {
            class EmptySignature extends Signature { }

            const inputFields = EmptySignature.getInputFields();
            const outputFields = EmptySignature.getOutputFields();
            const format = EmptySignature.getPromptFormat();

            expect(Object.keys(inputFields)).toHaveLength(0);
            expect(Object.keys(outputFields)).toHaveLength(0);
            expect(format).toBe(' -> ');
        });

        it('should support custom field prefixes', () => {
            class CustomSignature extends Signature {
                @InputField({ prefix: 'Query:' })
                question!: string;

                @OutputField({ prefix: 'Response:' })
                answer!: string;
            }

            const inputFields = CustomSignature.getInputFields();
            const outputFields = CustomSignature.getOutputFields();

            expect(inputFields.question.prefix).toBe('Query:');
            expect(outputFields.answer.prefix).toBe('Response:');
        });
    });
}); 