import { buildPrompt, parseOutput } from './parsing';
import { Signature, InputField, OutputField } from '../core/signature';

// Mock signature class for testing
class TestSignature extends Signature {
    static description = 'Test signature for parsing';
}

describe('Parsing Utils', () => {
    describe('buildPrompt', () => {
        it('should build prompt from string signature', () => {
            const signature = 'question -> answer';
            const inputs = { question: 'What is the capital of France?' };

            const prompt = buildPrompt(signature, inputs);

            expect(prompt).toContain('question: What is the capital of France?');
            expect(prompt).toContain('Provide the answer in this format:');
            expect(prompt).toContain('answer: [your response]');
        });

        it('should handle multiple inputs and outputs', () => {
            const signature = 'question, context -> answer, confidence: float';
            const inputs = {
                question: 'What is the capital?',
                context: 'France is a country in Europe.'
            };

            const prompt = buildPrompt(signature, inputs);

            expect(prompt).toContain('question: What is the capital?');
            expect(prompt).toContain('context: France is a country in Europe.');
            expect(prompt).toContain('Provide the following fields:');
            expect(prompt).toContain('answer (string): [your response]');
            expect(prompt).toContain('confidence (float): [your response]');
        });

        it('should skip undefined inputs', () => {
            const signature = 'question, context -> answer';
            const inputs = { question: 'What is the capital?' };

            const prompt = buildPrompt(signature, inputs);

            expect(prompt).toContain('question: What is the capital?');
            expect(prompt).not.toContain('context:');
        });
    });

    describe('parseOutput', () => {
        it('should parse simple output from string signature', () => {
            const signature = 'question -> answer';
            const rawOutput = 'answer: Paris is the capital of France.';

            const parsed = parseOutput(signature, rawOutput);

            expect(parsed.answer).toBe('Paris is the capital of France.');
        });

        it('should parse multiple outputs', () => {
            const signature = 'question -> answer, confidence: float';
            const rawOutput = `answer: Paris
confidence: 0.95`;

            const parsed = parseOutput(signature, rawOutput);

            expect(parsed.answer).toBe('Paris');
            expect(parsed.confidence).toBe(0.95);
        });

        it('should convert types correctly', () => {
            const signature = 'input -> number_val: int, float_val: float, bool_val: bool, array_val: array';
            const rawOutput = `number_val: 42
float_val: 3.14
bool_val: true
array_val: ["a", "b", "c"]`;

            const parsed = parseOutput(signature, rawOutput);

            expect(parsed.number_val).toBe(42);
            expect(parsed.float_val).toBe(3.14);
            expect(parsed.bool_val).toBe(true);
            expect(parsed.array_val).toEqual(['a', 'b', 'c']);
        });

        it('should handle invalid JSON gracefully', () => {
            const signature = 'input -> array_val: array';
            const rawOutput = 'array_val: item1, item2, item3';

            const parsed = parseOutput(signature, rawOutput);

            expect(parsed.array_val).toEqual(['item1', 'item2', 'item3']);
        });

        it('should return null for missing fields', () => {
            const signature = 'input -> missing_field';
            const rawOutput = 'some_other_field: value';

            const parsed = parseOutput(signature, rawOutput);

            expect(parsed.missing_field).toBeNull();
        });
    });

    describe('Type conversion', () => {
        it('should handle boolean conversion', () => {
            const signature = 'input -> bool_val: bool';

            expect(parseOutput(signature, 'bool_val: true').bool_val).toBe(true);
            expect(parseOutput(signature, 'bool_val: false').bool_val).toBe(false);
            expect(parseOutput(signature, 'bool_val: 1').bool_val).toBe(true);
            expect(parseOutput(signature, 'bool_val: TRUE').bool_val).toBe(true);
        });

        it('should handle number conversion with fallback', () => {
            const signature = 'input -> num_val: int';

            expect(parseOutput(signature, 'num_val: 42').num_val).toBe(42);
            expect(parseOutput(signature, 'num_val: not_a_number').num_val).toBe('not_a_number');
        });

        it('should auto-detect JSON objects', () => {
            const signature = 'input -> obj_val';
            const rawOutput = 'obj_val: {"key": "value", "number": 42}';

            const parsed = parseOutput(signature, rawOutput);

            expect(parsed.obj_val).toEqual({ key: 'value', number: 42 });
        });
    });
}); 