import { ValidationError, type FieldValidationIssue } from './errors';
import {
    buildRepairObservation,
    buildRepairPrompt,
    describeValidationIssues,
    isRepeatedFailure,
    listFailingFields,
    MAX_REPAIR_ATTEMPTS,
    normaliseRepairAttempts,
} from './repair';

const issues: FieldValidationIssue[] = [
    {
        field: 'confidence',
        expected: 'number',
        received: 'very high',
        message: 'Expected number, received string',
    },
    {
        field: 'answer',
        expected: 'string',
        received: undefined,
        message: 'field not found in model output',
    },
];

describe('listFailingFields', () => {
    it('joins the field names in order', () => {
        expect(listFailingFields(issues)).toBe('confidence, answer');
    });

    it('de-duplicates a field that failed more than once', () => {
        const repeated: FieldValidationIssue[] = [
            { field: 'score', expected: 'number', received: 'a', message: 'bad' },
            { field: 'score', expected: 'number', received: 'a', message: 'also bad' },
        ];

        expect(listFailingFields(repeated)).toBe('score');
    });
});

describe('describeValidationIssues', () => {
    it('reports the declared type and the received value per field', () => {
        const description = describeValidationIssues(issues);

        expect(description).toBe(
            '  - confidence: expected number, received "very high"\n' +
                '  - answer: expected string, received nothing'
        );
    });

    it('renders an absent value as nothing', () => {
        expect(describeValidationIssues(issues)).toContain(
            '  - answer: expected string, received nothing'
        );
    });

    it('omits the coercion message, which describes the post-coercion value', () => {
        const coerced: FieldValidationIssue[] = [
            {
                field: 'confidence',
                expected: 'number',
                received: 'very high',
                message: 'Invalid input: expected number, received NaN',
            },
        ];

        expect(describeValidationIssues(coerced)).not.toContain('NaN');
    });
});

describe('buildRepairPrompt', () => {
    it('keeps the original prompt, the failures and the previous response', () => {
        const error = new ValidationError(issues, 'answer: Paris\nconfidence: very high');

        const prompt = buildRepairPrompt('question: Capital of France?', error);

        expect(prompt).toContain('question: Capital of France?');
        expect(prompt).toContain('failed validation for: confidence, answer');
        expect(prompt).toContain('expected number, received "very high"');
        expect(prompt).toContain('Previous response:\nanswer: Paris\nconfidence: very high');
    });

    it('asks for labelled lines on the text path', () => {
        const error = new ValidationError(issues, 'raw');

        expect(buildRepairPrompt('base', error, 'text')).toContain('"field: value" line');
    });

    it('asks for a corrected object on the structured path', () => {
        const error = new ValidationError(issues, '{}');

        expect(buildRepairPrompt('base', error, 'structured')).toContain(
            'Return the corrected object'
        );
    });
});

describe('buildRepairObservation', () => {
    it('names the malformed fields and the shape RespAct expects back', () => {
        const error = new ValidationError(issues, 'answer: Paris');

        const observation = buildRepairObservation(error);

        expect(observation).toContain('missing or malformed for: confidence, answer');
        expect(observation).toContain('expected number, received "very high"');
        expect(observation).toContain('Provide a Final Answer with every required field');
    });
});

describe('normaliseRepairAttempts', () => {
    it('defaults an absent value to zero', () => {
        expect(normaliseRepairAttempts(undefined)).toBe(0);
    });

    it('clamps a negative value to zero', () => {
        expect(normaliseRepairAttempts(-3)).toBe(0);
    });

    it('truncates a fractional value', () => {
        expect(normaliseRepairAttempts(2.7)).toBe(2);
    });

    it('rejects NaN', () => {
        expect(normaliseRepairAttempts(Number.NaN)).toBe(0);
    });

    it('reads Infinity as the maximum rather than as none', () => {
        expect(normaliseRepairAttempts(Number.POSITIVE_INFINITY)).toBe(MAX_REPAIR_ATTEMPTS);
    });

    it('caps an implausibly large value', () => {
        expect(normaliseRepairAttempts(100_000)).toBe(MAX_REPAIR_ATTEMPTS);
    });
});

describe('isRepeatedFailure', () => {
    it('recognises two identical failures', () => {
        const first = new ValidationError(issues, 'answer: Paris');
        const second = new ValidationError(issues, 'answer: Paris');

        expect(isRepeatedFailure(first, second)).toBe(true);
    });

    it('separates failures whose raw output differs', () => {
        const first = new ValidationError(issues, 'answer: Paris');
        const second = new ValidationError(issues, 'answer: Lyon');

        expect(isRepeatedFailure(first, second)).toBe(false);
    });

    it('separates failures whose fields differ', () => {
        const other: FieldValidationIssue[] = [
            { field: 'answer', expected: 'string', received: undefined, message: 'missing' },
        ];
        const first = new ValidationError(issues, 'raw');
        const second = new ValidationError(other, 'raw');

        expect(isRepeatedFailure(first, second)).toBe(false);
    });
});
