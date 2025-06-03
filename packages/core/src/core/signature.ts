import 'reflect-metadata';
import { FieldConfig, ParsedSignature } from '../types/signature';

// Symbol keys for decorator metadata
const INPUT_FIELDS = Symbol('inputFields');
const OUTPUT_FIELDS = Symbol('outputFields');

// Minimal interface for the context object apparently being passed
interface DecoratorPropertyContext {
    name: string | symbol;
    // kind?: 'field'; // Stage 3 decorators also have a 'kind' property
}

export function InputField(config: Partial<FieldConfig> = {}) {
    return function (target: any, propertyKey: string | symbol | any) {
        // Handle both legacy and modern decorator contexts
        const key = typeof propertyKey === 'object' && propertyKey.name
            ? String(propertyKey.name)
            : String(propertyKey);

        if (!target.constructor[INPUT_FIELDS]) {
            target.constructor[INPUT_FIELDS] = {};
        }
        target.constructor[INPUT_FIELDS][key] = {
            description: config.description || `Input field: ${key}`,
            prefix: config.prefix,
            type: config.type || 'string',
            required: config.required !== false
        };
    };
}

export function OutputField(config: Partial<FieldConfig> = {}) {
    return function (target: any, propertyKey: string | symbol | any) {
        // Handle both legacy and modern decorator contexts
        const key = typeof propertyKey === 'object' && propertyKey.name
            ? String(propertyKey.name)
            : String(propertyKey);

        if (!target.constructor[OUTPUT_FIELDS]) {
            target.constructor[OUTPUT_FIELDS] = {};
        }
        target.constructor[OUTPUT_FIELDS][key] = {
            description: config.description || `Output field: ${key}`,
            prefix: config.prefix,
            type: config.type || 'string',
            required: config.required !== false
        };
    };
}

export abstract class Signature {
    static description?: string;

    static getInputFields(): Record<string, FieldConfig> {
        return (this as any)[INPUT_FIELDS] || {};
    }

    static getOutputFields(): Record<string, FieldConfig> {
        return (this as any)[OUTPUT_FIELDS] || {};
    }

    static getPromptFormat(): string {
        const inputs = Object.keys(this.getInputFields());
        const outputs = Object.keys(this.getOutputFields());
        return `${inputs.join(', ')} -> ${outputs.join(', ')}`;
    }

    static parseStringSignature(signature: string): ParsedSignature {
        const [inputPart, outputPart] = signature.split('->').map(s => s.trim());

        const parseFields = (part: string) => {
            return part.split(',').map(field => {
                const trimmed = field.trim();
                const [name, type] = trimmed.split(':').map(s => s.trim());
                return { name, type: type || 'string' };
            });
        };

        const inputFields = parseFields(inputPart);
        const outputFields = parseFields(outputPart);

        return {
            inputs: inputFields.map(f => f.name),
            outputs: outputFields.map(f => f.name),
            types: {
                ...Object.fromEntries(inputFields.map(f => [f.name, f.type])),
                ...Object.fromEntries(outputFields.map(f => [f.name, f.type]))
            }
        };
    }
} 