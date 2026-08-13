import type { FieldConfig, ParsedSignature } from '../types/signature';

// Symbol keys for decorator metadata
const INPUT_FIELDS = Symbol('inputFields');
const OUTPUT_FIELDS = Symbol('outputFields');

/**
 * Record a field on the owning class.
 *
 * These are legacy (`experimentalDecorators`) decorators: they receive the
 * prototype and a property key, and store the field on the constructor so
 * `getInputFields()` works without instantiating the class.
 *
 * Under TC39 stage-3 decorators — what you get when `experimentalDecorators` is
 * off — a field decorator is instead called with `undefined` and a context
 * object, and cannot reach the class at decoration time. Rather than fail with
 * `Cannot read properties of undefined`, detect that and say what to change.
 */
function defineField(
    kind: 'input' | 'output',
    symbol: symbol,
    target: any,
    propertyKey: string | symbol | any,
    config: Partial<FieldConfig>
): void {
    if (target === undefined || target === null) {
        const name =
            typeof propertyKey === 'object' && propertyKey?.name
                ? String(propertyKey.name)
                : String(propertyKey);
        throw new Error(
            `@${kind === 'input' ? 'InputField' : 'OutputField'} on "${name}" requires legacy decorators. ` +
                `Set "experimentalDecorators": true in your tsconfig.json ` +
                `(and make sure the file is covered by that tsconfig).`
        );
    }

    const key =
        typeof propertyKey === 'object' && propertyKey.name
            ? String(propertyKey.name)
            : String(propertyKey);

    const owner = target.constructor;
    // Own property, not inherited: two signatures extending a common base must
    // not share one field map.
    if (!Object.prototype.hasOwnProperty.call(owner, symbol)) {
        owner[symbol] = { ...(owner[symbol] ?? {}) };
    }

    owner[symbol][key] = {
        description:
            config.description || `${kind === 'input' ? 'Input' : 'Output'} field: ${key}`,
        prefix: config.prefix,
        type: config.type || 'string',
        required: config.required !== false,
    };
}

export function InputField(config: Partial<FieldConfig> = {}) {
    return function (target: any, propertyKey: string | symbol | any) {
        defineField('input', INPUT_FIELDS, target, propertyKey, config);
    };
}

export function OutputField(config: Partial<FieldConfig> = {}) {
    return function (target: any, propertyKey: string | symbol | any) {
        defineField('output', OUTPUT_FIELDS, target, propertyKey, config);
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
        const [inputPart, outputPart] = signature.split('->').map((s) => s.trim());

        const parseFields = (part: string) => {
            return part.split(',').map((field) => {
                const trimmed = field.trim();
                const [name, type] = trimmed.split(':').map((s) => s.trim());
                return { name, type: type || 'string' };
            });
        };

        const inputFields = parseFields(inputPart);
        const outputFields = parseFields(outputPart);

        return {
            inputs: inputFields.map((f) => f.name),
            outputs: outputFields.map((f) => f.name),
            types: {
                ...Object.fromEntries(inputFields.map((f) => [f.name, f.type])),
                ...Object.fromEntries(outputFields.map((f) => [f.name, f.type])),
            },
        };
    }
}
