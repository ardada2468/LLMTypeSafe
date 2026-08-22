import type { z } from 'zod';
import type { FieldConfig, ParsedSignature } from '../types/signature';

/**
 * Field type marking an input as an image rather than text.
 *
 * Only inputs may be images: a model returns text, so an output field declared
 * `image` would be a promise nothing can keep.
 */
export const IMAGE_FIELD_TYPE = 'image';

/** True when a field config declares an image input. */
export function isImageFieldType(type: string | undefined): boolean {
    return type === IMAGE_FIELD_TYPE;
}

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
        // Only meaningful for `type: 'enum'`, but copied unconditionally so the
        // field map stays a faithful record of what was declared.
        values: config.values,
    };
}

export function InputField(config: Partial<FieldConfig> = {}) {
    return function (target: any, propertyKey: string | symbol | any) {
        defineField('input', INPUT_FIELDS, target, propertyKey, config);
    };
}

/**
 * Declare an image input. Sugar for `@InputField({ type: 'image' })`.
 *
 * The decorated property holds an {@link ImageInput}: an `https://` URL, a
 * `data:` URI, or an explicit source object.
 *
 * ```ts
 * class DescribeReceipt extends Signature {
 *     @ImageField({ description: 'photo of the receipt' })
 *     receipt!: ImageInput;
 *
 *     @OutputField({ description: 'total charged', type: 'number' })
 *     total!: number;
 * }
 * ```
 */
export function ImageField(config: Omit<Partial<FieldConfig>, 'type'> = {}) {
    return function (target: any, propertyKey: string | symbol | any) {
        defineField('input', INPUT_FIELDS, target, propertyKey, {
            ...config,
            type: IMAGE_FIELD_TYPE,
        });
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

/**
 * A signature declared with zod schemas rather than decorators or a string.
 *
 * Decorators record fields on a static at runtime, so TypeScript learns nothing
 * about them; a zod object carries its shape in the type system, which is what
 * lets `Predict` infer the result type with no explicit type argument. It also
 * unlocks the field kinds the flat decorator type list cannot express — enums,
 * unions, nested objects, and numeric bounds.
 *
 * Build one with {@link signature}; the `kind` brand is what every consumer
 * branches on.
 */
export interface ZodSignature<
    TInput extends z.ZodObject = z.ZodObject,
    TOutput extends z.ZodObject = z.ZodObject,
> {
    readonly kind: 'zod-signature';
    readonly description?: string;
    readonly input: TInput;
    readonly output: TOutput;
}

/** Any zod signature, whatever its input and output shapes. */
export type AnyZodSignature = ZodSignature<any, any>;

/** Anything that declares fields: a `Signature` subclass or a zod signature. */
export type SignatureSource = typeof Signature | AnyZodSignature;

/** Every accepted signature form, including the string shorthand. */
export type SignatureLike = SignatureSource | string;

export interface ZodSignatureDefinition<
    TInput extends z.ZodObject,
    TOutput extends z.ZodObject,
> {
    /** Task description, placed at the top of the prompt. */
    description?: string;
    input: TInput;
    output: TOutput;
}

function assertZodObject(value: unknown, side: 'input' | 'output'): void {
    if (
        typeof value !== 'object' ||
        value === null ||
        typeof (value as { shape?: unknown }).shape !== 'object'
    ) {
        throw new Error(
            `signature(): "${side}" must be a z.object({ ... }). ` +
                `Received ${value === null ? 'null' : typeof value}.`
        );
    }
}

/**
 * Declare a signature from zod schemas.
 *
 * ```ts
 * const AnalyzeReview = signature({
 *     description: 'Analyze a product review.',
 *     input: z.object({ review: z.string() }),
 *     output: z.object({
 *         sentiment: z.enum(['positive', 'negative', 'neutral']),
 *         rating: z.number().int().min(1).max(5),
 *     }),
 * });
 *
 * const result = await new Predict(AnalyzeReview).forward({ review });
 * result.sentiment; // 'positive' | 'negative' | 'neutral'
 * ```
 *
 * Needs no `experimentalDecorators`, and the zod schema is used verbatim for
 * validation, so every constraint you express is enforced.
 */
export function signature<TInput extends z.ZodObject, TOutput extends z.ZodObject>(
    definition: ZodSignatureDefinition<TInput, TOutput>
): ZodSignature<TInput, TOutput> {
    assertZodObject(definition?.input, 'input');
    assertZodObject(definition?.output, 'output');

    return {
        kind: 'zod-signature',
        description: definition.description,
        input: definition.input,
        output: definition.output,
    };
}

/** Narrow an unknown signature value to a zod signature. */
export function isZodSignature(value: unknown): value is AnyZodSignature {
    return (
        typeof value === 'object' &&
        value !== null &&
        (value as { kind?: unknown }).kind === 'zod-signature'
    );
}
