import { z } from 'zod';
import type { FieldConfig, ParsedSignature } from '../types/signature';
import type { ImageInput } from '../types/language-model';
import { TsDspyError } from './errors';
import { type History, toHistory } from './history';

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

/**
 * Field type marking an input as conversation {@link History}. Adapters render
 * it as prior user/assistant turns rather than as a field of the current turn.
 */
export const HISTORY_FIELD_TYPE = 'history';

/** True when a field config declares a history input. */
export function isHistoryFieldType(type: string | undefined): boolean {
    return type === HISTORY_FIELD_TYPE;
}

// --- zod field helpers -------------------------------------------------------
//
// A zod schema has no slot for "this is an image", so the helpers below are
// recognised by their validator function. `z.custom(fn)` keeps `fn` on the
// schema definition, and zod copies the definition through `.describe()`,
// `.optional()` and friends, so the mark survives whatever the caller chains.

function looksLikeImage(value: unknown): boolean {
    if (typeof value === 'string') return value.length > 0;
    return (
        typeof value === 'object' &&
        value !== null &&
        ('kind' in value || (value as { type?: unknown }).type === 'image')
    );
}

function looksLikeHistory(value: unknown): boolean {
    return toHistory(value) !== undefined;
}

/**
 * A zod schema for an image input, so a zod signature can declare one:
 *
 * ```ts
 * const Caption = signature({
 *     input: z.object({ photo: image().describe('the photo') }),
 *     output: z.object({ caption: z.string() }),
 * });
 * ```
 *
 * Accepts anything an image field takes: an `https://` URL, a `data:` URI, or
 * an image source object.
 */
export function image(): z.ZodType<ImageInput> {
    return z.custom<ImageInput>(looksLikeImage, { message: 'Expected an image' });
}

/**
 * A zod schema for a {@link History} input, so a zod signature can carry the
 * earlier turns of a conversation.
 */
export function history(): z.ZodType<History> {
    return z.custom<History>(looksLikeHistory, { message: 'Expected a History' });
}

/**
 * Which special input kind a zod field was declared with — `image` or
 * `history` — looking through `.optional()`, `.describe()` and similar
 * wrappers. `undefined` for an ordinary field.
 */
export function zodSpecialFieldType(schema: z.ZodType): 'image' | 'history' | undefined {
    let current: any = schema;
    for (let depth = 0; depth < 16 && current; depth++) {
        const def = current._zod?.def;
        if (!def) return undefined;
        if (def.type === 'custom') {
            if (def.fn === looksLikeImage) return 'image';
            if (def.fn === looksLikeHistory) return 'history';
            return undefined;
        }
        current = def.innerType ?? (def.type === 'pipe' ? def.in : undefined);
    }
    return undefined;
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

    /**
     * Parse the `"inputs -> outputs"` shorthand, e.g. `"question -> answer: number"`.
     *
     * @throws {TsDspyError} when the string is not exactly one `->` separating a
     * non-empty input list from a non-empty output list of unique field names.
     */
    static parseStringSignature(signature: string): ParsedSignature {
        const fail = (reason: string): never => {
            throw new TsDspyError(
                `Invalid string signature ${JSON.stringify(signature)}: ${reason}. ` +
                    'Expected the form "input1, input2 -> output1, output2: type".'
            );
        };

        if (typeof signature !== 'string') {
            fail('a string signature must be a string');
        }

        // Split outside quotes and parentheses, so a description may contain
        // commas, colons and arrows, and `enum(a|b)` stays one type.
        const sides = splitOutside(signature, '->');
        if (sides.length < 2) fail('missing "->" between inputs and outputs');
        if (sides.length > 2) fail('found more than one "->"');

        const [inputPart, outputPart] = sides.map((s) => s.trim());
        if (inputPart === '') fail('no input fields before "->"');
        if (outputPart === '') fail('no output fields after "->"');

        const seen = new Set<string>();
        const parseFields = (part: string, side: 'input' | 'output') => {
            return splitOutside(part, ',').map((field) => {
                let rest = field.trim();

                // A trailing quoted string is the field's description.
                let description: string | undefined;
                const quoted = /\s*(["'])((?:\\.|(?!\1)[^\\])*)\1\s*$/.exec(rest);
                if (quoted) {
                    description = quoted[2].replace(/\\(["'\\])/g, '$1');
                    rest = rest.slice(0, quoted.index).trim();
                }

                const colon = rest.indexOf(':');
                let name = (colon === -1 ? rest : rest.slice(0, colon)).trim();
                const type = colon === -1 ? '' : rest.slice(colon + 1).trim();

                const optional = name.endsWith('?');
                if (optional) name = name.slice(0, -1).trim();

                if (!name) fail(`empty ${side} field name`);
                if (seen.has(name)) fail(`duplicate field name "${name}"`);
                seen.add(name);
                return { name, type: type || 'string', description, optional };
            });
        };

        const inputFields = parseFields(inputPart, 'input');
        const outputFields = parseFields(outputPart, 'output');
        const all = [...inputFields, ...outputFields];

        return {
            inputs: inputFields.map((f) => f.name),
            outputs: outputFields.map((f) => f.name),
            types: Object.fromEntries(all.map((f) => [f.name, f.type])),
            descriptions: Object.fromEntries(
                all
                    .filter((f) => f.description !== undefined)
                    .map((f) => [f.name, f.description as string])
            ),
            optional: all.filter((f) => f.optional).map((f) => f.name),
        };
    }
}

/**
 * Split `text` on `separator`, ignoring separators inside quotes or
 * parentheses. Unbalanced quotes simply run to the end of the text.
 */
function splitOutside(text: string, separator: string): string[] {
    const parts: string[] = [];
    let depth = 0;
    let quote: string | undefined;
    let start = 0;

    for (let index = 0; index < text.length; index++) {
        const char = text[index];
        if (quote) {
            if (char === '\\') index++;
            else if (char === quote) quote = undefined;
            continue;
        }
        if (char === '"' || char === "'") quote = char;
        else if (char === '(') depth++;
        else if (char === ')') depth = Math.max(0, depth - 1);
        else if (depth === 0 && text.startsWith(separator, index)) {
            parts.push(text.slice(start, index));
            index += separator.length - 1;
            start = index + 1;
        }
    }
    parts.push(text.slice(start));
    return parts;
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
 * Declare a signature from zod schemas, or give a string signature its
 * instructions.
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
 *
 * The string form, `signature('question -> answer: int', 'Answer the maths
 * question.')`, returns a `Signature` class equivalent to the shorthand with
 * those instructions attached — a plain string has nowhere to keep them.
 */
export function signature<TInput extends z.ZodObject, TOutput extends z.ZodObject>(
    definition: ZodSignatureDefinition<TInput, TOutput>
): ZodSignature<TInput, TOutput>;
export function signature(shorthand: string, instructions?: string): typeof Signature;
export function signature(
    definition: ZodSignatureDefinition<z.ZodObject, z.ZodObject> | string,
    instructions?: string
): AnyZodSignature | typeof Signature {
    if (typeof definition === 'string') {
        return toSignatureClass(definition, instructions);
    }

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

// --- deriving signatures -------------------------------------------------------

/**
 * One field to add with {@link extendSignature}: a field config, as the
 * decorators take, or a zod schema.
 */
export type FieldSpec = Partial<FieldConfig> | z.ZodType;

/** Fields to add, in order, keyed by name. */
export type FieldSpecs = Record<string, FieldSpec>;

export interface ExtendSignatureOptions {
    /** Inputs placed before the existing ones. */
    prependInputs?: FieldSpecs;
    /** Inputs placed after the existing ones. */
    appendInputs?: FieldSpecs;
    /** Outputs placed before the existing ones — `reasoning` for chain of thought. */
    prependOutputs?: FieldSpecs;
    /** Outputs placed after the existing ones. */
    appendOutputs?: FieldSpecs;
    /** Existing inputs to drop. */
    removeInputs?: string[];
    /** Existing outputs to drop. */
    removeOutputs?: string[];
    /** Replace the instructions (the signature's description). */
    instructions?: string;
}

function isZodSchema(value: unknown): value is z.ZodType {
    return (
        typeof value === 'object' &&
        value !== null &&
        typeof (value as { safeParse?: unknown }).safeParse === 'function'
    );
}

/** The field-config view of a zod field, for adding it to a class signature. */
function zodToFieldConfig(
    name: string,
    schema: z.ZodType,
    side: 'input' | 'output'
): FieldConfig {
    const special = zodSpecialFieldType(schema);
    const required = !schema.safeParse(undefined).success;
    const description =
        schema.description || `${side === 'input' ? 'Input' : 'Output'} field: ${name}`;
    if (special) return { description, type: special, required };

    let node: Record<string, unknown> = {};
    try {
        node = z.toJSONSchema(schema, { io: 'input', unrepresentable: 'any' }) as Record<
            string,
            unknown
        >;
    } catch {
        // Undescribable: treated as free text.
    }
    if (Array.isArray(node.enum)) {
        return { description, type: 'enum', values: node.enum.map(String), required };
    }
    const type = Array.isArray(node.type)
        ? node.type.find((entry) => entry !== 'null')
        : node.type;
    const items = (node.items as { type?: unknown } | undefined)?.type;
    const label =
        type === 'integer'
            ? 'int'
            : type === 'number' || type === 'boolean' || type === 'object'
              ? type
              : type === 'array'
                ? items === 'number' || items === 'integer'
                    ? 'number[]'
                    : items === 'string'
                      ? 'string[]'
                      : 'array'
                : 'string';
    return { description, type: label, required };
}

/**
 * The plain zod schema for a field config, for adding it to a zod signature.
 *
 * No coercion here: a zod signature's output schema is wrapped in the
 * text-coercion layer when it is validated, so the schema itself stays exactly
 * what the model must produce — which is also what its JSON Schema needs.
 */
function fieldConfigToPlainZod(config: Partial<FieldConfig>): z.ZodType {
    const raw = (config.type ?? 'string').trim();
    const inlineEnum = /^enum\s*\((.*)\)$/is.exec(raw);
    const type = inlineEnum ? 'enum' : raw.toLowerCase();

    let schema: z.ZodType;
    switch (type) {
        case 'number':
        case 'float':
            schema = z.number();
            break;
        case 'int':
        case 'integer':
            schema = z.number().int();
            break;
        case 'boolean':
        case 'bool':
            schema = z.boolean();
            break;
        case 'string[]':
            schema = z.array(z.string());
            break;
        case 'number[]':
            schema = z.array(z.number());
            break;
        case 'array':
        case 'list':
            schema = z.array(z.string());
            break;
        case 'object':
        case 'json':
            schema = z.object({});
            break;
        case 'enum': {
            const members = inlineEnum
                ? inlineEnum[1].split('|').map((member) => member.trim())
                : (config.values ?? []);
            if (members.length === 0) {
                throw new TsDspyError("Field type 'enum' requires a non-empty set of values.");
            }
            schema = z.enum(members as [string, ...string[]]);
            break;
        }
        case IMAGE_FIELD_TYPE:
            schema = image();
            break;
        case HISTORY_FIELD_TYPE:
            schema = history();
            break;
        default:
            schema = z.string();
    }

    if (config.description) schema = schema.describe(config.description);
    return config.required === false ? schema.optional() : schema;
}

/** A field spec as a complete field config, with the decorator defaults applied. */
function specToFieldConfig(
    name: string,
    spec: FieldSpec,
    side: 'input' | 'output'
): FieldConfig {
    if (isZodSchema(spec)) return zodToFieldConfig(name, spec, side);
    return {
        description:
            spec.description || `${side === 'input' ? 'Input' : 'Output'} field: ${name}`,
        prefix: spec.prefix,
        type: spec.type || 'string',
        required: spec.required !== false,
        values: spec.values,
    };
}

/**
 * Turn a string signature into an equivalent `Signature` class, carrying its
 * inline descriptions, optional markers and types, plus `instructions` as the
 * class description.
 */
export function toSignatureClass(shorthand: string, instructions?: string): typeof Signature {
    const parsed = Signature.parseStringSignature(shorthand);
    const config = (name: string, side: 'input' | 'output'): FieldConfig => ({
        description:
            parsed.descriptions[name] ??
            `${side === 'input' ? 'Input' : 'Output'} field: ${name}`,
        type: parsed.types[name] ?? 'string',
        required: !parsed.optional.includes(name),
    });
    return defineSignatureClass(
        Object.fromEntries(parsed.inputs.map((name) => [name, config(name, 'input')])),
        Object.fromEntries(parsed.outputs.map((name) => [name, config(name, 'output')])),
        instructions
    );
}

/** Build an anonymous `Signature` subclass from field maps. */
function defineSignatureClass(
    inputs: Record<string, FieldConfig>,
    outputs: Record<string, FieldConfig>,
    description?: string
): typeof Signature {
    const Derived = class extends Signature {} as unknown as typeof Signature & {
        [key: symbol]: Record<string, FieldConfig>;
    };
    Derived[INPUT_FIELDS] = inputs;
    Derived[OUTPUT_FIELDS] = outputs;
    if (description !== undefined) Derived.description = description;
    return Derived;
}

function withoutKeys<T>(
    record: Record<string, T>,
    keys: string[] | undefined
): Record<string, T> {
    if (!keys?.length) return { ...record };
    return Object.fromEntries(Object.entries(record).filter(([key]) => !keys.includes(key)));
}

/**
 * Derive a new signature from an existing one: add inputs or outputs before or
 * after the existing ones, drop some, or replace the instructions. The original
 * is never modified.
 *
 * ```ts
 * // What ChainOfThought does: a leading `reasoning` output.
 * const withReasoning = extendSignature(QA, {
 *     prependOutputs: { reasoning: { description: 'Think step by step' } },
 * });
 * ```
 *
 * Works on all three forms. A zod signature stays a zod signature, so its
 * schemas — enums, bounds, refinements on individual fields — are kept
 * exactly; refinements on the output *object* itself are not carried over,
 * because the object has to be rebuilt to reorder its keys. A class or string
 * signature comes back as a `Signature` class. New fields may be given as
 * field configs or zod schemas whichever the base is.
 */
export function extendSignature(
    base: AnyZodSignature,
    options: ExtendSignatureOptions
): AnyZodSignature;
export function extendSignature(
    base: typeof Signature | string,
    options: ExtendSignatureOptions
): typeof Signature;
export function extendSignature(
    base: SignatureLike,
    options: ExtendSignatureOptions
): SignatureSource;
export function extendSignature(
    base: SignatureLike,
    options: ExtendSignatureOptions
): SignatureSource {
    if (isZodSignature(base)) {
        const zodFields = (specs: FieldSpecs | undefined) =>
            Object.fromEntries(
                Object.entries(specs ?? {}).map(([name, spec]) => [
                    name,
                    isZodSchema(spec) ? spec : fieldConfigToPlainZod(spec),
                ])
            );
        const inputShape = withoutKeys(
            base.input.shape as Record<string, z.ZodType>,
            options.removeInputs
        );
        const outputShape = withoutKeys(
            base.output.shape as Record<string, z.ZodType>,
            options.removeOutputs
        );
        return {
            kind: 'zod-signature',
            description: options.instructions ?? base.description,
            input: z.object({
                ...zodFields(options.prependInputs),
                ...inputShape,
                ...zodFields(options.appendInputs),
            }),
            output: z.object({
                ...zodFields(options.prependOutputs),
                ...outputShape,
                ...zodFields(options.appendOutputs),
            }),
        };
    }

    const source = typeof base === 'string' ? toSignatureClass(base) : base;
    const configs = (specs: FieldSpecs | undefined, side: 'input' | 'output') =>
        Object.fromEntries(
            Object.entries(specs ?? {}).map(([name, spec]) => [
                name,
                specToFieldConfig(name, spec, side),
            ])
        );

    return defineSignatureClass(
        {
            ...configs(options.prependInputs, 'input'),
            ...withoutKeys(source.getInputFields(), options.removeInputs),
            ...configs(options.appendInputs, 'input'),
        },
        {
            ...configs(options.prependOutputs, 'output'),
            ...withoutKeys(source.getOutputFields(), options.removeOutputs),
            ...configs(options.appendOutputs, 'output'),
        },
        options.instructions ?? source.description
    );
}

/** A signature's instructions: the description, or `undefined` for a bare string signature. */
export function getSignatureInstructions(signature: SignatureLike): string | undefined {
    if (typeof signature === 'string') return undefined;
    return signature.description;
}
