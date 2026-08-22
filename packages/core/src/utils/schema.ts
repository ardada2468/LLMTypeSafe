import { z } from 'zod';
import { Signature } from '../core/signature';
import type { FieldConfig } from '../types/signature';

/** JSON Schema fragment describing one field. */
type JsonSchemaProperty = Record<string, unknown>;

/** A JSON Schema object suitable for provider structured-output modes. */
export interface OutputJsonSchema {
    type: 'object';
    properties: Record<string, JsonSchemaProperty>;
    required: string[];
    additionalProperties: false;
}

const TRUTHY = new Set(['true', '1', 'yes', 'y', 'on']);
const FALSY = new Set(['false', '0', 'no', 'n', 'off']);

/**
 * Resolve the output fields of a signature, whether it is a `Signature` subclass
 * or a string signature like `"question -> answer: float"`.
 */
export function getOutputFieldConfigs(
    signature: typeof Signature | string
): Record<string, FieldConfig> {
    if (typeof signature !== 'string') {
        return signature.getOutputFields();
    }

    const parsed = Signature.parseStringSignature(signature);
    const configs: Record<string, FieldConfig> = {};
    for (const name of parsed.outputs) {
        // Normalise here so callers see the same shape whichever way the field
        // was declared: `enum(a|b)` in a string signature and
        // `{ type: 'enum', values: ['a', 'b'] }` on a decorator both arrive as
        // `{ type: 'enum', values: [...] }`.
        const resolved = resolveFieldType({
            description: '',
            type: parsed.types[name] ?? 'string',
        });
        configs[name] = {
            description: `Output field: ${name}`,
            type: resolved.type,
            required: true,
            ...(resolved.values ? { values: resolved.values } : {}),
        };
    }
    return configs;
}

/** A field's declared type after enum members have been resolved. */
interface ResolvedFieldType {
    type: string;
    /** Present only when `type` is `'enum'`. */
    values?: string[];
}

/** Split the body of an inline `enum(a|b|c)` declaration into its members. */
function splitEnumMembers(body: string): string[] {
    const members = body
        .split('|')
        .map((member) => member.trim())
        .filter((member) => member.length > 0);
    return [...new Set(members)];
}

/**
 * Normalise a field's declared type, lower-casing it and resolving the members
 * of an enum.
 *
 * Enum members can be declared two ways and both land here:
 *
 * - decorator form &mdash; `@OutputField({ type: 'enum', values: ['a', 'b'] })`
 * - string-signature form &mdash; `'review -> sentiment: enum(a|b)'`, pipe-separated
 *   because `parseStringSignature` splits fields on commas
 */
function resolveFieldType(config: FieldConfig): ResolvedFieldType {
    const raw = (config.type ?? 'string').trim();

    if (/^enum\b/i.test(raw)) {
        const inline = /^enum\s*\((.*)\)$/is.exec(raw);
        if (inline) {
            return { type: 'enum', values: splitEnumMembers(inline[1]) };
        }
        if (raw.toLowerCase() === 'enum') {
            return { type: 'enum', values: config.values ? [...new Set(config.values)] : [] };
        }
        // Letting a malformed declaration fall through to `string` would validate
        // anything at all, which is the exact permissiveness this type removes. A
        // mistyped `enum(a|b` has to be an error, not a silently open field.
        throw new Error(
            `Unrecognised enum declaration ${JSON.stringify(raw)}. Write it as ` +
                '`enum(a|b|c)`, members separated by `|` because commas already ' +
                "separate fields, or as `type: 'enum'` with a `values` array."
        );
    }

    return { type: raw.toLowerCase() };
}

/**
 * An enum with no members would validate nothing and constrain nothing, which is
 * the silent-permissiveness this field type exists to remove. Say so instead.
 */
function enumMembersOrThrow(resolved: ResolvedFieldType): [string, ...string[]] {
    const values = resolved.values ?? [];
    if (values.length === 0) {
        throw new Error(
            "Field type 'enum' requires a non-empty set of values. Declare them as " +
                "`values: ['a', 'b']` alongside `type: 'enum'`, or inline in a string " +
                'signature as `enum(a|b)`.'
        );
    }
    return values as [string, ...string[]];
}

/** Strings that came out of a text response need trimming before coercion. */
function preprocessNumber(value: unknown): unknown {
    if (typeof value !== 'string') return value;
    const trimmed = value.trim();
    // Number('') === 0, which would silently accept an empty field.
    if (trimmed === '') return value;
    // Tolerate thousands separators and a trailing unit-free percent sign.
    return Number(trimmed.replace(/,/g, '').replace(/%$/, ''));
}

function preprocessBoolean(value: unknown): unknown {
    if (typeof value !== 'string') return value;
    const normalized = value.trim().toLowerCase();
    if (TRUTHY.has(normalized)) return true;
    if (FALSY.has(normalized)) return false;
    return value;
}

function preprocessArray(value: unknown): unknown {
    if (Array.isArray(value)) return value;
    if (typeof value !== 'string') return value;
    const trimmed = value.trim();
    if (trimmed.startsWith('[')) {
        try {
            return JSON.parse(trimmed);
        } catch {
            // Fall through to delimiter splitting.
        }
    }
    const parts = trimmed
        .split(/[,;\n]/)
        .map((part) => part.trim())
        .filter((part) => part.length > 0);
    return parts.length > 0 ? parts : value;
}

function preprocessObject(value: unknown): unknown {
    if (typeof value !== 'string') return value;
    const trimmed = value.trim();
    if (!trimmed.startsWith('{') && !trimmed.startsWith('[')) return value;
    try {
        return JSON.parse(trimmed);
    } catch {
        return value;
    }
}

/**
 * Match a model's answer against the declared members, ignoring surrounding
 * whitespace and case &mdash; a model asked for `positive` will happily write
 * `Positive`, and that is the same answer.
 *
 * Anything with no match is passed through untouched so `z.enum` fails on the
 * value the model actually produced rather than on a mangled version of it.
 */
function preprocessEnum(values: string[]) {
    return (value: unknown): unknown => {
        if (typeof value !== 'string') return value;
        const normalized = value.trim().toLowerCase();
        return values.find((member) => member.toLowerCase() === normalized) ?? value;
    };
}

/**
 * Build a zod schema for a single field.
 *
 * Coercion stays lenient — model output is text, so `"42"` should satisfy a
 * `number` field — but anything that cannot be coerced now fails loudly instead
 * of falling back to the raw string.
 */
export function fieldConfigToZod(config: FieldConfig): z.ZodType {
    const resolved = resolveFieldType(config);

    let schema: z.ZodType;
    switch (resolved.type) {
        case 'number':
        case 'float':
            schema = z.preprocess(preprocessNumber, z.number());
            break;

        case 'int':
        case 'integer':
            schema = z.preprocess(preprocessNumber, z.number().int());
            break;

        case 'boolean':
        case 'bool':
            schema = z.preprocess(preprocessBoolean, z.boolean());
            break;

        case 'string[]':
            schema = z.preprocess(preprocessArray, z.array(z.string()));
            break;

        case 'number[]':
            schema = z.preprocess((value) => {
                const arr = preprocessArray(value);
                return Array.isArray(arr) ? arr.map(preprocessNumber) : arr;
            }, z.array(z.number()));
            break;

        case 'array':
        case 'list':
            schema = z.preprocess(preprocessArray, z.array(z.unknown()));
            break;

        case 'object':
        case 'json':
            schema = z.preprocess(preprocessObject, z.looseObject({}));
            break;

        case 'enum': {
            const members = enumMembersOrThrow(resolved);
            schema = z.preprocess(preprocessEnum(members), z.enum(members));
            break;
        }

        case 'string':
        default:
            schema = z.string();
            break;
    }

    return config.required === false ? schema.optional() : schema;
}

/** Build a zod object schema validating every output field of a signature. */
export function buildOutputSchema(signature: typeof Signature | string): z.ZodType {
    const fields = getOutputFieldConfigs(signature);
    const shape: Record<string, z.ZodType> = {};
    for (const [name, config] of Object.entries(fields)) {
        shape[name] = fieldConfigToZod(config);
    }
    // Loose: providers may return extra keys, and dropping them silently is worse
    // than passing them through for the caller to inspect.
    return z.looseObject(shape);
}

/**
 * Map one declared field type onto a JSON Schema fragment.
 *
 * Every fragment here has to survive OpenAI's strict mode, which is stricter
 * than JSON Schema itself in two ways that bite the open-ended types:
 *
 * - every object in the schema, nested ones included, must carry
 *   `additionalProperties: false`
 * - `items` must be a real schema; the empty `{}` that means "any element" is
 *   rejected outright
 */
function jsonSchemaTypeFor(resolved: ResolvedFieldType): JsonSchemaProperty {
    switch (resolved.type) {
        case 'number':
        case 'float':
            return { type: 'number' };
        case 'int':
        case 'integer':
            return { type: 'integer' };
        case 'boolean':
        case 'bool':
            return { type: 'boolean' };
        case 'string[]':
            return { type: 'array', items: { type: 'string' } };
        case 'number[]':
            return { type: 'array', items: { type: 'number' } };
        case 'array':
        case 'list':
            // This used to emit `items: {}`, which OpenAI rejects. Naming `string`
            // as the element type is the trade, and it is a real one: it does
            // constrain what a structured provider may emit, so a list of figures
            // now arrives as `['1', '2']`. Declare `number[]` when the elements
            // have a type worth naming. Validation is unmoved either way — the zod
            // side is `z.array(z.unknown())`.
            return { type: 'array', items: { type: 'string' } };
        case 'object':
        case 'json':
            // This used to emit `additionalProperties: true`, which OpenAI rejects.
            // Strict mode has no way to express a free-form object at all, so this
            // closes it — and one schema goes to every structured provider, so a
            // bare `object` field is effectively pinned to `{}` on all of them, not
            // only OpenAI. Declare the keys you actually want as their own signature
            // fields. Only the text path, taken when a model reports
            // `supportsStructuredOutput: false`, still accepts an arbitrary object.
            return {
                type: 'object',
                properties: {},
                required: [],
                additionalProperties: false,
            };
        case 'enum':
            return { type: 'string', enum: [...enumMembersOrThrow(resolved)] };
        case 'string':
        default:
            return { type: 'string' };
    }
}

/**
 * Build a JSON Schema for a signature's outputs, for provider structured-output
 * modes (OpenAI `response_format`, Gemini `responseSchema`, Anthropic
 * `output_config.format`).
 *
 * This is hand-built rather than derived from the zod schema via `z.toJSONSchema`
 * because the zod schemas carry preprocessing steps that have no JSON Schema
 * representation — and because OpenAI's strict mode has requirements a generic
 * conversion will not satisfy: every property must appear in `required`, and
 * `additionalProperties` must be `false`. Optional fields are expressed as
 * nullable instead of being omitted from `required`.
 */
export function buildOutputJsonSchema(signature: typeof Signature | string): OutputJsonSchema {
    const fields = getOutputFieldConfigs(signature);
    const properties: Record<string, JsonSchemaProperty> = {};

    for (const [name, config] of Object.entries(fields)) {
        const base = jsonSchemaTypeFor(resolveFieldType(config));
        const property: JsonSchemaProperty = { ...base };

        if (config.description) {
            property.description = config.description;
        }
        if (config.required === false && typeof base.type === 'string') {
            // The array form of `type` is what OpenAI strict mode documents for
            // nullable fields, Anthropic takes plain JSON Schema, and Gemini's
            // `responseJsonSchema` documents it too ("to allow a property to be
            // null, include 'null' in the type array"). Gemini's older
            // `responseSchema` keyword models nullability with a `nullable`
            // boolean instead, so a provider that switches back to it would need
            // to translate this.
            property.type = [base.type, 'null'];
            if (Array.isArray(property.enum)) {
                // `enum` still constrains the value when the type union allows
                // null, so null has to be a member or the field can never be null.
                property.enum = [...property.enum, null];
            }
        }
        properties[name] = property;
    }

    return {
        type: 'object',
        properties,
        required: Object.keys(properties),
        additionalProperties: false,
    };
}
