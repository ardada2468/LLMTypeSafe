import { z } from 'zod';
import {
    isZodSignature,
    Signature,
    type AnyZodSignature,
    type SignatureLike,
} from '../core/signature';
import { TsDspyError } from '../core/errors';
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
export function getOutputFieldConfigs(signature: SignatureLike): Record<string, FieldConfig> {
    if (isZodSignature(signature)) {
        return zodOutputFieldConfigs(signature);
    }
    if (typeof signature !== 'string') {
        return signature.getOutputFields();
    }

    const parsed = Signature.parseStringSignature(signature);
    const configs: Record<string, FieldConfig> = {};
    for (const name of parsed.outputs) {
        configs[name] = {
            description: `Output field: ${name}`,
            type: parsed.types[name] ?? 'string',
            required: true,
        };
    }
    return configs;
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

// --- zod signatures ---------------------------------------------------------
//
// Everything a zod signature needs — the field list, the prompt hints, the
// coercion layer and the provider JSON Schema — is derived from one place:
// `z.toJSONSchema()` of each output field. Deriving from the JSON Schema rather
// than from zod's internal class hierarchy means enums, integer formats, nested
// objects and unions are all described the same way, whatever combination of
// wrappers the caller built them from.

/** Coerce one raw field value towards the type its schema expects. */
type Coercer = (value: unknown) => unknown;

function isPlainObject(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * JSON Schema for one zod schema, never throwing.
 *
 * Used for description only — prompt hints, field labels, coercion choices — so
 * a type zod cannot represent degrades to "no information" rather than taking
 * the whole prompt down. The provider path calls `z.toJSONSchema` directly and
 * does let it throw.
 */
function describeAsJsonSchema(schema: z.ZodType): Record<string, unknown> {
    try {
        const produced = z.toJSONSchema(schema, {
            io: 'input',
            unrepresentable: 'any',
        }) as Record<string, unknown>;
        delete produced.$schema;
        return produced;
    } catch {
        return {};
    }
}

/** The non-null half of a JSON Schema `type`, which may be a union array. */
function primaryJsonType(node: Record<string, unknown>): string | undefined {
    const type = node.type;
    if (typeof type === 'string') return type;
    if (Array.isArray(type)) {
        return type.find(
            (entry): entry is string => typeof entry === 'string' && entry !== 'null'
        );
    }
    return undefined;
}

/** Short label for a JSON Schema node, in the vocabulary of `FieldConfig.type`. */
function jsonSchemaLabel(node: Record<string, unknown>): string {
    if (Array.isArray(node.enum)) return 'enum';
    if (node.const !== undefined) return 'literal';
    if (Array.isArray(node.anyOf) || Array.isArray(node.oneOf)) return 'union';

    switch (primaryJsonType(node)) {
        case 'integer':
            return 'int';
        case 'number':
            return 'number';
        case 'boolean':
            return 'boolean';
        case 'object':
            return 'object';
        case 'array': {
            const items = isPlainObject(node.items) ? jsonSchemaLabel(node.items) : 'unknown';
            return items === 'unknown' ? 'array' : `${items}[]`;
        }
        case 'string':
            return 'string';
        default:
            return 'string';
    }
}

/** True when the schema accepts a missing value, so the field is optional. */
function acceptsUndefined(schema: z.ZodType): boolean {
    return schema.safeParse(undefined).success;
}

/**
 * Human-readable constraint hint for one zod field, for text prompts.
 *
 * The structured-output path constrains decoding directly, but the text path
 * only has the prompt to go on — a model told `sentiment (one of: positive,
 * negative, neutral)` is far likelier to return a value the enum accepts.
 */
export function zodFieldHint(schema: z.ZodType): string {
    const node = describeAsJsonSchema(schema);
    const parts: string[] = [];

    if (Array.isArray(node.enum)) {
        parts.push(`one of: ${node.enum.map((value) => String(value)).join(', ')}`);
    } else if (node.const !== undefined) {
        parts.push(`exactly: ${String(node.const)}`);
    } else {
        parts.push(jsonSchemaLabel(node));
    }

    const min = typeof node.minimum === 'number' ? node.minimum : undefined;
    const max = typeof node.maximum === 'number' ? node.maximum : undefined;
    if (min !== undefined && max !== undefined) parts.push(`between ${min} and ${max}`);
    else if (min !== undefined) parts.push(`at least ${min}`);
    else if (max !== undefined) parts.push(`at most ${max}`);

    if (typeof node.minItems === 'number' && node.minItems > 0) {
        parts.push(`at least ${node.minItems} item${node.minItems === 1 ? '' : 's'}`);
    }
    if (typeof node.maxItems === 'number') {
        parts.push(`at most ${node.maxItems} item${node.maxItems === 1 ? '' : 's'}`);
    }
    if (typeof node.format === 'string') parts.push(node.format);

    if (acceptsUndefined(schema)) parts.push('optional');

    return parts.join(', ');
}

/** How a raw text value should be massaged before the zod schema sees it. */
function coercerForJsonSchema(node: Record<string, unknown>): Coercer | null {
    // An enum or literal is already a string in the source text; coercing it
    // would only risk turning a valid option into something the enum rejects.
    if (Array.isArray(node.enum) || node.const !== undefined) return null;

    switch (primaryJsonType(node)) {
        case 'integer':
        case 'number':
            return preprocessNumber;
        case 'boolean':
            return preprocessBoolean;
        case 'object':
            return preprocessObject;
        case 'array': {
            const itemType = isPlainObject(node.items)
                ? primaryJsonType(node.items)
                : undefined;
            if (itemType === 'number' || itemType === 'integer') {
                return (value) => {
                    const array = preprocessArray(value);
                    return Array.isArray(array) ? array.map(preprocessNumber) : array;
                };
            }
            if (itemType === 'boolean') {
                return (value) => {
                    const array = preprocessArray(value);
                    return Array.isArray(array) ? array.map(preprocessBoolean) : array;
                };
            }
            return preprocessArray;
        }
        case 'string':
            return null;
        default:
            // Unrepresentable types: parse the value when it looks like JSON,
            // and otherwise leave the string for zod to judge.
            return preprocessObject;
    }
}

/**
 * Coerce towards whichever branch of a union the value can satisfy.
 *
 * A union has no single target type, so guessing one is wrong: given
 * `z.union([z.number(), z.literal('unknown')])` a text response of `7` must
 * become a number while `unknown` must stay a string. Each branch's coercion is
 * tried in turn and the first result the field actually accepts wins; if none
 * does, the original value is returned so the error message quotes what the
 * model really said.
 */
function unionCoercer(field: z.ZodType, node: Record<string, unknown>): Coercer {
    const branches = (Array.isArray(node.anyOf) ? node.anyOf : node.oneOf) as unknown[];
    const branchCoercers = (Array.isArray(branches) ? branches : [])
        .filter(isPlainObject)
        .map(coercerForJsonSchema)
        .filter((coercer): coercer is Coercer => coercer !== null);

    // JSON text is worth a try for any branch that is an object or an array.
    branchCoercers.push(preprocessObject);

    return (value) => {
        if (field.safeParse(value).success) return value;

        for (const coerce of branchCoercers) {
            const candidate = coerce(value);
            if (candidate !== value && field.safeParse(candidate).success) {
                return candidate;
            }
        }
        return value;
    };
}

/** The coercion one output field needs, if any. */
function coercerForField(field: z.ZodType): Coercer | null {
    const node = describeAsJsonSchema(field);
    if (Array.isArray(node.anyOf) || Array.isArray(node.oneOf)) {
        return unionCoercer(field, node);
    }
    return coercerForJsonSchema(node);
}

/** Field configs for a zod signature's outputs, for prompts and error messages. */
function zodOutputFieldConfigs(signature: AnyZodSignature): Record<string, FieldConfig> {
    const shape = signature.output.shape as Record<string, z.ZodType>;
    const configs: Record<string, FieldConfig> = {};

    for (const [name, field] of Object.entries(shape)) {
        configs[name] = {
            description: field.description || `Output field: ${name}`,
            type: jsonSchemaLabel(describeAsJsonSchema(field)),
            required: !acceptsUndefined(field),
        };
    }
    return configs;
}

/**
 * Wrap a zod signature's output schema in the text-coercion layer.
 *
 * The caller's schema is used verbatim, so every refinement and constraint they
 * wrote is enforced. Coercion happens at the object level rather than per field
 * so that optionality, defaults and object-level refinements all survive
 * untouched — wrapping individual fields in `z.preprocess` would lose them.
 */
function buildZodOutputSchema(signature: AnyZodSignature): z.ZodType {
    const shape = signature.output.shape as Record<string, z.ZodType>;
    const coercers: Array<[string, Coercer]> = [];

    for (const [name, field] of Object.entries(shape)) {
        const coercer = coercerForField(field);
        if (coercer) coercers.push([name, coercer]);
    }
    if (coercers.length === 0) return signature.output;

    return z.preprocess((raw) => {
        if (!isPlainObject(raw)) return raw;
        const coerced: Record<string, unknown> = { ...raw };
        for (const [name, coerce] of coercers) {
            if (coerced[name] !== undefined) {
                coerced[name] = coerce(coerced[name]);
            }
        }
        return coerced;
    }, signature.output);
}

/** Peel the wrappers that do not change a schema's structural shape. */
function unwrapZodSchema(schema: z.ZodType): z.ZodType {
    let current: z.ZodType = schema;

    // Bounded: a pathological chain of wrappers must not spin forever.
    for (let depth = 0; depth < 16; depth += 1) {
        if (
            current instanceof z.ZodOptional ||
            current instanceof z.ZodNullable ||
            current instanceof z.ZodDefault ||
            current instanceof z.ZodPrefault ||
            current instanceof z.ZodNonOptional ||
            current instanceof z.ZodReadonly ||
            current instanceof z.ZodCatch ||
            current instanceof z.ZodLazy
        ) {
            current = current.unwrap() as z.ZodType;
            continue;
        }
        if (current instanceof z.ZodPipe) {
            current = current.in as z.ZodType;
            continue;
        }
        return current;
    }
    return current;
}

/**
 * Drop the nulls a provider sends for fields that are really absent.
 *
 * Strict mode has no way to say "may be omitted", so an optional field is sent
 * as nullable and comes back as `null` when the model has nothing to say. That
 * null means "absent" and has to go before validation. A field the caller
 * declared `.nullable()` is a different matter — there `null` is a legitimate
 * value the model was asked for, and deleting it would make the field
 * impossible to satisfy. Optionality is therefore read from the schema, field by
 * field, all the way down: `strictifyJsonSchema` nullifies optional properties
 * at every level of nesting, so nested objects need the same treatment.
 */
function stripOptionalNulls(schema: z.ZodType | undefined, value: unknown): unknown {
    if (schema === undefined || value === null || value === undefined) return value;

    const inner = unwrapZodSchema(schema);

    if (inner instanceof z.ZodObject && isPlainObject(value)) {
        const shape = inner.shape as Record<string, z.ZodType>;
        const cleaned: Record<string, unknown> = {};

        for (const [key, item] of Object.entries(value)) {
            const field = shape[key];
            if (item === null && field !== undefined && acceptsUndefined(field)) continue;
            cleaned[key] = stripOptionalNulls(field, item);
        }
        return cleaned;
    }

    if (inner instanceof z.ZodArray && Array.isArray(value)) {
        const element = inner.element as z.ZodType;
        return value.map((item) => stripOptionalNulls(element, item));
    }

    return value;
}

/**
 * Prepare a provider's structured response for validation against a signature.
 *
 * Both the zod path and the decorator path express optional fields as nullable
 * in the JSON Schema, so both have nulls standing in for absent values; only the
 * zod path can also declare a field where `null` is the answer.
 */
export function stripAbsentNulls(
    signature: SignatureLike,
    raw: Record<string, any> | null | undefined
): Record<string, unknown> {
    const source = raw ?? {};

    if (isZodSignature(signature)) {
        const stripped = stripOptionalNulls(signature.output, source);
        return isPlainObject(stripped) ? stripped : source;
    }

    const cleaned: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(source)) {
        if (value !== null) cleaned[key] = value;
    }
    return cleaned;
}

/** Add `null` to a JSON Schema node's accepted types. */
function makeNullable(node: Record<string, unknown>): Record<string, unknown> {
    // `const` and `$ref` pin the value down and cannot be widened in place —
    // `{ type: ['string','null'], const: 'done' }` still admits only 'done', and
    // a `$ref` carries no type at all. Both have to become a union instead.
    if (node.const !== undefined || typeof node.$ref === 'string') {
        const { description, ...rest } = node;
        const wrapped: Record<string, unknown> = { anyOf: [rest, { type: 'null' }] };
        if (description !== undefined) wrapped.description = description;
        return wrapped;
    }

    const nullable: Record<string, unknown> = { ...node };

    if (typeof nullable.type === 'string') {
        if (nullable.type !== 'null') nullable.type = [nullable.type, 'null'];
    } else if (Array.isArray(nullable.type)) {
        if (!nullable.type.includes('null')) nullable.type = [...nullable.type, 'null'];
    } else if (Array.isArray(nullable.anyOf)) {
        const alreadyNull = nullable.anyOf.some(
            (branch) => isPlainObject(branch) && branch.type === 'null'
        );
        if (!alreadyNull) nullable.anyOf = [...nullable.anyOf, { type: 'null' }];
    }

    // A nullable enum has to list null as an option too, or the enum keyword
    // contradicts the widened type.
    if (Array.isArray(nullable.enum) && !nullable.enum.includes(null)) {
        nullable.enum = [...nullable.enum, null];
    }

    return nullable;
}

const SUBSCHEMA_KEYS = ['items', 'additionalProperties', 'contains', 'not', 'propertyNames'];
const SUBSCHEMA_LIST_KEYS = ['anyOf', 'oneOf', 'allOf', 'prefixItems'];
const DEFINITION_KEYS = ['$defs', 'definitions'];

/**
 * Rewrite a JSON Schema tree into OpenAI strict mode's dialect.
 *
 * Strict mode is narrower than JSON Schema: every object must list all of its
 * properties in `required` and set `additionalProperties: false`, at every level
 * of nesting. A property zod left out of `required` is therefore added and made
 * nullable instead, which is how strict mode spells "optional".
 */
function strictifyJsonSchema(value: unknown): unknown {
    if (Array.isArray(value)) return value.map(strictifyJsonSchema);
    if (!isPlainObject(value)) return value;

    const node: Record<string, unknown> = { ...value };

    for (const key of SUBSCHEMA_KEYS) {
        if (isPlainObject(node[key])) node[key] = strictifyJsonSchema(node[key]);
    }
    for (const key of SUBSCHEMA_LIST_KEYS) {
        if (Array.isArray(node[key])) {
            node[key] = (node[key] as unknown[]).map(strictifyJsonSchema);
        }
    }
    for (const key of DEFINITION_KEYS) {
        const defs = node[key];
        if (isPlainObject(defs)) {
            node[key] = Object.fromEntries(
                Object.entries(defs).map(([name, schema]) => [
                    name,
                    strictifyJsonSchema(schema),
                ])
            );
        }
    }

    if (isPlainObject(node.properties)) {
        const required = new Set(
            Array.isArray(node.required) ? (node.required as unknown[]).map(String) : []
        );
        const properties: Record<string, unknown> = {};

        for (const [name, child] of Object.entries(node.properties)) {
            const strict = strictifyJsonSchema(child);
            properties[name] =
                required.has(name) || !isPlainObject(strict) ? strict : makeNullable(strict);
        }

        node.properties = properties;
        node.required = Object.keys(properties);
        node.additionalProperties = false;
    }

    return node;
}

/**
 * Build a JSON Schema for a zod signature's outputs.
 *
 * Unlike the decorator and string paths this *is* derived from zod — the schema
 * is the caller's own, with no preprocessing baked in, so `z.toJSONSchema` sees
 * exactly what the model must produce. `io: 'input'` matters: if the caller
 * added a `.transform()`, the model owes us the value going in, not the one
 * coming out. The result is then rewritten for OpenAI strict mode.
 */
function buildZodOutputJsonSchema(signature: AnyZodSignature): OutputJsonSchema {
    let produced: Record<string, unknown>;
    try {
        produced = z.toJSONSchema(signature.output, { io: 'input' }) as Record<string, unknown>;
    } catch (cause) {
        // Not every zod type has a JSON Schema spelling — `z.date()` is the
        // usual culprit. Left bare, this surfaces as an anonymous error from
        // deep inside `forward()`, and only on structured-output providers,
        // while the same signature works fine on text ones.
        throw new TsDspyError(
            `This signature's output cannot be converted to JSON Schema, which the ` +
                `provider's structured-output mode requires: ${
                    cause instanceof Error ? cause.message : String(cause)
                }. Express the field as a JSON-representable type — a date as ` +
                `z.iso.datetime(), for instance.`,
            { cause }
        );
    }

    delete produced.$schema;
    return strictifyJsonSchema(produced) as unknown as OutputJsonSchema;
}

/**
 * Build a zod schema for a single field.
 *
 * Coercion stays lenient — model output is text, so `"42"` should satisfy a
 * `number` field — but anything that cannot be coerced now fails loudly instead
 * of falling back to the raw string.
 */
export function fieldConfigToZod(config: FieldConfig): z.ZodType {
    const type = (config.type ?? 'string').toLowerCase();

    let schema: z.ZodType;
    switch (type) {
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

        case 'string':
        default:
            schema = z.string();
            break;
    }

    return config.required === false ? schema.optional() : schema;
}

/** Build a zod object schema validating every output field of a signature. */
export function buildOutputSchema(signature: SignatureLike): z.ZodType {
    if (isZodSignature(signature)) {
        return buildZodOutputSchema(signature);
    }

    const fields = getOutputFieldConfigs(signature);
    const shape: Record<string, z.ZodType> = {};
    for (const [name, config] of Object.entries(fields)) {
        shape[name] = fieldConfigToZod(config);
    }
    // Loose: providers may return extra keys, and dropping them silently is worse
    // than passing them through for the caller to inspect.
    return z.looseObject(shape);
}

function jsonSchemaTypeFor(type: string): JsonSchemaProperty {
    switch (type.toLowerCase()) {
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
            return { type: 'array', items: {} };
        case 'object':
        case 'json':
            return { type: 'object', additionalProperties: true };
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
 *
 * Zod signatures take the other route — see {@link buildZodOutputJsonSchema} —
 * because their schemas carry no preprocessing and are the caller's own.
 */
export function buildOutputJsonSchema(signature: SignatureLike): OutputJsonSchema {
    if (isZodSignature(signature)) {
        return buildZodOutputJsonSchema(signature);
    }

    const fields = getOutputFieldConfigs(signature);
    const properties: Record<string, JsonSchemaProperty> = {};

    for (const [name, config] of Object.entries(fields)) {
        const base = jsonSchemaTypeFor(config.type ?? 'string');
        const property: JsonSchemaProperty = { ...base };

        if (config.description) {
            property.description = config.description;
        }
        if (config.required === false && typeof base.type === 'string') {
            property.type = [base.type, 'null'];
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
