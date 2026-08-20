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
 */
export function buildOutputJsonSchema(signature: typeof Signature | string): OutputJsonSchema {
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
