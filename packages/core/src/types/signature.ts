export interface FieldConfig {
    description: string;
    prefix?: string;
    type?: string;
    required?: boolean;
}

export interface ISignature {
    inputFields: Record<string, FieldConfig>;
    outputFields: Record<string, FieldConfig>;
    instructions?: string;
    description?: string;
}

export interface ParsedSignature {
    inputs: string[];
    outputs: string[];
    types: Record<string, string>;
}

// New Utility Types Start

export type FieldTypeMapping<FieldTypeStr extends string | undefined> =
    FieldTypeStr extends 'string'
        ? string
        : FieldTypeStr extends 'number'
          ? number
          : FieldTypeStr extends 'float'
            ? number
            : FieldTypeStr extends 'int'
              ? number
              : FieldTypeStr extends 'integer'
                ? number
                : FieldTypeStr extends 'boolean'
                  ? boolean
                  : FieldTypeStr extends 'bool'
                    ? boolean
                    : FieldTypeStr extends 'string[]'
                      ? string[]
                      : FieldTypeStr extends 'number[]'
                        ? number[]
                        : FieldTypeStr extends 'array'
                          ? unknown[]
                          : FieldTypeStr extends 'list'
                            ? unknown[]
                            : FieldTypeStr extends 'object'
                              ? Record<string, any>
                              : FieldTypeStr extends 'json'
                                ? Record<string, any>
                                : string; // Default to string for unknown or undefined types

// Type-only import: erased at compile time, so it introduces no runtime cycle
// with core/signature.ts, which imports the types in this file.
import type { Signature } from '../core/signature';

// Helper to get the FieldConfig record from a Signature class's static getOutputFields method
export type GetOutputFieldsReturnType<S extends typeof Signature> = ReturnType<
    S['getOutputFields']
>;

/**
 * Derives the output shape (e.g. `{ answer: string, score: number }`) from a
 * signature whose field configs are known as literal types.
 *
 * Decorators record fields on a static at runtime, so `getOutputFields()` is
 * declared as `Record<string, FieldConfig>` and TypeScript sees no per-field
 * literal types. In that case this resolves to `Record<string, any>` rather than
 * mapping every field to `string`, which is what it used to do — that claimed a
 * field declared `type: 'number'` was a `string`, the exact class of lie the
 * runtime validation exists to prevent.
 *
 * To get precise output types, pass the shape explicitly:
 *
 * ```ts
 * type QAOutput = { answer: string; confidence: number };
 * const qa = new Predict<typeof AnswerQuestion, QAOutput>(AnswerQuestion);
 * ```
 *
 * Runtime validation is enforced from the signature either way.
 */
export type SignatureOutput<S extends typeof Signature> = S extends {
    getOutputFields: () => infer OFs;
}
    ? string extends keyof OFs
        ? // Index signature: no literal field information survives, so don't invent any.
          Record<string, any>
        : OFs extends Record<string, FieldConfig>
          ? {
                -readonly [
                    K in keyof OFs as OFs[K]['required'] extends false ? never : K
                ]: FieldTypeMapping<OFs[K]['type']>;
            } & {
                -readonly [
                    K in keyof OFs as OFs[K]['required'] extends false ? K : never
                ]?: FieldTypeMapping<OFs[K]['type']>;
            }
          : Record<string, any>
    : Record<string, any>;

// New Utility Types End
