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
import type { Signature, SignatureSource, ZodSignature } from '../core/signature';
import type { z } from 'zod';

// Helper to get the FieldConfig record from a Signature class's static getOutputFields method
export type GetOutputFieldsReturnType<S extends typeof Signature> = ReturnType<
    S['getOutputFields']
>;

/**
 * Derives the output shape (e.g. `{ answer: string, score: number }`) from a
 * signature whose field configs are known as literal types.
 *
 * A zod signature short-circuits all of this: its output schema already carries
 * per-field types, so the result is simply `z.infer` of it — enums stay unions
 * of literals, optional fields stay optional, and no type argument is needed.
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
export type SignatureOutput<S extends SignatureSource> =
    S extends ZodSignature<any, infer OutputSchema>
        ? z.infer<OutputSchema>
        : S extends {
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

/**
 * Derives the input shape a signature's `forward()` accepts.
 *
 * Only zod signatures carry that information at compile time; the decorator and
 * string forms stay `Record<string, any>`, exactly as before.
 */
export type SignatureInput<S extends SignatureSource> =
    S extends ZodSignature<infer InputSchema, any> ? z.input<InputSchema> : Record<string, any>;

// New Utility Types End
