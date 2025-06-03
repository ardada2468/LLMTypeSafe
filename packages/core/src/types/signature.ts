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
    FieldTypeStr extends 'string' ? string :
    FieldTypeStr extends 'number' ? number :
    FieldTypeStr extends 'float' ? number :
    FieldTypeStr extends 'int' ? number :
    FieldTypeStr extends 'integer' ? number :
    FieldTypeStr extends 'boolean' ? boolean :
    FieldTypeStr extends 'bool' ? boolean :
    FieldTypeStr extends 'array' ? any[] :
    FieldTypeStr extends 'list' ? any[] :
    FieldTypeStr extends 'object' ? Record<string, any> :
    FieldTypeStr extends 'json' ? Record<string, any> :
    string; // Default to string for unknown or undefined types

// Helper to get the FieldConfig record from a Signature class's static getOutputFields method
export type GetOutputFieldsReturnType<S extends typeof import('../core/signature').Signature> = ReturnType<S['getOutputFields']>;

// Derives the output shape (e.g., { answer: string, score: number }) from a Signature class
export type SignatureOutput<S extends typeof import('../core/signature').Signature> =
    // Check if S has a callable getOutputFields method
    S extends { getOutputFields: () => infer OFs } ?
    OFs extends Record<string, FieldConfig> ? // Check if the return type is what we expect
    {
        // Required fields: Iterate over keys K of OFs (FieldConfig record)
        // Check if OFs[K] (the FieldConfig for field K) has 'required' not explicitly false
        -readonly [K in keyof OFs as OFs[K]['required'] extends false ? never : K]: FieldTypeMapping<OFs[K]['type']>;
    } &
    {
        // Optional fields: Iterate over keys K of OFs
        // Check if OFs[K] has 'required' as false
        -readonly [K in keyof OFs as OFs[K]['required'] extends false ? K : never]?: FieldTypeMapping<OFs[K]['type']>;
    }
    : Record<string, any> // Fallback if getOutputFields doesn't return Record<string, FieldConfig>
    : Record<string, any>; // Fallback if S doesn't have getOutputFields (e.g., for string signatures where TSignature is `typeof Signature`)

// New Utility Types End 