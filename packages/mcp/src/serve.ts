import {
    History,
    TsDspyError,
    VERSION,
    getInputFieldConfigs,
    getSignatureInstructions,
    isHistoryFieldType,
    isImageFieldType,
    isZodSignature,
    type FieldConfig,
    type Module,
    type SignatureLike,
} from '@ts-dspy/core';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import type { CallToolResult } from '@modelcontextprotocol/sdk/types.js';
import { z } from 'zod';

export interface ServeProgramOptions {
    /** The server's name, reported to clients during the handshake. */
    name: string;
    /** The server's version. Defaults to the installed `@ts-dspy/core` version. */
    version?: string;
    /**
     * The name clients call the program by. Defaults to `name`. MCP tool names
     * should match `[A-Za-z0-9_.-]{1,128}`.
     */
    toolName?: string;
    /**
     * What the tool does, as a client's model will read it. Defaults to the
     * signature's instructions, which is usually exactly that.
     */
    description?: string;
    /**
     * The signature whose input fields become the tool's arguments. Defaults
     * to the program's own (`Predict`, `ChainOfThought` and `ReAct` all have
     * one); required for a custom `Module` that composes others and declares
     * none itself.
     */
    signature?: SignatureLike;
}

/**
 * Expose a ts-dspy program as a tool on a new MCP server, so Claude Desktop,
 * an IDE, or any other MCP client can call it.
 *
 * ```ts
 * const server = serveProgram(new ChainOfThought('question -> answer'), {
 *     name: 'qa',
 *     description: 'Answer a question with reasoning',
 * });
 * await server.connect(new StdioServerTransport());
 * ```
 *
 * The tool's input schema is derived from the signature's input fields — one
 * argument per field, typed, described, and optional when the field is — so
 * the client's model is told exactly what to send. A call runs
 * `program.forward(arguments)` and returns the prediction's fields as JSON
 * text. A program that throws produces an `isError` result carrying the
 * message rather than a protocol error, which is how MCP expects a tool to
 * report its own failures, so the calling model can see what went wrong.
 *
 * The server is returned unconnected: pick a transport (`StdioServerTransport`,
 * `StreamableHTTPServerTransport`, …) and `connect` it. Register more tools,
 * resources or prompts on it first if you like — it is the SDK's own
 * `McpServer`.
 */
export function serveProgram(program: Module, options: ServeProgramOptions): McpServer {
    const signature =
        options.signature ?? (program as unknown as { signature?: SignatureLike }).signature;
    if (signature === undefined) {
        throw new TsDspyError(
            `serveProgram cannot derive arguments for ${program.constructor.name}: it has no ` +
                'signature of its own. Pass { signature } naming its input fields.'
        );
    }

    const server = new McpServer({ name: options.name, version: options.version ?? VERSION });
    const inputSchema = inputSchemaFor(signature);
    const historyFields = isZodSignature(signature) ? [] : historyFieldsOf(signature);

    server.registerTool(
        options.toolName ?? options.name,
        {
            description:
                options.description ??
                getSignatureInstructions(signature) ??
                `Run the ${program.constructor.name} program.`,
            inputSchema,
        },
        async (
            args: Record<string, unknown>,
            extra?: { signal?: AbortSignal }
        ): Promise<CallToolResult> => {
            try {
                // A client that cancels or times out must stop the program too,
                // or an abandoned request keeps spending tokens to the end.
                const prediction = await program.forward(
                    reviveInputs(args, historyFields),
                    extra?.signal ? { signal: extra.signal } : undefined
                );
                return {
                    content: [{ type: 'text', text: JSON.stringify(prediction.toObject()) }],
                };
            } catch (error) {
                return {
                    isError: true,
                    content: [
                        {
                            type: 'text',
                            text: error instanceof Error ? error.message : String(error),
                        },
                    ],
                };
            }
        }
    );

    return server;
}

/**
 * A Zod object with one property per input field.
 *
 * A zod signature's own input object is used as is: it is the caller's schema
 * and carries no preprocessing. For decorator and string signatures the
 * schema is built here rather than with core's `fieldConfigToZod`, whose
 * validators wrap every non-string type in a coercing `z.preprocess`. That is
 * right for parsing a model's output, but a preprocess step has no JSON
 * Schema form, so the client would be told each argument accepts anything.
 * Clients send JSON, so plain types are both accurate and sufficient.
 */
export function inputSchemaFor(signature: SignatureLike): z.ZodObject {
    if (isZodSignature(signature)) {
        return signature.input as unknown as z.ZodObject;
    }
    const shape: Record<string, z.ZodType> = {};
    for (const [name, config] of Object.entries(getInputFieldConfigs(signature))) {
        let schema = jsonTypeFor(config);
        if (config.description) schema = schema.describe(config.description);
        shape[name] = config.required === false ? schema.optional() : schema;
    }
    return z.object(shape);
}

function jsonTypeFor(config: FieldConfig): z.ZodType {
    const raw = (config.type ?? 'string').trim();
    if (isImageFieldType(raw)) {
        return z.string().describe('An https:// image URL or a data: URI');
    }
    if (isHistoryFieldType(raw)) {
        // Earlier turns, each an object of the signature's field values.
        return z.array(z.record(z.string(), z.unknown()));
    }
    const inlineEnum = /^enum\s*\((.*)\)$/is.exec(raw);
    if (inlineEnum) {
        return enumOf(
            inlineEnum[1]
                .split('|')
                .map((value) => value.trim())
                .filter(Boolean)
        );
    }

    switch (raw.toLowerCase()) {
        case 'number':
        case 'float':
            return z.number();
        case 'int':
        case 'integer':
            return z.number().int();
        case 'boolean':
        case 'bool':
            return z.boolean();
        case 'string[]':
            return z.array(z.string());
        case 'number[]':
            return z.array(z.number());
        case 'array':
        case 'list':
            return z.array(z.unknown());
        case 'object':
        case 'json':
            return z.record(z.string(), z.unknown());
        case 'enum':
            return enumOf(config.values ?? []);
        default:
            return z.string();
    }
}

function enumOf(values: string[]): z.ZodType {
    // An empty enum would reject every call; core rejects it at prediction
    // time with a clearer message, so accept any string here.
    return values.length > 0 ? z.enum(values as [string, ...string[]]) : z.string();
}

function historyFieldsOf(signature: SignatureLike): string[] {
    return Object.entries(getInputFieldConfigs(signature))
        .filter(([, config]) => isHistoryFieldType(config.type?.trim()))
        .map(([name]) => name);
}

/** JSON has no `History`; rebuild one from the array of turns a client sent. */
function reviveInputs(
    args: Record<string, unknown>,
    historyFields: string[]
): Record<string, unknown> {
    if (historyFields.length === 0) return args;
    const inputs = { ...args };
    for (const name of historyFields) {
        const value = inputs[name];
        if (Array.isArray(value)) inputs[name] = new History(value);
    }
    return inputs;
}
