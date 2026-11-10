import { Tool, TsDspyError } from '@ts-dspy/core';
import type { Client } from '@modelcontextprotocol/sdk/client/index.js';
import type {
    ContentBlock,
    Tool as McpToolDefinition,
} from '@modelcontextprotocol/sdk/types.js';

/** Which of a server's tools {@link mcpTools} converts, and under what names. */
export interface McpToolsOptions {
    /**
     * Only these tools, by their name on the server. Applied before `exclude`.
     * Omit to take every tool the server lists.
     */
    include?: string[];
    /** Skip these tools, by their name on the server. */
    exclude?: string[];
    /**
     * Prepended verbatim to every tool's name, e.g. `'github_'`. Two servers
     * that both expose a `search` tool would otherwise collide in one ReAct
     * agent, whose tools are keyed by name. Calls still reach the server under
     * the tool's original name.
     */
    prefix?: string;
    /**
     * Longest one tool call may take, in milliseconds. Defaults to the MCP
     * SDK's own limit (60 seconds); raise it for slow tools, lower it to keep
     * an agent responsive.
     */
    timeoutMs?: number;
}

/** A tool backed by an MCP server: takes the JSON arguments, resolves to the result's text. */
export type McpTool = Tool<Record<string, unknown>, string>;

/**
 * Thrown when an MCP tool reports `isError: true`.
 *
 * MCP reports a tool's own failure — a bad argument, a missing record — as a
 * successful protocol response carrying `isError`, so that the model can read
 * the message and try again. Throwing it here is what makes that work in
 * ts-dspy: `ReAct` turns a throwing tool into an `Error executing …`
 * observation, so the model sees exactly the message the server wrote.
 */
export class McpToolError extends TsDspyError {
    /** The tool's name on the server. */
    readonly toolName: string;

    constructor(toolName: string, message: string) {
        super(`MCP tool ${toolName} failed: ${message}`);
        this.toolName = toolName;
    }
}

/**
 * Convert the tools an MCP server exposes into ts-dspy {@link Tool}s, ready to
 * hand to `ReAct` or to declare in `LLMCallOptions.tools`.
 *
 * ```ts
 * const client = await connectMcp({ command: 'npx', args: ['-y', '@modelcontextprotocol/server-everything'] });
 * const agent = new ReAct('question -> answer', { tools: await mcpTools(client) });
 * ```
 *
 * Each tool keeps the server's name, description and JSON Schema for its
 * arguments unchanged — MCP and every provider speak JSON Schema already, so
 * no translation can lose information. Calling one runs `tools/call` on the
 * client; the reply's content is flattened to text (see
 * {@link mcpResultToText}), and an `isError` reply throws {@link McpToolError}.
 *
 * The tool list is read once, here. A server that changes its tools later
 * (`notifications/tools/list_changed`) needs another call to pick them up.
 */
export async function mcpTools(
    client: Client,
    options: McpToolsOptions = {}
): Promise<McpTool[]> {
    const definitions = await listAllTools(client);
    const include = options.include ? new Set(options.include) : undefined;
    const exclude = new Set(options.exclude ?? []);
    const prefix = options.prefix ?? '';

    return definitions
        .filter((tool) => (!include || include.has(tool.name)) && !exclude.has(tool.name))
        .map((definition) => toTool(client, definition, prefix, options.timeoutMs));
}

/** `tools/list` is paginated; follow the cursor until the server stops sending one. */
async function listAllTools(client: Client): Promise<McpToolDefinition[]> {
    const tools: McpToolDefinition[] = [];
    let cursor: string | undefined;
    do {
        const page = await client.listTools(cursor ? { cursor } : undefined);
        tools.push(...page.tools);
        cursor = page.nextCursor;
    } while (cursor);
    return tools;
}

function toTool(
    client: Client,
    definition: McpToolDefinition,
    prefix: string,
    timeoutMs?: number
): McpTool {
    const serverName = definition.name;
    return new Tool<Record<string, unknown>, string>({
        name: `${prefix}${serverName}`,
        // MCP's `description` is optional, but a model choosing between tools
        // needs something; the title is the next best thing the server offers.
        description: definition.description ?? definition.title ?? `MCP tool: ${serverName}`,
        // Passed through as JSON Schema, not converted to Zod: the server
        // validates its own arguments, and a second, lossy validation here
        // could only reject calls the server would have accepted.
        parameters: definition.inputSchema as Record<string, unknown>,
        fn: async (args) => {
            const result = await client.callTool(
                {
                    name: serverName,
                    arguments: isRecord(args) ? args : {},
                },
                undefined,
                timeoutMs !== undefined ? { timeout: timeoutMs } : undefined
            );
            const text = mcpResultToText(result);
            if (result.isError === true) {
                throw new McpToolError(serverName, text || 'the tool reported an error');
            }
            return text;
        },
    });
}

/**
 * Flatten a `tools/call` result to the text a model reads as an observation.
 *
 * Text blocks are joined with newlines. Everything else is described rather
 * than dropped, so the model at least knows something came back: an image or
 * audio clip by its media type, an embedded resource by its text when it has
 * some and its URI otherwise, a resource link by its URI. A result with no
 * content but `structuredContent` falls back to that object as JSON, and a
 * pre-2024-11 server's `toolResult` is serialised the same way.
 */
export function mcpResultToText(result: Record<string, unknown>): string {
    const content = Array.isArray(result.content) ? (result.content as ContentBlock[]) : [];
    if (content.length > 0) {
        return content.map(describeBlock).join('\n');
    }
    if (result.structuredContent !== undefined) {
        return JSON.stringify(result.structuredContent);
    }
    if ('toolResult' in result) {
        const legacy = result.toolResult;
        return typeof legacy === 'string' ? legacy : JSON.stringify(legacy);
    }
    return '';
}

function describeBlock(block: ContentBlock): string {
    switch (block.type) {
        case 'text':
            return block.text;
        case 'image':
            return `[image: ${block.mimeType}]`;
        case 'audio':
            return `[audio: ${block.mimeType}]`;
        case 'resource': {
            const resource = block.resource;
            if ('text' in resource && typeof resource.text === 'string') return resource.text;
            return `[resource: ${resource.uri}${resource.mimeType ? ` (${resource.mimeType})` : ''}]`;
        }
        case 'resource_link':
            return `[resource link: ${block.uri}${block.name ? ` (${block.name})` : ''}]`;
        default:
            // A content type newer than this package: say so instead of
            // silently dropping it.
            return `[${(block as { type?: string }).type ?? 'unknown'} content]`;
    }
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}
