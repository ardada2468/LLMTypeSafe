import { VERSION } from '@ts-dspy/core';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';

/** Launch a local MCP server as a child process and talk to it over stdio. */
export interface McpStdioOptions {
    /** Executable to run, e.g. `'npx'` or `'uvx'`. */
    command: string;
    args?: string[];
    /**
     * Extra environment variables for the server process. Merged over the SDK's
     * safe default environment (`PATH`, `HOME`, …) rather than over the full
     * `process.env`, so the parent's secrets are not handed to every server.
     */
    env?: Record<string, string>;
    /** Working directory for the server process. */
    cwd?: string;
}

/** Connect to a remote MCP server over Streamable HTTP. */
export interface McpHttpOptions {
    /** The server's MCP endpoint, e.g. `https://example.com/mcp`. */
    url: string | URL;
    /** Sent with every request — typically `Authorization`. */
    headers?: Record<string, string>;
}

export type McpConnectOptions = McpStdioOptions | McpHttpOptions;

/** How the client introduces itself during the MCP handshake. */
export interface McpClientInfo {
    /** Defaults to `ts-dspy`. */
    name?: string;
    /** Defaults to the installed `@ts-dspy/core` version. */
    version?: string;
}

/**
 * Open a connected MCP {@link Client}, over stdio for a local server or
 * Streamable HTTP for a remote one, depending on whether `command` or `url`
 * is given.
 *
 * ```ts
 * const local = await connectMcp({ command: 'npx', args: ['-y', '@modelcontextprotocol/server-filesystem', '.'] });
 * const remote = await connectMcp({ url: 'https://example.com/mcp', headers: { Authorization: `Bearer ${token}` } });
 * try {
 *     const tools = await mcpTools(local);
 * } finally {
 *     await local.close();
 * }
 * ```
 *
 * The returned client is the SDK's own, so anything this package does not
 * wrap — resources, prompts, sampling — is still available on it. Close it
 * when done: for stdio that terminates the child process, which otherwise
 * keeps Node alive.
 *
 * Streamable HTTP only; the older HTTP+SSE transport is deprecated by the
 * protocol. Build a `Client` with the SDK's `SSEClientTransport` yourself if a
 * server still needs it — {@link mcpTools} accepts any connected client.
 */
export async function connectMcp(
    options: McpConnectOptions,
    info: McpClientInfo = {}
): Promise<Client> {
    const client = new Client({
        name: info.name ?? 'ts-dspy',
        version: info.version ?? VERSION,
    });

    const transport =
        'command' in options
            ? new StdioClientTransport({
                  command: options.command,
                  args: options.args,
                  env: options.env,
                  cwd: options.cwd,
              })
            : new StreamableHTTPClientTransport(
                  typeof options.url === 'string' ? new URL(options.url) : options.url,
                  options.headers ? { requestInit: { headers: options.headers } } : undefined
              );

    await client.connect(transport);
    return client;
}
