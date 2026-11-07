/* global process */
// A minimal MCP server over stdio, launched as a child process by the
// connectMcp test. It reads GREETING from its environment to prove that the
// `env` option reaches the server.
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';

const server = new McpServer({ name: 'greeter', version: '1.0.0' });
server.registerTool(
    'greet',
    { description: 'Greet someone', inputSchema: { name: z.string() } },
    async ({ name }) => ({
        content: [{ type: 'text', text: `${process.env.GREETING ?? 'hello'}, ${name}` }],
    })
);
await server.connect(new StdioServerTransport());
