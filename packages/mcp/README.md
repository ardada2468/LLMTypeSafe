# @ts-dspy/mcp

[Model Context Protocol](https://modelcontextprotocol.io) integration for
[TS-DSPy](https://github.com/ardada2468/LLMTypeSafe), built on the official
`@modelcontextprotocol/sdk`. It works in both directions: use any MCP server's
tools in a ts-dspy agent, and expose a ts-dspy program as a tool any MCP client
can call.

```bash
npm install @ts-dspy/core @ts-dspy/mcp
```

## MCP tools in ReAct

```ts
import { ReAct } from '@ts-dspy/core';
import { connectMcp, mcpTools } from '@ts-dspy/mcp';

// A local server over stdio...
const client = await connectMcp({
  command: 'npx',
  args: ['-y', '@modelcontextprotocol/server-filesystem', '.'],
});
// ...or a remote one over Streamable HTTP:
// const client = await connectMcp({ url: 'https://example.com/mcp', headers: { Authorization: `Bearer ${token}` } });

try {
  const tools = await mcpTools(client, { exclude: ['write_file'], prefix: 'fs_' });
  const agent = new ReAct('question -> answer', { tools });
  console.log((await agent.forward({ question: 'What is in README.md?' })).answer);
} finally {
  await client.close(); // stops the stdio child process
}
```

`mcpTools` converts each MCP tool into a core `Tool`, keeping its name,
description and JSON Schema unchanged. Calling one runs `tools/call`: text
content is joined, images, audio and resources are described rather than
dropped, and an `isError` result throws `McpToolError` — which `ReAct` turns
into an observation, so the model reads the server's message and can retry. Pass `timeoutMs` to change how long one tool call may take (the MCP SDK's default is 60 seconds).

`include` / `exclude` filter by the tool's name on the server; `prefix` renames
the tools ts-dspy sees (to keep two servers' `search` tools apart) while calls
still use the original name. Any connected SDK `Client` works, not only the one
`connectMcp` returns.

## Serving a program over MCP

```ts
import { ChainOfThought } from '@ts-dspy/core';
import { serveProgram } from '@ts-dspy/mcp';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';

const server = serveProgram(new ChainOfThought('question -> answer'), {
  name: 'qa',
  description: 'Answer a question, reasoning step by step',
});
await server.connect(new StdioServerTransport());
```

The tool's input schema comes from the signature's input fields — typed,
described, and optional where the field is — and a call returns the
prediction's fields as JSON text. A program that throws returns an `isError`
result carrying the message. For a custom `Module` without a signature of its
own, pass `{ signature }`.

Requires Node.js 22+. Ships ESM and CommonJS.

Full documentation: <https://github.com/ardada2468/LLMTypeSafe#readme>

## License

MIT
