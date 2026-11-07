/**
 * MCP tools in a ReAct agent.
 *
 * `connectMcp` starts (or connects to) a Model Context Protocol server, and
 * `mcpTools` turns the tools it lists into ts-dspy `Tool`s. By default this
 * runs the reference filesystem server over stdio, read-only, on this
 * examples directory (it is fetched with `npx` on first use).
 *
 *   export OPENAI_API_KEY="sk-..."
 *   npm run example:mcp
 *
 * Point it at a remote server instead with MCP_URL (Streamable HTTP):
 *
 *   MCP_URL="https://example.com/mcp" npm run example:mcp
 */
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { ReAct, configure } from '@ts-dspy/core';
import { OpenAILM } from '@ts-dspy/openai';
import { connectMcp, mcpTools } from '@ts-dspy/mcp';
import { requireEnv, section } from './utils';

async function main(): Promise<void> {
    configure({ lm: new OpenAILM({ apiKey: requireEnv('OPENAI_API_KEY') }) });

    const examplesDir = dirname(fileURLToPath(import.meta.url));
    const url = process.env.MCP_URL;

    section('Connect');
    const client = url
        ? await connectMcp({ url })
        : await connectMcp({
              command: 'npx',
              args: ['-y', '@modelcontextprotocol/server-filesystem', examplesDir],
          });

    try {
        // Keep the agent read-only, and prefix the names so tools from two
        // servers could never collide. Calls still use the server's own names.
        const tools = await mcpTools(client, {
            exclude: ['write_file', 'edit_file', 'create_directory', 'move_file'],
            prefix: url ? '' : 'fs_',
        });
        console.log(`tools: ${tools.map((tool) => tool.name).join(', ')}`);

        section('ReAct');
        const agent = new ReAct('question -> answer', {
            tools,
            maxSteps: 6,
            onEvent: (event) => {
                if (event.type === 'tool_call') {
                    console.log(`  [step ${event.step}] ${event.tool}(${event.input})`);
                } else if (event.type === 'tool_error') {
                    console.log(
                        `  [step ${event.step}] ${event.tool} failed: ${String(event.error)}`
                    );
                }
            },
        });

        const result = await agent.forward({
            question: url
                ? 'What tools do you have, and what can you tell me with them?'
                : `Which example files in ${examplesDir} use the ReAct module? List their names.`,
        });

        console.log(`\nanswer: ${result.answer}`);
        console.log(`steps:  ${result.steps}`);
    } finally {
        // Closes the transport; for stdio this also stops the child process.
        await client.close();
    }
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});
