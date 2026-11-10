import { createServer, type Server as HttpServer } from 'node:http';
import type { AddressInfo } from 'node:net';
import { fileURLToPath } from 'node:url';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import {
    ChainOfThought,
    InputField,
    Module,
    OutputField,
    Predict,
    Prediction,
    ReAct,
    Signature,
    signature,
} from '@ts-dspy/core';
import { MockLM } from '@ts-dspy/core/testing';
import { z } from 'zod';
import {
    McpToolError,
    connectMcp,
    inputSchemaFor,
    mcpResultToText,
    mcpTools,
    serveProgram,
} from './index';

/** Connect a real SDK client to `server` over the in-memory transport. */
async function connect(server: McpServer): Promise<Client> {
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    const client = new Client({ name: 'test-client', version: '1.0.0' });
    await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
    return client;
}

/** A small calculator server with a deliberately varied set of tools. */
function calculatorServer(): McpServer {
    const server = new McpServer({ name: 'calc', version: '1.0.0' });
    server.registerTool(
        'add',
        {
            description: 'Add two numbers',
            inputSchema: { a: z.number(), b: z.number() },
        },
        async ({ a, b }) => ({ content: [{ type: 'text', text: String(a + b) }] })
    );
    server.registerTool(
        'divide',
        {
            description: 'Divide a by b',
            inputSchema: { a: z.number(), b: z.number() },
        },
        async ({ a, b }) =>
            b === 0
                ? { isError: true, content: [{ type: 'text', text: 'division by zero' }] }
                : { content: [{ type: 'text', text: String(a / b) }] }
    );
    server.registerTool(
        'chart',
        { title: 'Render a chart', inputSchema: { label: z.string() } },
        async ({ label }) => ({
            content: [
                { type: 'text', text: `chart for ${label}` },
                { type: 'image', data: 'aGVsbG8=', mimeType: 'image/png' },
            ],
        })
    );
    return server;
}

describe('mcpTools', () => {
    it('converts every listed tool, keeping its name, description and schema', async () => {
        const client = await connect(calculatorServer());
        const tools = await mcpTools(client);

        expect(tools.map((tool) => tool.name)).toEqual(['add', 'divide', 'chart']);
        const add = tools[0];
        expect(add.description).toBe('Add two numbers');
        expect(add.parameters).toMatchObject({
            type: 'object',
            properties: { a: { type: 'number' }, b: { type: 'number' } },
            required: ['a', 'b'],
        });
        // The spec handed to a provider is the same JSON Schema.
        expect(add.toSpec().parameters).toBe(add.parameters);
        // A tool without a description falls back to its title.
        expect(tools[2].description).toBe('Render a chart');
        await client.close();
    });

    it('calls the server and returns its text', async () => {
        const client = await connect(calculatorServer());
        const [add] = await mcpTools(client, { include: ['add'] });
        expect(await add.call({ a: 2, b: 3 })).toBe('5');
        await client.close();
    });

    it('throws McpToolError when the tool reports isError', async () => {
        const client = await connect(calculatorServer());
        const [divide] = await mcpTools(client, { include: ['divide'] });
        await expect(divide.call({ a: 1, b: 0 })).rejects.toThrow(McpToolError);
        await expect(divide.call({ a: 1, b: 0 })).rejects.toThrow(/division by zero/);
        await client.close();
    });

    it('describes non-text content instead of dropping it', async () => {
        const client = await connect(calculatorServer());
        const [chart] = await mcpTools(client, { include: ['chart'] });
        expect(await chart.call({ label: 'sales' })).toBe(
            'chart for sales\n[image: image/png]'
        );
        await client.close();
    });

    it('filters with include and exclude, and prefixes names but not calls', async () => {
        const client = await connect(calculatorServer());
        const tools = await mcpTools(client, {
            include: ['add', 'divide'],
            exclude: ['divide'],
            prefix: 'calc_',
        });
        expect(tools.map((tool) => tool.name)).toEqual(['calc_add']);
        expect(await tools[0].call({ a: 1, b: 1 })).toBe('2');
        await client.close();
    });

    it('follows pagination cursors', async () => {
        const pages = [
            { tools: [{ name: 'one', inputSchema: { type: 'object' } }], nextCursor: 'p2' },
            { tools: [{ name: 'two', inputSchema: { type: 'object' } }] },
        ];
        const seen: unknown[] = [];
        const client = {
            listTools: async (params?: unknown) => {
                seen.push(params);
                return pages.shift()!;
            },
        } as unknown as Client;

        const tools = await mcpTools(client);
        expect(tools.map((tool) => tool.name)).toEqual(['one', 'two']);
        expect(tools[0].description).toBe('MCP tool: one');
        expect(seen).toEqual([undefined, { cursor: 'p2' }]);
    });
});

describe('mcpResultToText', () => {
    it('renders every content block type', () => {
        expect(
            mcpResultToText({
                content: [
                    { type: 'text', text: 'a' },
                    { type: 'audio', data: '', mimeType: 'audio/wav' },
                    { type: 'resource', resource: { uri: 'file:///x', text: 'inline' } },
                    {
                        type: 'resource',
                        resource: { uri: 'file:///y', blob: 'AA==', mimeType: 'image/png' },
                    },
                    { type: 'resource_link', uri: 'file:///z', name: 'zed' },
                    { type: 'hologram' },
                ],
            })
        ).toBe(
            [
                'a',
                '[audio: audio/wav]',
                'inline',
                '[resource: file:///y (image/png)]',
                '[resource link: file:///z (zed)]',
                '[hologram content]',
            ].join('\n')
        );
    });

    it('falls back to structuredContent, then a legacy toolResult', () => {
        expect(mcpResultToText({ content: [], structuredContent: { n: 1 } })).toBe('{"n":1}');
        expect(mcpResultToText({ toolResult: { ok: true } })).toBe('{"ok":true}');
        expect(mcpResultToText({ toolResult: 'plain' })).toBe('plain');
        expect(mcpResultToText({})).toBe('');
    });
});

describe('ReAct with MCP tools', () => {
    function step(thought: string, tool: string, args: Record<string, unknown>): string {
        return (
            `[[ ## next_thought ## ]]\n${thought}\n\n` +
            `[[ ## next_tool_name ## ]]\n${tool}\n\n` +
            `[[ ## next_tool_args ## ]]\n${JSON.stringify(args)}\n\n[[ ## completed ## ]]`
        );
    }

    it('runs the agent loop against a real MCP server', async () => {
        const client = await connect(calculatorServer());
        const lm = new MockLM({
            responses: [
                step('Add first.', 'add', { a: 19, b: 23 }),
                step('Now try dividing by zero.', 'divide', { a: 42, b: 0 }),
                step('Done.', 'finish', {}),
                '[[ ## reasoning ## ]]\n19 + 23 = 42\n\n[[ ## answer ## ]]\n42\n\n[[ ## completed ## ]]',
            ],
        });
        const agent = new ReAct('question -> answer', {
            lm,
            tools: await mcpTools(client, { exclude: ['chart'] }),
        });

        const result = await agent.forward({ question: 'What is 19 + 23?' });

        expect(result.answer).toBe('42');
        expect(result.trajectory.observation_0).toBe('42');
        // The server's own error message reaches the model as an observation.
        expect(String(result.trajectory.observation_1)).toMatch(/division by zero/);
        // Both tools were advertised to the model with their MCP descriptions.
        const firstPrompt = JSON.stringify(lm.calls[0]);
        expect(firstPrompt).toContain('Add two numbers');
        expect(firstPrompt).toContain('Divide a by b');
        await client.close();
    });
});

describe('serveProgram', () => {
    class Classify extends Signature {
        static description = 'Classify the sentiment of a review.';
    }
    // Decorators are applied by hand, as in the packaging check, so this file
    // does not depend on the experimentalDecorators setting.
    InputField({ description: 'The review text' })(Classify.prototype, 'review');
    InputField({ description: 'Star rating', type: 'int', required: false })(
        Classify.prototype,
        'stars'
    );
    InputField({ description: 'Locale', type: 'enum', values: ['en', 'fr'] })(
        Classify.prototype,
        'locale'
    );
    OutputField({ description: 'positive or negative' })(Classify.prototype, 'sentiment');

    it('derives the input schema from the signature and returns outputs as JSON', async () => {
        const lm = new MockLM({
            respond: () => '[[ ## sentiment ## ]]\npositive\n\n[[ ## completed ## ]]',
        });
        const server = serveProgram(new Predict(Classify, lm), {
            name: 'sentiment',
            version: '2.0.0',
        });
        const client = await connect(server);

        const { tools } = await client.listTools();
        expect(tools).toHaveLength(1);
        expect(tools[0].name).toBe('sentiment');
        expect(tools[0].description).toBe('Classify the sentiment of a review.');
        expect(tools[0].inputSchema).toMatchObject({
            type: 'object',
            properties: {
                review: { type: 'string', description: 'The review text' },
                stars: { type: 'integer', description: 'Star rating' },
                locale: { type: 'string', enum: ['en', 'fr'] },
            },
            required: ['review', 'locale'],
        });
        expect(client.getServerVersion()).toMatchObject({
            name: 'sentiment',
            version: '2.0.0',
        });

        const result = await client.callTool({
            name: 'sentiment',
            arguments: { review: 'Loved it', locale: 'en' },
        });
        expect(result.isError).toBeFalsy();
        expect(JSON.parse(mcpResultToText(result))).toEqual({ sentiment: 'positive' });
        expect(JSON.stringify(lm.calls[0])).toContain('Loved it');
        await client.close();
    });

    it('rejects arguments that do not match the schema before running the program', async () => {
        const lm = new MockLM({ respond: () => 'unused' });
        const client = await connect(serveProgram(new Predict(Classify, lm), { name: 's' }));
        const result = await client.callTool({
            name: 's',
            arguments: { review: 'x', locale: 'de' },
        });
        expect(result.isError).toBe(true);
        expect(lm.calls).toHaveLength(0);
        await client.close();
    });

    it('cancels the program when the client cancels the call', async () => {
        let seen: AbortSignal | undefined;
        class Slow extends Module {
            signature = signature({
                input: z.object({ question: z.string() }),
                output: z.object({ answer: z.string() }),
            });
            async forward(_inputs: Record<string, any>, options?: { signal?: AbortSignal }) {
                seen = options?.signal;
                await new Promise((_, reject) =>
                    options?.signal?.addEventListener('abort', () =>
                        reject(new Error('stopped'))
                    )
                );
                return new Prediction({ answer: 'never' });
            }
        }
        const client = await connect(serveProgram(new Slow(), { name: 'slow' }));
        const controller = new AbortController();
        const call = client.callTool(
            { name: 'slow', arguments: { question: 'q' } },
            undefined,
            { signal: controller.signal }
        );
        await new Promise((resolve) => setTimeout(resolve, 30));
        controller.abort();

        await expect(call).rejects.toThrow();
        await new Promise((resolve) => setTimeout(resolve, 30));
        expect(seen?.aborted).toBe(true);
        await client.close();
    });

    it('reports a failing program as an isError result', async () => {
        const lm = new MockLM({ respond: () => 'this is not in the expected format' });
        const client = await connect(
            serveProgram(new Predict('question -> answer: int', lm), {
                name: 'qa',
                toolName: 'ask',
                description: 'Answer a question',
            })
        );
        const { tools } = await client.listTools();
        expect(tools[0]).toMatchObject({ name: 'ask', description: 'Answer a question' });

        const result = await client.callTool({ name: 'ask', arguments: { question: 'hi' } });
        expect(result.isError).toBe(true);
        expect(mcpResultToText(result).length).toBeGreaterThan(0);
        await client.close();
    });

    it('accepts zod signatures and custom modules with an explicit signature', async () => {
        const qa = signature({
            input: z.object({ question: z.string().describe('The question') }),
            output: z.object({ answer: z.string() }),
        });
        class Echo extends Module {
            async forward(inputs: Record<string, any>) {
                return new Prediction({ answer: `echo: ${inputs.question}` });
            }
        }

        expect(() => serveProgram(new Echo(), { name: 'echo' })).toThrow(/no signature/);

        const client = await connect(serveProgram(new Echo(), { name: 'echo', signature: qa }));
        const { tools } = await client.listTools();
        expect(tools[0].inputSchema).toMatchObject({
            properties: { question: { type: 'string', description: 'The question' } },
        });
        const result = await client.callTool({ name: 'echo', arguments: { question: 'hi' } });
        expect(JSON.parse(mcpResultToText(result))).toEqual({ answer: 'echo: hi' });
        await client.close();
    });

    it('round-trips: a served program used as a tool through mcpTools', async () => {
        const lm = new MockLM({
            respond: () =>
                '[[ ## reasoning ## ]]\nsimple\n\n[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]',
        });
        const client = await connect(
            serveProgram(new ChainOfThought('question -> answer', lm), { name: 'qa' })
        );
        const [tool] = await mcpTools(client);
        expect(JSON.parse(await tool.call({ question: 'Capital of France?' }))).toEqual({
            reasoning: 'simple',
            answer: 'Paris',
        });
        await client.close();
    });

    it('maps every field type to a plain JSON Schema type', () => {
        const schema = z.toJSONSchema(
            inputSchemaFor(
                'a: float, b: bool, c: string[], d: number[], e: list, f: json, g: image, h: history, i: enum(x|y) -> out'
            )
        ) as { properties: Record<string, Record<string, unknown>> };
        expect(schema.properties.a).toMatchObject({ type: 'number' });
        expect(schema.properties.b).toMatchObject({ type: 'boolean' });
        expect(schema.properties.c).toMatchObject({ type: 'array', items: { type: 'string' } });
        expect(schema.properties.d).toMatchObject({ type: 'array', items: { type: 'number' } });
        expect(schema.properties.e).toMatchObject({ type: 'array' });
        expect(schema.properties.f).toMatchObject({ type: 'object' });
        expect(schema.properties.g).toMatchObject({ type: 'string' });
        expect(schema.properties.h).toMatchObject({ type: 'array' });
        expect(schema.properties.i).toMatchObject({ enum: ['x', 'y'] });
    });
});

describe('connectMcp', () => {
    let http: HttpServer | undefined;

    afterEach(async () => {
        await new Promise<void>((resolve) => (http ? http.close(() => resolve()) : resolve()));
        http = undefined;
    });

    it('connects over Streamable HTTP and sends the configured headers', async () => {
        const seenAuth: Array<string | undefined> = [];
        http = createServer(async (req, res) => {
            seenAuth.push(req.headers.authorization);
            // Stateless mode: a fresh server and transport per request.
            const server = calculatorServer();
            const transport = new StreamableHTTPServerTransport({
                sessionIdGenerator: undefined,
            });
            res.on('close', () => {
                void transport.close();
                void server.close();
            });
            await server.connect(transport);
            await transport.handleRequest(req, res);
        });
        await new Promise<void>((resolve) => http!.listen(0, '127.0.0.1', resolve));
        const { port } = http.address() as AddressInfo;

        const client = await connectMcp(
            { url: `http://127.0.0.1:${port}/mcp`, headers: { Authorization: 'Bearer test' } },
            { name: 'http-test' }
        );
        const [add] = await mcpTools(client, { include: ['add'] });
        expect(await add.call({ a: 4, b: 5 })).toBe('9');
        expect(seenAuth.length).toBeGreaterThan(0);
        expect(seenAuth.every((value) => value === 'Bearer test')).toBe(true);
        await client.close();
    });

    it('launches a stdio server as a child process', async () => {
        const fixture = fileURLToPath(new URL('./fixtures/stdio-server.mjs', import.meta.url));
        const client = await connectMcp({
            command: process.execPath,
            args: [fixture],
            env: { GREETING: 'howdy' },
        });
        try {
            const [greet] = await mcpTools(client);
            expect(greet.name).toBe('greet');
            expect(await greet.call({ name: 'Ada' })).toBe('howdy, Ada');
        } finally {
            await client.close();
        }
    });
});
