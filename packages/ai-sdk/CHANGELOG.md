# @ts-dspy/ai-sdk

## 0.7.0

### Minor Changes

- df12e87: Three new integration packages:

  - **`@ts-dspy/mcp`** (Model Context Protocol, on `@modelcontextprotocol/sdk`): `mcpTools(client, { include, exclude, prefix })` turns an MCP server's tools into core `Tool`s for `ReAct` — JSON Schema passed through unchanged, text content joined, non-text content described, `isError` thrown as `McpToolError` so the model sees the server's message. `connectMcp({ command, args, env, cwd })` / `connectMcp({ url, headers })` opens a stdio or Streamable HTTP client. `serveProgram(program, { name, version, toolName, description, signature })` returns an `McpServer` exposing a program as a tool whose input schema is derived from the signature's input fields and whose result is the prediction as JSON text.
  - **`@ts-dspy/otel`** (OpenTelemetry): `OpenTelemetryCallback` creates `ts-dspy.module <Class>`, `chat <model>` and `execute_tool <name>` spans with GenAI semantic-convention attributes (`gen_ai.operation.name`, `gen_ai.provider.name`/`gen_ai.system`, `gen_ai.request.*`, `gen_ai.usage.*`, `gen_ai.response.finish_reasons`, `gen_ai.tool.name`), parented through ts-dspy's `parentCallId` so the hierarchy is correct under concurrency. Errors are recorded with `recordException`, `error.type` and an `ERROR` status. Prompt/completion capture is opt-in (`captureContent`). `instrument()` / `uninstrument()` register it globally.
  - **`@ts-dspy/ai-sdk`** (Vercel AI SDK 7): `AISDKLM` wraps any AI SDK `LanguageModel` as a core `BaseLM` — chat with image parts, `chatStream`, `chatWithTools` (tools declared without `execute`, calls returned to ts-dspy), `generateStructured` via `Output.object` + `jsonSchema()`, cached/reasoning token usage, `signal`/`timeout`/`retries`, configurable capabilities, and `APICallError`/`RetryError` mapped onto core's `LMError` subclasses.

### Patch Changes

- Updated dependencies [a026be5]
- Updated dependencies [a016219]
- Updated dependencies [fbc1ea8]
- Updated dependencies [fa3b16b]
- Updated dependencies [0b048b1]
- Updated dependencies [df12e87]
- Updated dependencies [a455632]
- Updated dependencies [d805fc8]
  - @ts-dspy/core@0.7.0
