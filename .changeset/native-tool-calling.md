---
'@ts-dspy/anthropic': minor
'@ts-dspy/gemini': minor
'@ts-dspy/openai': minor
'@ts-dspy/core': minor
---

Native tool calling, end to end.

All three providers reported `supportsFunctionCalling: true` while implementing
nothing, and `RespAct` ran ReAct purely by text prompting — regex-extracting
`Action:`/`Action Input:` from raw completions. That capped every tool at exactly
one string argument, ruled out parallel calls, and left the loop at the mercy of
the model formatting its output correctly. The flag is now honest.

`LLMCallOptions` gains `tools` and `toolChoice`, and `ILanguageModel` gains
`chatWithTools`, which returns text, tool calls, and a normalised finish reason
from one turn. `BaseLM` supplies a text-only default, so the capability flag —
not feature detection — is what callers branch on. Each provider translates the
declarations into its own request shape (OpenAI `tools`/`tool_calls`, Anthropic
`input_schema`/`tool_use`, Gemini `functionDeclarations`/`functionCall`) and
reads the calls back out.

`RespAct` uses that path whenever the model supports it and tools are declared,
and keeps the text-parsing loop as the fallback for local models and providers
without native tool calling — the same task completes either way. Tools can now
declare a JSON Schema or Zod schema for their arguments and receive a validated
object instead of a single string; bare functions and `{ description, function }`
keep working unchanged. Parallel tool calls in one turn are executed and reported
individually, and the whole `RespActEvent` surface stays meaningful on both
paths. `forceTextMode` pins a tool-capable model to the text loop.

**Breaking:** `ToolCall` is reshaped for cross-provider use. It was a copy of
OpenAI's encoding — a required `id`, a `type: 'function'` literal, and a nested
`function.arguments` JSON *string* — which no other provider can populate
faithfully. It is now `{ id?, name, arguments, rawArguments? }`, where
`arguments` is always a parsed object and `id` is optional because Gemini's
function calls have none. The dead `ChatMessage.functionCall` field is removed;
`ChatMessage` gains `toolCallId` to correlate a tool result with its call.

That correlation also fixes a silent role collapse in all three converters:
`tool` and `function` turns were downgraded to `user` text, and Anthropic could
then merge a tool result into the preceding user turn. Anthropic additionally
dropped `tool_use` blocks on the floor (`textOf` keeps only `text` blocks) and
ignored `input_json_delta` while streaming; both are now surfaced.
