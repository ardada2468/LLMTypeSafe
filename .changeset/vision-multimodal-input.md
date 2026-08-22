---
'@ts-dspy/anthropic': minor
'@ts-dspy/gemini': minor
'@ts-dspy/openai': minor
'@ts-dspy/core': minor
---

Send images, not just text. `ChatMessage.content` is widened from `string` to
`string | ContentPart[]`, where a `ContentPart` is either text or an image
carried as an `https://` URL, a `data:` URI, or base64 plus a media type. Plain
strings remain valid content and behave exactly as before, so text-only code —
`generate()`, `generateStructured()`, and every module — is untouched.

Signature inputs can now be declared as images with `@ImageField` (or the
`image` type in a string signature), and the new `buildPromptContent()` renders
such a signature as content parts, returning a plain string when every input is
text. `buildPrompt()` still returns a string, rendering an image input as an
`[image: image/png]` placeholder.

Each provider maps parts onto its own SDK shape: OpenAI `image_url` parts (only
on user turns, since system and assistant messages accept text alone),
Anthropic `image` blocks with a base64 or URL source, and Gemini `inlineData` or
`fileData`. Anthropic's merging of consecutive same-role turns now concatenates
block arrays rather than strings; it previously merged only when both turns were
strings, which silently skipped the merge for image turns and produced two
adjacent user messages that the Messages API rejects.

Widening `ChatMessage.content` is a breaking change to a public type — code
that treats it as a `string` without narrowing will need a narrowing step. Per
the pre-1.0 convention this ships as a minor.

`supportsVision` is reported per model rather than hardcoded to `true`: false
for `gpt-3.5`, `o1-mini` and `o3-mini`, for `claude-3-5-haiku` and older Claude
models, and for Gemini embedding models.
