---
'@ts-dspy/core': minor
---

Emit strict-mode-correct JSON Schema, and add an `enum` field type.

`buildOutputJsonSchema` used to describe an `object`/`json` field as
`{ type: 'object', additionalProperties: true }` and a bare `array`/`list` field
as `{ type: 'array', items: {} }`. OpenAI's strict structured output rejects
both: it requires `additionalProperties: false` on every object in the document,
nested ones included, and will not accept an empty `items` schema. Any signature
with such a field was therefore refused by the API on the provider path. Objects
now emit a closed, empty object and arrays declare `items: { type: 'string' }`.

Two consequences worth knowing, both documented on the site. One schema is built
per signature and handed to whichever provider is configured, so these shapes
land everywhere, not only on OpenAI. Strict mode cannot express a free-form
object at all, which means a bare `object` field is now pinned to `{}` on every
provider with structured output — declare the keys you want as their own
signature fields instead. And a bare `array` now tells the provider its elements
are strings, so a list of figures arrives as `['1', '2']`; declare `number[]`
when the elements have a type worth naming. Only the text path, taken when a
model reports `supportsStructuredOutput: false`, is unchanged.

The new `enum` field type pins an output to a closed set, so the model cannot
invent a fourth value that still passes validation. Declare members with
`@OutputField({ type: 'enum', values: ['positive', 'negative', 'neutral'] })`,
or inline in a string signature as `sentiment: enum(positive|negative|neutral)`
— pipe-separated, because commas already separate fields. Matching trims and
ignores case, in the same lenient spirit as the other coercions, and returns the
declared spelling; anything outside the set is a `ValidationError` that names the
members. The set is emitted into the provider schema as `enum`, so it constrains
decoding rather than only the check afterwards, and it is named in the prompt on
the text path, where nothing else could carry it. An optional enum admits `null`
into its member list so `type: [base, 'null']` and `enum` do not contradict each
other. An enum with no members, or a malformed inline declaration such as
`enum(a|b`, throws rather than degrading to an unconstrained string.
