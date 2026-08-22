---
'@ts-dspy/core': minor
---

Add zod-native signatures: a third signature form, alongside decorated classes
and the string shorthand, built with the new `signature()` factory from a pair of
`z.object()` schemas.

Decorators record fields at runtime, so TypeScript could never infer per-field
types from a signature class — callers had to hand-write a `TOutput` type
argument to get real types back, and the decorator field-type list had no
spelling for an enum, a union, a nested object, or a numeric bound. A zod
signature carries its shape in the type system instead, so `Predict`,
`ChainOfThought` and `RespAct` infer the result type with no type argument, and
the input keys are typed too. It also needs no `experimentalDecorators`, which
was a real adoption barrier for projects that cannot enable it.

The caller's zod schema is used verbatim as the validator, so every constraint
they express is enforced. Text responses are still coerced leniently — `"42"`
satisfies a number field, `"a, b"` satisfies a `string[]` — with the coercion
applied at the object level so optionality, defaults and object-level
refinements survive. On the provider structured-output path the schema is
converted with `z.toJSONSchema()` and then rewritten for OpenAI strict mode:
every property listed in `required`, `additionalProperties: false` on every
object including nested ones, and optional fields expressed as
`type: [base, 'null']`.

Decorator and string signatures are unchanged.
