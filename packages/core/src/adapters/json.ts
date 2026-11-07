import type { ChatMessage, ILanguageModel } from '../types/language-model';
import { type SignatureLike } from '../core/signature';
import { ValidationError } from '../core/errors';
import { parseJsonResponse } from '../core/base-lm';
import { invokeLM, providerOptions } from '../core/lm-call';
import { buildOutputJsonSchema } from '../utils/schema';
import { contentToText, hasImageContent } from '../utils/content';
import { parsePartialJson } from '../utils/partial-json';
import {
    Adapter,
    ContentBuilder,
    demoTurns,
    describeFields,
    historyTurns,
    signatureLayout,
    writeInputFields,
    type AdapterCallInput,
    type AdapterFormatInput,
    type SignatureLayout,
} from './base';

/**
 * Asks for the outputs as one JSON object.
 *
 * On a model advertising `supportsStructuredOutput`, the call goes through
 * `lm.generateStructured` with the signature's JSON Schema, so the provider
 * constrains decoding rather than merely being asked for JSON. Everywhere else
 * — models without native structured output, prompts carrying images (the
 * structured methods take a string prompt), and streams — the schema travels
 * in the prompt and the reply is parsed leniently, tolerating code fences and
 * prose around the object.
 *
 * Inputs are written under `[[ ## field ## ]]` markers, as {@link ChatAdapter}
 * writes them; demos and history replies are JSON objects.
 */
export class JSONAdapter extends Adapter {
    readonly name: string = 'JSONAdapter';

    format(input: AdapterFormatInput): ChatMessage[] {
        const layout = signatureLayout(input.signature, input.instructions);
        const messages: ChatMessage[] = [
            { role: 'system', content: this.systemMessage(input.signature, layout) },
        ];

        for (const turn of [
            ...demoTurns(input.demos, layout),
            ...historyTurns(input, layout),
        ]) {
            const user = new ContentBuilder();
            writeInputFields(user, layout.inputs, turn.inputs, inputBlock);
            if (!user.empty) messages.push({ role: 'user', content: user.build() });
            if (Object.keys(turn.outputs).length > 0) {
                messages.push({ role: 'assistant', content: JSON.stringify(turn.outputs) });
            }
        }

        const user = new ContentBuilder();
        const wrote = writeInputFields(user, layout.inputs, input.inputs, inputBlock);
        const order = layout.outputs
            .map((field) => `\`${field.name}\`${field.hint ? ` (must be ${field.hint})` : ''}`)
            .join(', then ');
        user.text(
            `${wrote ? '\n\n' : ''}Respond with a JSON object in the following order of fields: ${order}.`
        );
        messages.push({ role: 'user', content: user.build() });

        return messages;
    }

    private systemMessage(signature: SignatureLike, layout: SignatureLayout): string {
        let schemaText: string;
        try {
            schemaText = JSON.stringify(buildOutputJsonSchema(signature), null, 2);
        } catch {
            // A type with no JSON Schema spelling (`z.date()`): describe the
            // keys instead. The native structured call, which needs the schema
            // itself, reports the problem properly.
            schemaText = layout.outputs
                .map((field) => `"${field.name}": ${field.type}`)
                .join('\n');
        }
        return [
            describeFields(layout),
            '',
            'Respond with JSON matching this schema. Output only the JSON object, with no surrounding prose or code fences.',
            schemaText,
            '',
            `In adhering to this structure, your objective is:\n${layout.instructions}`,
        ].join('\n');
    }

    parse(_signature: SignatureLike, completion: string): Record<string, unknown> {
        const parsed = parseJsonResponse<unknown>(completion, () => notAnObject(completion));
        if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
            throw notAnObject(completion);
        }
        return parsed as Record<string, unknown>;
    }

    parsePartial(_signature: SignatureLike, text: string): Record<string, unknown> {
        const value = parsePartialJson<unknown>(text);
        if (value === null || typeof value !== 'object' || Array.isArray(value)) return {};

        const fields: Record<string, unknown> = {};
        for (const [key, entry] of Object.entries(value as Record<string, unknown>)) {
            // Optional fields are nullable in the JSON Schema; a null is the
            // model declining to fill the field in, not a value to show.
            if (entry !== null) fields[key] = entry;
        }
        return fields;
    }

    protected repairClosing(_signature: SignatureLike): string {
        return 'Return the corrected object with every required field, using the declared type for each.';
    }

    /**
     * Native structured output when the model has it and the prompt is text;
     * a plain chat call otherwise.
     */
    protected async request(
        lm: ILanguageModel,
        messages: ChatMessage[],
        input: AdapterCallInput
    ): Promise<string> {
        const structured =
            lm.getCapabilities().supportsStructuredOutput &&
            !messages.some((message) => hasImageContent(message.content));
        if (!structured) return super.request(lm, messages, input);

        const schema = buildOutputJsonSchema(input.signature);
        const prompt = flattenMessages(messages);
        const options = providerOptions(input.options);
        const result = await invokeLM(
            {
                lm,
                method: 'generateStructured',
                traceText: prompt,
                prompt,
                options,
                span: input.span,
                moduleId: input.moduleId,
            },
            () => lm.generateStructured<unknown>(prompt, schema, options),
            (value) => JSON.stringify(value)
        );
        return JSON.stringify(result);
    }
}

function inputBlock(name: string, value: string): string {
    return `[[ ## ${name} ## ]]\n${value}`;
}

function notAnObject(raw: string): ValidationError {
    return new ValidationError(
        [
            {
                field: '(root)',
                expected: 'object',
                received: raw,
                message: 'model did not return a JSON object',
            },
        ],
        raw
    );
}

/**
 * Messages as the single prompt `generateStructured` takes. The system message
 * leads; a lone user turn follows as is; a conversation with demos, history or
 * a repair turn keeps its roles as `User:`/`Assistant:` labels.
 */
export function flattenMessages(messages: ChatMessage[]): string {
    const system = messages.filter((message) => message.role === 'system');
    const turns = messages.filter((message) => message.role !== 'system');
    const parts = system.map((message) => contentToText(message.content));
    if (turns.length === 1) {
        parts.push(contentToText(turns[0].content));
    } else {
        for (const message of turns) {
            const label = message.role === 'assistant' ? 'Assistant' : 'User';
            parts.push(`${label}:\n${contentToText(message.content)}`);
        }
    }
    return parts.join('\n\n');
}
