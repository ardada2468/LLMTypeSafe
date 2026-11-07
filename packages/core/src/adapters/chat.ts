import type { ChatMessage, ILanguageModel } from '../types/language-model';
import { type SignatureLike } from '../core/signature';
import { ValidationError } from '../core/errors';
import { parseJsonResponse } from '../core/base-lm';
import { extractLabelledFields } from '../utils/parsing';
import {
    Adapter,
    ContentBuilder,
    demoTurns,
    describeFields,
    historyTurns,
    outputNames,
    pickOutputs,
    signatureLayout,
    writeInputFields,
    type AdapterCallInput,
    type AdapterField,
    type AdapterFormatInput,
    type SignatureLayout,
} from './base';
import { JSONAdapter } from './json';

export interface ChatAdapterOptions {
    /**
     * Retry once with {@link JSONAdapter} when a reply contains none of the
     * output fields at all — the model ignored the format entirely. Defaults
     * to `true`, as in DSPy. A reply that has the fields but fails validation
     * is left to `repairAttempts` instead: re-asking in JSON would not fix a
     * wrong value, and would double the cost of every repair.
     */
    jsonFallback?: boolean;
}

/**
 * DSPy's default prompt format: every field travels under a
 * `[[ ## field_name ## ]]` marker, and the reply ends with
 * `[[ ## completed ## ]]`.
 *
 * The system message lists the input and output fields with their types and
 * descriptions, shows the structure every exchange follows — with a note on
 * each typed output (enum members, numbers, JSON for objects and arrays) — and
 * closes with the task's instructions. Demos and history turns follow as
 * user/assistant pairs in the same structure, then the real user turn.
 *
 * Parsing splits the reply on the markers. A reply without any — common on
 * small models — is read as `field: value` lines (or as JSON, if it is a JSON
 * object) before giving up; one that yields no output field at all falls back
 * to a {@link JSONAdapter} call unless `jsonFallback` is off.
 */
export class ChatAdapter extends Adapter {
    readonly name: string = 'ChatAdapter';
    protected readonly jsonFallback: boolean;

    constructor(options: ChatAdapterOptions = {}) {
        super();
        this.jsonFallback = options.jsonFallback ?? true;
    }

    // ------------------------------------------------------------ format

    format(input: AdapterFormatInput): ChatMessage[] {
        const layout = signatureLayout(input.signature, input.instructions);
        const messages: ChatMessage[] = [
            { role: 'system', content: this.systemMessage(layout) },
        ];

        for (const turn of [
            ...demoTurns(input.demos, layout),
            ...historyTurns(input, layout),
        ]) {
            const user = new ContentBuilder();
            writeInputFields(user, layout.inputs, turn.inputs, (name, value) =>
                this.fieldBlock(name, value)
            );
            if (!user.empty) messages.push({ role: 'user', content: user.build() });
            if (Object.keys(turn.outputs).length > 0) {
                messages.push({
                    role: 'assistant',
                    content: this.assistantMessage(layout, turn.outputs),
                });
            }
        }

        const user = new ContentBuilder();
        const wrote = writeInputFields(user, layout.inputs, input.inputs, (name, value) =>
            this.fieldBlock(name, value)
        );
        user.text(`${wrote ? '\n\n' : ''}${this.responseInstruction(layout)}`);
        messages.push({ role: 'user', content: user.build() });

        return messages;
    }

    /** Field listing, structure template, then the objective. */
    protected systemMessage(layout: SignatureLayout): string {
        const structure = [
            ...layout.inputs.map((field) => this.fieldBlock(field.name, `{${field.name}}`)),
            ...layout.outputs.map((field) =>
                this.fieldBlock(field.name, `{${field.name}}${this.structureNote(field)}`)
            ),
            ...(this.completedMarker() ? [this.completedMarker()] : []),
        ].join('\n\n');

        return [
            describeFields(layout),
            '',
            'All interactions will be structured in the following way, with the appropriate values filled in.',
            '',
            structure,
            '',
            `In adhering to this structure, your objective is:\n${layout.instructions}`,
        ].join('\n');
    }

    /** The note after a typed output's placeholder in the structure template. */
    protected structureNote(field: AdapterField): string {
        return field.hint ? `        # note: the value you produce must be ${field.hint}` : '';
    }

    /** One field: its marker, then its value. */
    protected fieldBlock(name: string, value: string): string {
        return `[[ ## ${name} ## ]]\n${value}`;
    }

    /** The marker closing a reply, or `undefined` for a format without one. */
    protected completedMarker(): string | undefined {
        return '[[ ## completed ## ]]';
    }

    /** A demo or history turn's reply: the outputs it has, then the completed marker. */
    protected assistantMessage(
        layout: SignatureLayout,
        outputs: Record<string, unknown>
    ): string {
        const builder = new ContentBuilder();
        writeInputFields(builder, layout.outputs, outputs, (name, value) =>
            this.fieldBlock(name, value)
        );
        const marker = this.completedMarker();
        if (marker) builder.text(`\n\n${marker}`);
        return builder.build() as string;
    }

    /** The closing line of the real user turn: which fields to produce, in order. */
    protected responseInstruction(layout: SignatureLayout): string {
        const fields = layout.outputs.map(
            (field) =>
                `\`[[ ## ${field.name} ## ]]\`${field.hint ? ` (must be ${field.hint})` : ''}`
        );
        return (
            `Respond with the corresponding output fields, starting with the field ${fields.join(', then ')}, ` +
            'and then ending with the marker for `[[ ## completed ## ]]`.'
        );
    }

    // ------------------------------------------------------------- parse

    parse(signature: SignatureLike, completion: string): Record<string, unknown> {
        const names = outputNames(signature);
        const sections = this.readSections(completion, names);
        if (sections) return pickNonEmpty(sections, names);
        return tolerantParse(completion, names);
    }

    parsePartial(signature: SignatureLike, text: string): Record<string, unknown> {
        const names = outputNames(signature);
        if (this.startsStructured(text)) {
            const sections = this.readSections(this.dropIncompleteMarker(text), names, true);
            return sections ? pickNonEmpty(sections, names) : {};
        }
        // Still the beginning of `answer:` — showing "answ" as the answer would
        // be worse than showing nothing for one more chunk.
        if (isPartialLabel(text, names)) return {};
        return extractLabelledFields(text, names);
    }

    /**
     * The text under each marker, keyed by field name, or `null` when the
     * reply has no markers at all. Text before the first marker is ignored.
     */
    protected readSections(
        text: string,
        _names: string[],
        _partial = false
    ): Record<string, string> | null {
        const marker = /\[\[\s*##\s*([^\]#]+?)\s*##\s*\]\]/g;
        const found: Array<{ name: string; start: number; end: number }> = [];
        for (let match = marker.exec(text); match; match = marker.exec(text)) {
            found.push({
                name: match[1].trim(),
                start: match.index,
                end: match.index + match[0].length,
            });
        }
        if (found.length === 0) return null;

        const sections: Record<string, string> = {};
        found.forEach((entry, index) => {
            const content = text.slice(entry.end, found[index + 1]?.start ?? text.length);
            // The first occurrence wins: a model that repeats a field usually
            // got it right the first time.
            if (!(entry.name in sections)) sections[entry.name] = content.trim();
        });
        return sections;
    }

    /** Whether a reply so far looks like it is using the markers. */
    protected startsStructured(text: string): boolean {
        return text.trimStart().startsWith('[') || text.includes('[[');
    }

    /** Cut a trailing, half-written marker so its text is not shown as a value. */
    protected dropIncompleteMarker(text: string): string {
        const open = text.lastIndexOf('[[');
        if (open !== -1 && text.indexOf(']]', open) === -1) return text.slice(0, open);
        return text.endsWith('[') ? text.slice(0, -1) : text;
    }

    protected repairClosing(_signature: SignatureLike): string {
        return (
            'Reply again with every output field under its own `[[ ## field ## ]]` marker, ' +
            'using the declared type for each, and end with `[[ ## completed ## ]]`.'
        );
    }

    // -------------------------------------------------------------- call

    async call(lm: ILanguageModel, input: AdapterCallInput): Promise<Record<string, any>> {
        return this.withJsonFallback(lm, input, () => super.call(lm, input));
    }

    /**
     * A streamed reply gets the same JSON fallback as {@link call}: without
     * it, a program run under `streamify` (or `Predict.stream`) would fail on
     * a reply that the same program, run normally, recovers from.
     */
    completeFrom(
        lm: ILanguageModel,
        messages: ChatMessage[],
        completion: string,
        input: AdapterCallInput
    ): Promise<Record<string, any>> {
        return this.withJsonFallback(lm, input, () =>
            super.completeFrom(lm, messages, completion, input)
        );
    }

    private async withJsonFallback(
        lm: ILanguageModel,
        input: AdapterCallInput,
        run: () => Promise<Record<string, any>>
    ): Promise<Record<string, any>> {
        try {
            return await run();
        } catch (error) {
            if (!this.shouldFallBack(input.signature, error)) throw error;
            try {
                return await new JSONAdapter().call(lm, input);
            } catch (fallbackError) {
                // A cancelled call stays cancelled. Otherwise the fallback was
                // a best-effort extra, and the reply that failed is the more
                // useful thing to report.
                if (input.options?.signal?.aborted) throw fallbackError;
                throw error;
            }
        }
    }

    /**
     * Fall back only when the reply held none of the output fields: the model
     * ignored the format, which a JSON request can recover. Language-model
     * errors — context length, rate limits — are never retried here.
     */
    private shouldFallBack(signature: SignatureLike, error: unknown): boolean {
        if (!this.jsonFallback || !(error instanceof ValidationError)) return false;
        return Object.keys(this.parse(signature, error.rawOutput)).length === 0;
    }
}

function pickNonEmpty(
    values: Record<string, unknown>,
    names: string[]
): Record<string, unknown> {
    const picked = pickOutputs(values, names);
    for (const [key, value] of Object.entries(picked)) {
        if (value === '') delete picked[key];
    }
    return picked;
}

/**
 * Read a reply that used no markers: a JSON object carrying output fields,
 * or else `field: value` lines (a lone field may be the bare reply).
 */
function tolerantParse(text: string, names: string[]): Record<string, unknown> {
    const trimmed = text.trim();
    if (trimmed.startsWith('{') || trimmed.startsWith('```')) {
        try {
            const parsed = parseJsonResponse<unknown>(trimmed, () => new Error('not JSON'));
            if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
                const picked = pickOutputs(parsed as Record<string, unknown>, names);
                if (Object.keys(picked).length > 0) return picked;
            }
        } catch {
            // Not JSON after all; read it as labelled text.
        }
    }
    return extractLabelledFields(text, names);
}

/** Whether the whole buffer is still the beginning of a field's own label. */
function isPartialLabel(buffer: string, fieldNames: string[]): boolean {
    const seen = buffer.trimStart().toLowerCase();
    if (seen === '') return true;
    return fieldNames.some((name) => `${name.toLowerCase()}:`.startsWith(seen));
}
