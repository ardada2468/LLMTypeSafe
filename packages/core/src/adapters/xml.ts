import { type SignatureLike } from '../core/signature';
import { ChatAdapter, type ChatAdapterOptions } from './chat';
import { type AdapterField, type SignatureLayout } from './base';

/**
 * Like {@link ChatAdapter}, with every field wrapped in XML tags —
 * `<answer>Paris</answer>` — instead of `[[ ## answer ## ]]` markers. Some
 * models, Claude among them, follow tag structure especially reliably.
 *
 * Inherits the rest of `ChatAdapter`: the system message layout, tolerant
 * parsing of replies that ignore the format, and the JSON fallback.
 */
export class XMLAdapter extends ChatAdapter {
    readonly name: string = 'XMLAdapter';

    constructor(options: ChatAdapterOptions = {}) {
        super(options);
    }

    protected fieldBlock(name: string, value: string): string {
        return `<${name}>\n${value}\n</${name}>`;
    }

    protected structureNote(field: AdapterField): string {
        return field.hint ? ` (the value you produce must be ${field.hint})` : '';
    }

    protected completedMarker(): string | undefined {
        return undefined;
    }

    protected responseInstruction(layout: SignatureLayout): string {
        const fields = layout.outputs.map(
            (field) => `\`<${field.name}>\`${field.hint ? ` (must be ${field.hint})` : ''}`
        );
        return `Respond with the corresponding output fields wrapped in XML tags, starting with ${fields.join(', then ')}.`;
    }

    protected readSections(
        text: string,
        names: string[],
        partial = false
    ): Record<string, string> | null {
        const sections: Record<string, string> = {};
        for (const name of names) {
            const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const closed = new RegExp(`<${escaped}>([\\s\\S]*?)</${escaped}>`, 'i').exec(text);
            if (closed) {
                sections[name] = closed[1].trim();
                continue;
            }
            if (partial) {
                // Still being written: everything after the opening tag.
                const open = new RegExp(`<${escaped}>([\\s\\S]*)$`, 'i').exec(text);
                if (open) sections[name] = open[1].trim();
            }
        }
        // No declared field tag means this is not our format, however many
        // angle brackets the prose contains (`List<String>`, `<div>`): let the
        // tolerant label parser have it.
        return Object.keys(sections).length > 0 ? sections : null;
    }

    protected startsStructured(text: string): boolean {
        return text.trimStart().startsWith('<');
    }

    protected dropIncompleteMarker(text: string): string {
        const open = text.lastIndexOf('<');
        return open !== -1 && text.indexOf('>', open) === -1 ? text.slice(0, open) : text;
    }

    protected repairClosing(_signature: SignatureLike): string {
        return 'Reply again with every output field wrapped in its own XML tag, using the declared type for each.';
    }
}
