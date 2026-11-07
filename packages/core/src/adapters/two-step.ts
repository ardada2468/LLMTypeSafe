import type { ChatMessage, ILanguageModel } from '../types/language-model';
import { extendSignature, type SignatureLike } from '../core/signature';
import { extractLabelledFields } from '../utils/parsing';
import { getInputFieldConfigs } from '../utils/schema';
import {
    Adapter,
    ContentBuilder,
    demoTurns,
    describeFieldList,
    historyTurns,
    outputNames,
    signatureLayout,
    writeInputFields,
    type AdapterCallInput,
    type AdapterFormatInput,
    type SignatureLayout,
} from './base';
import { defaultAdapter } from './select';

export interface TwoStepAdapterOptions {
    /**
     * The model that turns the main model's free-form answer into the output
     * fields. A small, cheap model is usually plenty.
     */
    extractionLM: ILanguageModel;
    /**
     * Adapter for the extraction call. Defaults to the usual choice for
     * `extractionLM`: {@link JSONAdapter} when it has native structured
     * output, {@link ChatAdapter} otherwise.
     */
    extractionAdapter?: Adapter;
}

/**
 * Two models, two steps: the main model answers in natural language from a
 * plain prompt, then `extractionLM` pulls the output fields out of that
 * answer.
 *
 * Built for reasoning models, which do their best work unconstrained and
 * often follow field-marker formats poorly. The main call never sees a
 * structure template; the extraction call — through {@link ChatAdapter} or
 * {@link JSONAdapter} — does all the formatting work, and its own
 * `repairAttempts` handle a validation failure.
 *
 * {@link parse} on its own only reads `field: value` lines; extraction proper
 * needs the second model, so it happens in {@link call}. Streaming falls back
 * to one ordinary call.
 */
export class TwoStepAdapter extends Adapter {
    readonly name: string = 'TwoStepAdapter';
    readonly extractionLM: ILanguageModel;
    readonly extractionAdapter?: Adapter;

    constructor(options: TwoStepAdapterOptions) {
        super();
        this.extractionLM = options.extractionLM;
        this.extractionAdapter = options.extractionAdapter;
    }

    format(input: AdapterFormatInput): ChatMessage[] {
        const layout = signatureLayout(input.signature, input.instructions);
        const messages: ChatMessage[] = [{ role: 'system', content: systemMessage(layout) }];

        for (const turn of [
            ...demoTurns(input.demos, layout),
            ...historyTurns(input, layout),
        ]) {
            const user = new ContentBuilder();
            writeInputFields(user, layout.inputs, turn.inputs, labelLine, '\n');
            if (!user.empty) messages.push({ role: 'user', content: user.build() });
            const reply = new ContentBuilder();
            writeInputFields(reply, layout.outputs, turn.outputs, labelLine, '\n');
            if (!reply.empty) messages.push({ role: 'assistant', content: reply.build() });
        }

        const user = new ContentBuilder();
        writeInputFields(user, layout.inputs, input.inputs, labelLine, '\n');
        if (user.empty) user.text('Begin.');
        messages.push({ role: 'user', content: user.build() });
        return messages;
    }

    parse(signature: SignatureLike, completion: string): Record<string, unknown> {
        return extractLabelledFields(completion, outputNames(signature));
    }

    async call(lm: ILanguageModel, input: AdapterCallInput): Promise<Record<string, any>> {
        const messages = this.formatMessages(input);
        const answer = await this.request(lm, messages, input);

        const extraction = extendSignature(input.signature, {
            removeInputs: Object.keys(getInputFieldConfigs(input.signature)),
            prependInputs: {
                text: { description: 'A response containing the information to extract' },
            },
            instructions:
                'The input is a text that should contain all the information needed to produce ' +
                'the output fields. Your job is to extract those fields from the text verbatim, ' +
                'without adding anything it does not say.',
        });
        const adapter = this.extractionAdapter ?? defaultAdapter(this.extractionLM);

        // `lm` and `adapter` chose the main call's model and format; the
        // extraction call has its own.
        const { lm: _lm, adapter: _adapter, ...options } = input.options ?? {};
        return adapter.call(this.extractionLM, {
            signature: extraction,
            inputs: { text: answer },
            options,
            span: input.span,
            moduleId: input.moduleId,
        });
    }
}

function labelLine(name: string, value: string): string {
    return `${name}: ${value}`;
}

function systemMessage(layout: SignatureLayout): string {
    const parts = ['You are a helpful assistant that can solve tasks based on user input.', ''];
    if (layout.inputs.length > 0) {
        parts.push('As input, you will be provided with:', describeFieldList(layout.inputs));
    }
    parts.push(
        'Your outputs must contain:',
        describeFieldList(layout.outputs),
        'You should lay out your outputs in detail so that your answer can be understood by another agent.',
        '',
        `Specific instructions: ${layout.instructions}`
    );
    return parts.join('\n');
}
