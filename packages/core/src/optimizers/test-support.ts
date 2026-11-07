/**
 * Toy tasks for the optimizer tests. Not exported from the package.
 *
 * The task is "double a number". The mock task model answers correctly or
 * not according to a *policy* over what is in the prompt — demos present,
 * a magic word in the instructions, the sampling temperature — so a test can
 * build a world where one candidate prompt really is better than another and
 * assert that the optimizer finds it, with no real model and no randomness.
 */
import { Example } from '../core/example';
import { type Prediction } from '../core/prediction';
import { MockLM } from '../testing/mock-lm';
import { contentToText } from '../utils/content';
import type { ChatMessage, LLMCallOptions } from '../types/language-model';

/** What the mock sees in one call. */
export interface PromptView {
    /** The system message: field list, structure, instructions. */
    system: string;
    /** The real input fields of the call, by name. */
    inputs: Record<string, string>;
    /** Demos rendered as user/assistant pairs before the real turn. */
    demos: Array<{ user: string; assistant: string }>;
    /** Output fields the call asks for, in order. */
    outputs: string[];
    options?: LLMCallOptions;
}

/** Parse `[[ ## name ## ]]\nvalue` sections out of a ChatAdapter message. */
export function sections(text: string): Record<string, string> {
    const result: Record<string, string> = {};
    const pattern = /\[\[ ## (\w+) ## \]\]\n([\s\S]*?)(?=\n*\[\[ ## |\n*$)/g;
    for (const match of text.matchAll(pattern)) result[match[1]] = match[2].trim();
    return result;
}

export function viewPrompt(messages: ChatMessage[], options?: LLMCallOptions): PromptView {
    const texts = messages.map((message) => ({
        role: message.role,
        text: contentToText(message.content),
    }));
    const system = texts.find((t) => t.role === 'system')?.text ?? '';
    const last = texts.at(-1)?.text ?? '';
    const demos: PromptView['demos'] = [];
    for (let i = 1; i < texts.length - 1; i += 2) {
        demos.push({ user: texts[i].text, assistant: texts[i + 1]?.text ?? '' });
    }
    const asked =
        /Respond with the corresponding output fields, starting with the field ([\s\S]*)/.exec(
            last
        );
    const outputs = [...(asked?.[1] ?? last).matchAll(/`\[\[ ## (\w+) ## \]\]`/g)]
        .map((m) => m[1])
        .filter((name) => name !== 'completed');
    const inputs = sections(last.split(/\n\nRespond with/)[0]);
    return { system, inputs, demos, outputs, options };
}

/** A ChatAdapter-formatted reply. */
export function reply(fields: Record<string, unknown>): string {
    return (
        Object.entries(fields)
            .map(([name, value]) => `[[ ## ${name} ## ]]\n${String(value)}`)
            .join('\n\n') + '\n\n[[ ## completed ## ]]'
    );
}

/** `{ question: 'double 3', answer: '6' }` for each n. */
export function doublingSet(ns: number[]): Example[] {
    return ns.map((n) =>
        new Example({ question: `double ${n}`, answer: String(n * 2) }).withInputs('question')
    );
}

export function numberIn(text: string | undefined): number {
    return Number(/-?\d+/.exec(text ?? '')?.[0] ?? NaN);
}

/**
 * A task model for the doubling task: answers `2n` when `policy` says the
 * prompt deserves it, `n` otherwise. Fills a `reasoning` field if asked.
 */
export function doublingLM(
    policy: (view: PromptView, n: number) => boolean,
    extra: (view: PromptView) => string | undefined = () => undefined
): MockLM {
    return new MockLM({
        respond: (messages, options) => {
            const view = viewPrompt(messages, options);
            const special = extra(view);
            if (special !== undefined) return special;
            const n = numberIn(view.inputs.question);
            const answer = policy(view, n) ? n * 2 : n;
            const fields: Record<string, unknown> = {};
            for (const name of view.outputs) {
                fields[name] = name === 'answer' ? answer : `thinking about ${n}`;
            }
            return reply(fields);
        },
    });
}

export const exactAnswer = (example: Example, prediction: Prediction): boolean =>
    String(example.get('answer')) === String(prediction.get('answer'));

/** The `question` values of the demos a call rendered. */
export function demoQuestions(view: PromptView): string[] {
    return view.demos.map((demo) => sections(demo.user).question ?? '');
}
