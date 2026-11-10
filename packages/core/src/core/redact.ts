import type { ChatMessage, ChatResult, MessageContent } from '../types/language-model';
import type { LMMiddleware, LMRequest } from './middleware';

/** A kind of sensitive value {@link redact} can recognise out of the box. */
export type BuiltInEntity = 'email' | 'phone' | 'ssn' | 'creditCard' | 'ipAddress';

/** A custom pattern for {@link redact}. */
export interface RedactionPattern {
    /** Names the placeholder: `ACCOUNT` yields `[ACCOUNT_1]`, `[ACCOUNT_2]`, … */
    name: string;
    /** What to find. A missing `g` flag is added. */
    pattern: RegExp;
}

export interface RedactOptions {
    /**
     * Built-in entities to redact. Defaults to all of them:
     * `email`, `phone`, `ssn`, `creditCard`, `ipAddress`.
     */
    entities?: BuiltInEntity[];
    /** Further patterns, applied before the built-ins. */
    patterns?: RedactionPattern[];
    /**
     * Put the original values back into the model's reply, so the caller sees
     * a normal answer while the provider never saw the data. Default true.
     * Streams are never restored — a placeholder can be split across chunks.
     */
    restore?: boolean;
    /** Which message roles to redact. Defaults to every role. */
    roles?: Array<ChatMessage['role']>;
    /** Called once per call that redacted anything, with counts by entity. */
    onRedact?: (counts: Record<string, number>) => void;
}

const BUILT_IN: Record<BuiltInEntity, { name: string; pattern: RegExp }> = {
    email: {
        name: 'EMAIL',
        pattern: /[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}/g,
    },
    ssn: { name: 'SSN', pattern: /\b\d{3}-\d{2}-\d{4}\b/g },
    creditCard: { name: 'CARD', pattern: /\b\d(?:[ -]?\d){12,18}\b/g },
    ipAddress: {
        name: 'IP',
        pattern: /\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b/g,
    },
    phone: {
        name: 'PHONE',
        // Not part of a longer run of digit groups (an order or card number).
        pattern:
            /(?<![\w.]|\d[ .-])(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?|\d{2,4}[ .-])\d{3,4}[ .-]\d{3,4}(?![\w-]|[ .-]\d)/g,
    },
};

// Longest, most specific shapes first: a card number must not be half-eaten
// as a phone number.
const BUILT_IN_ORDER: BuiltInEntity[] = ['email', 'ssn', 'creditCard', 'ipAddress', 'phone'];

/** Whether a digit string passes the Luhn check every payment card number does. */
function passesLuhn(candidate: string): boolean {
    const digits = candidate.replace(/\D/g, '');
    if (digits.length < 13 || digits.length > 19) return false;
    let sum = 0;
    for (let i = 0; i < digits.length; i++) {
        let digit = Number(digits[digits.length - 1 - i]);
        if (i % 2 === 1) {
            digit *= 2;
            if (digit > 9) digit -= 9;
        }
        sum += digit;
    }
    return sum % 10 === 0;
}

/** One call's worth of placeholders: value ↔ token, stable within the call. */
class Vault {
    private readonly byValue = new Map<string, string>();
    private readonly byToken = new Map<string, string>();
    readonly counts: Record<string, number> = {};

    constructor(private readonly rules: ReadonlyArray<{ name: string; pattern: RegExp }>) {}

    get size(): number {
        return this.byToken.size;
    }

    redact(text: string): string {
        let out = text;
        for (const { name, pattern } of this.rules) {
            out = out.replace(pattern, (match) => {
                if (name === 'CARD' && !passesLuhn(match)) return match;
                // The same value gets the same token, so the model can still
                // tell that two mentions refer to one thing.
                let token = this.byValue.get(match);
                if (!token) {
                    this.counts[name] = (this.counts[name] ?? 0) + 1;
                    token = `[${name}_${this.counts[name]}]`;
                    this.byValue.set(match, token);
                    this.byToken.set(token, match);
                }
                return token;
            });
        }
        return out;
    }

    restore(text: string): string {
        if (this.byToken.size === 0) return text;
        return text.replace(
            /\[[A-Z][A-Z0-9_]*_\d+\]/g,
            (token) => this.byToken.get(token) ?? token
        );
    }

    restoreDeep(value: unknown): unknown {
        if (typeof value === 'string') return this.restore(value);
        if (Array.isArray(value)) return value.map((item) => this.restoreDeep(item));
        if (value !== null && typeof value === 'object') {
            const out: Record<string, unknown> = {};
            for (const [key, item] of Object.entries(value)) {
                if (key !== '__proto__') out[key] = this.restoreDeep(item);
            }
            return out;
        }
        return value;
    }
}

function redactContent(content: MessageContent, vault: Vault): MessageContent {
    if (typeof content === 'string') return vault.redact(content);
    if (!Array.isArray(content)) return content;
    return content.map((part) =>
        part.type === 'text' ? { ...part, text: vault.redact(part.text) } : part
    ) as MessageContent;
}

/**
 * Keep personal data out of prompts. Emails, phone numbers, US social security
 * numbers, payment card numbers and IP addresses — plus any `patterns` of your
 * own — are replaced with placeholders (`[EMAIL_1]`) before the request leaves
 * the process, and by default put back into the reply, so callers see a normal
 * answer while the provider, its logs, and anything downstream of this
 * middleware (a cache, a fallback model, traces) never see the values.
 *
 * ```ts
 * const lm = wrapLM(new OpenAILM(), redact({ onRedact: (counts) => metrics.pii(counts) }));
 * ```
 *
 * Put it **outermost** in `wrapLM` so every other layer works on redacted
 * text. Pattern matching is a safety net, not a guarantee: it does not find
 * names, addresses, or free-form identifiers. Images are passed through
 * untouched.
 */
export function redact(options: RedactOptions = {}): LMMiddleware {
    const { restore = true, roles, onRedact } = options;
    const rules = [
        ...(options.patterns ?? []).map(({ name, pattern }) => ({
            name: name.toUpperCase().replace(/[^A-Z0-9_]/g, '_'),
            pattern: new RegExp(
                pattern.source,
                pattern.flags.includes('g') ? pattern.flags : `${pattern.flags}g`
            ),
        })),
        ...BUILT_IN_ORDER.filter((entity) =>
            (options.entities ?? BUILT_IN_ORDER).includes(entity)
        ).map((entity) => BUILT_IN[entity]),
    ];

    return async (request, next) => {
        const vault = new Vault(rules);
        const redacted: LMRequest = {
            ...request,
            ...(request.prompt !== undefined ? { prompt: vault.redact(request.prompt) } : {}),
            ...(request.messages
                ? {
                      messages: request.messages.map((message) =>
                          roles && !roles.includes(message.role)
                              ? message
                              : { ...message, content: redactContent(message.content, vault) }
                      ),
                  }
                : {}),
        };

        if (vault.size > 0) {
            try {
                onRedact?.({ ...vault.counts });
            } catch {
                // An observer must not fail the call.
            }
        }

        const result = await next(redacted);
        if (!restore || vault.size === 0 || request.method === 'chatStream') return result;

        if (request.method === 'chatWithTools') {
            const chat = result as ChatResult;
            return {
                ...chat,
                content: vault.restore(chat.content),
                ...(chat.toolCalls
                    ? {
                          // Tools must receive real values, not placeholders.
                          toolCalls: chat.toolCalls.map((call) => ({
                              ...call,
                              arguments: vault.restoreDeep(call.arguments) as Record<
                                  string,
                                  unknown
                              >,
                          })),
                      }
                    : {}),
            } satisfies ChatResult;
        }
        return vault.restoreDeep(result);
    };
}
