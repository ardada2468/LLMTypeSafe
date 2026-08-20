/** Helpers shared by the examples. */

/**
 * Read a required environment variable, failing with an actionable message.
 *
 * Examples never fall back to a placeholder key: a placeholder turns a missing
 * credential into a confusing 401 from the provider instead of a clear error
 * here, and committing anything key-shaped invites leaking a real one.
 */
export function requireEnv(name: string): string {
    const value = process.env[name];
    if (!value) {
        console.error(
            `Missing ${name}.\n\n` +
                `Set it before running this example:\n` +
                `  export ${name}="your-key-here"\n`
        );
        process.exit(1);
    }
    return value;
}

/**
 * Evaluate a basic arithmetic expression.
 *
 * A recursive-descent parser over `+ - * / % ^ ( )` and numbers. The examples
 * previously passed model output straight to `eval()`, which executes arbitrary
 * JavaScript — never do that with text a model produced.
 */
export function evaluateArithmetic(expression: string): number {
    const tokens = tokenize(expression);
    let position = 0;

    const peek = () => tokens[position];
    const consume = () => tokens[position++];

    // expression := term (('+' | '-') term)*
    function parseExpression(): number {
        let value = parseTerm();
        while (peek() === '+' || peek() === '-') {
            const operator = consume();
            const right = parseTerm();
            value = operator === '+' ? value + right : value - right;
        }
        return value;
    }

    // term := factor (('*' | '/' | '%') factor)*
    function parseTerm(): number {
        let value = parseFactor();
        while (peek() === '*' || peek() === '/' || peek() === '%') {
            const operator = consume();
            const right = parseFactor();
            if ((operator === '/' || operator === '%') && right === 0) {
                throw new Error('Division by zero');
            }
            if (operator === '*') value *= right;
            else if (operator === '/') value /= right;
            else value %= right;
        }
        return value;
    }

    // factor := unary ('^' factor)?   — right-associative
    function parseFactor(): number {
        const base = parseUnary();
        if (peek() === '^') {
            consume();
            return base ** parseFactor();
        }
        return base;
    }

    function parseUnary(): number {
        if (peek() === '-') {
            consume();
            return -parseUnary();
        }
        if (peek() === '+') {
            consume();
            return parseUnary();
        }
        return parsePrimary();
    }

    function parsePrimary(): number {
        const token = consume();
        if (token === undefined) {
            throw new Error('Unexpected end of expression');
        }
        if (token === '(') {
            const value = parseExpression();
            if (consume() !== ')') {
                throw new Error('Unbalanced parentheses');
            }
            return value;
        }
        const value = Number(token);
        if (Number.isNaN(value)) {
            throw new Error(`Unexpected token: ${token}`);
        }
        return value;
    }

    const result = parseExpression();
    if (position < tokens.length) {
        throw new Error(`Unexpected trailing input: ${tokens.slice(position).join(' ')}`);
    }
    if (!Number.isFinite(result)) {
        throw new Error('Expression did not evaluate to a finite number');
    }
    return result;
}

function tokenize(expression: string): string[] {
    const tokens = expression.match(/\d+\.?\d*|[+\-*/%^()]/g);
    if (!tokens) {
        throw new Error(`Not an arithmetic expression: ${expression}`);
    }
    // Reject anything the tokenizer skipped, e.g. identifiers or function calls.
    const consumed = tokens.join('').length;
    const stripped = expression.replace(/\s+/g, '').length;
    if (consumed !== stripped) {
        throw new Error(`Expression contains unsupported characters: ${expression}`);
    }
    return tokens;
}

/** Print a labelled section header. */
export function section(title: string): void {
    console.log(`\n${'='.repeat(60)}\n${title}\n${'='.repeat(60)}`);
}
