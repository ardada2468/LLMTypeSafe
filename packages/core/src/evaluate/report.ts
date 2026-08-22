import type { EvaluationReport, EvaluationResult } from './types';

export interface FormatReportOptions {
    /** Maximum per-example rows to render. Defaults to `20`. */
    maxRows?: number;
    /** Characters before a cell is elided. Defaults to `48`. */
    maxCellWidth?: number;
    /** Render every example, not just the failures. Defaults to `true`. */
    includePassing?: boolean;
}

function truncate(text: string, width: number): string {
    const flat = text.replace(/\s+/g, ' ').trim();
    return flat.length > width ? `${flat.slice(0, Math.max(1, width - 1))}…` : flat;
}

function renderValue(value: unknown): string {
    if (value === undefined) return '—';
    if (typeof value === 'string') return value;
    try {
        return JSON.stringify(value) ?? String(value);
    } catch {
        return String(value);
    }
}

function detailFor(result: EvaluationResult, width: number): string {
    if (result.error) {
        return truncate(`${result.error.name}: ${result.error.message}`, width);
    }

    const predicted = (result.prediction?.toObject() ?? {}) as Record<string, unknown>;
    const fields = Object.keys(result.expected);
    if (fields.length === 0) return '';

    const parts = fields.map(
        (field) =>
            `${field}: ${renderValue(result.expected[field])} → ` +
            `${renderValue(predicted[field])}`
    );
    return truncate(parts.join('; '), width);
}

function pad(text: string, width: number): string {
    return text.length >= width ? text : text + ' '.repeat(width - text.length);
}

function padStart(text: string, width: number): string {
    return text.length >= width ? text : ' '.repeat(width - text.length) + text;
}

/**
 * Render a report as plain text.
 *
 * The library never prints: this hands back a string so the caller decides
 * whether it goes to a console, a log line, or a CI annotation.
 */
export function formatReport(
    report: EvaluationReport,
    options: FormatReportOptions = {}
): string {
    const { maxRows = 20, maxCellWidth = 48, includePassing = true } = options;
    const { usage } = report;

    const lines: string[] = [];
    lines.push(
        `score ${report.score.toFixed(3)}  ` +
            `(${report.totalScore.toFixed(2)} / ${report.count})  ` +
            `errors ${report.errorCount}`
    );
    lines.push(
        `tokens ${usage.totalTokens} ` +
            `(${usage.promptTokens} prompt + ${usage.completionTokens} completion)  ` +
            `requests ${usage.requestCount}  ` +
            `mean latency ${usage.averageLatency.toFixed(1)} ms  ` +
            `wall ${usage.durationMs} ms`
    );

    const rows = includePassing
        ? report.results
        : report.results.filter((result) => result.score < 1 || result.error);
    const shown = rows.slice(0, Math.max(0, maxRows));

    if (shown.length > 0) {
        lines.push('');
        const indexWidth = Math.max(1, String(report.count).length);
        lines.push(`${padStart('#', indexWidth)}  ${pad('score', 5)}  detail`);
        for (const result of shown) {
            lines.push(
                `${padStart(String(result.index), indexWidth)}  ` +
                    `${pad(result.score.toFixed(3), 5)}  ` +
                    `${detailFor(result, maxCellWidth)}`
            );
        }
    }

    // Outside the block above: with `maxRows: 0` there are no rows to render,
    // and silence would read as a dataset with nothing in it.
    if (rows.length > shown.length) {
        if (shown.length === 0) lines.push('');
        lines.push(`… ${rows.length - shown.length} more`);
    }

    return lines.join('\n');
}
