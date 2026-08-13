#!/usr/bin/env node
/**
 * Generates assets/npm-downloads.svg — a bar chart of monthly npm downloads
 * for @ts-dspy/core (every provider package depends on core, so it is the
 * install count for the framework as a whole).
 *
 * Usage: node scripts/generate-downloads-chart.js
 *
 * Data source: https://api.npmjs.org/downloads (public, no auth).
 * Run on a schedule from .github/workflows/npm-downloads-chart.yml so the
 * committed SVG stays current.
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const PACKAGE = '@ts-dspy/core';
const MONTHS = 12;
// The repo is ESM ("type": "module"), so __dirname has to be derived.
const OUTPUT = path.join(
    path.dirname(fileURLToPath(import.meta.url)),
    '..',
    'assets',
    'npm-downloads.svg'
);

// npm's downloads API caps a single range request at 18 months.
const RANGE_LIMIT_MONTHS = 18;

function monthKey(date) {
    return `${date.getUTCFullYear()}-${String(date.getUTCMonth() + 1).padStart(2, '0')}`;
}

function isoDay(date) {
    return date.toISOString().slice(0, 10);
}

/** The last MONTHS complete months, oldest first. The in-progress month is
 *  excluded so a partial total never reads as a drop. */
function targetMonths() {
    const now = new Date();
    const months = [];
    for (let i = MONTHS; i >= 1; i--) {
        months.push(
            monthKey(new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth() - i, 1)))
        );
    }
    return months;
}

async function fetchDailyDownloads(from, to) {
    const url = `https://api.npmjs.org/downloads/range/${isoDay(from)}:${isoDay(to)}/${PACKAGE}`;
    const res = await fetch(url);
    if (!res.ok) {
        throw new Error(`npm downloads API returned ${res.status} for ${url}`);
    }
    const body = await res.json();
    return body.downloads || [];
}

async function monthlyTotals() {
    const now = new Date();
    const to = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), 0)); // last day of previous month
    const from = new Date(
        Date.UTC(now.getUTCFullYear(), now.getUTCMonth() - RANGE_LIMIT_MONTHS, 1)
    );

    const totals = new Map();
    for (const { day, downloads } of await fetchDailyDownloads(from, to)) {
        const key = day.slice(0, 7);
        totals.set(key, (totals.get(key) || 0) + downloads);
    }
    return targetMonths().map((key) => ({ key, downloads: totals.get(key) || 0 }));
}

const MONTH_NAMES = [
    'Jan',
    'Feb',
    'Mar',
    'Apr',
    'May',
    'Jun',
    'Jul',
    'Aug',
    'Sep',
    'Oct',
    'Nov',
    'Dec',
];

function shortLabel(key) {
    const [year, month] = key.split('-');
    const name = MONTH_NAMES[Number(month) - 1];
    return month === '01' ? `${name} ’${year.slice(2)}` : name;
}

function niceMax(value) {
    if (value <= 0) return 10;
    const magnitude = Math.pow(10, Math.floor(Math.log10(value)));
    for (const step of [1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10]) {
        const candidate = step * magnitude;
        if (candidate >= value) return candidate;
    }
    return 10 * magnitude;
}

function formatCount(value) {
    return value >= 1000 ? `${(value / 1000).toFixed(value >= 10000 ? 0 : 1)}k` : String(value);
}

function renderSvg(data) {
    const width = 840;
    const height = 310;
    const margin = { top: 62, right: 20, bottom: 46, left: 52 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;

    const max = niceMax(Math.max(...data.map((d) => d.downloads)));
    const total = data.reduce((sum, d) => sum + d.downloads, 0);
    const y = (value) => margin.top + plotHeight - (value / max) * plotHeight;

    const slot = plotWidth / data.length;
    const barWidth = Math.min(44, slot - 10); // ≥2px surface gap between bars
    const radius = 4;

    const ticks = [0, max / 2, max];
    const gridlines = ticks
        .map((tick) => {
            const ty = y(tick);
            return `    <line class="grid" x1="${margin.left}" y1="${ty.toFixed(1)}" x2="${(margin.left + plotWidth).toFixed(1)}" y2="${ty.toFixed(1)}" />
    <text class="axis" x="${margin.left - 10}" y="${(ty + 4).toFixed(1)}" text-anchor="end">${formatCount(tick)}</text>`;
        })
        .join('\n');

    // Direct-label only the peak and the most recent month.
    const peakIndex = data.reduce(
        (best, d, i) => (d.downloads > data[best].downloads ? i : best),
        0
    );
    const labelled = new Set([peakIndex, data.length - 1]);

    const bars = data
        .map((d, i) => {
            const x = margin.left + slot * i + (slot - barWidth) / 2;
            const top = y(d.downloads);
            const barHeight = Math.max(margin.top + plotHeight - top, d.downloads > 0 ? 2 : 0);
            const r = Math.min(radius, barHeight / 2);
            const bar = barHeight
                ? `    <rect class="bar" x="${x.toFixed(1)}" y="${(margin.top + plotHeight - barHeight).toFixed(1)}" width="${barWidth.toFixed(1)}" height="${barHeight.toFixed(1)}" rx="${r.toFixed(1)}" ry="${r.toFixed(1)}"><title>${shortLabel(d.key)} ${d.key.slice(0, 4)}: ${d.downloads.toLocaleString('en-US')} downloads</title></rect>`
                : '';
            const value =
                labelled.has(i) && d.downloads > 0
                    ? `\n    <text class="value" x="${(x + barWidth / 2).toFixed(1)}" y="${(margin.top + plotHeight - barHeight - 7).toFixed(1)}" text-anchor="middle">${d.downloads.toLocaleString('en-US')}</text>`
                    : '';
            const label = `\n    <text class="axis" x="${(x + barWidth / 2).toFixed(1)}" y="${(margin.top + plotHeight + 20).toFixed(1)}" text-anchor="middle">${shortLabel(d.key)}</text>`;
            return bar + value + label;
        })
        .join('\n');

    const range = `${shortLabel(data[0].key)} ${data[0].key.slice(0, 4)} – ${shortLabel(data[data.length - 1].key)} ${data[data.length - 1].key.slice(0, 4)}`;

    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img" aria-label="Monthly npm downloads of ${PACKAGE}: ${data.map((d) => `${shortLabel(d.key)} ${d.downloads}`).join(', ')}">
  <style>
    text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
    .title { font-size: 15px; font-weight: 600; fill: #0b0b0b; }
    .subtitle { font-size: 12px; fill: #52514e; }
    .axis { font-size: 11px; fill: #52514e; }
    .value { font-size: 11px; font-weight: 600; fill: #0b0b0b; }
    .grid { stroke: #d8d7d2; stroke-width: 1; }
    .bar { fill: #2a78d6; }
    @media (prefers-color-scheme: dark) {
      .title { fill: #f0f6fc; }
      .subtitle, .axis { fill: #b0afa5; }
      .value { fill: #f0f6fc; }
      .grid { stroke: #3a3a38; }
      .bar { fill: #3987e5; }
    }
  </style>
  <text class="title" x="${margin.left - 40}" y="24">npm downloads per month — ${PACKAGE}</text>
  <text class="subtitle" x="${margin.left - 40}" y="40">${range} · ${total.toLocaleString('en-US')} downloads over ${MONTHS} months</text>
${gridlines}
${bars}
</svg>
`;
}

async function main() {
    const data = await monthlyTotals();
    fs.mkdirSync(path.dirname(OUTPUT), { recursive: true });
    fs.writeFileSync(OUTPUT, renderSvg(data));
    console.log(
        `Wrote ${path.relative(process.cwd(), OUTPUT)} (${data.length} months, ${data.reduce((s, d) => s + d.downloads, 0)} downloads)`
    );
}

main().catch((error) => {
    console.error(error.message);
    process.exit(1);
});
