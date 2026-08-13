#!/usr/bin/env node
/**
 * Packs every workspace, installs the tarballs into a throwaway project, and
 * imports them from both ESM and CommonJS.
 *
 * This is the gate that catches broken installs. Version 0.4.2 shipped
 * @ts-dspy/openai declaring @ts-dspy/core@^0.3.0 — a range its own release did
 * not satisfy — so consumers silently resolved a stale core from the registry.
 * Nothing in the build, the type checker, or the unit tests could see that,
 * because they all run against the workspace symlinks rather than against what
 * npm actually installs.
 *
 * Usage: node scripts/verify-packaging.js
 */

import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const PACKAGES = ['@ts-dspy/core', '@ts-dspy/openai', '@ts-dspy/gemini', '@ts-dspy/anthropic'];

function run(command, args, options = {}) {
    return execFileSync(command, args, {
        encoding: 'utf8',
        stdio: ['ignore', 'pipe', 'pipe'],
        ...options,
    });
}

function packAll(destination) {
    const tarballs = {};
    for (const name of PACKAGES) {
        const output = run('npm', [
            'pack',
            '--workspace',
            name,
            '--pack-destination',
            destination,
            '--silent',
        ]);
        const file = output.trim().split('\n').pop().trim();
        tarballs[name] = path.join(destination, file);
        if (!fs.existsSync(tarballs[name])) {
            throw new Error(`npm pack did not produce ${tarballs[name]}`);
        }
    }
    return tarballs;
}

const ESM_CHECK = `
import { Signature, InputField, OutputField, Predict, parseOutput, ValidationError, BaseLM, configure } from '@ts-dspy/core';
import { OpenAILM } from '@ts-dspy/openai';
import { GeminiLM } from '@ts-dspy/gemini';
import { AnthropicLM } from '@ts-dspy/anthropic';
import assert from 'node:assert/strict';

// BaseLM only exists from 0.5.0 onward. If a provider pulled an older core from
// the registry, its own import of BaseLM would already have failed above.
assert.equal(typeof BaseLM, 'function', 'BaseLM missing from @ts-dspy/core');

// Every provider must extend the *same* core class object. Two copies of core
// on disk would make these fail while every other check still passed.
for (const [name, Provider] of [['openai', OpenAILM], ['gemini', GeminiLM], ['anthropic', AnthropicLM]]) {
    assert.ok(Object.create(Provider.prototype) instanceof BaseLM, name + ' does not extend the installed core BaseLM');
}

class QA extends Signature {}
InputField({ description: 'q' })(QA.prototype, 'question');
OutputField({ description: 'a' })(QA.prototype, 'answer');
OutputField({ description: 's', type: 'number' })(QA.prototype, 'score');

const parsed = parseOutput(QA, 'answer: Paris\\nscore: 0.9');
assert.equal(parsed.answer, 'Paris');
assert.equal(typeof parsed.score, 'number', 'typed field did not validate to a number');

assert.throws(() => parseOutput(QA, 'answer: Paris\\nscore: high'), ValidationError);

assert.equal(typeof configure, 'function');
assert.equal(new OpenAILM({ apiKey: 'x' }).getCapabilities().supportsStructuredOutput, true);

console.log('  ESM  ok');
`;

const CJS_CHECK = `
const { Signature, InputField, OutputField, parseOutput, ValidationError, BaseLM } = require('@ts-dspy/core');
const { OpenAILM } = require('@ts-dspy/openai');
const { GeminiLM } = require('@ts-dspy/gemini');
const { AnthropicLM } = require('@ts-dspy/anthropic');
const assert = require('node:assert/strict');

assert.equal(typeof BaseLM, 'function', 'BaseLM missing from @ts-dspy/core');
for (const [name, Provider] of [['openai', OpenAILM], ['gemini', GeminiLM], ['anthropic', AnthropicLM]]) {
    assert.ok(Object.create(Provider.prototype) instanceof BaseLM, name + ' does not extend the installed core BaseLM');
}

class QA extends Signature {}
InputField({ description: 'q' })(QA.prototype, 'question');
OutputField({ description: 's', type: 'number' })(QA.prototype, 'score');

assert.equal(parseOutput(QA, 'score: 42').score, 42);
assert.throws(() => parseOutput(QA, 'score: high'), ValidationError);

console.log('  CJS  ok');
`;

function main() {
    const workdir = fs.mkdtempSync(path.join(os.tmpdir(), 'ts-dspy-packaging-'));
    const consumer = path.join(workdir, 'consumer');
    fs.mkdirSync(consumer);

    try {
        console.log('Packing workspaces...');
        const tarballs = packAll(workdir);

        fs.writeFileSync(
            path.join(consumer, 'package.json'),
            JSON.stringify(
                {
                    name: 'packaging-check',
                    private: true,
                    version: '1.0.0',
                    type: 'module',
                    dependencies: Object.fromEntries(
                        PACKAGES.map((name) => [name, `file:${tarballs[name]}`])
                    ),
                },
                null,
                2
            )
        );

        console.log('Installing tarballs into a clean project...');
        run('npm', ['install', '--no-audit', '--no-fund', '--silent'], { cwd: consumer });

        // If a provider's declared core range did not match the core we packed,
        // npm resolves a second copy from the registry instead of deduping.
        const installed = JSON.parse(
            fs.readFileSync(
                path.join(consumer, 'node_modules', '@ts-dspy', 'core', 'package.json'),
                'utf8'
            )
        );
        const expected = JSON.parse(
            fs.readFileSync(path.join(REPO, 'packages', 'core', 'package.json'), 'utf8')
        ).version;
        if (installed.version !== expected) {
            throw new Error(
                `Installed @ts-dspy/core is ${installed.version}, expected ${expected}. ` +
                    `A provider's declared core range does not match this release.`
            );
        }
        const nested = PACKAGES.slice(1)
            .map((name) =>
                path.join(
                    consumer,
                    'node_modules',
                    ...name.split('/'),
                    'node_modules',
                    '@ts-dspy',
                    'core'
                )
            )
            .filter((dir) => fs.existsSync(dir));
        if (nested.length > 0) {
            throw new Error(
                `A second copy of @ts-dspy/core was installed under: ${nested.join(', ')}`
            );
        }

        console.log(`Importing @ts-dspy/core@${installed.version} as a consumer:`);
        fs.writeFileSync(path.join(consumer, 'check.mjs'), ESM_CHECK);
        fs.writeFileSync(path.join(consumer, 'check.cjs'), CJS_CHECK);

        for (const file of ['check.mjs', 'check.cjs']) {
            process.stdout.write(
                run('node', [file], { cwd: consumer, stdio: ['ignore', 'pipe', 'inherit'] })
            );
        }

        console.log('Packaging verified.');
    } finally {
        fs.rmSync(workdir, { recursive: true, force: true });
    }
}

main();
