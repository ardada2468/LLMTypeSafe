const typescript = require('@rollup/plugin-typescript');
const { nodeResolve } = require('@rollup/plugin-node-resolve');

module.exports = {
    input: 'src/index.ts',
    output: [
        {
            file: 'dist/index.js',
            format: 'cjs',
            sourcemap: true
        },
        {
            file: 'dist/index.esm.js',
            format: 'esm',
            sourcemap: true
        }
    ],
    plugins: [
        nodeResolve(),
        typescript({
            typescript: require('typescript'),
            tsconfig: './tsconfig.json',
            declaration: true,
            declarationDir: './dist',
            outputToFilesystem: true
        })
    ],
    external: ['reflect-metadata']
}; 