#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

const version = process.argv[2];
if (!version) {
    console.error('Please provide a version number');
    process.exit(1);
}

console.log(`Updating all packages to version ${version}`);

// Update root package.json
const rootPackage = JSON.parse(fs.readFileSync('package.json', 'utf8'));
rootPackage.version = version;
fs.writeFileSync('package.json', JSON.stringify(rootPackage, null, 2) + '\n');
console.log('Updated root package.json');

// Update lerna.json
const lernaConfig = JSON.parse(fs.readFileSync('lerna.json', 'utf8'));
lernaConfig.version = version;
fs.writeFileSync('lerna.json', JSON.stringify(lernaConfig, null, 2) + '\n');
console.log('Updated lerna.json');

// Update all package.json files in packages/*
const packagesDir = 'packages';
const packages = fs.readdirSync(packagesDir);

packages.forEach(packageName => {
    const packagePath = path.join(packagesDir, packageName);
    const packageJsonPath = path.join(packagePath, 'package.json');

    if (fs.existsSync(packageJsonPath)) {
        const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
        packageJson.version = version;

        // Update internal dependencies
        if (packageJson.dependencies) {
            Object.keys(packageJson.dependencies).forEach(dep => {
                if (dep.startsWith('@ts-dspy/')) {
                    packageJson.dependencies[dep] = `^${version}`;
                }
            });
        }

        fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson, null, 2) + '\n');
        console.log(`Updated ${packageName}/package.json`);
    }
});

console.log('All versions updated successfully!'); 