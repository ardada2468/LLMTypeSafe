#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

console.log('Publishing packages to NPM...');

const packagesDir = 'packages';
const packages = fs.readdirSync(packagesDir);

let publishSuccess = true;

packages.forEach(packageName => {
    const packagePath = path.join(packagesDir, packageName);
    const packageJsonPath = path.join(packagePath, 'package.json');

    if (fs.existsSync(packageJsonPath)) {
        const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
        const packageFullName = packageJson.name;
        const packageVersion = packageJson.version;

        console.log(`\nProcessing ${packageFullName}@${packageVersion}...`);

        try {
            // Check if this version is already published
            try {
                execSync(`npm view "${packageFullName}@${packageVersion}" version`, { stdio: 'pipe' });
                console.log(`Version ${packageVersion} of ${packageFullName} already exists on npm, skipping...`);
                return;
            } catch (e) {
                // Version doesn't exist, proceed with publishing
            }

            // Change to package directory and publish
            process.chdir(packagePath);
            console.log(`Publishing ${packageFullName}@${packageVersion} to npm...`);
            execSync('npm publish', { stdio: 'inherit' });
            console.log(`✅ Successfully published ${packageFullName}@${packageVersion}`);

            // Return to root directory
            process.chdir('../..');

        } catch (error) {
            console.error(`❌ Failed to publish ${packageFullName}@${packageVersion}:`, error.message);
            publishSuccess = false;
            // Return to root directory even if failed
            process.chdir('../..');
        }
    }
});

if (publishSuccess) {
    console.log('\n🎉 All packages published successfully!');
    process.exit(0);
} else {
    console.error('\n💥 Some packages failed to publish');
    process.exit(1);
} 