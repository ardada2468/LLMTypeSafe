#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

// Parse command line arguments
const args = process.argv.slice(2);
const version = args[0];
const dryRun = args.includes('--dry-run');
const useWorkspace = args.includes('--workspace');

// Validate input
if (!version) {
    console.error('Usage: node update-versions.js <version> [--dry-run] [--workspace]');
    console.error('');
    console.error('Examples:');
    console.error('  node update-versions.js 0.3.0');
    console.error('  node update-versions.js 0.3.0 --dry-run');
    console.error('  node update-versions.js 0.3.0 --workspace');
    process.exit(1);
}

// Validate semver format
const semverRegex = /^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/;
if (!semverRegex.test(version)) {
    console.error(`Error: "${version}" is not a valid semver version`);
    console.error('Expected format: x.y.z (e.g., 1.0.0, 2.1.3, 1.0.0-beta.1)');
    process.exit(1);
}

console.log(`${dryRun ? '[DRY RUN] ' : ''}Updating all packages to version ${version}`);
if (useWorkspace) {
    console.log('Using workspace protocol for internal dependencies');
}

const changes = [];

function safeReadJson(filePath) {
    try {
        return JSON.parse(fs.readFileSync(filePath, 'utf8'));
    } catch (error) {
        console.error(`Error reading ${filePath}: ${error.message}`);
        return null;
    }
}

function safeWriteJson(filePath, data) {
    try {
        if (!dryRun) {
            fs.writeFileSync(filePath, JSON.stringify(data, null, 2) + '\n');
        }
        return true;
    } catch (error) {
        console.error(`Error writing ${filePath}: ${error.message}`);
        return false;
    }
}

function updateDependencies(dependencies, packageName, type) {
    const updates = [];
    if (dependencies) {
        Object.keys(dependencies).forEach(dep => {
            if (dep.startsWith('@ts-dspy/')) {
                const oldVersion = dependencies[dep];
                const newVersion = useWorkspace ? '^' + version : '^' + version;

                if (oldVersion !== newVersion) {
                    dependencies[dep] = newVersion;
                    updates.push({
                        package: packageName,
                        type: type,
                        dependency: dep,
                        from: oldVersion,
                        to: newVersion
                    });
                }
            }
        });
    }
    return updates;
}

// Update root package.json
console.log('\n📦 Updating root package.json...');
const rootPackage = safeReadJson('package.json');
if (rootPackage) {
    const oldVersion = rootPackage.version;
    if (oldVersion !== version) {
        rootPackage.version = version;
        if (safeWriteJson('package.json', rootPackage)) {
            changes.push({
                file: 'package.json',
                type: 'version',
                from: oldVersion,
                to: version
            });
            console.log(`  ✅ Updated version: ${oldVersion} → ${version}`);
        }
    } else {
        console.log(`  ℹ️  Already at version ${version}`);
    }
}

// Update lerna.json
console.log('\n📋 Updating lerna.json...');
const lernaConfig = safeReadJson('lerna.json');
if (lernaConfig) {
    const oldVersion = lernaConfig.version;
    if (oldVersion !== version) {
        lernaConfig.version = version;
        if (safeWriteJson('lerna.json', lernaConfig)) {
            changes.push({
                file: 'lerna.json',
                type: 'version',
                from: oldVersion,
                to: version
            });
            console.log(`  ✅ Updated version: ${oldVersion} → ${version}`);
        }
    } else {
        console.log(`  ℹ️  Already at version ${version}`);
    }
}

// Update all package.json files in packages/*
console.log('\n📦 Updating package versions...');
const packagesDir = 'packages';

if (!fs.existsSync(packagesDir)) {
    console.error(`Error: ${packagesDir} directory not found`);
    process.exit(1);
}

const packages = fs.readdirSync(packagesDir).filter(item => {
    const itemPath = path.join(packagesDir, item);
    return fs.statSync(itemPath).isDirectory();
});

if (packages.length === 0) {
    console.log('  ℹ️  No packages found in packages/ directory');
} else {
    packages.forEach(packageName => {
        const packagePath = path.join(packagesDir, packageName);
        const packageJsonPath = path.join(packagePath, 'package.json');

        if (fs.existsSync(packageJsonPath)) {
            console.log(`\n  📦 Processing ${packageName}...`);
            const packageJson = safeReadJson(packageJsonPath);

            if (packageJson) {
                let hasChanges = false;

                // Update package version
                const oldVersion = packageJson.version;
                if (oldVersion !== version) {
                    packageJson.version = version;
                    hasChanges = true;
                    changes.push({
                        file: `${packageName}/package.json`,
                        type: 'version',
                        from: oldVersion,
                        to: version
                    });
                    console.log(`    ✅ Updated version: ${oldVersion} → ${version}`);
                }

                // Update dependencies
                const depUpdates = updateDependencies(packageJson.dependencies, packageName, 'dependencies');
                const devDepUpdates = updateDependencies(packageJson.devDependencies, packageName, 'devDependencies');
                const peerDepUpdates = updateDependencies(packageJson.peerDependencies, packageName, 'peerDependencies');

                const allDepUpdates = [...depUpdates, ...devDepUpdates, ...peerDepUpdates];
                if (allDepUpdates.length > 0) {
                    hasChanges = true;
                    changes.push(...allDepUpdates);
                    allDepUpdates.forEach(update => {
                        console.log(`    🔗 Updated ${update.type}: ${update.dependency} ${update.from} → ${update.to}`);
                    });
                }

                if (!hasChanges) {
                    console.log(`    ℹ️  No changes needed`);
                }

                if (hasChanges && !safeWriteJson(packageJsonPath, packageJson)) {
                    console.error(`    ❌ Failed to write changes`);
                }
            }
        } else {
            console.log(`  ⚠️  Skipping ${packageName} (no package.json found)`);
        }
    });
}

// Summary
console.log('\n' + '='.repeat(50));
console.log('📊 SUMMARY');
console.log('='.repeat(50));

if (changes.length === 0) {
    console.log('✅ No changes needed - all packages are already up to date!');
} else {
    console.log(`${dryRun ? '[DRY RUN] ' : ''}Made ${changes.length} changes:`);

    const versionChanges = changes.filter(c => c.type === 'version');
    const depChanges = changes.filter(c => c.type !== 'version');

    if (versionChanges.length > 0) {
        console.log(`\n📦 Version updates (${versionChanges.length}):`);
        versionChanges.forEach(change => {
            console.log(`  - ${change.file}: ${change.from} → ${change.to}`);
        });
    }

    if (depChanges.length > 0) {
        console.log(`\n🔗 Dependency updates (${depChanges.length}):`);
        depChanges.forEach(change => {
            console.log(`  - ${change.package} ${change.type}: ${change.dependency} ${change.from} → ${change.to}`);
        });
    }
}

if (dryRun) {
    console.log('\n💡 This was a dry run. No files were actually modified.');
    console.log('   Run without --dry-run to apply these changes.');
} else if (changes.length > 0) {
    console.log('\n✅ All versions updated successfully!');
    console.log('\n💡 Next steps:');
    console.log('   1. Review the changes with: git diff');
    console.log('   2. Build the packages: npm run build');
    console.log('   3. Run tests: npm test');
    console.log('   4. Commit changes: git add . && git commit -m "chore: bump version to ' + version + '"');
}

process.exit(0); 