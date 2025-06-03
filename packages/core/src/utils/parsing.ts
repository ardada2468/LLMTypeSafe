import { Signature } from '../core/signature';

export function buildPrompt(signature: typeof Signature | string, inputs: Record<string, any>): string {
    if (typeof signature === 'string') {
        return buildPromptFromString(signature, inputs);
    } else {
        return buildPromptFromClass(signature, inputs);
    }
}

function buildPromptFromString(signatureStr: string, inputs: Record<string, any>): string {
    const parsed = Signature.parseStringSignature(signatureStr);

    let prompt = '';

    // Add input fields
    for (const inputKey of parsed.inputs) {
        if (inputs[inputKey] !== undefined) {
            prompt += `${inputKey}: ${inputs[inputKey]}\n`;
        }
    }

    // Add output format - be more explicit about the format expected
    if (parsed.outputs.length === 1) {
        // For single output, make it clear the format expected
        const outputKey = parsed.outputs[0];
        prompt += `\nProvide the ${outputKey} in this format:\n${outputKey}: [your response]`;
    } else {
        // For multiple outputs, be explicit about each field
        prompt += '\nProvide the following fields:\n';
        for (const outputKey of parsed.outputs) {
            const typeInfo = parsed.types[outputKey] ? ` (${parsed.types[outputKey]})` : '';
            prompt += `${outputKey}${typeInfo}: [your response]\n`;
        }
    }

    return prompt.trim();
}

function buildPromptFromClass(signatureClass: typeof Signature, inputs: Record<string, any>): string {
    const inputFields = signatureClass.getInputFields();
    const outputFields = signatureClass.getOutputFields();

    let prompt = '';

    // Add description if available
    if (signatureClass.description) {
        prompt += `${signatureClass.description}\n\n`;
    }

    // Add input fields
    Object.entries(inputFields).forEach(([key, config]) => {
        if (inputs[key] !== undefined) {
            const prefix = config.prefix || `${key}:`;
            prompt += `${prefix} ${inputs[key]}\n`;
        }
    });

    // Add output format
    prompt += '\nProvide:\n';
    Object.entries(outputFields).forEach(([key, config]) => {
        const desc = config.description ? ` (${config.description})` : '';
        prompt += `${key}${desc}:\n`;
    });

    return prompt.trim();
}

export function parseOutput(signature: typeof Signature | string, rawOutput: string): Record<string, any> {
    if (typeof signature === 'string') {
        return parseOutputFromString(signature, rawOutput);
    } else {
        return parseOutputFromClass(signature, rawOutput);
    }
}

function parseOutputFromString(signatureStr: string, rawOutput: string): Record<string, any> {
    const parsed = Signature.parseStringSignature(signatureStr);
    const result: Record<string, any> = {};

    for (const outputKey of parsed.outputs) {
        const value = extractFieldValue(rawOutput, outputKey, parsed.types[outputKey]);
        result[outputKey] = value; // Always set the field, even if null
    }

    return result;
}

function parseOutputFromClass(signatureClass: typeof Signature, rawOutput: string): Record<string, any> {
    const outputFields = signatureClass.getOutputFields();
    const result: Record<string, any> = {};

    Object.entries(outputFields).forEach(([key, config]) => {
        const value = extractFieldValue(rawOutput, key, config.type);
        result[key] = value; // Always set the field, even if null
    });

    return result;
}

function extractFieldValue(text: string, fieldName: string, fieldType?: string): any {
    // Ensure text is a string
    if (typeof text !== 'string') {
        text = String(text);
    }

    // If the text is a simple value without field names, and we're looking for "answer", just return the text
    if (fieldName === 'answer' && !text.includes(':') && !text.includes('\n')) {
        // Simple single-line response without field names - treat as the answer
        let value = text.trim();
        value = value.replace(/^\*+|\*+$/g, ''); // Remove asterisks
        value = value.replace(/^["']|["']$/g, ''); // Remove quotes
        if (value) {
            return convertValue(value, fieldType);
        }
    }

    // Try multiple patterns for field extraction - more flexible patterns
    const patterns = [
        // Standard format: "fieldName: value"
        new RegExp(`${fieldName}\\s*:\\s*(.+?)(?=\\n\\s*\\w+\\s*:|$)`, 'is'),
        // Alternative format: "fieldName = value" or "fieldName value"
        new RegExp(`${fieldName}\\s*[=:]\\s*(.+?)(?=\\n|$)`, 'is'),
        // More flexible: field name followed by content
        new RegExp(`\\b${fieldName}\\b[:\\s=]*([^\\n]+)`, 'i'),
        // Try to find the value on the same line after the field name
        new RegExp(`${fieldName}[:\\s]*([^\\n]+?)(?=\\n|$)`, 'i'),
        // Very loose pattern - just look for anything after the field name
        new RegExp(`${fieldName}[^\\w]*([\\s\\S]*?)(?=\\n\\s*[A-Z]|$)`, 'i')
    ];

    for (const pattern of patterns) {
        const match = text.match(pattern);
        if (match && match[1]) {
            let value = match[1].trim();

            // Remove common artifacts
            value = value.replace(/^\*+|\*+$/g, ''); // Remove asterisks
            value = value.replace(/^["']|["']$/g, ''); // Remove quotes
            value = value.trim();

            if (value) {
                // Type conversion based on field type
                return convertValue(value, fieldType);
            }
        }
    }

    return null;
}

function convertValue(value: string, type?: string): any {
    if (!type || type === 'string') {
        // Auto-detect JSON for untyped fields
        if ((value.startsWith('{') && value.endsWith('}')) ||
            (value.startsWith('[') && value.endsWith(']'))) {
            try {
                return JSON.parse(value);
            } catch {
                return value;
            }
        }
        return value;
    }

    switch (type.toLowerCase()) {
        case 'number':
        case 'float':
            const num = parseFloat(value);
            return isNaN(num) ? value : num;

        case 'int':
        case 'integer':
            const int = parseInt(value);
            return isNaN(int) ? value : int;

        case 'boolean':
        case 'bool':
            return value.toLowerCase() === 'true' || value === '1';

        case 'array':
        case 'list':
            try {
                return JSON.parse(value);
            } catch {
                // Try to split by common delimiters
                return value.split(/[,;\n]/).map(s => s.trim()).filter(s => s);
            }

        case 'object':
        case 'json':
            try {
                return JSON.parse(value);
            } catch {
                return value;
            }

        default:
            // Try to auto-detect and parse JSON
            if ((value.startsWith('{') && value.endsWith('}')) ||
                (value.startsWith('[') && value.endsWith(']'))) {
                try {
                    return JSON.parse(value);
                } catch {
                    return value;
                }
            }
            return value;
    }
} 