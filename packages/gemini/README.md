# @ts-dspy/gemini

[![npm version](https://badge.fury.io/js/@ts-dspy%2Fgemini.svg)](https://badge.fury.io/js/@ts-dspy%2Fgemini)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Google Gemini API integration for TS-DSPy - enables type-safe LLM interactions with Gemini models for TypeScript applications.**

This package provides seamless integration between TS-DSPy and Google's Gemini language models, allowing you to build powerful, type-safe applications with Gemini 2.0 Flash and other Gemini models.

## 🚀 Features

- **Full Gemini Support**: Works with Gemini 2.0 Flash, Gemini 1.0 Pro, and other Gemini models
- **Type-Safe Integration**: Fully compatible with TS-DSPy signatures and modules
- **Structured Output**: Built-in JSON schema support for structured responses
- **Safety Settings**: Configurable content filtering and safety controls
- **Error Handling**: Robust error handling with detailed feedback
- **Flexible Configuration**: Support for all Gemini parameters
- **Advanced Context**: Support for large context windows (32k+ tokens)

## 📦 Installation

```bash
npm install @ts-dspy/gemini @ts-dspy/core

# Install ts-node for proper execution (recommended)
npm install -g ts-node
```

**⚠️ Important: Use `ts-node` to run TypeScript files directly. Transpiling to JavaScript may cause issues with decorators and type information.**

```bash
# Run your scripts with ts-node
npx ts-node your-script.ts

# Or install globally and use directly
npm install -g ts-node
ts-node your-script.ts
```

## 🔑 Setup

Get your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

```typescript
import { GeminiLM } from '@ts-dspy/gemini';

const lm = new GeminiLM({
  apiKey: process.env.GEMINI_API_KEY, // Your Gemini API key
  model: 'gemini-2.0-flash', // or 'gemini-1.0-pro', etc.
});
```

## 🎯 Quick Start

### Basic Usage

```typescript
import { Signature, InputField, OutputField, Predict, configure } from '@ts-dspy/core';
import { GeminiLM } from '@ts-dspy/gemini';

// Define your signature
class Translator extends Signature {
  static description = "Translate text between languages";

  @InputField({ description: "Text to translate" })
  text!: string;

  @InputField({ description: "Target language" })
  target_language!: string;

  @OutputField({ description: "Translated text" })
  translation!: string;

  @OutputField({ description: "Confidence score from 0-1", type: "number" })
  confidence!: number;
}

// Setup Gemini model
configure({
    lm: new GeminiLM({
        apiKey: process.env.GEMINI_API_KEY,
        model: 'gemini-2.0-flash'
    })
});

// Create and use predictor
const translator = new Predict(Translator);

const result = await translator.forward({
  text: "Hello, how are you?",
  target_language: "Spanish"
});

console.log(result.translation); // "Hola, ¿cómo estás?"
console.log(result.confidence); // 0.95
```

### With Global Configuration

```typescript
import { configure } from '@ts-dspy/core';
import { GeminiLM } from '@ts-dspy/gemini';

// Configure globally
configure({
  lm: new GeminiLM({
    apiKey: process.env.GEMINI_API_KEY,
    model: 'gemini-2.0-flash'
  })
});

// Now you can use modules without passing the language model
const predictor = new Predict("question -> answer");
```

## ⚙️ Configuration Options

### Model Configuration

```typescript
const lm = new GeminiLM({
  // Required
  apiKey: 'your-api-key',
  
  // Model selection
  model: 'gemini-2.0-flash', // gemini-2.0-flash, gemini-1.0-pro, etc.
});

// You can also set generation parameters through LLMCallOptions
const result = await predictor.forward(
  { question: "What is AI?" },
  {
    temperature: 0.7, // 0-1, controls creativity
    maxTokens: 1000, // Maximum tokens to generate
    topP: 0.9, // Nucleus sampling parameter
    stopSequences: ['END'], // Stop generation at these sequences
  }
);
```

### Supported Models

- **Gemini 2.0**: `gemini-2.0-flash` (latest and fastest)
- **Gemini 1.0**: `gemini-1.0-pro`
- **Legacy**: Other Gemini model variants

### Model Capabilities

```typescript
const capabilities = lm.getCapabilities();
console.log(capabilities);
// {
//   supportsStreaming: false,
//   supportsStructuredOutput: true,
//   supportsFunctionCalling: false,
//   supportsVision: false,
//   maxContextLength: 32768,
//   supportedFormats: ['json_object']
// }
```

## 📊 Structured Output

Gemini excels at generating structured JSON responses:

```typescript
import { Signature, InputField, OutputField } from '@ts-dspy/core';

class ProductAnalysis extends Signature {
  @InputField({ description: "Product description" })
  description!: string;

  @OutputField({ description: "Product category" })
  category!: string;

  @OutputField({ description: "Price range", type: "string" })
  priceRange!: 'budget' | 'mid-range' | 'premium';

  @OutputField({ description: "Key features as array", type: "array" })
  features!: string[];

  @OutputField({ description: "Sentiment score", type: "number" })
  sentiment!: number;
}

const analyzer = new Predict(ProductAnalysis);
const result = await analyzer.forward({
  description: "Latest smartphone with AI camera, long battery life, and premium design"
});

console.log(result.category); // "Electronics"
console.log(result.priceRange); // "premium"
console.log(result.features); // ["AI camera", "long battery", "premium design"]
console.log(result.sentiment); // 0.8
```

## 🛡️ Safety and Content Filtering

Gemini includes built-in safety settings to filter harmful content:

```typescript
// The GeminiLM automatically configures safety settings
// Default: BLOCK_MEDIUM_AND_ABOVE for harassment

// If content is blocked, you'll receive a descriptive error:
try {
  const result = await predictor.forward({ question: "Inappropriate content..." });
} catch (error) {
  if (error.message.includes('blockReason')) {
    console.log('Content was blocked by Gemini safety filters');
    // Handle content filtering gracefully
  }
}
```

## 🎨 Advanced Usage

### Chain of Thought Reasoning

```typescript
import { ChainOfThought } from '@ts-dspy/core';

const reasoner = new ChainOfThought("problem -> solution: int");
const result = await reasoner.forward({
  problem: "A store sells apples for $2 each and oranges for $3 each. If someone buys 4 apples and 3 oranges, how much do they pay in total?"
});

console.log(result.reasoning); // "First, calculate apples: 4 × $2 = $8..."
console.log(result.solution);  // 17
```

### ReAct Pattern with Tools

```typescript
import { RespAct } from '@ts-dspy/core';

const agent = new RespAct("question -> answer", {
    tools: {
        calculate: {
            description: "Performs mathematical calculations including arithmetic operations. Use this when you need to compute numerical results.",
            function: (expression: string) => {
                try {
                    return new Function('return ' + expression)();
                } catch (error) {
                    return `Error: ${error}`;
                }
            }
        },
        convertCurrency: {
            description: "Converts between currencies. Provide amount and currency codes (e.g., '100 USD to EUR').",
            function: async (query: string) => {
                // Implementation would call a currency API
                return "Converted amount: ...";
            }
        }
    },
    maxSteps: 5
});

const result = await agent.forward({
    question: "What is 15 * 24 + 100 - 50 in USD converted to EUR?"
});
```

### Sentiment Analysis Example

```typescript
class SentimentAnalysis extends Signature {
    @InputField({ description: "Text to analyze for sentiment" })
    text!: string;

    @OutputField({ description: "Sentiment classification" })
    sentiment!: 'positive' | 'negative' | 'neutral';

    @OutputField({ description: "Confidence score between 0 and 1" })
    confidence!: number;

    @OutputField({ description: "Key emotional indicators found" })
    emotions!: string[];

    static description = "Analyze the sentiment and emotions in the given text";
}

const classifier = new Predict(SentimentAnalysis);
const result = await classifier.forward({
    text: "I absolutely love this new framework! It's incredibly powerful and easy to use."
});

console.log(result.sentiment);   // "positive"
console.log(result.confidence);  // 0.92
console.log(result.emotions);    // ["love", "enthusiasm", "satisfaction"]
```

## 🔄 Chat and Conversation

```typescript
// Multi-turn conversations
const chatModel = new GeminiLM({
  apiKey: process.env.GEMINI_API_KEY,
  model: 'gemini-2.0-flash'
});

const messages = [
  { role: 'user', content: 'Hello! What is TypeScript?' },
  { role: 'assistant', content: 'TypeScript is a superset of JavaScript...' },
  { role: 'user', content: 'How does it help with large applications?' }
];

const response = await chatModel.chat(messages);
console.log(response);
```

## 🛡️ Error Handling

Comprehensive error handling for Gemini-specific issues:

```typescript
try {
  const result = await predictor.forward({ question: "Complex question" });
} catch (error) {
  if (error.message.includes('blockReason')) {
    console.log('Content blocked by safety filters');
    // Try rephrasing the question
  } else if (error.message.includes('API key')) {
    console.log('Invalid or missing API key');
    // Check your API key configuration
  } else if (error.message.includes('quota')) {
    console.log('API quota exceeded');
    // Implement retry with backoff
  } else {
    console.log('Unexpected error:', error.message);
  }
}
```

## 📈 Usage Tracking

Note: Gemini API currently doesn't provide detailed token usage statistics through the SDK:

```typescript
const lm = new GeminiLM({
  apiKey: process.env.GEMINI_API_KEY,
  model: 'gemini-2.0-flash'
});

// Make some predictions
const predictor = new Predict("question -> answer", lm);
await predictor.forward({ question: "What is AI?" });

// Get usage statistics (currently limited)
const usage = lm.getUsage();
console.log(`Requests made: ${usage.requestCount || 'Not available'}`);
console.log('Note: Detailed token usage not yet available in Gemini API');
```

## 🆚 Gemini vs OpenAI Comparison

| Feature | Gemini | OpenAI |
|---------|--------|--------|
| **Models** | Gemini 2.0 Flash, 1.0 Pro | GPT-4, GPT-3.5-turbo |
| **Context Length** | 32k+ tokens | 8k-128k tokens |
| **Structured Output** | ✅ Native JSON | ✅ JSON mode |
| **Streaming** | ❌ Not yet supported | ✅ Full support |
| **Function Calling** | ❌ Not in this SDK | ✅ Full support |
| **Safety Filtering** | ✅ Built-in | ⚠️ Moderation API |
| **Cost** | Generally lower | Higher for advanced models |
| **Speed** | Very fast (Flash model) | Variable by model |

## 🔧 Troubleshooting

### Common Issues

1. **API Key Issues**
   ```bash
   # Set your API key
   export GEMINI_API_KEY="your-key-here"
   ```

2. **Content Blocked**
   - Gemini has strict safety filters
   - Try rephrasing sensitive queries
   - Check the error message for specific block reasons

3. **Model Not Found**
   - Ensure you're using supported model names
   - Check [Google AI Studio](https://makersuite.google.com/) for available models

4. **Rate Limiting**
   - Implement exponential backoff
   - Check your API quotas in Google Cloud Console

## 📚 More Examples

Check out the complete examples in the [TS-DSPy repository](https://github.com/ardada2468/LLMTypeSafe/tree/main/examples):

- `basic-gemini-example.ts` - Comprehensive usage examples
- Integration with other TS-DSPy modules
- Real-world application patterns

## 🤝 Contributing

Contributions are welcome! Please see the main [TS-DSPy repository](https://github.com/ardada2468/LLMTypeSafe) for contribution guidelines.

## 📄 License

MIT License - see the [LICENSE](../../LICENSE) file for details.

## 🔗 Related Packages

- [`@ts-dspy/core`](https://www.npmjs.com/package/@ts-dspy/core) - Core TS-DSPy framework
- [`@ts-dspy/openai`](https://www.npmjs.com/package/@ts-dspy/openai) - OpenAI integration

## 🆘 Support

- 📖 [Documentation](https://github.com/ardada2468/LLMTypeSafe#readme)
- 🐛 [Issue Tracker](https://github.com/ardada2468/LLMTypeSafe/issues)
- 💬 [Discussions](https://github.com/ardada2468/LLMTypeSafe/discussions)

---

**Made with ❤️ for the TypeScript + AI community** 