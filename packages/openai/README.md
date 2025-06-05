# @ts-dspy/openai

[![npm version](https://badge.fury.io/js/@ts-dspy%2Fopenai.svg)](https://badge.fury.io/js/@ts-dspy%2Fopenai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**OpenAI ChatGPT integration for TS-DSPy - enables type-safe LLM interactions with GPT-3.5, GPT-4, and other OpenAI models.**

This package provides seamless integration between TS-DSPy and OpenAI's language models, allowing you to build powerful, type-safe applications with GPT models.

## 🚀 Features

- **Full OpenAI Support**: Works with GPT-4, GPT-3.5-turbo, and other OpenAI models
- **Type-Safe Integration**: Fully compatible with TS-DSPy signatures and modules
- **Usage Tracking**: Built-in token usage and cost tracking
- **Streaming Support**: Stream responses for real-time applications
- **Error Handling**: Robust error handling with retry mechanisms
- **Flexible Configuration**: Support for all OpenAI parameters

## 📦 Installation

```bash
npm install @ts-dspy/openai @ts-dspy/core

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

Get your OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys).

```typescript
import { OpenAILM } from '@ts-dspy/openai';

const lm = new OpenAILM({
  apiKey: process.env.OPENAI_API_KEY, // Your OpenAI API key
  model: 'gpt-4', // or 'gpt-3.5-turbo', 'gpt-4-turbo', etc.
});
```

## 🎯 Quick Start

### Basic Usage

```typescript
import { Signature, InputField, OutputField, Predict, configure } from '@ts-dspy/core';
import { OpenAILM } from '@ts-dspy/openai';

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

// Setup OpenAI model
configure({
    lm: new OpenAILM({
        apiKey: process.env.OPENAI_API_KEY,
        model: 'gpt-4'
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
import { OpenAILM } from '@ts-dspy/openai';

// Configure globally
configure({
  lm: new OpenAILM({
    apiKey: process.env.OPENAI_API_KEY,
    model: 'gpt-4-turbo'
  })
});

// Now you can use modules without passing the language model
const predictor = new Predict("question -> answer");
```

## ⚙️ Configuration Options

### Model Configuration

```typescript
const lm = new OpenAILM({
  // Required
  apiKey: 'your-api-key',
  
  // Model selection
  model: 'gpt-4', // gpt-4, gpt-4-turbo, gpt-3.5-turbo, etc.
  
  // Generation parameters
  temperature: 0.7, // 0-2, controls randomness
  maxTokens: 1000, // Maximum tokens to generate
  topP: 1.0, // Nucleus sampling parameter
  frequencyPenalty: 0, // -2 to 2, penalize frequent tokens
  presencePenalty: 0, // -2 to 2, penalize present tokens
  
  // Advanced options
  timeout: 30000, // Request timeout in milliseconds
  maxRetries: 3, // Number of retry attempts
  organization: 'org-id', // OpenAI organization ID (optional)
  
  // Custom base URL (for proxies, etc.)
  baseURL: 'https://api.openai.com/v1'
});
```

### Supported Models

- **GPT-4**: `gpt-4`, `gpt-4-turbo`, `gpt-4-turbo-preview`
- **GPT-3.5**: `gpt-3.5-turbo`, `gpt-3.5-turbo-16k`
- **Legacy**: `gpt-4-0613`, `gpt-3.5-turbo-0613`, etc.

## 📊 Usage Tracking

Track token usage and costs:

```typescript
const lm = new OpenAILM({
  apiKey: process.env.OPENAI_API_KEY,
  model: 'gpt-4'
});

// Make some predictions
const predictor = new Predict("question -> answer", lm);
await predictor.forward({ question: "What is AI?" });

// Get usage statistics
const usage = lm.getUsage();
console.log(`Tokens: ${usage.totalTokens}`);
console.log(`Cost: $${usage.totalCost}`);
console.log(`Requests: ${usage.requestCount}`);
```

## 🔄 Streaming Responses

For real-time applications:

```typescript
const lm = new OpenAILM({
  apiKey: process.env.OPENAI_API_KEY,
  model: 'gpt-4',
  stream: true
});

// Stream responses
const stream = await lm.generateStream("Tell me a story about AI");

for await (const chunk of stream) {
  process.stdout.write(chunk);
}
```

## 🛡️ Error Handling

The OpenAI integration includes robust error handling:

```typescript
import { OpenAIError, RateLimitError, AuthenticationError } from '@ts-dspy/openai';

try {
  const result = await predictor.forward({ question: "What is life?" });
} catch (error) {
  if (error instanceof RateLimitError) {
    console.log('Rate limit hit, retrying...');
    // Handle rate limiting
  } else if (error instanceof AuthenticationError) {
    console.log('Invalid API key');
    // Handle auth issues
  } else if (error instanceof OpenAIError) {
    console.log('OpenAI API error:', error.message);
    // Handle other OpenAI errors
  }
}
```

## 🎨 Advanced Usage

### ReAct Pattern with Tools

```typescript
import { RespAct } from '@ts-dspy/core';

const agent = new RespAct("question -> answer", {
    tools: {
        calculate: {
            description: "Performs mathematical calculations. Use for any arithmetic operations.",
            function: (expr: string) => eval(expr)
        },
        search: {
            description: "Searches for information online. Use when you need current or factual information.",
            function: async (query: string) => await searchWeb(query)
        }
    },
    maxSteps: 5
});

const result = await agent.forward({
    question: "What's the square root of 144 plus the current population of Tokyo?"
});
```

### Chain of Thought Reasoning

```typescript
import { ChainOfThought } from '@ts-dspy/core';

const reasoner = new ChainOfThought("problem -> solution: int");
const result = await reasoner.forward({
    problem: "If I have 3 apples and buy 5 more, then eat 2, how many do I have?"
});

console.log(result.reasoning); // Step-by-step explanation
console.log(result.solution);  // 6
```

### Custom System Messages

```typescript
class CustomSignature extends Signature {
  static description = "You are a helpful AI assistant specialized in code review";
  
  @InputField({ description: "Code to review" })
  code!: string;
  
  @OutputField({ description: "Review feedback" })
  feedback!: string;
}
```

### Function Calling (Tools)

```typescript
const lm = new OpenAILM({
  apiKey: process.env.OPENAI_API_KEY,
  model: 'gpt-4',
  tools: [
    {
      type: 'function',
      function: {
        name: 'get_weather',
        description: 'Get current weather',
        parameters: {
          type: 'object',
          properties: {
            location: { type: 'string' }
          }
        }
      }
    }
  ]
});
```

### Multiple Models

```typescript
const gpt4 = new OpenAILM({
  apiKey: process.env.OPENAI_API_KEY,
  model: 'gpt-4'
});

const gpt35 = new OpenAILM({
  apiKey: process.env.OPENAI_API_KEY,
  model: 'gpt-3.5-turbo'
});

// Use different models for different tasks
const complexReasoning = new Predict(ComplexSignature, gpt4);
const simpleTask = new Predict(SimpleSignature, gpt35);
```

## 🔧 Environment Variables

Create a `.env` file:

```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_ORG_ID=your_org_id_here  # Optional
```

## 💰 Cost Optimization

Tips for managing costs:

```typescript
// Use cheaper models for simple tasks
const simpleLM = new OpenAILM({
  apiKey: process.env.OPENAI_API_KEY,
  model: 'gpt-3.5-turbo', // Cheaper than GPT-4
  maxTokens: 100 // Limit response length
});

// Use higher temperature for creative tasks, lower for factual
const creativeLM = new OpenAILM({
  apiKey: process.env.OPENAI_API_KEY,
  model: 'gpt-4',
  temperature: 0.9 // More creative
});

const factualLM = new OpenAILM({
  apiKey: process.env.OPENAI_API_KEY,
  model: 'gpt-4',
  temperature: 0.1 // More deterministic
});
```

## 🧪 Testing

```typescript
import { MockOpenAILM } from '@ts-dspy/openai/testing';

// Use mock for testing
const mockLM = new MockOpenAILM();
mockLM.setResponse("Mocked response");

const predictor = new Predict("question -> answer", mockLM);
const result = await predictor.forward({ question: "Test?" });
console.log(result.answer); // "Mocked response"
```

## 📚 API Reference

### OpenAILM

Main class for OpenAI integration.

**Constructor Options:**
- `apiKey`: OpenAI API key (required)
- `model`: Model name (required)
- `temperature`: Sampling temperature (0-2)
- `maxTokens`: Maximum tokens to generate
- `topP`: Nucleus sampling parameter
- `frequencyPenalty`: Frequency penalty (-2 to 2)
- `presencePenalty`: Presence penalty (-2 to 2)
- `timeout`: Request timeout in ms
- `maxRetries`: Number of retry attempts

**Methods:**
- `generate(prompt: string): Promise<string>` - Generate text
- `generateStream(prompt: string): AsyncIterable<string>` - Stream generation
- `getUsage(): UsageStats` - Get usage statistics

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](https://github.com/your-org/ts-dspy) for details.

## 📄 License

MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Packages

- [`@ts-dspy/core`](https://www.npmjs.com/package/@ts-dspy/core) - Core TS-DSPy library
- More integrations coming soon!

## 📖 Learn More

- [TS-DSPy Documentation](https://ts-dspy.dev)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Examples](https://github.com/your-org/ts-dspy/tree/main/examples) 