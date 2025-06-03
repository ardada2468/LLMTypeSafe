# @ts-dspy/core

[![npm version](https://badge.fury.io/js/@ts-dspy%2Fcore.svg)](https://badge.fury.io/js/@ts-dspy%2Fcore)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Core library for building type-safe LLM applications with structured input/output signatures, automatic validation, and reasoning patterns.**

TS-DSPy brings the power of [DSPy](https://github.com/stanfordnlp/dspy) to TypeScript, enabling you to build robust, composable, and type-safe LLM applications with structured prompting and automatic optimization.

## 🚀 Features

- **Type-Safe Signatures**: Define structured input/output schemas with TypeScript decorators
- **Automatic Validation**: Built-in validation for LLM inputs and outputs
- **Modular Architecture**: Composable modules for complex reasoning patterns
- **Multiple LLM Support**: Works with any language model through a unified interface
- **ReAct Pattern**: Built-in Reasoning and Acting with intelligent tool integration
- **Enhanced Tool Descriptions**: Provide detailed tool descriptions for better AI decision-making

## 📦 Installation

```bash
npm install @ts-dspy/core
```

For OpenAI integration:
```bash
npm install @ts-dspy/core @ts-dspy/openai
```

## 🎯 Quick Start

### 1. Define a Signature

```typescript
import { Signature, InputField, OutputField } from '@ts-dspy/core';

class QuestionAnswering extends Signature {
  static description = "Answer questions based on given context";

  @InputField({ description: "The question to answer" })
  question!: string;

  @InputField({ description: "Context information" })
  context!: string;

  @OutputField({ description: "The answer to the question" })
  answer!: string;

  @OutputField({ description: "Confidence score", type: "number" })
  confidence!: number;
}
```

### 2. Use with a Module

```typescript
import { Predict, configure } from '@ts-dspy/core';
import { OpenAILM } from '@ts-dspy/openai';

// Configure your language model
configure({
    lm: new OpenAILM({ 
        apiKey: process.env.OPENAI_API_KEY,
        model: 'gpt-4'
    })
});

// Create a prediction module
const qa = new Predict(QuestionAnswering);

// Use it
const result = await qa.forward({
  question: "What is the capital of France?",
  context: "France is a country in Europe. Its capital city is Paris."
});

console.log(result.answer); // "Paris"
console.log(result.confidence); // 0.95
```

### 3. String-based Signatures (Alternative)

```typescript
import { Predict } from '@ts-dspy/core';

const summarizer = new Predict("text -> summary");

const result = await summarizer.forward({
  text: "Long text to summarize..."
});

console.log(result.summary);
```

## 🏗️ Core Concepts

### Signatures

Signatures define the structure of your LLM interactions:

```typescript
class MySignature extends Signature {
  @InputField({ description: "Input description", required: true })
  input!: string;

  @OutputField({ description: "Output description", type: "string" })
  output!: string;
}
```

**Field Options:**
- `description`: Human-readable description for the LLM
- `type`: TypeScript type (`string`, `number`, `boolean`, etc.)
- `required`: Whether the field is required (default: `true`)
- `prefix`: Custom prefix for prompts

### Modules

Modules are the building blocks of TS-DSPy applications:

#### Predict: Basic LLM Prediction
```typescript
const predictor = new Predict("context, question -> answer");
const result = await predictor.forward({
    context: "The sky is blue because of Rayleigh scattering.",
    question: "Why is the sky blue?"
});
```

#### ChainOfThought: Step-by-Step Reasoning
```typescript
import { ChainOfThought } from '@ts-dspy/core';

const reasoner = new ChainOfThought("problem -> solution: int");
const result = await reasoner.forward({
    problem: "If I have 3 apples and buy 5 more, then eat 2, how many do I have?"
});

console.log(result.reasoning); // "First I have 3 apples..."
console.log(result.solution);  // 6
```

#### RespAct: Tool-Using Agents
```typescript
import { RespAct } from '@ts-dspy/core';

const agent = new RespAct("question -> answer", {
    tools: {
        calculate: {
            description: "Performs mathematical calculations",
            function: (expr: string) => eval(expr)
        },
        search: {
            description: "Searches for information online",
            function: async (query: string) => await searchWeb(query)
        }
    },
    maxSteps: 5
});
```

### Examples and Few-Shot Learning

```typescript
import { Example } from '@ts-dspy/core';

const examples = [
  new Example({
    question: "What is 2+2?",
    answer: "4",
    explanation: "Simple addition"
  }).withInputs("question"),
  
  new Example({
    question: "What is the capital of Spain?",
    answer: "Madrid",
    explanation: "Madrid is the capital and largest city of Spain"
  }).withInputs("question")
];

// Use examples in your modules
const predictor = new Predict(MySignature);
// Examples can be used for few-shot prompting or optimization
```

## 🛠️ Enhanced Tool Integration

TS-DSPy features an advanced tool system with intelligent descriptions:

```typescript
const financialAgent = new RespAct("question -> answer", {
    tools: {
        fetchStockPrice: {
            description: "Retrieves current stock price for a ticker symbol (e.g., AAPL, GOOGL). Returns price in USD. Use when you need current market data.",
            function: async (symbol: string) => {
                const response = await fetch(`/api/stocks/${symbol}`);
                return response.json();
            }
        },
        
        calculatePortfolioValue: {
            description: "Calculates total portfolio value given holdings. Takes array of {symbol, shares} objects. Use for portfolio analysis.",
            function: (holdings: Array<{symbol: string, shares: number}>) => {
                return holdings.reduce((total, holding) => 
                    total + (getStockPrice(holding.symbol) * holding.shares), 0
                );
            }
        }
    }
});
```

## 🔧 Configuration

```typescript
import { configure } from '@ts-dspy/core';
import { OpenAILM } from '@ts-dspy/openai';

// Global configuration
configure({
  lm: new OpenAILM({ 
    apiKey: process.env.OPENAI_API_KEY,
    model: 'gpt-4'
  }),
  cache: true,
  tracing: true
});
```

## 🎨 Advanced Usage

### Custom Language Models

Implement the `ILanguageModel` interface:

```typescript
import { ILanguageModel, UsageStats } from '@ts-dspy/core';

class MyCustomLM implements ILanguageModel {
  async generate(prompt: string): Promise<string> {
    // Your implementation
  }

  async getUsageStats(): Promise<UsageStats> {
    // Return usage statistics
  }
}
```

### Chaining Modules

```typescript
const step1 = new Predict("question -> research_query");
const step2 = new Predict("research_query -> answer");

// Chain the operations
const query = await step1.forward({ question: "How does photosynthesis work?" });
const answer = await step2.forward({ research_query: query.research_query });
```

### Usage Tracking & Cost Monitoring

```typescript
import { OpenAILM } from '@ts-dspy/openai';

const lm = new OpenAILM({ apiKey: process.env.OPENAI_API_KEY });
const module = new Predict("question -> answer", lm);

await module.forward({ question: "Hello world!" });

// Get detailed usage statistics
const usage = lm.getUsage();
console.log(`Tokens: ${usage.totalTokens}`);
console.log(`Cost: $${usage.totalCost}`);
console.log(`Requests: ${usage.requestCount}`);
```

## 📚 API Reference

### Core Classes

- **`Signature`**: Base class for defining input/output schemas
- **`Module`**: Base class for all TS-DSPy modules
- **`Predict`**: Basic prediction module
- **`ChainOfThought`**: Reasoning module with step-by-step thinking
- **`RespAct`**: Tool-using agent module
- **`Example`**: Data structure for examples and demonstrations
- **`Prediction`**: Wrapper for module outputs

### Decorators

- **`@InputField(options)`**: Marks a field as input
- **`@OutputField(options)`**: Marks a field as output

### Functions

- **`configure(options)`**: Global configuration
- **`buildPrompt(signature, inputs)`**: Build prompts from signatures
- **`parseOutput(signature, output)`**: Parse LLM outputs

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](https://github.com/your-org/ts-dspy) for details.

## 📄 License

MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Packages

- [`@ts-dspy/openai`](https://www.npmjs.com/package/@ts-dspy/openai) - OpenAI integration
- More integrations coming soon!

## 📖 Learn More

- [Documentation](https://ts-dspy.dev)
- [Examples](https://github.com/your-org/ts-dspy/tree/main/examples)
- [DSPy Paper](https://arxiv.org/abs/2310.03714) - Original research 