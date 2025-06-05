# TS-DSPy: TypeScript-First LLM Framework

<div align="center">

[![npm version](https://badge.fury.io/js/@ts-dspy%2Fcore.svg)](https://badge.fury.io/js/@ts-dspy%2Fcore)
[![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=flat&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Node.js](https://img.shields.io/badge/Node.js-18%2B-green)](https://nodejs.org/)

**A production-ready TypeScript framework for building reliable, type-safe LLM applications with structured outputs, reasoning patterns, and intelligent tool integration.**

[Quick Start](#-quick-start) • [Examples](#-examples) • [API Reference](#-api-reference) • [Contributing](#-contributing)

</div>

---

## 🌟 Why TS-DSPy?

TS-DSPy brings the powerful paradigms of [Stanford's DSPy framework](https://github.com/stanfordnlp/dspy) to the TypeScript ecosystem with full type safety, modern tooling, and production-grade features. Whether you're building AI chatbots, data processing pipelines, or intelligent agents, TS-DSPy provides the abstractions you need.


### ✨ Key Features

- **🔒 Type-Safe Signatures**: Define input/output schemas with automatic validation and TypeScript inference
- **🧠 ReAct Pattern**: Built-in Reasoning and Acting with intelligent tool integration
- **🛠️ Enhanced Tool Descriptions**: Provide detailed tool descriptions for better AI decision-making
- **🔌 Multiple LLM Support**: Currently supports OpenAI with extensible architecture for other providers
- **⚡ Automatic Parsing**: Convert raw LLM outputs to structured TypeScript objects
- **🛡️ Robust Error Handling**: Comprehensive validation with automatic retries and fallbacks
- **📊 Usage Tracking**: Built-in token usage and cost monitoring
- **🎯 Zero Config**: Works out of the box with minimal setup
- **📦 Modular Design**: Pluggable architecture for LLM providers and custom modules

---

## 📦 Installation

```bash
# Core framework
npm install @ts-dspy/core

# OpenAI integration
npm install @ts-dspy/openai
```

**Requirements:**
- Node.js 18+
- TypeScript 5.0+

---

## 🚀 Quick Start

Look ./examples/basic-usage.ts to test this package as well as the examples below

### Basic Prediction

```typescript
import { configure, Predict } from '@ts-dspy/core';
import { OpenAILM } from '@ts-dspy/openai';

// Configure your LLM provider
configure({
    lm: new OpenAILM({ 
        apiKey: process.env.OPENAI_API_KEY,
        model: 'gpt-4'
    })
});

// Simple question-answering
const qa = new Predict("question -> answer");
const result = await qa.forward({ 
    question: "What is the capital of France?" 
});

console.log(result.answer); // "Paris"
```

### Type-Safe Signatures

```typescript
import { Signature, InputField, OutputField } from '@ts-dspy/core';

class SentimentAnalysis extends Signature {
    @InputField({ description: "Text to analyze for sentiment" })
    text!: string;

    @OutputField({ description: "Sentiment classification" })
    sentiment!: 'positive' | 'negative' | 'neutral';

    @OutputField({ description: "Confidence score between 0 and 1" })
    confidence!: number;

    static description = "Analyze the sentiment of the given text";
}

const classifier = new Predict(SentimentAnalysis);
const result = await classifier.forward({ 
    text: "I love this framework!" 
});

// Full TypeScript autocomplete and type safety
console.log(result.sentiment);   // Type: 'positive' | 'negative' | 'neutral'
console.log(result.confidence);  // Type: number
```

---

## 🎯 Core Concepts

### 1. Signatures: Define Your Interface

Signatures are the foundation of TS-DSPy, defining the input/output structure for your LLM interactions.

#### String Signatures (Quick & Simple)
```typescript
// Basic signature
const qa = new Predict("question -> answer");

// Multi-output with types
const analyzer = new Predict("text -> sentiment: string, score: float, summary");
```

#### Class-Based Signatures (Type-Safe & Structured)
```typescript
class DataExtraction extends Signature {
    @InputField({ description: "Raw text to extract data from" })
    text!: string;

    @OutputField({ description: "Extracted person names" })
    names!: string[];

    @OutputField({ description: "Extracted dates in ISO format" })
    dates!: string[];

    @OutputField({ description: "Confidence level" })
    confidence!: number;
}
```

### 2. Modules: Pre-Built Reasoning Patterns

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
const reasoner = new ChainOfThought("problem -> solution: int");
const result = await reasoner.forward({
    problem: "If I have 3 apples and buy 5 more, then eat 2, how many do I have?"
});

console.log(result.reasoning); // "First I have 3 apples..."
console.log(result.solution);  // 6
```

#### RespAct: Tool-Using Agents
```typescript
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

---

## 🛠️ Enhanced Tool Integration

TS-DSPy features an advanced tool system with intelligent descriptions that help LLMs make better decisions about when and how to use tools.

### Enhanced Tool Format

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
        },

        convertCurrency: {
            description: "Converts amounts between currencies using live rates. Params: amount (number), from (currency code), to (currency code).",
            function: (amount: number, from: string, to: string) => {
                return convertCurrency(amount, from, to);
            }
        }
    }
});

// The LLM now has context about when and how to use each tool
const result = await financialAgent.forward({
    question: "I own 100 AAPL shares and 50 TSLA shares. What's my portfolio worth in EUR?"
});
```

### Benefits of Tool Descriptions

- **🎯 Better Tool Selection**: LLMs make more intelligent decisions about which tools to use
- **📝 Self-Documenting**: Your code becomes more readable and maintainable  
- **🔧 Improved Parameters**: Descriptions guide proper parameter formatting
- **⚡ Reduced Errors**: Clear descriptions prevent tool misuse
- **🔄 Backward Compatible**: Legacy function-only tools still work

---

## 📊 Advanced Features

### Usage Tracking & Cost Monitoring

```typescript
const lm = new OpenAILM({ apiKey: process.env.OPENAI_API_KEY });
const module = new Predict("question -> answer", lm);

await module.forward({ question: "Hello world!" });

// Get detailed usage statistics
const usage = lm.getUsage();
console.log(`Tokens: ${usage.totalTokens}`);
console.log(`Cost: $${usage.totalCost}`);
console.log(`Requests: ${usage.requestCount}`);
```

### Configuration & Tracing

```typescript
import { configure, getTraceHistory } from '@ts-dspy/core';

configure({
    lm: new OpenAILM({ apiKey: process.env.OPENAI_API_KEY }),
    cache: true,        // Enable response caching
    tracing: true,      // Record all interactions
    maxRetries: 3,      // Auto-retry failed requests
    timeout: 30000      // Request timeout in ms
});

// After running modules, analyze performance
const traces = getTraceHistory();
traces.forEach(trace => {
    console.log(`Module: ${trace.moduleId}`);
    console.log(`Duration: ${trace.duration}ms`);
    console.log(`Tokens: ${trace.usage.totalTokens}`);
    console.log(`Cost: $${trace.usage.totalCost}`);
});
```

### Multiple LLM Providers

```typescript
const fastLM = new OpenAILM({ 
    apiKey: process.env.OPENAI_API_KEY,
    model: 'gpt-3.5-turbo'  // Fast for simple tasks
});

const smartLM = new OpenAILM({ 
    apiKey: process.env.OPENAI_API_KEY,
    model: 'gpt-4'  // Powerful for complex reasoning
});

// Use different LLMs for different tasks
const quickQA = new Predict("question -> answer", fastLM);
const complexReasoner = new ChainOfThought("problem -> solution", smartLM);
```

---

## 📚 Examples

### 1. Content Analysis Pipeline

```typescript
class ContentAnalysis extends Signature {
    @InputField({ description: "Article text to analyze" })
    article!: string;

    @OutputField({ description: "Main topics covered" })
    topics!: string[];

    @OutputField({ description: "Article sentiment" })
    sentiment!: 'positive' | 'negative' | 'neutral';

    @OutputField({ description: "Reading difficulty (1-10)" })
    difficulty!: number;

    @OutputField({ description: "Key takeaways" })
    takeaways!: string[];
}

const analyzer = new Predict(ContentAnalysis);
const result = await analyzer.forward({
    article: "Your article text here..."
});
```

### 2. Research Assistant Agent

```typescript
const researcher = new RespAct("research_query -> comprehensive_answer", {
    tools: {
        searchAcademic: {
            description: "Searches academic papers and journals. Use for scholarly research and citations.",
            function: async (query: string) => await searchScholar(query)
        },
        
        searchWeb: {
            description: "Searches general web content. Use for current events and general information.",
            function: async (query: string) => await searchGoogle(query)
        },
        
        summarizeText: {
            description: "Summarizes long text into key points. Use when you have lengthy content to process.",
            function: (text: string) => summarizeContent(text)
        }
    },
    maxSteps: 8
});

const result = await researcher.forward({
    research_query: "What are the latest developments in quantum computing algorithms?"
});
```

### 3. Data Processing Chain

```typescript
// Multi-step data processing with type safety
class DataProcessor extends Signature {
    @InputField({ description: "Raw CSV data string" })
    csvData!: string;

    @OutputField({ description: "Processed and cleaned data" })
    cleanedData!: Array<Record<string, any>>;

    @OutputField({ description: "Data quality issues found" })
    issues!: string[];

    @OutputField({ description: "Suggested improvements" })
    improvements!: string[];
}

const processor = new ChainOfThought(DataProcessor);
const result = await processor.forward({
    csvData: "name,age,email\nJohn,25,john@example.com\n..."
});
```

---

## 🏗️ Architecture

TS-DSPy follows a clean, modular architecture:

```
@ts-dspy/
├── core/                   # Core framework
│   ├── types/             # TypeScript interfaces & types
│   ├── core/              # Base classes (Signature, Module, Prediction)
│   ├── modules/           # Built-in modules (Predict, ChainOfThought, RespAct)
│   └── utils/             # Utilities (parsing, caching, validation)
│
├── openai/                # OpenAI integration
│   ├── models/            # OpenAI language model implementation
│   └── utils/             # OpenAI-specific utilities
│
└── [future providers]/    # Anthropic, Cohere, etc. (coming soon)
```

### Core Classes

- **`Signature`**: Abstract base for defining input/output schemas
- **`Module`**: Abstract base for all LLM modules  
- **`Prediction`**: Type-safe container for module outputs
- **`Example`**: Container for training/evaluation examples

### Decorators

- **`@InputField(config)`**: Mark class properties as input fields
- **`@OutputField(config)`**: Mark class properties as output fields

---

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
npm test

# Run with coverage
npm run test:coverage

# Test specific package
cd packages/core && npm test

# Run examples
npm run run:example:openai
```

---

## 🔧 Development

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/your-username/ts-dspy.git
cd ts-dspy

# Install dependencies
npm install

# Build all packages
npm run build

# Run tests
npm test

# Start development mode
npm run dev
```

### Creating Custom LLM Providers

Implement the `ILanguageModel` interface:

```typescript
import { ILanguageModel, LLMCallOptions, ChatMessage } from '@ts-dspy/core';

export class CustomLM implements ILanguageModel {
    async generate(prompt: string, options?: LLMCallOptions): Promise<string> {
        // Your implementation
    }
    
    async chat(messages: ChatMessage[], options?: LLMCallOptions): Promise<string> {
        // Your implementation  
    }
    
    getUsage() {
        // Return usage statistics
    }
}
```

### Contributing Custom Modules

Extend the `Module` base class:

```typescript
import { Module, Signature, Prediction } from '@ts-dspy/core';

export class CustomModule extends Module {
    constructor(signature: string | typeof Signature, lm?: ILanguageModel) {
        super(signature, lm);
    }

    async forward(inputs: Record<string, any>): Promise<Prediction> {
        // Your module implementation
    }
}
```

---

## 📋 API Reference

### Configuration

```typescript
// Global configuration
configure({
    lm: new OpenAILM({ apiKey: 'your-key' }),
    cache: boolean,
    tracing: boolean,
    maxRetries: number,
    timeout: number
});

// Get current configuration
const config = getDefaultLM();
```

### Core Modules

```typescript
// Basic prediction
const predict = new Predict("input -> output");

// Reasoning with chain of thought
const reasoner = new ChainOfThought("problem -> solution");

// Tool-using agent
const agent = new RespAct("question -> answer", { 
    tools: { /* your tools */ },
    maxSteps: 5 
});
```

### Utilities

```typescript
// Manual prompt building
const prompt = buildPrompt(signature, inputs, examples);

// Parse LLM output
const parsed = parseOutput(rawOutput, signature);
```

---

## 📦 Publishing Information

This package is published to NPM as a scoped monorepo:

- **Core Package**: `@ts-dspy/core`
- **OpenAI Integration**: `@ts-dspy/openai`
- **License**: MIT
- **Author**: Arnav Dadarya
- **Node.js**: 18+ required
- **TypeScript**: 5.0+ required

### Installation

```bash
# For most users (core + OpenAI)
npm install @ts-dspy/core @ts-dspy/openai

# Just the core framework
npm install @ts-dspy/core

# Specific version
npm install @ts-dspy/core@^0.1.0
```

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** with tests
4. **Run the test suite**: `npm test`
5. **Submit a pull request**

### Contribution Guidelines

- Follow TypeScript best practices
- Add tests for new features
- Update documentation as needed
- Ensure all CI checks pass
- Follow conventional commit format

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**Copyright (c) 2024 Arnav Dadarya**

---

## 🙏 Acknowledgments

- Inspired by [Stanford's DSPy framework](https://github.com/stanfordnlp/dspy)
- Built with ❤️ for the TypeScript community
- Thanks to all contributors and early adopters

---

## 🔗 Links & Resources

- **[Examples](./examples/)** - Comprehensive usage examples
- **[Issues](https://github.com/your-username/ts-dspy/issues)** - Bug reports and feature requests  
- **[Discussions](https://github.com/your-username/ts-dspy/discussions)** - Community discussions
- **[Changelog](https://github.com/your-username/ts-dspy/releases)** - Release notes

---

<div align="center">

**Star ⭐ the repo if TS-DSPy helped you build better LLM applications!**

Made with TypeScript • Powered by AI • Built for Developers

</div> 
