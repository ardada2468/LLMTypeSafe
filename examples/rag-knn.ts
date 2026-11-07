/**
 * Retrieval-augmented generation, with demos picked per question.
 *
 * - `InMemoryVectorStore` indexes a handful of documents with an embedder.
 * - `Retrieve` fetches the passages nearest a query.
 * - `KNNFewShot` shows the model the training examples most similar to each
 *   incoming question, instead of one fixed set of demos.
 *
 *   export OPENAI_API_KEY="sk-..."
 *   npm run example:rag
 */
import {
    ChainOfThought,
    Example,
    InMemoryVectorStore,
    KNNFewShot,
    Module,
    Retrieve,
    configure,
    type Prediction,
} from '@ts-dspy/core';
import { OpenAIEmbedder, OpenAILM } from '@ts-dspy/openai';
import { requireEnv, section } from './utils';

const DOCUMENTS = [
    'Our standard plan costs $12 per user per month, billed annually.',
    'Refunds are available within 30 days of purchase for annual plans.',
    'Single sign-on (SAML and OIDC) is included in the Enterprise plan only.',
    'Data is stored in the EU (Frankfurt) or the US (Oregon), chosen at signup.',
    'The API is rate-limited to 600 requests per minute per workspace.',
    'Support replies within one business day; Enterprise has a 4-hour SLA.',
];

/** Retrieve passages, then answer from them. */
class SupportRAG extends Module {
    retrieve = new Retrieve({ k: 2 });
    answer = new ChainOfThought('context: string[], question -> answer');

    async forward(inputs: { question: string }): Promise<Prediction & Record<string, any>> {
        const { passages } = await this.retrieve.forward({ query: inputs.question });
        return this.answer.forward({
            context: passages.map((passage) => passage.text),
            question: inputs.question,
        });
    }
}

// Labelled examples. KNNFewShot embeds their inputs once, then picks the
// nearest ones for every call.
const trainset = [
    ['How much is the standard plan?', '$12 per user per month, billed annually.'],
    ['Can I get my money back?', 'Yes, within 30 days of purchase on annual plans.'],
    ['Do you support SAML?', 'Yes, on the Enterprise plan.'],
    ['Where is my data kept?', 'In the EU (Frankfurt) or the US (Oregon).'],
    ['How fast does support answer?', 'Within one business day; 4 hours on Enterprise.'],
].map(([question, answer]) => new Example({ question, answer }).withInputs('question'));

async function main(): Promise<void> {
    const apiKey = requireEnv('OPENAI_API_KEY');
    const embedder = new OpenAIEmbedder({ apiKey });

    section('Index');
    const store = new InMemoryVectorStore(embedder);
    await store.addAll(DOCUMENTS);
    console.log(`indexed ${store.size} documents`);

    // `retriever` is a setting like `lm`: every Retrieve without its own uses it.
    configure({ lm: new OpenAILM({ apiKey }), retriever: store });

    section('Retrieve');
    const { passages } = await new Retrieve({ k: 2 }).forward({ query: 'single sign-on' });
    for (const passage of passages) {
        console.log(`  ${passage.score?.toFixed(3)}  ${passage.text}`);
    }

    section('RAG with KNN demos');
    // The returned program wraps a copy of the student; each call embeds the
    // question and uses the 2 nearest training examples as demos for every
    // predictor in the program.
    const program = new KNNFewShot({ k: 2, embedder, trainset }).compile(new SupportRAG());

    const question = 'Is SSO available on the standard plan?';
    const neighbours = await program.neighbours({ question });
    console.log('demos chosen for this question:');
    for (const { example, similarity } of neighbours) {
        console.log(`  ${similarity.toFixed(3)}  ${example.get('question')}`);
    }

    const result = await program.forward({ question });
    console.log(`\nanswer:    ${result.get('answer')}`);
    console.log(`reasoning: ${String(result.get('reasoning')).slice(0, 200)}...`);
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});
