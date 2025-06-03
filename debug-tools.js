// Simulating the buildInitialPrompt method from RespAct
function buildInitialPrompt(tools, inputs) {
    const toolNames = Object.keys(tools).join(', ');

    return `You have access to the following tools: ${toolNames}

Question: ${inputs.question || JSON.stringify(inputs)}

Use this format:
Thought: [your reasoning about what to do next]
Action: [tool name]
Action Input: [input to the tool]
Observation: [result from tool]
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: [final reasoning]
Final Answer: [your final answer]

Begin!`;
}

// Example tools
const tools = {
    calculate: (expression) => eval(expression),
    search: (query) => `Mock results for: ${query}`,
    getWeather: (location) => `Weather in ${location}: 72°F, sunny`,
    convertCurrency: (amount, from, to) => `${amount} ${from} = ${amount * 1.1} ${to}`
};

// Example question
const inputs = { question: 'What is 10 + 5 * 2 and what is the weather in Paris?' };

console.log('=== PROMPT SENT TO LLM ===');
console.log(buildInitialPrompt(tools, inputs));

console.log('\n=== HOW IT WORKS ===');
console.log('1. Tool names are extracted from the tools object and listed in the prompt');
console.log('2. The LLM is given a specific format to follow (Thought/Action/Action Input/Observation)');
console.log('3. The LLM chooses which tool to use based on the question and available tools');
console.log('4. RespAct extracts the tool call and executes it, then feeds the result back');

console.log('\n=== EXAMPLE LLM RESPONSE ===');
console.log(`Thought: I need to calculate 10 + 5 * 2 first, then get weather information.
Action: calculate
Action Input: 10 + 5 * 2
Observation: 20
Thought: Now I need to get the weather in Paris.
Action: getWeather
Action Input: Paris
Observation: Weather in Paris: 72°F, sunny
Thought: I have both pieces of information needed.
Final Answer: 10 + 5 * 2 = 20, and the weather in Paris is 72°F and sunny.`); 