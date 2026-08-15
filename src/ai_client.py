// Senate AI - AI Client (Node.js)
// Uses Puter.js for AI calls - no API key needed

const { puter } = require('@heyputer/puter.js');

async function callAI(prompt, maxTokens = 500) {
    try {
        const response = await puter.ai.chat(prompt, {
            model: 'gpt-4o-mini',
            max_tokens: maxTokens,
            temperature: 0.7
        });
        
        // Puter returns different formats - handle both
        if (typeof response === 'string') {
            return response;
        } else if (response && response.message) {
            return response.message;
        } else if (response && response.text) {
            return response.text;
        } else if (response && response.content) {
            return response.content;
        }
        
        return null;
    } catch (e) {
        console.error(`AI call failed: ${e.message}`);
        return null;
    }
}

// Read prompt from stdin and output result to stdout
let input = '';
process.stdin.on('data', chunk => {
    input += chunk;
});

process.stdin.on('end', async () => {
    const result = await callAI(input);
    if (result) {
        process.stdout.write(result);
    } else {
        process.exit(1);
    }
});
