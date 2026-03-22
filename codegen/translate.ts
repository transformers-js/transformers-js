import OpenAI from "openai";
import { readFileSync } from "node:fs";
import { join } from "node:path";

// Uses GitHub Models API (models.github.ai) so only GITHUB_TOKEN is needed —
// no separate ANTHROPIC_API_KEY. GITHUB_TOKEN is auto-provisioned in Actions.
const client = new OpenAI({
    baseURL: "https://models.github.ai/inference",
    apiKey: process.env.GITHUB_TOKEN ?? process.env.GH_TOKEN ?? "",
});

const SYSTEM_PROMPT = readFileSync(
    join(import.meta.dirname, "prompts/image-processor.md"),
    "utf-8",
);

export async function translate(pythonSource: string, sourcePath: string): Promise<string> {
    const response = await client.chat.completions.create({
        model: "claude-opus-4-6",
        max_tokens: 8192,
        messages: [
            { role: "system", content: SYSTEM_PROMPT },
            {
                role: "user",
                content: `Translate this file to TypeScript.\n\nSource path: ${sourcePath}\n\n\`\`\`python\n${pythonSource}\n\`\`\``,
            },
        ],
    });

    const text = response.choices[0]?.message?.content ?? "";
    const match = text.match(/```typescript\n([\s\S]+?)\n```/);
    if (match?.[1]) return match[1];
    return text;
}
