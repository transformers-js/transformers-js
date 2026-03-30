import Anthropic from "@anthropic-ai/sdk";
import { readFileSync } from "node:fs";
import { join } from "node:path";

// Local-use only. The CI sync now runs via gh-aw (see sync-preprocessing.md).
// Requires ANTHROPIC_API_KEY in the environment.
const client = new Anthropic();

const SYSTEM_PROMPT = readFileSync(
    join(import.meta.dirname, "prompts/image-processor.md"),
    "utf-8",
);

export async function translate(pythonSource: string, sourcePath: string): Promise<string> {
    const response = await client.messages.create({
        model: "claude-opus-4-6",
        max_tokens: 8192,
        system: SYSTEM_PROMPT,
        messages: [
            {
                role: "user",
                content: `Translate this file to TypeScript.\n\nSource path: ${sourcePath}\n\n\`\`\`python\n${pythonSource}\n\`\`\``,
            },
        ],
    });

    const text = response.content[0]?.type === "text" ? response.content[0].text : "";
    const match = text.match(/```typescript\n([\s\S]+?)\n```/);
    if (match?.[1]) return match[1];
    return text;
}
