import Anthropic from "@anthropic-ai/sdk";
import { readFileSync } from "node:fs";
import { join } from "node:path";

const client = new Anthropic();

const SYSTEM_PROMPT = readFileSync(
    join(import.meta.dirname, "prompts/image-processor.md"),
    "utf-8",
);

export async function translate(pythonSource: string, sourcePath: string): Promise<string> {
    const stream = client.messages.stream({
        model: "claude-opus-4-6",
        max_tokens: 8192,
        thinking: { type: "adaptive" },
        system: SYSTEM_PROMPT,
        messages: [
            {
                role: "user",
                content: `Translate this file to TypeScript.\n\nSource path: ${sourcePath}\n\n\`\`\`python\n${pythonSource}\n\`\`\``,
            },
        ],
    });

    const message = await stream.finalMessage();

    for (const block of message.content) {
        if (block.type === "text") {
            const match = block.text.match(/```typescript\n([\s\S]+?)\n```/);
            if (match?.[1]) return match[1];
            return block.text;
        }
    }

    throw new Error("No text content in translation response");
}
