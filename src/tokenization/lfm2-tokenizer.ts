import { Tokenizer } from "@huggingface/tokenizers";
import { fetchJSON } from "../runtime/hub.js";

export interface Message {
    role: "system" | "user" | "assistant";
    content: string;
}

/**
 * LFM2 tokenizer.
 *
 * Uses the same @huggingface/tokenizers backend as CLIPTokenizer.
 * Chat formatting mirrors the model's Jinja2 template:
 *   <|im_start|>role\ncontent<|im_end|>\n
 * with a trailing <|im_start|>assistant\n to prime generation.
 */
export class LFM2Tokenizer {
    private constructor(private readonly tokenizer: Tokenizer) {}

    static async fromHub(modelId: string): Promise<LFM2Tokenizer> {
        const [tokenizerJson, tokenizerConfig] = await Promise.all([
            fetchJSON<object>(modelId, "tokenizer.json"),
            fetchJSON<object>(modelId, "tokenizer_config.json"),
        ]);
        return new LFM2Tokenizer(new Tokenizer(tokenizerJson, tokenizerConfig));
    }

    /** Apply chat template and encode to token ids. */
    encodeChat(messages: Message[], tools?: object[]): number[] {
        const text = this.applyChatTemplate(messages, tools);
        const enc = this.tokenizer.encode(text, { add_special_tokens: false });
        return enc.ids;
    }

    /** Decode token ids back to a string. */
    decode(ids: number[]): string {
        return this.tokenizer.decode(ids, { skip_special_tokens: true });
    }

    /**
     * Mirrors the model's Jinja2 chat template:
     * - Extracts system message if first
     * - Injects tools as "List of tools: [...]" appended to system prompt
     * - Formats: <|im_start|>role\ncontent<|im_end|>\n…<|im_start|>assistant\n
     */
    private applyChatTemplate(messages: Message[], tools?: object[]): string {
        const msgs = [...messages];
        let systemPrompt = "";

        const first = msgs[0];
        if (first?.role === "system") {
            systemPrompt = first.content;
            msgs.shift();
        }

        if (tools && tools.length > 0) {
            const toolsStr = `List of tools: [${tools.map(t => JSON.stringify(t)).join(", ")}]`;
            systemPrompt = systemPrompt ? `${systemPrompt}\n${toolsStr}` : toolsStr;
        }

        let out = "";
        if (systemPrompt) {
            out += `<|im_start|>system\n${systemPrompt}<|im_end|>\n`;
        }
        for (const { role, content } of msgs) {
            out += `<|im_start|>${role}\n${content}<|im_end|>\n`;
        }
        out += "<|im_start|>assistant\n";
        return out;
    }
}
