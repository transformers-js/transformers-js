import { Tokenizer } from "@huggingface/tokenizers";
import { fetchJSON } from "../runtime/hub.js";

// CLIP processes at most 77 tokens (BOS + up to 75 content tokens + EOS).
const MAX_LENGTH = 77;

export interface TokenizerOutput {
    input_ids: BigInt64Array;
    attention_mask: BigInt64Array;
}

export class CLIPTokenizer {
    private constructor(private readonly tokenizer: Tokenizer) {}

    static fromTokenizer(tokenizer: Tokenizer): CLIPTokenizer {
        return new CLIPTokenizer(tokenizer);
    }

    static async fromHub(modelId: string): Promise<CLIPTokenizer> {
        const [tokenizerJson, tokenizerConfig] = await Promise.all([
            fetchJSON<object>(modelId, "tokenizer.json"),
            fetchJSON<object>(modelId, "tokenizer_config.json"),
        ]);
        return new CLIPTokenizer(new Tokenizer(tokenizerJson, tokenizerConfig));
    }

    /** Encode a string to padded int64 tensors of length MAX_LENGTH (77). */
    encode(text: string): TokenizerOutput {
        const encoding = this.tokenizer.encode(text, { add_special_tokens: true });

        const ids = encoding.ids.slice(0, MAX_LENGTH);
        const mask = encoding.attention_mask.slice(0, MAX_LENGTH);

        // Pad to MAX_LENGTH with zeros
        while (ids.length < MAX_LENGTH) {
            ids.push(0);
            mask.push(0);
        }

        return {
            input_ids:     BigInt64Array.from(ids, BigInt),
            attention_mask: BigInt64Array.from(mask, BigInt),
        };
    }

    /** Encode multiple texts in one call. Returns flat tensors of shape [n, MAX_LENGTH]. */
    encodeBatch(texts: string[]): TokenizerOutput {
        const n = texts.length;
        const input_ids     = new BigInt64Array(n * MAX_LENGTH);
        const attention_mask = new BigInt64Array(n * MAX_LENGTH);

        for (let i = 0; i < n; i++) {
            const { input_ids: ids, attention_mask: mask } = this.encode(texts[i]!);
            input_ids.set(ids, i * MAX_LENGTH);
            attention_mask.set(mask, i * MAX_LENGTH);
        }

        return { input_ids, attention_mask };
    }
}
