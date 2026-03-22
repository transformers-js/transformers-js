import { describe, it, expect } from "vitest";
import { CLIPTokenizer } from "./clip-tokenizer.js";
import { Tokenizer } from "@huggingface/tokenizers";

// Minimal tokenizer.json that produces deterministic ids for testing.
// Uses a ByteLevel BPE model with a tiny vocabulary.
const TOKENIZER_JSON = {
    version: "1.0",
    truncation: null,
    padding: null,
    added_tokens: [
        { id: 0, content: "<|startoftext|>", single_word: false, lstrip: false, rstrip: false, normalized: false, special: true },
        { id: 1, content: "<|endoftext|>",   single_word: false, lstrip: false, rstrip: false, normalized: false, special: true },
    ],
    normalizer: { type: "Lowercase" },
    pre_tokenizer: { type: "Whitespace" },
    post_processor: {
        type: "TemplateProcessing",
        single: [
            { SpecialToken: { id: "<|startoftext|>", type_id: 0 } },
            { Sequence: { id: "A", type_id: 0 } },
            { SpecialToken: { id: "<|endoftext|>", type_id: 0 } },
        ],
        pair: [],
        special_tokens: {
            "<|startoftext|>": { id: "<|startoftext|>", ids: [0], tokens: ["<|startoftext|>"] },
            "<|endoftext|>":   { id: "<|endoftext|>",   ids: [1], tokens: ["<|endoftext|>"]   },
        },
    },
    decoder: null,
    model: {
        type: "WordLevel",
        vocab: { "<|startoftext|>": 0, "<|endoftext|>": 1, "hello": 2, "world": 3, "[UNK]": 4 },
        unk_token: "[UNK]",
    },
};

const TOKENIZER_CONFIG = {
    tokenizer_class: "CLIPTokenizer",
    unk_token: "[UNK]",
    bos_token: "<|startoftext|>",
    eos_token: "<|endoftext|>",
};

function makeTokenizer(): CLIPTokenizer {
    const tok = new Tokenizer(TOKENIZER_JSON, TOKENIZER_CONFIG);
    return new (CLIPTokenizer as any)(tok);
}

describe("CLIPTokenizer.encode", () => {
    it("always returns length 77", () => {
        const tok = makeTokenizer();
        const { input_ids, attention_mask } = tok.encode("hello world");
        expect(input_ids.length).toBe(77);
        expect(attention_mask.length).toBe(77);
    });

    it("pads short sequences with zeros", () => {
        const tok = makeTokenizer();
        const { input_ids, attention_mask } = tok.encode("hello");
        // Tokens beyond the real sequence should be 0n
        const hasTrailingZeros = Array.from(input_ids).slice(-1)[0] === 0n;
        expect(hasTrailingZeros).toBe(true);
    });

    it("attention mask is 1 for real tokens, 0 for padding", () => {
        const tok = makeTokenizer();
        const { attention_mask } = tok.encode("hello");
        const ones = Array.from(attention_mask).filter((v) => v === 1n).length;
        const zeros = Array.from(attention_mask).filter((v) => v === 0n).length;
        expect(ones + zeros).toBe(77);
        expect(ones).toBeGreaterThan(0);
        expect(zeros).toBeGreaterThan(0);
    });

    it("returns BigInt64Arrays", () => {
        const tok = makeTokenizer();
        const { input_ids, attention_mask } = tok.encode("hello");
        expect(input_ids).toBeInstanceOf(BigInt64Array);
        expect(attention_mask).toBeInstanceOf(BigInt64Array);
    });

    it("encodeBatch produces [n, 77] flat tensors", () => {
        const tok = makeTokenizer();
        const texts = ["hello", "world", "hello world"];
        const { input_ids, attention_mask } = tok.encodeBatch(texts);
        expect(input_ids.length).toBe(texts.length * 77);
        expect(attention_mask.length).toBe(texts.length * 77);
    });
});
