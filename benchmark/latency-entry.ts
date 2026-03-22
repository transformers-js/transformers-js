// Minimal browser entry point for the latency benchmark.
// Excludes tokenizer (not needed for ViT) to avoid @huggingface/tokenizers dependency.
export { initRuntime } from "../src/runtime/index.js";
export { ViTForImageClassification } from "../src/models/vit.js";
export type { ImageData } from "../src/preprocessing/ops.js";
