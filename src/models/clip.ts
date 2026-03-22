import { fetchRaw } from "../runtime/hub.js";
import { ONNXSession } from "../runtime/session.js";
import { ImageProcessor } from "../preprocessing/image-processor.js";
import { CLIPTokenizer } from "../tokenization/clip-tokenizer.js";
import type { Device } from "../runtime/index.js";
import type { ImageData } from "../preprocessing/ops.js";

export class CLIPModel {
    private constructor(
        private readonly visionSession: ONNXSession,
        private readonly textSession: ONNXSession,
        readonly processor: ImageProcessor,
        readonly tokenizer: CLIPTokenizer,
    ) {}

    static async fromHub(modelId: string, device: Device = "webgpu"): Promise<CLIPModel> {
        const [visionBuffer, textBuffer, processor, tokenizer] = await Promise.all([
            fetchRaw(modelId, "onnx/vision_model.onnx"),
            fetchRaw(modelId, "onnx/text_model.onnx"),
            ImageProcessor.fromHub(modelId),
            CLIPTokenizer.fromHub(modelId),
        ]);

        const [visionSession, textSession] = await Promise.all([
            ONNXSession.load(visionBuffer, device),
            ONNXSession.load(textBuffer, device),
        ]);

        return new CLIPModel(visionSession, textSession, processor, tokenizer);
    }

    /** Returns a normalized L2 embedding of shape [hidden_size]. */
    async encodeImage(image: ImageData): Promise<Float32Array> {
        const { config } = this.processor;
        const pixelValues = await this.processor.preprocess(image);
        const dims = [1, 3, config.size.height, config.size.width] as const;

        const out = await this.visionSession.run({
            pixel_values: { data: pixelValues, dims },
        });

        return l2Normalize(out["pooler_output"] ?? out["last_hidden_state"]!);
    }

    /** Returns a normalized L2 embedding of shape [hidden_size]. */
    async encodeText(text: string): Promise<Float32Array> {
        const { input_ids, attention_mask } = this.tokenizer.encode(text);
        const seqLen = input_ids.length; // always MAX_LENGTH (77)
        const dims = [1, seqLen] as const;

        const out = await this.textSession.run({
            input_ids:      { data: input_ids,      dims },
            attention_mask: { data: attention_mask, dims },
        });

        return l2Normalize(out["pooler_output"] ?? out["last_hidden_state"]!);
    }

    dispose(): void {
        this.visionSession.dispose();
        this.textSession.dispose();
    }
}

/** L2-normalize a vector in-place, return it. */
export function l2Normalize(vec: Float32Array): Float32Array {
    let norm = 0;
    for (const v of vec) norm += v * v;
    norm = Math.sqrt(norm);
    if (norm === 0) return vec;
    for (let i = 0; i < vec.length; i++) vec[i]! / norm; // read-only: create new
    const out = new Float32Array(vec.length);
    for (let i = 0; i < vec.length; i++) out[i] = vec[i]! / norm;
    return out;
}

/** Dot product of two L2-normalized vectors = cosine similarity. */
export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
    let dot = 0;
    for (let i = 0; i < a.length; i++) dot += a[i]! * b[i]!;
    return dot;
}
