import { ViTForImageClassification } from "../models/vit.js";
import type { Device } from "../runtime/index.js";
import type { ImageData } from "../preprocessing/ops.js";

export interface ClassificationResult {
    label: string;
    score: number;
}

export interface ImageClassificationOptions {
    device?: Device;
    quantized?: boolean;
    topK?: number;
}

export class ImageClassificationPipeline {
    private constructor(private readonly model: ViTForImageClassification) {}

    static async create(
        modelId: string,
        options: ImageClassificationOptions = {},
    ): Promise<ImageClassificationPipeline> {
        const model = await ViTForImageClassification.fromHub(modelId, options);
        return new ImageClassificationPipeline(model);
    }

    async run(image: ImageData, topK = 5): Promise<ClassificationResult[]> {
        const logits = await this.model.run(image);
        return topKSoftmax(logits, topK, (i) => this.model.label(i));
    }

    dispose(): void {
        this.model.dispose();
    }
}

// ── Math utils ─────────────────────────────────────────────────────────────

export function softmax(logits: Float32Array): Float32Array {
    let max = -Infinity;
    for (let i = 0; i < logits.length; i++) if (logits[i]! > max) max = logits[i]!;
    const exps = logits.map((v) => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return new Float32Array(exps.map((v) => v / sum));
}

export function topKSoftmax(
    logits: Float32Array,
    k: number,
    label: (i: number) => string,
): ClassificationResult[] {
    const probs = softmax(logits);
    return Array.from(probs)
        .map((score, i) => ({ label: label(i), score }))
        .sort((a, b) => b.score - a.score)
        .slice(0, k);
}
