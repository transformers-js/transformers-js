import { CLIPModel, cosineSimilarity } from "../models/clip.js";
import { softmax } from "./image-classification.js";
import type { Device } from "../runtime/index.js";
import type { ImageData } from "../preprocessing/ops.js";

export interface ZeroShotResult {
    label: string;
    score: number;
}

export interface ZeroShotOptions {
    device?: Device;
    /** Template wrapping each label before encoding. Default: "a photo of a {label}" */
    template?: (label: string) => string;
}

export class ZeroShotImageClassificationPipeline {
    private readonly template: (label: string) => string;

    private constructor(
        private readonly model: CLIPModel,
        options: ZeroShotOptions,
    ) {
        this.template = options.template ?? ((label) => `a photo of a ${label}`);
    }

    static async create(
        modelId: string,
        options: ZeroShotOptions = {},
    ): Promise<ZeroShotImageClassificationPipeline> {
        const model = await CLIPModel.fromHub(modelId, options.device ?? "webgpu");
        return new ZeroShotImageClassificationPipeline(model, options);
    }

    async run(image: ImageData, labels: string[]): Promise<ZeroShotResult[]> {
        const [imageEmb, ...textEmbs] = await Promise.all([
            this.model.encodeImage(image),
            ...labels.map((label) => this.model.encodeText(this.template(label))),
        ]);

        const logits = new Float32Array(
            textEmbs.map((textEmb) => cosineSimilarity(imageEmb!, textEmb)),
        );

        const probs = softmax(logits);
        return labels
            .map((label, i) => ({ label, score: probs[i]! }))
            .sort((a, b) => b.score - a.score);
    }

    dispose(): void {
        this.model.dispose();
    }
}
