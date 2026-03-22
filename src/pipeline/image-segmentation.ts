import { SAMModel } from "../models/sam.js";
import type { SAMPrompt, SAMMask, SAMEmbedding } from "../models/sam.js";
import type { Device } from "../runtime/index.js";
import type { ImageData } from "../preprocessing/ops.js";

export type { SAMPrompt, SAMMask };

export interface SegmentationOptions {
    device?: Device;
}

/**
 * Interactive segmentation pipeline backed by SAM.
 *
 * For a single image with multiple prompts, use encodeImage() once then
 * call predict() for each prompt — avoids re-running the expensive encoder.
 *
 * For a one-shot single prompt, use segment() which does both in one call.
 */
export class ImageSegmentationPipeline {
    private constructor(private readonly model: SAMModel) {}

    static async create(
        modelId: string,
        options: SegmentationOptions = {},
    ): Promise<ImageSegmentationPipeline> {
        const model = await SAMModel.fromHub(modelId, options.device ?? "webgpu");
        return new ImageSegmentationPipeline(model);
    }

    /** Encode image once. Reuse the result across multiple predict() calls. */
    encodeImage(image: ImageData): Promise<SAMEmbedding> {
        return this.model.encodeImage(image);
    }

    /** Run decoder with a prompt against a pre-encoded image. */
    predict(embedding: SAMEmbedding, prompt: SAMPrompt): Promise<SAMMask[]> {
        return this.model.predict(embedding, prompt);
    }

    /** One-shot: encode + predict. Use when you have one prompt per image. */
    segment(image: ImageData, prompt: SAMPrompt): Promise<SAMMask[]> {
        return this.model.run(image, prompt);
    }

    dispose(): void {
        this.model.dispose();
    }
}
