import { ImageClassificationPipeline } from "./image-classification.js";
import { ZeroShotImageClassificationPipeline } from "./zero-shot-classification.js";
import { ImageSegmentationPipeline } from "./image-segmentation.js";
import type { Device } from "../runtime/index.js";

export { ImageClassificationPipeline } from "./image-classification.js";
export { ZeroShotImageClassificationPipeline } from "./zero-shot-classification.js";
export { ImageSegmentationPipeline } from "./image-segmentation.js";

export interface PipelineOptions {
    model: string;
    device?: Device;
    quantized?: boolean;
}

export function pipeline(
    task: "image-classification",
    options: PipelineOptions,
): Promise<ImageClassificationPipeline>;
export function pipeline(
    task: "zero-shot-image-classification",
    options: PipelineOptions,
): Promise<ZeroShotImageClassificationPipeline>;
export function pipeline(
    task: "image-segmentation",
    options: PipelineOptions,
): Promise<ImageSegmentationPipeline>;
export async function pipeline(
    task: "image-classification" | "zero-shot-image-classification" | "image-segmentation",
    options: PipelineOptions,
): Promise<ImageClassificationPipeline | ZeroShotImageClassificationPipeline | ImageSegmentationPipeline> {
    const { model, ...rest } = options;
    switch (task) {
        case "image-classification":
            return ImageClassificationPipeline.create(model, rest);
        case "zero-shot-image-classification":
            return ZeroShotImageClassificationPipeline.create(model, rest);
        case "image-segmentation":
            return ImageSegmentationPipeline.create(model, rest);
    }
}
