import { ImageClassificationPipeline } from "./image-classification.js";
import { ZeroShotImageClassificationPipeline } from "./zero-shot-classification.js";
import { ImageSegmentationPipeline } from "./image-segmentation.js";
import { TextGenerationPipeline } from "./text-generation.js";
import type { Device } from "../runtime/index.js";
import type { LFM2Precision } from "../models/lfm2.js";

export { ImageClassificationPipeline } from "./image-classification.js";
export { ZeroShotImageClassificationPipeline } from "./zero-shot-classification.js";
export { ImageSegmentationPipeline } from "./image-segmentation.js";
export { TextGenerationPipeline } from "./text-generation.js";

export interface PipelineOptions {
    model: string;
    device?: Device;
    quantized?: boolean;
    /** For text-generation: quantization variant. Default: "q8". */
    precision?: LFM2Precision;
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
export function pipeline(
    task: "text-generation",
    options: PipelineOptions,
): Promise<TextGenerationPipeline>;
export async function pipeline(
    task: "image-classification" | "zero-shot-image-classification" | "image-segmentation" | "text-generation",
    options: PipelineOptions,
): Promise<ImageClassificationPipeline | ZeroShotImageClassificationPipeline | ImageSegmentationPipeline | TextGenerationPipeline> {
    const { model, ...rest } = options;
    switch (task) {
        case "image-classification":
            return ImageClassificationPipeline.create(model, rest);
        case "zero-shot-image-classification":
            return ZeroShotImageClassificationPipeline.create(model, rest);
        case "image-segmentation":
            return ImageSegmentationPipeline.create(model, rest);
        case "text-generation":
            return TextGenerationPipeline.create(model, rest);
    }
}
