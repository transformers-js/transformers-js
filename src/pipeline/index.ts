import { TextGenerationPipeline } from "./text-generation.js";
import { ImageTextToTextPipeline } from "./image-text-to-text.js";
import type { Device } from "../runtime/index.js";
import type { LFM2Precision } from "../models/lfm2.js";
import type { LFM2VLPrecision } from "../models/lfm2-vl.js";

export { TextGenerationPipeline } from "./text-generation.js";
export { ImageTextToTextPipeline } from "./image-text-to-text.js";

export interface PipelineOptions {
    model: string;
    device?: Device;
    precision?: LFM2Precision | LFM2VLPrecision;
}

export function pipeline(
    task: "text-generation",
    options: PipelineOptions,
): Promise<TextGenerationPipeline>;
export function pipeline(
    task: "image-text-to-text",
    options: PipelineOptions,
): Promise<ImageTextToTextPipeline>;
export async function pipeline(
    task: "text-generation" | "image-text-to-text",
    options: PipelineOptions,
): Promise<TextGenerationPipeline | ImageTextToTextPipeline> {
    const { model, ...rest } = options;
    switch (task) {
        case "text-generation":
            return TextGenerationPipeline.create(model, rest);
        case "image-text-to-text":
            return ImageTextToTextPipeline.create(model, rest);
    }
}
