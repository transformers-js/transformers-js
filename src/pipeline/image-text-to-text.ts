import { LFM2VLForConditionalGeneration, type LFM2VLOptions, type VLGenerateOptions } from "../models/lfm2-vl.js";
import type { Message } from "../tokenization/lfm2-tokenizer.js";
import type { ImageData } from "../preprocessing/ops.js";

export type { Message, LFM2VLOptions, VLGenerateOptions };

export class ImageTextToTextPipeline {
    private constructor(private readonly model: LFM2VLForConditionalGeneration) {}

    static async create(modelId: string, options: LFM2VLOptions = {}): Promise<ImageTextToTextPipeline> {
        const model = await LFM2VLForConditionalGeneration.fromHub(modelId, options);
        return new ImageTextToTextPipeline(model);
    }

    /** Send an image + conversation and get the assistant reply. */
    run(messages: Message[], image: ImageData, options: VLGenerateOptions = {}): Promise<string> {
        return this.model.chat(messages, image, options);
    }

    dispose(): void {
        this.model.dispose();
    }
}
