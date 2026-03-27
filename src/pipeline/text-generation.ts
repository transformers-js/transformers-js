import { LFM2ForCausalLM, type LFM2Options, type GenerateOptions } from "../models/lfm2.js";
import type { Message } from "../tokenization/lfm2-tokenizer.js";

export type { Message, LFM2Options, GenerateOptions };

export class TextGenerationPipeline {
    private constructor(private readonly model: LFM2ForCausalLM) {}

    static async create(modelId: string, options: LFM2Options = {}): Promise<TextGenerationPipeline> {
        const model = await LFM2ForCausalLM.fromHub(modelId, options);
        return new TextGenerationPipeline(model);
    }

    /**
     * Send a conversation and get the assistant reply.
     * Pass `onChunk` for streaming text output as tokens are generated.
     */
    run(messages: Message[], options: GenerateOptions = {}): Promise<string> {
        return this.model.chat(messages, options);
    }

    dispose(): void {
        this.model.dispose();
    }
}
