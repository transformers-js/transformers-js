import { fetchRaw, fetchJSON } from "../runtime/hub.js";
import { ONNXSession } from "../runtime/session.js";
import { LFM2Tokenizer, type Message } from "../tokenization/lfm2-tokenizer.js";
import { generate, type LFM2ModelConfig } from "../generation/loop.js";
import type { SamplingOptions } from "../generation/sampling.js";
import type { Device } from "../runtime/index.js";

export type LFM2Precision = "q8" | "q4" | "fp16";

export interface LFM2Options {
    device?: Device;
    /** Quantization variant. Default: "q8" (~1.7GB, recommended balance). */
    precision?: LFM2Precision;
}

export interface GenerateOptions {
    maxNewTokens?: number;
    sampling?: SamplingOptions;
    /** Tool definitions to inject into the system prompt (JSON Schema format). */
    tools?: object[];
}

interface FullConfig extends LFM2ModelConfig {
    eos_token_id: number;
    transformers?: {
        js_config?: {
            use_external_data_format?: Record<string, number>;
        };
    };
}

const ONNX_FILE: Record<LFM2Precision, string> = {
    q8:   "onnx/model_q8.onnx",
    q4:   "onnx/model_q4.onnx",
    fp16: "onnx/model_fp16.onnx",
};

const DATA_FILE: Record<LFM2Precision, string> = {
    q8:   "onnx/model_q8.onnx_data",
    q4:   "onnx/model_q4.onnx_data",
    fp16: "onnx/model_fp16.onnx_data",
};

export class LFM2ForCausalLM {
    private constructor(
        private readonly session: ONNXSession,
        private readonly tokenizer: LFM2Tokenizer,
        private readonly modelCfg: LFM2ModelConfig,
        private readonly eosTokenId: number,
        private readonly hasPositionIds: boolean,
        private readonly inputNames: string[],
    ) {}

    static async fromHub(modelId: string, options: LFM2Options = {}): Promise<LFM2ForCausalLM> {
        const { device = "webgpu", precision = "q8" } = options;

        const onnxFile = ONNX_FILE[precision];
        const dataFile = DATA_FILE[precision];

        const [modelBuffer, dataBuffer, config, tokenizer] = await Promise.all([
            fetchRaw(modelId, onnxFile),
            fetchRaw(modelId, dataFile),
            fetchJSON<FullConfig>(modelId, "config.json"),
            LFM2Tokenizer.fromHub(modelId),
        ]);

        const externalData = [{ path: dataFile.split("/").pop()!, data: dataBuffer }];
        const session = await ONNXSession.load(modelBuffer, device, externalData);

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const inputNames: string[] = (session as any).session.inputNames ?? [];
        const hasPositionIds = inputNames.includes("position_ids");

        return new LFM2ForCausalLM(session, tokenizer, config, config.eos_token_id, hasPositionIds, inputNames);
    }

    async chat(messages: Message[], options: GenerateOptions = {}): Promise<string> {
        const promptIds = this.tokenizer.encodeChat(messages, options.tools);
        const genCfg = {
            eosTokenId: this.eosTokenId,
            ...(options.maxNewTokens !== undefined ? { maxNewTokens: options.maxNewTokens } : {}),
            ...(options.sampling !== undefined ? { sampling: options.sampling } : {}),
        };
        const generatedIds = await generate(
            this.session,
            promptIds,
            this.modelCfg,
            genCfg,
            this.hasPositionIds,
            this.inputNames,
        );
        return this.tokenizer.decode(generatedIds);
    }

    dispose(): void {
        this.session.dispose();
    }
}
