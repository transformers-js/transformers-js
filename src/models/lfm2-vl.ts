import { fetchRaw, fetchJSON } from "../runtime/hub.js";
import { ONNXSession } from "../runtime/session.js";
import { LFM2Tokenizer, type Message } from "../tokenization/lfm2-tokenizer.js";
import { initCache, updateCache, type LFM2ModelConfig } from "../generation/loop.js";
import { argmax, sampleTopP, type SamplingOptions } from "../generation/sampling.js";
import { preprocessVLImage } from "../preprocessing/lfm2-vl.js";
import type { Device } from "../runtime/index.js";
import type { ImageData } from "../preprocessing/ops.js";

export type LFM2VLPrecision = "q4" | "q4f16" | "q8" | "fp16";

export interface LFM2VLOptions {
    device?: Device;
    /**
     * Decoder precision. Encoder always uses fp16.
     * Available precisions depend on export:
     * - LiquidAI exports: "q4", "q8", "fp16"
     * - Community exports: "q4", "q4f16", "fp16"
     * Default: "q4"
     */
    precision?: LFM2VLPrecision;
    /** Base URL for a self-hosted mirror. See LFM2Options.mirrorBaseUrl. */
    mirrorBaseUrl?: string;
}

export interface VLPhaseTiming {
    preprocessMs: number;
    visionEncoderMs: number;
    embedTokensMs: number;
    decoderPrefillMs: number;
    firstDecodeMs: number;
}

export interface VLGenerateOptions {
    maxNewTokens?: number;
    sampling?: SamplingOptions;
    /** If provided, filled with per-phase latency breakdowns. */
    timing?: VLPhaseTiming;
}

interface VLConfig {
    image_token_id: number;
    max_tiles: number;
    use_thumbnail?: boolean;
    text_config: LFM2ModelConfig & { eos_token_id: number };
}

/**
 * LiquidAI official exports and onnx-community exports use different file
 * naming conventions and different vision encoder input formats.
 *
 * liquidai:  embed_images_fp16.onnx | decoder_q4.onnx | CHW images [N,3,512,512]
 * community: vision_encoder_fp16.onnx | decoder_model_merged_q4.onnx | NaFlex patches [N,1024,768]
 */
type VLExportFlavor = "liquidai" | "community";

function detectFlavor(modelId: string): VLExportFlavor {
    return modelId.startsWith("LiquidAI/") ? "liquidai" : "community";
}

const VISION_ENCODER: Record<VLExportFlavor, [string, string]> = {
    liquidai:  ["onnx/embed_images_fp16.onnx",  "onnx/embed_images_fp16.onnx_data"],
    community: ["onnx/vision_encoder_fp16.onnx", "onnx/vision_encoder_fp16.onnx_data"],
};

const DECODER: Record<VLExportFlavor, Partial<Record<LFM2VLPrecision, [string, string]>>> = {
    liquidai: {
        q4:   ["onnx/decoder_q4.onnx",   "onnx/decoder_q4.onnx_data"],
        q8:   ["onnx/decoder_q8.onnx",   "onnx/decoder_q8.onnx_data"],
        fp16: ["onnx/decoder_fp16.onnx", "onnx/decoder_fp16.onnx_data"],
    },
    community: {
        q4:    ["onnx/decoder_model_merged_q4.onnx",    "onnx/decoder_model_merged_q4.onnx_data"],
        q4f16: ["onnx/decoder_model_merged_q4f16.onnx", "onnx/decoder_model_merged_q4f16.onnx_data"],
        fp16:  ["onnx/decoder_model_merged_fp16.onnx",  "onnx/decoder_model_merged_fp16.onnx_data"],
    },
};

export class LFM2VLForConditionalGeneration {
    private constructor(
        private readonly embedImages: ONNXSession,
        private readonly embedTokens: ONNXSession,
        private readonly decoder: ONNXSession,
        private readonly tokenizer: LFM2Tokenizer,
        private readonly modelCfg: LFM2ModelConfig,
        private readonly eosTokenId: number,
        private readonly imageTokenId: number,
        private readonly maxTiles: number,
        private readonly useThumbnail: boolean,
        private readonly flavor: VLExportFlavor,
        private readonly hasPositionIds: boolean,
        private readonly hiddenSize: number,
        private readonly decoderInputNames: string[],
    ) {}

    static async fromHub(modelId: string, options: LFM2VLOptions = {}): Promise<LFM2VLForConditionalGeneration> {
        const { device = "webgpu", precision = "q4", mirrorBaseUrl } = options;
        const flavor = detectFlavor(modelId);

        const [visionFile, visionDataFile] = VISION_ENCODER[flavor];
        const decoderFiles = DECODER[flavor][precision];
        if (!decoderFiles) {
            throw new Error(`Precision "${precision}" is not available for ${flavor} exports.`);
        }
        const [decoderFile, decoderDataFile] = decoderFiles;

        // LiquidAI embed_tokens is self-contained (no .onnx_data); community has external data.
        const embedTokensFile = "onnx/embed_tokens_fp16.onnx";
        const embedTokensDataFile = flavor === "community" ? "onnx/embed_tokens_fp16.onnx_data" : null;

        const fetchList: Promise<ArrayBuffer>[] = [
            fetchRaw(modelId, visionFile, mirrorBaseUrl),
            fetchRaw(modelId, visionDataFile, mirrorBaseUrl),
            fetchRaw(modelId, embedTokensFile, mirrorBaseUrl),
            fetchRaw(modelId, decoderFile, mirrorBaseUrl),
            fetchRaw(modelId, decoderDataFile, mirrorBaseUrl),
        ];
        if (embedTokensDataFile) fetchList.splice(3, 0, fetchRaw(modelId, embedTokensDataFile, mirrorBaseUrl));

        const config = await fetchJSON<VLConfig>(modelId, "config.json", mirrorBaseUrl);
        const [tokenizer, ...buffers] = await Promise.all([
            LFM2Tokenizer.fromHub(modelId, mirrorBaseUrl),
            ...fetchList,
        ]);

        let idx = 0;
        const visionBuffer     = buffers[idx++]!;
        const visionDataBuffer = buffers[idx++]!;
        const embedTokensBuffer = buffers[idx++]!;
        const embedTokensDataBuffer = embedTokensDataFile ? buffers[idx++]! : null;
        const decoderBuffer    = buffers[idx++]!;
        const decoderDataBuffer = buffers[idx++]!;

        const embedTokensExtData = embedTokensDataBuffer
            ? [{ path: "embed_tokens_fp16.onnx_data", data: embedTokensDataBuffer }]
            : undefined;

        const [embedImagesSession, embedTokensSession, decoderSession] = await Promise.all([
            ONNXSession.load(visionBuffer, device, [
                { path: visionDataFile.split("/").pop()!, data: visionDataBuffer },
            ]),
            ONNXSession.load(embedTokensBuffer, device, embedTokensExtData),
            ONNXSession.load(decoderBuffer, device, [
                { path: decoderDataFile.split("/").pop()!, data: decoderDataBuffer },
            ]),
        ]);

        const decInputNames: string[] = decoderSession.inputNames;
        const textCfg = config.text_config;

        return new LFM2VLForConditionalGeneration(
            embedImagesSession,
            embedTokensSession,
            decoderSession,
            tokenizer as LFM2Tokenizer,
            textCfg,
            textCfg.eos_token_id,
            config.image_token_id,
            config.max_tiles,
            config.use_thumbnail ?? false,
            flavor,
            decInputNames.includes("position_ids"),
            textCfg.hidden_size,
            decInputNames,
        );
    }

    async chat(
        messages: Message[],
        image: ImageData,
        options: VLGenerateOptions = {},
    ): Promise<string> {
        const { maxNewTokens = 512, sampling } = options;
        const t = options.timing != null ? () => performance.now() : null;

        // ── 1. Image preprocessing ─────────────────────────────────────────
        const t0 = t?.();
        const { pixelValues, pixelAttentionMask, spatialShapes, numTiles } =
            await preprocessVLImage(image, this.maxTiles, this.useThumbnail, this.flavor);
        const t1 = t?.();

        // Dispatch to the correct vision encoder input format.
        let imgOut: Record<string, { data: unknown; dims: readonly number[] }>;
        if (this.flavor === "liquidai") {
            // CHW image format: [num_tiles, 3, 512, 512]
            imgOut = await this.embedImages.run({
                pixel_values:         { data: pixelValues,        dims: [numTiles, 3, 512, 512] },
                pixel_attention_mask: { data: pixelAttentionMask, dims: [numTiles, 512, 512] },
                spatial_shapes:       { data: spatialShapes,       dims: [numTiles, 2] },
            });
        } else {
            // NaFlex patch format: [num_tiles, max_patches, patch_dim]
            const MAX_PATCHES = 1024, PATCH_DIM = 768;
            imgOut = await this.embedImages.run({
                pixel_values:         { data: pixelValues,        dims: [numTiles, MAX_PATCHES, PATCH_DIM] },
                pixel_attention_mask: { data: pixelAttentionMask, dims: [numTiles, MAX_PATCHES] },
                spatial_shapes:       { data: spatialShapes,       dims: [numTiles, 2] },
            });
        }

        const imageFeatures = imgOut["image_features"]!.data as Float32Array;
        const imgEmbedTokens = imageFeatures.length / this.hiddenSize;
        const t2 = t?.();

        // ── 2. Tokenize with image placeholder ─────────────────────────────
        const vlMessages = injectImageToken(messages);
        const promptIds = this.tokenizer.encodeChat(vlMessages);

        // ── 3. Embed tokens ────────────────────────────────────────────────
        const inputIds = new BigInt64Array(promptIds.map(BigInt));
        const tokOut = await this.embedTokens.run({
            input_ids: { data: inputIds, dims: [1, promptIds.length] },
        });
        const tokenEmbeds = tokOut["inputs_embeds"]!.data as Float32Array;
        const t3 = t?.();

        // ── 4. Splice image embeddings at image_token positions ────────────
        const prefillEmbeds = spliceImageEmbeds(
            tokenEmbeds,
            promptIds,
            this.imageTokenId,
            imageFeatures,
            imgEmbedTokens,
            this.hiddenSize,
        );
        const prefillSeqLen = prefillEmbeds.length / this.hiddenSize;

        // ── 5. Prefill decoder ─────────────────────────────────────────────
        const cache = initCache(this.decoderInputNames, this.modelCfg);
        const attnMask = new BigInt64Array(prefillSeqLen).fill(1n);
        const hasNumLogitsToKeep = this.decoderInputNames.includes("num_logits_to_keep");

        const prefillInputs: Record<string, import("../runtime/session.js").TensorInput> = {
            inputs_embeds:  { data: prefillEmbeds, dims: [1, prefillSeqLen, this.hiddenSize] },
            attention_mask: { data: attnMask,       dims: [1, prefillSeqLen] },
            ...cache,
        };
        if (this.hasPositionIds) {
            prefillInputs["position_ids"] = {
                data: new BigInt64Array(prefillSeqLen).map((_, i) => BigInt(i)),
                dims: [1, prefillSeqLen],
            };
        }
        if (hasNumLogitsToKeep) {
            prefillInputs["num_logits_to_keep"] = { data: new BigInt64Array([1n]), dims: [1] };
        }

        const prefillOut = await this.decoder.run(prefillInputs);
        updateCache(cache, prefillOut);

        const logitsDims = prefillOut["logits"]!.dims;
        const vocabSize  = logitsDims[logitsDims.length - 1]!;
        const logitsData = prefillOut["logits"]!.data as Float32Array;
        const lastLogits = logitsData.subarray(logitsData.length - vocabSize);
        let nextToken = sampling ? sampleTopP(lastLogits, sampling) : argmax(lastLogits);
        const generated: number[] = [nextToken];
        const t4 = t?.();

        // ── 6. Decode loop ─────────────────────────────────────────────────
        let pastLen = prefillSeqLen;

        while (nextToken !== this.eosTokenId && generated.length < maxNewTokens) {
            const singleId = new BigInt64Array([BigInt(nextToken)]);
            const embedOut = await this.embedTokens.run({
                input_ids: { data: singleId, dims: [1, 1] },
            });
            const singleEmbed = embedOut["inputs_embeds"]!.data as Float32Array;

            const decInputs: Record<string, import("../runtime/session.js").TensorInput> = {
                inputs_embeds:  { data: singleEmbed, dims: [1, 1, this.hiddenSize] },
                attention_mask: { data: new BigInt64Array(pastLen + 1).fill(1n), dims: [1, pastLen + 1] },
                ...cache,
            };
            if (this.hasPositionIds) {
                decInputs["position_ids"] = {
                    data: new BigInt64Array([BigInt(pastLen)]),
                    dims: [1, 1],
                };
            }

            const out = await this.decoder.run(decInputs);
            updateCache(cache, out);
            pastLen++;

            nextToken = sampling
                ? sampleTopP(out["logits"]!.data as Float32Array, sampling)
                : argmax(out["logits"]!.data as Float32Array);
            generated.push(nextToken);
        }

        if (generated[generated.length - 1] === this.eosTokenId) generated.pop();

        if (options.timing != null && t0 != null && t1 != null && t2 != null && t3 != null && t4 != null) {
            const t5 = performance.now();
            options.timing.preprocessMs     = t1 - t0;
            options.timing.visionEncoderMs  = t2 - t1;
            options.timing.embedTokensMs    = t3 - t2;
            options.timing.decoderPrefillMs = t4 - t3;
            options.timing.firstDecodeMs    = t5 - t4;
        }

        return this.tokenizer.decode(generated);
    }

    dispose(): void {
        this.embedImages.dispose();
        this.embedTokens.dispose();
        this.decoder.dispose();
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

function injectImageToken(messages: Message[]): Message[] {
    const out: Message[] = [];
    let injected = false;
    for (const msg of messages) {
        if (!injected && msg.role === "user") {
            out.push({ ...msg, content: `<image>\n${msg.content}` });
            injected = true;
        } else {
            out.push(msg);
        }
    }
    return out;
}

function spliceImageEmbeds(
    tokenEmbeds: Float32Array,
    promptIds: number[],
    imageTokenId: number,
    imageFeatures: Float32Array,
    imgEmbedTokens: number,
    hiddenSize: number,
): Float32Array {
    const outSeqLen = promptIds.reduce(
        (acc, id) => acc + (id === imageTokenId ? imgEmbedTokens : 1),
        0,
    );
    const out = new Float32Array(outSeqLen * hiddenSize);
    let outPos = 0, imgFeaturePos = 0;

    for (let i = 0; i < promptIds.length; i++) {
        if (promptIds[i] === imageTokenId) {
            const src = imageFeatures.subarray(
                imgFeaturePos * hiddenSize,
                (imgFeaturePos + imgEmbedTokens) * hiddenSize,
            );
            out.set(src, outPos * hiddenSize);
            outPos += imgEmbedTokens;
            imgFeaturePos += imgEmbedTokens;
        } else {
            out.set(tokenEmbeds.subarray(i * hiddenSize, (i + 1) * hiddenSize), outPos * hiddenSize);
            outPos++;
        }
    }
    return out;
}
