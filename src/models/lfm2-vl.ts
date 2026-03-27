import { fetchRaw, fetchJSON } from "../runtime/hub.js";
import { ONNXSession } from "../runtime/session.js";
import { LFM2Tokenizer, type Message } from "../tokenization/lfm2-tokenizer.js";
import { initCache, updateCache, type LFM2ModelConfig } from "../generation/loop.js";
import { argmax, sampleTopP, type SamplingOptions } from "../generation/sampling.js";
import { preprocessVLImage, TOKENS_PER_TILE } from "../preprocessing/lfm2-vl.js";
import type { Device } from "../runtime/index.js";
import type { ImageData } from "../preprocessing/ops.js";

export type LFM2VLPrecision = "q4" | "q4f16" | "fp16";

export interface LFM2VLOptions {
    device?: Device;
    /**
     * Decoder precision. Encoder always uses fp16.
     * - "q4"    (~1.5GB, WebGPU recommended)
     * - "q4f16" (~mixed precision, q4 weights + fp16 activations)
     * - "fp16"  (~3.2GB, server)
     * Default: "q4"
     */
    precision?: LFM2VLPrecision;
    /** Base URL for a self-hosted mirror. See LFM2Options.mirrorBaseUrl. */
    mirrorBaseUrl?: string;
}

export interface VLGenerateOptions {
    maxNewTokens?: number;
    sampling?: SamplingOptions;
}

interface VLConfig {
    image_token_id: number;
    max_tiles: number;
    use_thumbnail?: boolean;
    text_config: LFM2ModelConfig & { eos_token_id: number };
}

// Encoder and embed_tokens always fp16; only decoder varies.
const DECODER_FILE: Record<LFM2VLPrecision, [string, string]> = {
    q4:    ["onnx/decoder_model_merged_q4.onnx",    "onnx/decoder_model_merged_q4.onnx_data"],
    q4f16: ["onnx/decoder_model_merged_q4f16.onnx", "onnx/decoder_model_merged_q4f16.onnx_data"],
    fp16:  ["onnx/decoder_model_merged_fp16.onnx",  "onnx/decoder_model_merged_fp16.onnx_data"],
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
        private readonly hasPositionIds: boolean,
        private readonly hiddenSize: number,
        private readonly decoderInputNames: string[],
    ) {}

    static async fromHub(modelId: string, options: LFM2VLOptions = {}): Promise<LFM2VLForConditionalGeneration> {
        const { device = "webgpu", precision = "q4", mirrorBaseUrl } = options;
        const [decoderFile, decoderData] = DECODER_FILE[precision];

        const [
            visionEncoderBuffer, visionEncoderData,
            embedTokensBuffer, embedTokensData,
            decoderBuffer, decoderDataBuffer,
            config,
            tokenizer,
        ] = await Promise.all([
            fetchRaw(modelId, "onnx/vision_encoder_fp16.onnx", mirrorBaseUrl),
            fetchRaw(modelId, "onnx/vision_encoder_fp16.onnx_data", mirrorBaseUrl),
            fetchRaw(modelId, "onnx/embed_tokens_fp16.onnx", mirrorBaseUrl),
            fetchRaw(modelId, "onnx/embed_tokens_fp16.onnx_data", mirrorBaseUrl),
            fetchRaw(modelId, decoderFile, mirrorBaseUrl),
            fetchRaw(modelId, decoderData, mirrorBaseUrl),
            fetchJSON<VLConfig>(modelId, "config.json", mirrorBaseUrl),
            LFM2Tokenizer.fromHub(modelId, mirrorBaseUrl),
        ]);

        const [embedImagesSession, embedTokensSession, decoderSession] = await Promise.all([
            ONNXSession.load(visionEncoderBuffer, device, [
                { path: "vision_encoder_fp16.onnx_data", data: visionEncoderData },
            ]),
            ONNXSession.load(embedTokensBuffer, device, [
                { path: "embed_tokens_fp16.onnx_data", data: embedTokensData },
            ]),
            ONNXSession.load(decoderBuffer, device, [
                { path: decoderData.split("/").pop()!, data: decoderDataBuffer },
            ]),
        ]);

        const decInputNames: string[] = decoderSession.inputNames;
        const hasPositionIds = decInputNames.includes("position_ids");

        const textCfg = config.text_config;

        return new LFM2VLForConditionalGeneration(
            embedImagesSession,
            embedTokensSession,
            decoderSession,
            tokenizer,
            textCfg,
            textCfg.eos_token_id,
            config.image_token_id,
            config.max_tiles,
            config.use_thumbnail ?? false,
            hasPositionIds,
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

        // ── 1. Image preprocessing ─────────────────────────────────────────
        const { pixelValues, pixelAttentionMask, spatialShapes, numTiles } =
            await preprocessVLImage(image, this.maxTiles, this.useThumbnail);

        // The community ONNX export processes one tile at a time: [3, 512, 512].
        // Run vision encoder per tile and concatenate features.
        const TILE_PX  = 3 * 512 * 512;
        const MASK_PX  = 512 * 512;
        const tileFeatureArrays: Float32Array[] = [];
        for (let i = 0; i < numTiles; i++) {
            const imgOut = await this.embedImages.run({
                pixel_values:         { data: pixelValues.subarray(i * TILE_PX, (i + 1) * TILE_PX),       dims: [3, 512, 512] },
                pixel_attention_mask: { data: pixelAttentionMask.subarray(i * MASK_PX, (i + 1) * MASK_PX), dims: [512, 512] },
                spatial_shapes:       { data: spatialShapes.subarray(i * 2, (i + 1) * 2),                  dims: [2] },
            });
            tileFeatureArrays.push(imgOut["image_features"]!.data as Float32Array);
        }
        // image_features: [numTiles * tokensPerTile * hiddenSize] (flattened)
        const featTotalLen = tileFeatureArrays.reduce((s, f) => s + f.length, 0);
        const imageFeatures = new Float32Array(featTotalLen);
        let featOffset = 0;
        for (const f of tileFeatureArrays) { imageFeatures.set(f, featOffset); featOffset += f.length; }
        const imgEmbedTokens = featTotalLen / this.hiddenSize;

        // ── 2. Tokenize with image placeholder ─────────────────────────────
        // Inject <image> before the first user message content.
        const vlMessages = injectImageToken(messages);
        const promptIds = this.tokenizer.encodeChat(vlMessages);

        // ── 3. Embed tokens ────────────────────────────────────────────────
        const inputIds = new BigInt64Array(promptIds.map(BigInt));
        const tokOut = await this.embedTokens.run({
            input_ids: { data: inputIds, dims: [1, promptIds.length] },
        });
        const tokenEmbeds = tokOut["inputs_embeds"]!.data as Float32Array;

        // ── 4. Splice image embeddings at image_token positions ────────────
        // Each image_token_id placeholder expands to imgEmbedTokens embedding rows.
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
            inputs_embeds: { data: prefillEmbeds, dims: [1, prefillSeqLen, this.hiddenSize] },
            attention_mask: { data: attnMask, dims: [1, prefillSeqLen] },
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
        const vocabSize = logitsDims[logitsDims.length - 1]!;
        const logitsData = prefillOut["logits"]!.data as Float32Array;
        const lastLogits = logitsData.subarray(logitsData.length - vocabSize);
        let nextToken = sampling ? sampleTopP(lastLogits, sampling) : argmax(lastLogits);
        const generated: number[] = [nextToken];

        // ── 6. Decode loop ─────────────────────────────────────────────────
        let pastLen = prefillSeqLen;

        while (nextToken !== this.eosTokenId && generated.length < maxNewTokens) {
            // Embed single new token
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
        return this.tokenizer.decode(generated);
    }

    dispose(): void {
        this.embedImages.dispose();
        this.embedTokens.dispose();
        this.decoder.dispose();
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/**
 * Inject the <image> token before the first user message's content so the
 * tokenizer sees it as a special token and maps it to image_token_id=396.
 */
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

/**
 * Build a new inputs_embeds Float32Array where each image_token_id in promptIds
 * is replaced by imgEmbedTokens rows of image embeddings from imageFeatures.
 *
 * imageFeatures layout: [numTiles * TOKENS_PER_TILE, hiddenSize] (flattened).
 */
function spliceImageEmbeds(
    tokenEmbeds: Float32Array,   // [seqLen, hiddenSize] flattened (from embed_tokens output [1, seqLen, hiddenSize])
    promptIds: number[],
    imageTokenId: number,
    imageFeatures: Float32Array, // [numTiles * tokensPerTile, hiddenSize] flattened
    imgEmbedTokens: number,
    hiddenSize: number,
): Float32Array {
    // Count output tokens: replace each image_token_id with imgEmbedTokens rows
    const outSeqLen = promptIds.reduce(
        (acc, id) => acc + (id === imageTokenId ? imgEmbedTokens : 1),
        0,
    );
    const out = new Float32Array(outSeqLen * hiddenSize);

    let outPos = 0;
    let imgFeaturePos = 0; // position in imageFeatures (in tokens)

    for (let i = 0; i < promptIds.length; i++) {
        if (promptIds[i] === imageTokenId) {
            // Copy imgEmbedTokens rows from imageFeatures
            const src = imageFeatures.subarray(
                imgFeaturePos * hiddenSize,
                (imgFeaturePos + imgEmbedTokens) * hiddenSize,
            );
            out.set(src, outPos * hiddenSize);
            outPos += imgEmbedTokens;
            imgFeaturePos += imgEmbedTokens;
        } else {
            // Copy one token embedding from tokenEmbeds (batch dim 0 already stripped)
            out.set(tokenEmbeds.subarray(i * hiddenSize, (i + 1) * hiddenSize), outPos * hiddenSize);
            outPos++;
        }
    }

    return out;
}
