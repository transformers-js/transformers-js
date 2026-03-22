import { fetchRaw } from "../runtime/hub.js";
import { ONNXSession } from "../runtime/session.js";
import { SAMImageProcessor, scalePoint, scaleBox, postProcessMask } from "../preprocessing/sam.js";
import type { SAMPreprocessed } from "../preprocessing/sam.js";
import type { Device, ModelOptions } from "../runtime/index.js";
import type { ImageData } from "../preprocessing/ops.js";

// ── Public types ─────────────────────────────────────────────────────────────────

/** A click prompt. label: 1 = foreground, 0 = background. */
export interface SAMPoint {
    x: number;
    y: number;
    label: 0 | 1;
}

export interface SAMBox {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
}

export interface SAMPrompt {
    points?: SAMPoint[];
    boxes?: SAMBox[];
}

/**
 * Cached image encoding. Pass to predict() to avoid re-encoding the same image
 * when running multiple prompts interactively.
 */
export interface SAMEmbedding {
    imageEmbeddings: Float32Array;
    imagePositionalEmbeddings: Float32Array;
    preprocessed: SAMPreprocessed;
}

export interface SAMMask {
    /** Binary mask [H, W] in row-major order. 1 = inside mask, 0 = outside. */
    data: Float32Array;
    /** IOU quality score predicted by the model. */
    score: number;
    width: number;
    height: number;
}

// ── Model ────────────────────────────────────────────────────────────────────────

export class SAMModel {
    private constructor(
        private readonly encoderSession: ONNXSession,
        private readonly decoderSession: ONNXSession,
        private readonly processor: SAMImageProcessor,
    ) {}

    static async fromHub(modelId: string, options: ModelOptions = {}): Promise<SAMModel> {
        const { device = "webgpu", quantized = false } = options;
        const suffix = quantized ? "_quantized" : "";
        const [encoderBuf, decoderBuf, processor] = await Promise.all([
            fetchRaw(modelId, `onnx/encoder_model${suffix}.onnx`),
            fetchRaw(modelId, `onnx/decoder_model${suffix}.onnx`),
            SAMImageProcessor.fromHub(modelId),
        ]);

        const [encoderSession, decoderSession] = await Promise.all([
            ONNXSession.load(encoderBuf, device),
            ONNXSession.load(decoderBuf, device),
        ]);

        return new SAMModel(encoderSession, decoderSession, processor);
    }

    /**
     * Encode the image once. Reuse the returned embedding for multiple predict() calls.
     * This is the expensive step (~200ms for vit-base on CPU).
     */
    async encodeImage(image: ImageData): Promise<SAMEmbedding> {
        const preprocessed = await this.processor.preprocess(image);

        const out = await this.encoderSession.run({
            pixel_values: { data: preprocessed.pixelValues, dims: [1, 3, 1024, 1024] },
        });

        return {
            imageEmbeddings:            out["image_embeddings"]!.data,
            imagePositionalEmbeddings:  out["image_positional_embeddings"]!.data,
            preprocessed,
        };
    }

    /**
     * Run the decoder with a prompt. Fast (~10ms). Call many times per encodeImage().
     * Returns up to 3 candidate masks sorted by predicted IOU score.
     */
    async predict(embedding: SAMEmbedding, prompt: SAMPrompt): Promise<SAMMask[]> {
        const { preprocessed } = embedding;

        const { inputPoints, inputLabels, numPoints } = buildPointTensors(
            prompt.points ?? [],
            prompt.boxes  ?? [],
            preprocessed.scale,
        );

        const maskInput    = new Float32Array(256 * 256);   // no prior mask
        const hasMaskInput = new Float32Array([0]);
        const origSizes    = BigInt64Array.from(
            [BigInt(preprocessed.originalSize.height), BigInt(preprocessed.originalSize.width)],
        );

        const out = await this.decoderSession.run({
            image_embeddings:            { data: embedding.imageEmbeddings,           dims: [1, 256, 64, 64] },
            image_positional_embeddings: { data: embedding.imagePositionalEmbeddings, dims: [1, 256, 64, 64] },
            input_points:  { data: inputPoints,  dims: [1, numPoints, 2] },
            input_labels:  { data: inputLabels,  dims: [1, numPoints] },
            mask_input:    { data: maskInput,    dims: [1, 1, 256, 256] },
            has_mask_input:{ data: hasMaskInput, dims: [1] },
            orig_sizes:    { data: origSizes,    dims: [1, 2] },
        });

        return parseMasks(out, preprocessed);
    }

    /** Convenience: encode + predict in one call. Use when you have a single prompt. */
    async run(image: ImageData, prompt: SAMPrompt): Promise<SAMMask[]> {
        const embedding = await this.encodeImage(image);
        return this.predict(embedding, prompt);
    }

    dispose(): void {
        this.encoderSession.dispose();
        this.decoderSession.dispose();
    }
}

// ── Internals ────────────────────────────────────────────────────────────────────

function buildPointTensors(
    points: SAMPoint[],
    boxes: SAMBox[],
    scale: number,
): { inputPoints: Float32Array; inputLabels: Float32Array; numPoints: number } {
    const coords: number[] = [];
    const labels: number[] = [];

    // Points
    for (const pt of points) {
        const [sx, sy] = scalePoint(pt.x, pt.y, scale);
        coords.push(sx, sy);
        labels.push(pt.label);
    }

    // Boxes: encoded as two points with labels 2 (top-left) and 3 (bottom-right)
    for (const box of boxes) {
        const [sx1, sy1, sx2, sy2] = scaleBox(box.x1, box.y1, box.x2, box.y2, scale);
        coords.push(sx1, sy1, sx2, sy2);
        labels.push(2, 3);
    }

    // SAM requires at least one point. Add a padding point if prompt is empty.
    if (coords.length === 0) {
        coords.push(0, 0);
        labels.push(-1); // -1 = padding
    }

    const numPoints = labels.length;
    return {
        inputPoints: new Float32Array(coords),
        inputLabels: new Float32Array(labels),
        numPoints,
    };
}

async function parseMasks(
    out: Record<string, { data: Float32Array; dims: readonly number[] }>,
    preprocessed: SAMPreprocessed,
): Promise<SAMMask[]> {
    const predMasks   = out["pred_masks"]!;
    const iouPreds    = out["iou_predictions"]!;
    const numMasks    = iouPreds.dims[1] ?? iouPreds.data.length;

    // pred_masks dims: [1, num_masks, 1, H, W]
    const maskH = predMasks.dims[3] ?? 256;
    const maskW = predMasks.dims[4] ?? 256;
    const maskPixels = maskH * maskW;

    const masks: SAMMask[] = [];
    for (let m = 0; m < numMasks; m++) {
        const logits = predMasks.data.slice(m * maskPixels, (m + 1) * maskPixels);
        const binary = await postProcessMask(logits, maskH, maskW, preprocessed);
        masks.push({
            data: binary,
            score: iouPreds.data[m]!,
            width: preprocessed.originalSize.width,
            height: preprocessed.originalSize.height,
        });
    }

    return masks.sort((a, b) => b.score - a.score);
}
