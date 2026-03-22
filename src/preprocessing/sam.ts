import { resize, rescale, normalize, pad, hwcToChw } from "./ops.js";
import { fetchJSON } from "../runtime/hub.js";
import type { ImageData } from "./ops.js";

const MODEL_SIZE = 1024;

// SAM uses ImageNet mean/std in [0, 1] space (after rescaling by 1/255)
const SAM_MEAN = [0.485, 0.456, 0.406];
const SAM_STD  = [0.229, 0.224, 0.225];

export interface SAMPreprocessed {
    /** CHW float32 tensor, shape [3 * 1024 * 1024] */
    pixelValues: Float32Array;
    originalSize: { width: number; height: number };
    /** Scale applied to original coordinates to reach model input space */
    scale: number;
    /** Width after resize (before padding) */
    resizedWidth: number;
    /** Height after resize (before padding) */
    resizedHeight: number;
}

export class SAMImageProcessor {
    // SAM preprocessing is fully determined by MODEL_SIZE — no config needed.
    // fromHub is provided for API consistency with other processors.
    static async fromHub(_modelId: string): Promise<SAMImageProcessor> {
        return new SAMImageProcessor();
    }

    async preprocess(image: ImageData): Promise<SAMPreprocessed> {
        const { width: W, height: H } = image;

        // Resize longest edge to MODEL_SIZE, preserve aspect ratio
        const scale = MODEL_SIZE / Math.max(W, H);
        const rW = Math.round(W * scale);
        const rH = Math.round(H * scale);

        let img = await resize(image, { width: rW, height: rH }, "bilinear");
        img = rescale(img, 1 / 255);
        img = normalize(img, SAM_MEAN, SAM_STD);
        // Pad bottom + right to MODEL_SIZE × MODEL_SIZE
        img = pad(img, { top: 0, left: 0, bottom: MODEL_SIZE - rH, right: MODEL_SIZE - rW });

        return {
            pixelValues: hwcToChw(img),
            originalSize: { width: W, height: H },
            scale,
            resizedWidth: rW,
            resizedHeight: rH,
        };
    }
}

// ── Coordinate transforms ────────────────────────────────────────────────────────

/** Transform a point from original image space to model input space. */
export function scalePoint(x: number, y: number, scale: number): [number, number] {
    return [x * scale, y * scale];
}

/** Transform a box from original image space to model input space. */
export function scaleBox(
    x1: number, y1: number, x2: number, y2: number, scale: number,
): [number, number, number, number] {
    return [x1 * scale, y1 * scale, x2 * scale, y2 * scale];
}

// ── Mask post-processing ─────────────────────────────────────────────────────────

/**
 * Upscale a low-resolution logit mask to the original image size and threshold.
 *
 * SAM decoder outputs masks at 256×256. We:
 * 1. Upscale to MODEL_SIZE × MODEL_SIZE
 * 2. Crop to the non-padded region (resizedWidth × resizedHeight)
 * 3. Upscale to originalSize
 * 4. Threshold at logit 0 (sigmoid 0.5)
 */
export async function postProcessMask(
    logits: Float32Array,
    maskH: number,
    maskW: number,
    preprocessed: SAMPreprocessed,
): Promise<Float32Array> {
    const { originalSize, resizedWidth, resizedHeight } = preprocessed;

    // Step 1 — upscale from maskH×maskW to MODEL_SIZE×MODEL_SIZE
    let img: ImageData = { data: logits, width: maskW, height: maskH, channels: 1 };
    img = await resize(img, { width: MODEL_SIZE, height: MODEL_SIZE }, "bilinear");

    // Step 2 — crop to the non-padded region
    const cropData = new Float32Array(resizedWidth * resizedHeight);
    for (let y = 0; y < resizedHeight; y++) {
        for (let x = 0; x < resizedWidth; x++) {
            cropData[y * resizedWidth + x] = img.data[y * MODEL_SIZE + x]!;
        }
    }
    img = { data: cropData, width: resizedWidth, height: resizedHeight, channels: 1 };

    // Step 3 — upscale to original size
    img = await resize(img, originalSize, "bilinear");

    // Step 4 — threshold at logit 0
    const binary = new Float32Array(img.data.length);
    for (let i = 0; i < img.data.length; i++) {
        binary[i] = img.data[i]! > 0 ? 1 : 0;
    }
    return binary;
}
