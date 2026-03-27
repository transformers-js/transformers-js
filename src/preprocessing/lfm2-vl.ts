import { resize, rescale, normalize, hwcToChw, crop, type ImageData } from "./ops.js";

/** Crop grid size used when tiling the input image. */
const TILE_SIZE = 512;
/** Encoder input resolution — SigLip2 native size, independent of TILE_SIZE. */
const ENCODER_SIZE = 768;

const IMAGE_MEAN = [0.5, 0.5, 0.5];
const IMAGE_STD  = [0.5, 0.5, 0.5];

export interface VLImageTensors {
    pixelValues:         Float32Array;    // [numTiles, 3, 512, 512]
    pixelAttentionMask:  BigInt64Array;   // [numTiles, 512, 512]
    spatialShapes:       BigInt64Array;   // [numTiles, 2]
    numTiles:            number;
}

/**
 * Choose the (rows, cols) tiling that maximises image resolution within
 * maxContentTiles. The thumbnail is handled separately, so this returns
 * only the content-tile grid.
 */
function bestTiling(w: number, h: number, maxContent: number): [number, number] {
    let bestRows = 1, bestCols = 1, bestScale = 0;
    for (let rows = 1; rows <= maxContent; rows++) {
        for (let cols = 1; cols <= maxContent; cols++) {
            if (rows * cols > maxContent) continue;
            const scale = Math.min(
                (rows * TILE_SIZE) / h,
                (cols * TILE_SIZE) / w,
            );
            if (scale > bestScale) {
                bestScale = scale;
                [bestRows, bestCols] = [rows, cols];
            }
        }
    }
    return [bestRows, bestCols];
}

async function normalizeTile(image: ImageData): Promise<Float32Array> {
    const resized    = await resize(image, { width: ENCODER_SIZE, height: ENCODER_SIZE }, "bilinear");
    const rescaled   = rescale(resized, 1 / 255);
    const normalized = normalize(rescaled, IMAGE_MEAN, IMAGE_STD);
    return hwcToChw(normalized);
}

/**
 * Tile an image for LFM2-VL:
 *   - Split into up to (maxTiles-1) content tiles arranged in a bestTiling grid
 *   - Append one thumbnail (whole image resized to 512×512)
 *
 * Returns flat Float32Array tensors ready to feed embed_images.onnx.
 */
export async function preprocessVLImage(
    image: ImageData,
    maxTiles = 10,
    useThumbnail = false,
): Promise<VLImageTensors> {
    const maxContent = useThumbnail ? maxTiles - 1 : maxTiles; // thumbnail occupies one slot
    const [rows, cols] = bestTiling(image.width, image.height, maxContent);

    const tilePxls: Float32Array[] = [];

    // Content tiles — divide image into rows×cols crops
    const cropW = Math.floor(image.width  / cols);
    const cropH = Math.floor(image.height / rows);

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const left   = c * cropW;
            const top    = r * cropH;
            const right  = c < cols - 1 ? left + cropW : image.width;
            const bottom = r < rows - 1 ? top  + cropH : image.height;
            tilePxls.push(await normalizeTile(crop(image, { left, top, right, bottom })));
        }
    }

    // Thumbnail — whole image at 512×512 (only when model uses it)
    if (useThumbnail) tilePxls.push(await normalizeTile(image));

    const numTiles    = tilePxls.length;
    const pixPerTile  = 3 * ENCODER_SIZE * ENCODER_SIZE;
    const pixelValues = new Float32Array(numTiles * pixPerTile);
    for (let i = 0; i < numTiles; i++) pixelValues.set(tilePxls[i]!, i * pixPerTile);

    // Attention mask and spatial shapes are kept for LiquidAI exports that expect them.
    // Community exports filter them out via inputNames validation.
    const pixelAttentionMask = new BigInt64Array(numTiles * ENCODER_SIZE * ENCODER_SIZE).fill(1n);
    const spatialShapes = new BigInt64Array(numTiles * 2);

    return { pixelValues, pixelAttentionMask, spatialShapes, numTiles };
}
