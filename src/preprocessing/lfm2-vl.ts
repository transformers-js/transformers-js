import { resize, rescale, normalize, crop, type ImageData } from "./ops.js";

const TILE_SIZE = 512;         // crop grid size
const PATCH_SIZE = 16;         // encoder patch size
const PATCHES_PER_SIDE = TILE_SIZE / PATCH_SIZE; // 32
const MAX_PATCHES = PATCHES_PER_SIDE * PATCHES_PER_SIDE; // 1024
const PATCH_DIM = PATCH_SIZE * PATCH_SIZE * 3; // 768

const IMAGE_MEAN = [0.5, 0.5, 0.5];
const IMAGE_STD  = [0.5, 0.5, 0.5];

export interface VLImageTensors {
    /** [num_tiles, MAX_PATCHES, PATCH_DIM] — NaFlex patch sequence */
    pixelValues:         Float32Array;
    /** [num_tiles, MAX_PATCHES] — 1 for valid patches, 0 for padding */
    pixelAttentionMask:  BigInt64Array;
    /** [num_tiles, 2] — [patches_h, patches_w] per tile */
    spatialShapes:       BigInt64Array;
    numTiles:            number;
}

/**
 * Choose the (rows, cols) tiling that maximises image resolution within
 * maxContentTiles.
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

/**
 * Resize a tile to TILE_SIZE×TILE_SIZE, rescale, normalize, then extract
 * 16×16 patches in raster order. Each patch is flattened in CHW order
 * (all red values, then green, then blue) to a 768-dim vector.
 *
 * Returns patches [MAX_PATCHES, PATCH_DIM] and a mask [MAX_PATCHES] of 1n.
 */
async function tileToPatches(image: ImageData): Promise<{ patches: Float32Array; mask: BigInt64Array }> {
    const resized    = await resize(image, { width: TILE_SIZE, height: TILE_SIZE }, "bilinear");
    const rescaled   = rescale(resized, 1 / 255);
    const normalized = normalize(rescaled, IMAGE_MEAN, IMAGE_STD);

    const { data } = normalized; // HWC [TILE_SIZE, TILE_SIZE, 3]
    const W = TILE_SIZE, C = 3;
    const ph = PATCH_SIZE, pw = PATCH_SIZE;

    const patches = new Float32Array(MAX_PATCHES * PATCH_DIM);

    for (let pRow = 0; pRow < PATCHES_PER_SIDE; pRow++) {
        for (let pCol = 0; pCol < PATCHES_PER_SIDE; pCol++) {
            const patchIdx = pRow * PATCHES_PER_SIDE + pCol;
            for (let c = 0; c < C; c++) {
                for (let h = 0; h < ph; h++) {
                    for (let w = 0; w < pw; w++) {
                        const pixelPos = (pRow * ph + h) * W * C + (pCol * pw + w) * C + c;
                        const patchPos = patchIdx * PATCH_DIM + c * ph * pw + h * pw + w;
                        patches[patchPos] = data[pixelPos]!;
                    }
                }
            }
        }
    }

    return { patches, mask: new BigInt64Array(MAX_PATCHES).fill(1n) };
}

/**
 * Tile an image for LFM2-VL and return NaFlex patch tensors:
 *   pixel_values [num_tiles, MAX_PATCHES, PATCH_DIM]
 *   pixel_attention_mask [num_tiles, MAX_PATCHES]
 *   spatial_shapes [num_tiles, 2]
 */
export async function preprocessVLImage(
    image: ImageData,
    maxTiles = 10,
    useThumbnail = false,
): Promise<VLImageTensors> {
    const maxContent = useThumbnail ? maxTiles - 1 : maxTiles;
    const [rows, cols] = bestTiling(image.width, image.height, maxContent);

    const patchArrays: Float32Array[] = [];
    const maskArrays: BigInt64Array[] = [];

    const cropW = Math.floor(image.width  / cols);
    const cropH = Math.floor(image.height / rows);

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const left   = c * cropW;
            const top    = r * cropH;
            const right  = c < cols - 1 ? left + cropW : image.width;
            const bottom = r < rows - 1 ? top  + cropH : image.height;
            const { patches, mask } = await tileToPatches(crop(image, { left, top, right, bottom }));
            patchArrays.push(patches);
            maskArrays.push(mask);
        }
    }

    if (useThumbnail) {
        const { patches, mask } = await tileToPatches(image);
        patchArrays.push(patches);
        maskArrays.push(mask);
    }

    const numTiles = patchArrays.length;
    const pixelValues = new Float32Array(numTiles * MAX_PATCHES * PATCH_DIM);
    const pixelAttentionMask = new BigInt64Array(numTiles * MAX_PATCHES);
    for (let i = 0; i < numTiles; i++) {
        pixelValues.set(patchArrays[i]!, i * MAX_PATCHES * PATCH_DIM);
        pixelAttentionMask.set(maskArrays[i]!, i * MAX_PATCHES);
    }

    const spatialShapes = new BigInt64Array(numTiles * 2);
    for (let i = 0; i < numTiles; i++) {
        spatialShapes[i * 2]     = BigInt(PATCHES_PER_SIDE); // 32
        spatialShapes[i * 2 + 1] = BigInt(PATCHES_PER_SIDE); // 32
    }

    return { pixelValues, pixelAttentionMask, spatialShapes, numTiles };
}
