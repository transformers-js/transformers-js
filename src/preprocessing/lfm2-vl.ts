import { resize, rescale, normalize, crop, type ImageData } from "./ops.js";

const TILE_SIZE = 512;
const PATCH_SIZE = 16;
const PATCHES_PER_SIDE = TILE_SIZE / PATCH_SIZE; // 32
const MAX_PATCHES = PATCHES_PER_SIDE * PATCHES_PER_SIDE; // 1024
const PATCH_DIM = PATCH_SIZE * PATCH_SIZE * 3; // 768

const IMAGE_MEAN = [0.5, 0.5, 0.5];
const IMAGE_STD  = [0.5, 0.5, 0.5];

export interface VLImageTensors {
    /** community: [num_tiles, MAX_PATCHES, PATCH_DIM]; liquidai: [num_tiles, 3, 512, 512] */
    pixelValues:         Float32Array;
    /** community: [num_tiles, MAX_PATCHES]; liquidai: [num_tiles, 512, 512] */
    pixelAttentionMask:  BigInt64Array;
    /** [num_tiles, 2] — [patches_h, patches_w] per tile */
    spatialShapes:       BigInt64Array;
    numTiles:            number;
}

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
 * Resize a tile to 512×512, rescale, normalize, then extract 16×16 patches
 * in raster order. Each patch is flattened in CHW order to a 768-dim vector.
 * Returns patches [MAX_PATCHES, PATCH_DIM] and mask [MAX_PATCHES] of 1n.
 */
async function tileToPatches(image: ImageData): Promise<{ patches: Float32Array; mask: BigInt64Array }> {
    const resized    = await resize(image, { width: TILE_SIZE, height: TILE_SIZE }, "bilinear");
    const rescaled   = rescale(resized, 1 / 255);
    const normalized = normalize(rescaled, IMAGE_MEAN, IMAGE_STD);

    const { data } = normalized; // HWC [512, 512, 3]
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
 * Resize a tile to 512×512, rescale, normalize, then convert HWC→CHW.
 * Returns chw [3, 512, 512] and mask [512×512] of 1n.
 */
async function tileToCHW(image: ImageData): Promise<{ chw: Float32Array; mask: BigInt64Array }> {
    const resized    = await resize(image, { width: TILE_SIZE, height: TILE_SIZE }, "bilinear");
    const rescaled   = rescale(resized, 1 / 255);
    const normalized = normalize(rescaled, IMAGE_MEAN, IMAGE_STD);

    const { data } = normalized; // HWC [512, 512, 3]
    const H = TILE_SIZE, W = TILE_SIZE, C = 3;
    const chw = new Float32Array(C * H * W);
    for (let c = 0; c < C; c++) {
        for (let h = 0; h < H; h++) {
            for (let w = 0; w < W; w++) {
                chw[c * H * W + h * W + w] = data[(h * W + w) * C + c]!;
            }
        }
    }
    return { chw, mask: new BigInt64Array(H * W).fill(1n) };
}

export async function preprocessVLImage(
    image: ImageData,
    maxTiles = 10,
    useThumbnail = false,
    flavor: "liquidai" | "community" = "community",
): Promise<VLImageTensors> {
    const maxContent = useThumbnail ? maxTiles - 1 : maxTiles;
    const [rows, cols] = bestTiling(image.width, image.height, maxContent);

    const cropW = Math.floor(image.width  / cols);
    const cropH = Math.floor(image.height / rows);

    if (flavor === "liquidai") {
        const chwArrays:  Float32Array[]   = [];
        const maskArrays: BigInt64Array[] = [];

        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const left   = c * cropW;
                const top    = r * cropH;
                const right  = c < cols - 1 ? left + cropW : image.width;
                const bottom = r < rows - 1 ? top  + cropH : image.height;
                const { chw, mask } = await tileToCHW(crop(image, { left, top, right, bottom }));
                chwArrays.push(chw);
                maskArrays.push(mask);
            }
        }
        if (useThumbnail) {
            const { chw, mask } = await tileToCHW(image);
            chwArrays.push(chw);
            maskArrays.push(mask);
        }

        const numTiles = chwArrays.length;
        const tilePixels = 3 * TILE_SIZE * TILE_SIZE;
        const pixelValues = new Float32Array(numTiles * tilePixels);
        const pixelAttentionMask = new BigInt64Array(numTiles * TILE_SIZE * TILE_SIZE);
        for (let i = 0; i < numTiles; i++) {
            pixelValues.set(chwArrays[i]!, i * tilePixels);
            pixelAttentionMask.set(maskArrays[i]!, i * TILE_SIZE * TILE_SIZE);
        }

        const spatialShapes = new BigInt64Array(numTiles * 2);
        for (let i = 0; i < numTiles; i++) {
            spatialShapes[i * 2]     = BigInt(PATCHES_PER_SIDE);
            spatialShapes[i * 2 + 1] = BigInt(PATCHES_PER_SIDE);
        }

        return { pixelValues, pixelAttentionMask, spatialShapes, numTiles };
    }

    // community: NaFlex patch format
    const patchArrays: Float32Array[]  = [];
    const maskArrays:  BigInt64Array[] = [];

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
        spatialShapes[i * 2]     = BigInt(PATCHES_PER_SIDE);
        spatialShapes[i * 2 + 1] = BigInt(PATCHES_PER_SIDE);
    }

    return { pixelValues, pixelAttentionMask, spatialShapes, numTiles };
}
