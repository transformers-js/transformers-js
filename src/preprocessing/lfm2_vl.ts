/**
 * @generated
 * This file was automatically generated from:
 * src/transformers/models/lfm2_vl/image_processing_lfm2_vl.py
 * 
 * Do not edit manually.
 */

import { ImageData, ImageProcessor } from "../base.js";
import { 
    rescale, 
    normalize, 
    hwcToChw, 
    resize, 
    crop, 
    centerCrop, 
    pad 
} from "../ops.js";

export interface Lfm2VlImageProcessorConfig {
    downsample_factor?: number;
    do_image_splitting?: boolean;
    min_tiles?: number;
    max_tiles?: number;
    use_thumbnail?: boolean;
    min_image_tokens?: number;
    max_image_tokens?: number;
    encoder_patch_size?: number;
    tile_size?: number;
    max_pixels_tolerance?: number;
    do_resize?: boolean;
    size?: { height: number; width: number };
    resample?: string;
    do_rescale?: boolean;
    rescale_factor?: number;
    do_normalize?: boolean;
    do_pad?: boolean;
    return_row_col_info?: boolean;
    image_mean?: number[] | number;
    image_std?: number[] | number;
}

function roundByFactor(number: number, factor: number): number {
    return Math.round(number / factor) * factor;
}

function findClosestAspectRatio(
    aspectRatio: number,
    targetRatios: Array<[number, number]>,
    width: number,
    height: number,
    imageSize: number,
): [number, number] {
    let bestRatioDiff = Number.POSITIVE_INFINITY;
    let bestRatio: [number, number] = [1, 1];
    const area = width * height;

    for (const ratio of targetRatios) {
        const targetAspectRatio = ratio[0] / ratio[1];
        const ratioDiff = Math.abs(aspectRatio - targetAspectRatio);

        // update best ratio if we found a closer match
        if (ratioDiff < bestRatioDiff) {
            bestRatioDiff = ratioDiff;
            bestRatio = ratio;
        }
        // if equally close, prefer the ratio that better matches the original image area
        else if (ratioDiff === bestRatioDiff) {
            const targetArea = imageSize * imageSize * ratio[0] * ratio[1];
            if (area > 0.5 * targetArea) {
                bestRatio = ratio;
            }
        }
    }

    return bestRatio;
}

// Cache for getImageSizeForMaxNumPatches
const imageSizeCache = new Map<string, [number, number]>();

function getImageSizeForMaxNumPatches(
    imageHeight: number,
    imageWidth: number,
    patchSize: number,
    maxNumPatches: number,
    eps: number = 1e-5
): [number, number] {
    const cacheKey = `${imageHeight}-${imageWidth}-${patchSize}-${maxNumPatches}-${eps}`;
    if (imageSizeCache.has(cacheKey)) {
        return imageSizeCache.get(cacheKey)!;
    }

    const aspectRatio = imageWidth / imageHeight;
    let maxHeight = Math.sqrt(maxNumPatches / aspectRatio) * patchSize;
    let maxWidth = aspectRatio * maxHeight;

    maxHeight = Math.floor(maxHeight / patchSize) * patchSize;
    maxWidth = Math.floor(maxWidth / patchSize) * patchSize;

    // ensure at least one patch in each dimension
    maxHeight = Math.max(maxHeight, patchSize);
    maxWidth = Math.max(maxWidth, patchSize);

    const result: [number, number] = [maxHeight, maxWidth];
    imageSizeCache.set(cacheKey, result);
    return result;
}

export class Lfm2VlImageProcessor implements ImageProcessor {
    downsampleFactor: number = 2;
    doImageSplitting: boolean = true;
    minTiles: number = 2;
    maxTiles: number = 10;
    useThumbnail: boolean = true;
    minImageTokens: number = 64;
    maxImageTokens: number = 256;
    encoderPatchSize: number = 16;
    tileSize: number = 512;
    maxPixelsTolerance: number = 2.0;
    doResize: boolean = true;
    size: { height: number; width: number } = { height: 512, width: 512 };
    resample: string = "bilinear";
    doRescale: boolean = true;
    rescaleFactor: number = 1 / 255;
    doNormalize: boolean = true;
    doPad: boolean = true;
    returnRowColInfo: boolean = false;
    imageMean: number[] = [0.485, 0.456, 0.406]; // IMAGENET_STANDARD_MEAN
    imageStd: number[] = [0.229, 0.224, 0.225]; // IMAGENET_STANDARD_STD
    maxNumPatches: number;

    // Cache for target ratios
    private targetRatiosCache = new Map<string, Array<[number, number]>>();

    constructor(config: Lfm2VlImageProcessorConfig = {}) {
        if (config.downsample_factor !== undefined) this.downsampleFactor = config.downsample_factor;
        if (config.do_image_splitting !== undefined) this.doImageSplitting = config.do_image_splitting;
        if (config.min_tiles !== undefined) this.minTiles = config.min_tiles;
        if (config.max_tiles !== undefined) this.maxTiles = config.max_tiles;
        if (config.use_thumbnail !== undefined) this.useThumbnail = config.use_thumbnail;
        if (config.min_image_tokens !== undefined) this.minImageTokens = config.min_image_tokens;
        if (config.max_image_tokens !== undefined) this.maxImageTokens = config.max_image_tokens;
        if (config.encoder_patch_size !== undefined) this.encoderPatchSize = config.encoder_patch_size;
        if (config.tile_size !== undefined) this.tileSize = config.tile_size;
        if (config.max_pixels_tolerance !== undefined) this.maxPixelsTolerance = config.max_pixels_tolerance;
        if (config.do_resize !== undefined) this.doResize = config.do_resize;
        if (config.size !== undefined) this.size = config.size;
        if (config.resample !== undefined) this.resample = config.resample;
        if (config.do_rescale !== undefined) this.doRescale = config.do_rescale;
        if (config.rescale_factor !== undefined) this.rescaleFactor = config.rescale_factor;
        if (config.do_normalize !== undefined) this.doNormalize = config.do_normalize;
        if (config.do_pad !== undefined) this.doPad = config.do_pad;
        if (config.return_row_col_info !== undefined) this.returnRowColInfo = config.return_row_col_info;
        if (config.image_mean !== undefined) {
            this.imageMean = Array.isArray(config.image_mean) ? config.image_mean : [config.image_mean];
        }
        if (config.image_std !== undefined) {
            this.imageStd = Array.isArray(config.image_std) ? config.image_std : [config.image_std];
        }

        const maxThumbnailImagePatches = this.maxImageTokens * this.downsampleFactor ** 2;
        const tileSizePatches = this.doImageSplitting ? (this.tileSize / this.encoderPatchSize) ** 2 : 0;
        this.maxNumPatches = Math.max(maxThumbnailImagePatches, tileSizePatches);
    }

    private getTargetRatios(minTiles: number, maxTiles: number): Array<[number, number]> {
        const cacheKey = `${minTiles}-${maxTiles}`;
        if (this.targetRatiosCache.has(cacheKey)) {
            return this.targetRatiosCache.get(cacheKey)!;
        }

        const ratios: Array<[number, number]> = [];
        for (let n = minTiles; n <= maxTiles; n++) {
            for (let w = 1; w <= n; w++) {
                for (let h = 1; h <= n; h++) {
                    if (minTiles <= w * h && w * h <= maxTiles) {
                        ratios.push([w, h]);
                    }
                }
            }
        }

        // Remove duplicates and sort by area (w * h)
        const uniqueRatios = Array.from(new Set(ratios.map(r => JSON.stringify(r))))
            .map(s => JSON.parse(s) as [number, number])
            .sort((a, b) => a[0] * a[1] - b[0] * b[1]);

        this.targetRatiosCache.set(cacheKey, uniqueRatios);
        return uniqueRatios;
    }

    private getGridLayout(
        height: number,
        width: number,
        minTiles: number,
        maxTiles: number,
        tileSize: number,
    ): [number, number] {
        const aspectRatio = width / height;
        const targetRatios = this.getTargetRatios(minTiles, maxTiles);

        // find best matching grid configuration
        const [gridWidth, gridHeight] = findClosestAspectRatio(aspectRatio, targetRatios, width, height, tileSize);

        return [gridWidth, gridHeight];
    }

    preprocess(images: ImageData | ImageData[]): Promise<Record<string, any>> {
        throw new Error("Lfm2VlImageProcessor.preprocess() not yet implemented");
    }
}