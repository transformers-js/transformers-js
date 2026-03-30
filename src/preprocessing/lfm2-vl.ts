// @generated
// Source: src/transformers/models/lfm2_vl/image_processing_lfm2_vl.py
// Do not edit manually

import { resize, rescale, normalize, crop, centerCrop, pad, hwcToChw, type ImageData } from "./ops.js";
import { ImageProcessor } from "./base.js";

const IMAGENET_STANDARD_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STANDARD_STD = [0.229, 0.224, 0.225];

export interface Lfm2VlImageProcessorConfig {
    downsampleFactor?: number;
    doImageSplitting?: boolean;
    minTiles?: number;
    maxTiles?: number;
    useThumbnail?: boolean;
    minImageTokens?: number;
    maxImageTokens?: number;
    encoderPatchSize?: number;
    tileSize?: number;
    maxPixelsTolerance?: number;
    doPad?: boolean;
    returnRowColInfo?: boolean;
    doResize?: boolean;
    doRescale?: boolean;
    doNormalize?: boolean;
    imageMean?: number[];
    imageStd?: number[];
}

function roundByFactor(number: number, factor: number): number {
    return Math.round(number / factor) * factor;
}

function findClosestAspectRatio(
    aspectRatio: number,
    targetRatios: Array<[number, number]>,
    width: number,
    height: number,
    imageSize: number
): [number, number] {
    let bestRatioDiff = Number.POSITIVE_INFINITY;
    let bestRatio: [number, number] = [1, 1];
    const area = width * height;

    for (const ratio of targetRatios) {
        const targetAspectRatio = ratio[0] / ratio[1];
        const ratioDiff = Math.abs(aspectRatio - targetAspectRatio);

        if (ratioDiff < bestRatioDiff) {
            bestRatioDiff = ratioDiff;
            bestRatio = ratio;
        } else if (ratioDiff === bestRatioDiff) {
            const targetArea = imageSize * imageSize * ratio[0] * ratio[1];
            if (area > 0.5 * targetArea) {
                bestRatio = ratio;
            }
        }
    }

    return bestRatio;
}

function getImageSizeForMaxNumPatches(
    imageHeight: number,
    imageWidth: number,
    patchSize: number,
    maxNumPatches: number,
    eps = 1e-5
): [number, number] {
    function getScaledImageSize(scale: number, size: number, patchSize: number): number {
        const scaledSize = size * scale;
        const ceilSize = Math.ceil(scaledSize / patchSize) * patchSize;
        return Math.max(patchSize, ceilSize);
    }

    let scaleMin = eps / 10;
    let scaleMax = 100.0;

    while (scaleMax - scaleMin >= eps) {
        const scale = (scaleMin + scaleMax) / 2;
        const targetHeight = getScaledImageSize(scale, imageHeight, patchSize);
        const targetWidth = getScaledImageSize(scale, imageWidth, patchSize);
        const numPatches = (targetHeight / patchSize) * (targetWidth / patchSize);

        if (numPatches > maxNumPatches) {
            scaleMax = scale;
        } else {
            scaleMin = scale;
        }
    }

    const finalScale = scaleMin;
    const targetHeight = getScaledImageSize(finalScale, imageHeight, patchSize);
    const targetWidth = getScaledImageSize(finalScale, imageWidth, patchSize);

    return [targetHeight, targetWidth];
}

function convertImageToPatches(images: Float32Array, patchSize: number, height: number, width: number, channels: number): Float32Array {
    const patchesH = Math.floor(height / patchSize);
    const patchesW = Math.floor(width / patchSize);
    const numPatches = patchesH * patchesW;
    const patchDim = patchSize * patchSize * channels;
    
    const patches = new Float32Array(numPatches * patchDim);
    
    for (let pH = 0; pH < patchesH; pH++) {
        for (let pW = 0; pW < patchesW; pW++) {
            const patchIdx = pH * patchesW + pW;
            for (let c = 0; c < channels; c++) {
                for (let h = 0; h < patchSize; h++) {
                    for (let w = 0; w < patchSize; w++) {
                        const pixelH = pH * patchSize + h;
                        const pixelW = pW * patchSize + w;
                        const imageIdx = (pixelH * width + pixelW) * channels + c;
                        const patchIdx_full = patchIdx * patchDim + c * patchSize * patchSize + h * patchSize + w;
                        patches[patchIdx_full] = images[imageIdx];
                    }
                }
            }
        }
    }
    
    return patches;
}

function padAlongFirstDim(tensor: Float32Array, targetLength: number, patchDim: number): Float32Array {
    const currentLength = tensor.length / patchDim;
    if (currentLength >= targetLength) {
        return tensor;
    }
    
    const padded = new Float32Array(targetLength * patchDim);
    padded.set(tensor);
    return padded;
}

export class Lfm2VlImageProcessor implements ImageProcessor {
    private downsampleFactor = 2;
    private doImageSplitting = true;
    private minTiles = 2;
    private maxTiles = 10;
    private useThumbnail = true;
    private minImageTokens = 64;
    private maxImageTokens = 256;
    private encoderPatchSize = 16;
    private tileSize = 512;
    private maxPixelsTolerance = 2.0;
    private doPad = true;
    private returnRowColInfo = false;
    private doResize = true;
    private doRescale = true;
    private doNormalize = true;
    private imageMean = IMAGENET_STANDARD_MEAN;
    private imageStd = IMAGENET_STANDARD_STD;
    private maxNumPatches: number;
    private targetRatiosCache = new Map<string, Array<[number, number]>>();

    constructor(config: Lfm2VlImageProcessorConfig = {}) {
        Object.assign(this, config);
        
        const maxThumbnailImagePatches = this.maxImageTokens * (this.downsampleFactor ** 2);
        const tileSizePatches = this.doImageSplitting ? (this.tileSize / this.encoderPatchSize) ** 2 : 0;
        this.maxNumPatches = Math.max(maxThumbnailImagePatches, tileSizePatches);
    }

    private getTargetRatios(minTiles: number, maxTiles: number): Array<[number, number]> {
        const key = `${minTiles}-${maxTiles}`;
        if (this.targetRatiosCache.has(key)) {
            return this.targetRatiosCache.get(key)!;
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

        const uniqueRatios = Array.from(new Set(ratios.map(r => `${r[0]},${r[1]}`)))
            .map(s => s.split(',').map(Number) as [number, number])
            .sort((a, b) => a[0] * a[1] - b[0] * b[1]);

        this.targetRatiosCache.set(key, uniqueRatios);
        return uniqueRatios;
    }

    private getGridLayout(height: number, width: number, minTiles: number, maxTiles: number, tileSize: number): [number, number] {
        const aspectRatio = width / height;
        const targetRatios = this.getTargetRatios(minTiles, maxTiles);
        
        const [gridWidth, gridHeight] = findClosestAspectRatio(aspectRatio, targetRatios, width, height, tileSize);
        
        return [gridWidth, gridHeight];
    }

    private async cropImageToPatches(
        image: ImageData,
        gridWidth: number,
        gridHeight: number,
        tileSize: number,
        encoderPatchSize: number,
        doResize: boolean,
        doRescale: boolean,
        doNormalize: boolean,
        imageMean: number[],
        imageStd: number[]
    ): Promise<{ patches: Float32Array; spatialShapes: Array<[number, number]> }> {
        const cropWidth = Math.floor(image.width / gridWidth);
        const cropHeight = Math.floor(image.height / gridHeight);
        
        const allPatches: Float32Array[] = [];
        const spatialShapes: Array<[number, number]> = [];
        
        for (let row = 0; row < gridHeight; row++) {
            for (let col = 0; col < gridWidth; col++) {
                const left = col * cropWidth;
                const top = row * cropHeight;
                const right = col === gridWidth - 1 ? image.width : left + cropWidth;
                const bottom = row === gridHeight - 1 ? image.height : top + cropHeight;
                
                let croppedImage = crop(image, { left, top, right, bottom });
                
                if (doResize) {
                    croppedImage = await resize(croppedImage, { width: tileSize, height: tileSize }, "bilinear");
                }
                
                if (doRescale) {
                    croppedImage = rescale(croppedImage, 1 / 255);
                }
                
                if (doNormalize) {
                    croppedImage = normalize(croppedImage, imageMean, imageStd);
                }
                
                const patches = convertImageToPatches(
                    croppedImage.data,
                    encoderPatchSize,
                    croppedImage.height,
                    croppedImage.width,
                    croppedImage.channels
                );
                
                allPatches.push(patches);
                const patchesH = Math.floor(croppedImage.height / encoderPatchSize);
                const patchesW = Math.floor(croppedImage.width / encoderPatchSize);
                spatialShapes.push([patchesH, patchesW]);
            }
        }
        
        const totalPatches = allPatches.reduce((sum, p) => sum + p.length, 0);
        const combinedPatches = new Float32Array(totalPatches);
        let offset = 0;
        for (const patches of allPatches) {
            combinedPatches.set(patches, offset);
            offset += patches.length;
        }
        
        return { patches: combinedPatches, spatialShapes };
    }

    private async smartResize(
        image: ImageData,
        maxImageTokens: number,
        encoderPatchSize: number,
        downsampleFactor: number
    ): Promise<ImageData> {
        const maxNumPatches = maxImageTokens * (downsampleFactor ** 2);
        const [targetHeight, targetWidth] = getImageSizeForMaxNumPatches(
            image.height,
            image.width,
            encoderPatchSize * downsampleFactor,
            maxNumPatches
        );
        
        return await resize(image, { width: targetWidth, height: targetHeight }, "bilinear");
    }

    private isImageTooLarge(
        image: ImageData,
        maxImageTokens: number,
        encoderPatchSize: number,
        downsampleFactor: number,
        maxPixelsTolerance: number
    ): boolean {
        const maxPixels = maxImageTokens * (encoderPatchSize ** 2) * (downsampleFactor ** 2) * maxPixelsTolerance;
        const imagePixels = image.width * image.height;
        return imagePixels > maxPixels;
    }

    private async resizeAndSplit(
        image: ImageData,
        doImageSplitting: boolean,
        minTiles: number,
        maxTiles: number,
        useThumbnail: boolean,
        tileSize: number,
        minImageTokens: number,
        maxImageTokens: number,
        encoderPatchSize: number,
        downsampleFactor: number,
        maxPixelsTolerance: number,
        doResize: boolean,
        doRescale: boolean,
        doNormalize: boolean,
        imageMean: number[],
        imageStd: number[]
    ): Promise<{ patches: Float32Array; spatialShapes: Array<[number, number]> }> {
        if (!doImageSplitting) {
            const resizedImage = await this.smartResize(image, maxImageTokens, encoderPatchSize, downsampleFactor);
            
            let processedImage = resizedImage;
            if (doRescale) {
                processedImage = rescale(processedImage, 1 / 255);
            }
            if (doNormalize) {
                processedImage = normalize(processedImage, imageMean, imageStd);
            }
            
            const patches = convertImageToPatches(
                processedImage.data,
                encoderPatchSize,
                processedImage.height,
                processedImage.width,
                processedImage.channels
            );
            
            const patchesH = Math.floor(processedImage.height / encoderPatchSize);
            const patchesW = Math.floor(processedImage.width / encoderPatchSize);
            
            return { patches, spatialShapes: [[patchesH, patchesW]] };
        }

        const isLarge = this.isImageTooLarge(image, maxImageTokens, encoderPatchSize, downsampleFactor, maxPixelsTolerance);
        
        if (isLarge) {
            const [gridWidth, gridHeight] = this.getGridLayout(image.height, image.width, minTiles, maxTiles, tileSize);
            const { patches: tilePatches, spatialShapes: tileSpatialShapes } = await this.cropImageToPatches(
                image, gridWidth, gridHeight, tileSize, encoderPatchSize,
                doResize, doRescale, doNormalize, imageMean, imageStd
            );
            
            if (useThumbnail) {
                let thumbnailImage = await this.smartResize(image, maxImageTokens, encoderPatchSize, downsampleFactor);
                if (doRescale) {
                    thumbnailImage = rescale(thumbnailImage, 1 / 255);
                }
                if (doNormalize) {
                    thumbnailImage = normalize(thumbnailImage, imageMean, imageStd);
                }
                
                const thumbnailPatches = convertImageToPatches(
                    thumbnailImage.data,
                    encoderPatchSize,
                    thumbnailImage.height,
                    thumbnailImage.width,
                    thumbnailImage.channels
                );
                
                const thumbnailPatchesH = Math.floor(thumbnailImage.height / encoderPatchSize);
                const thumbnailPatchesW = Math.floor(thumbnailImage.width / encoderPatchSize);
                
                const combinedPatches = new Float32Array(tilePatches.length + thumbnailPatches.length);
                combinedPatches.set(tilePatches);
                combinedPatches.set(thumbnailPatches, tilePatches.length);
                
                return {
                    patches: combinedPatches,
                    spatialShapes: [...tileSpatialShapes, [thumbnailPatchesH, thumbnailPatchesW]]
                };
            }
            
            return { patches: tilePatches, spatialShapes: tileSpatialShapes };
        } else {
            let resizedImage = await this.smartResize(image, maxImageTokens, encoderPatchSize, downsampleFactor);
            if (doRescale) {
                resizedImage = rescale(resizedImage, 1 / 255);
            }
            if (doNormalize) {
                resizedImage = normalize(resizedImage, imageMean, imageStd);
            }
            
            const patches = convertImageToPatches(
                resizedImage.data,
                encoderPatchSize,
                resizedImage.height,
                resizedImage.width,
                resizedImage.channels
            );
            
            const patchesH = Math.floor(resizedImage.height / encoderPatchSize);
            const patchesW = Math.floor(resizedImage.width / encoderPatchSize);
            
            return { patches, spatialShapes: [[patchesH, patchesW]] };
        }
    }

    async preprocess(image: ImageData, config: Lfm2VlImageProcessorConfig = {}): Promise<Float32Array> {
        const {
            doImageSplitting = this.doImageSplitting,
            minTiles = this.minTiles,
            maxTiles = this.maxTiles,
            useThumbnail = this.useThumbnail,
            minImageTokens = this.minImageTokens,
            maxImageTokens = this.maxImageTokens,
            encoderPatchSize = this.encoderPatchSize,
            tileSize = this.tileSize,
            maxPixelsTolerance = this.maxPixelsTolerance,
            downsampleFactor = this.downsampleFactor,
            doPad = this.doPad,
            doResize = this.doResize,
            doRescale = this.doRescale,
            doNormalize = this.doNormalize,
            imageMean = this.imageMean,
            imageStd = this.imageStd
        } = config;

        let actualMinTiles = minTiles;
        let actualMaxTiles = maxTiles;
        
        if (!doImageSplitting) {
            actualMinTiles = 1;
            actualMaxTiles = 1;
        }

        if (doImageSplitting && actualMinTiles > actualMaxTiles) {
            throw new Error("minTiles must be less than or equal to maxTiles");
        }

        const { patches, spatialShapes } = await this.resizeAndSplit(
            image,
            doImageSplitting,
            actualMinTiles,
            actualMaxTiles,
            useThumbnail,
            tileSize,
            minImageTokens,
            maxImageTokens,
            encoderPatchSize,
            downsampleFactor,
            maxPixelsTolerance,
            doResize,
            doRescale,
            doNormalize,
            imageMean,
            imageStd
        );

        // For consistency with the existing interface, return just the patches
        // The full result with spatial shapes would be available through a different method
        return patches;
    }
}
