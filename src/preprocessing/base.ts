import type { ImageData } from "./ops.js";

export interface ImageProcessor {
    preprocess(image: ImageData, config?: Record<string, unknown>): Float32Array;
}

export interface FeatureExtractor {
    extract(audio: Float32Array, sampleRate: number): Float32Array;
}
