// Core tensor operations for image preprocessing.
// Generated files import from here — keep this API stable.

export interface ImageData {
    data: Float32Array;
    width: number;
    height: number;
    channels: number; // 1 (grayscale) | 3 (RGB) | 4 (RGBA)
}

export type ResampleFilter = "nearest" | "bilinear" | "bicubic";

/** Multiply every pixel by a scalar. Equivalent to numpy: arr * factor */
export function rescale(image: ImageData, factor: number): ImageData {
    const data = new Float32Array(image.data.length);
    for (let i = 0; i < image.data.length; i++) {
        data[i] = image.data[i]! * factor;
    }
    return { ...image, data };
}

/** Per-channel (value - mean) / std normalization. */
export function normalize(image: ImageData, mean: number[], std: number[]): ImageData {
    const data = new Float32Array(image.data.length);
    const c = image.channels;
    for (let i = 0; i < image.data.length; i++) {
        const ch = i % c;
        data[i] = (image.data[i]! - (mean[ch] ?? 0)) / (std[ch] ?? 1);
    }
    return { ...image, data };
}

/** Convert HWC layout to CHW. Equivalent to numpy: np.transpose(arr, (2, 0, 1)) */
export function hwcToChw(image: ImageData): Float32Array {
    const { data, width, height, channels: c } = image;
    const out = new Float32Array(data.length);
    for (let h = 0; h < height; h++) {
        for (let w = 0; w < width; w++) {
            for (let ch = 0; ch < c; ch++) {
                out[ch * height * width + h * width + w] = data[(h * width + w) * c + ch]!;
            }
        }
    }
    return out;
}

/** Crop an image to the given bounding box. */
export function crop(
    image: ImageData,
    box: { left: number; top: number; right: number; bottom: number },
): ImageData {
    const { left, top, right, bottom } = box;
    const newWidth = right - left;
    const newHeight = bottom - top;
    const { channels: c } = image;
    const data = new Float32Array(newWidth * newHeight * c);
    for (let h = 0; h < newHeight; h++) {
        for (let w = 0; w < newWidth; w++) {
            for (let ch = 0; ch < c; ch++) {
                data[(h * newWidth + w) * c + ch] =
                    image.data[((h + top) * image.width + (w + left)) * c + ch]!;
            }
        }
    }
    return { data, width: newWidth, height: newHeight, channels: c };
}

/** Center-crop an image to the target size. */
export function centerCrop(image: ImageData, size: { width: number; height: number }): ImageData {
    const left = Math.floor((image.width - size.width) / 2);
    const top = Math.floor((image.height - size.height) / 2);
    return crop(image, { left, top, right: left + size.width, bottom: top + size.height });
}

/** Pad an image with a constant value. */
export function pad(
    image: ImageData,
    padding: { top: number; bottom: number; left: number; right: number },
    value = 0,
): ImageData {
    const { top, bottom, left, right } = padding;
    const newWidth = image.width + left + right;
    const newHeight = image.height + top + bottom;
    const { channels: c } = image;
    const data = new Float32Array(newWidth * newHeight * c).fill(value);
    for (let h = 0; h < image.height; h++) {
        for (let w = 0; w < image.width; w++) {
            for (let ch = 0; ch < c; ch++) {
                data[((h + top) * newWidth + (w + left)) * c + ch] =
                    image.data[(h * image.width + w) * c + ch]!;
            }
        }
    }
    return { data, width: newWidth, height: newHeight, channels: c };
}

/** Resize an image to the target size.
 *  Async to accommodate both CPU and WebGPU paths.
 *  Injected by initRuntime() — call that before any preprocessing. */
export let resize: (
    image: ImageData,
    size: { width: number; height: number },
    filter?: ResampleFilter,
) => Promise<ImageData> = () =>
    Promise.reject(new Error("resize not initialized — call initRuntime() first"));

/** Injected by runtime/index.ts on startup. */
export function setResizeImpl(
    impl: (image: ImageData, size: { width: number; height: number }, filter?: ResampleFilter) => Promise<ImageData>,
): void {
    resize = impl;
}
