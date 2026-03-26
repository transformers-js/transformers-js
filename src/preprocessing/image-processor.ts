import { fetchJSON } from "../runtime/hub.js";
import { resize, rescale, normalize, hwcToChw, centerCrop } from "./ops.js";
import type { ImageData, ResampleFilter } from "./ops.js";
import type { ImageProcessor as IImageProcessor } from "./base.js";

// Mirrors the shape of preprocessor_config.json on the HF Hub.
// PIL resample integers: 0=NEAREST, 2=BILINEAR, 3=BICUBIC, 1=LANCZOS
// Lanczos maps to bicubic — not yet implemented; same approximation used in benchmark.
const PIL_RESAMPLE: Record<number, ResampleFilter> = {
    0: "nearest",
    2: "bilinear",
    3: "bicubic",
    1: "bicubic",
};

interface RawPreprocessorConfig {
    image_processor_type?: string;
    do_resize?: boolean;
    size?: { height?: number; width?: number; shortest_edge?: number };
    resample?: number;
    do_center_crop?: boolean;
    crop_size?: { height?: number; width?: number };
    do_rescale?: boolean;
    rescale_factor?: number;
    do_normalize?: boolean;
    image_mean?: number[];
    image_std?: number[];
    do_convert_rgb?: boolean;
}

export interface ProcessorConfig {
    do_resize: boolean;
    size: { height: number; width: number };
    resample: ResampleFilter;
    do_center_crop: boolean;
    crop_size: { height: number; width: number };
    do_rescale: boolean;
    rescale_factor: number;
    do_normalize: boolean;
    image_mean: number[];
    image_std: number[];
}

function normalizeSize(raw: RawPreprocessorConfig["size"], fallback: number): { height: number; width: number } {
    if (!raw) return { height: fallback, width: fallback };
    if (raw.shortest_edge != null) return { height: raw.shortest_edge, width: raw.shortest_edge };
    return { height: raw.height ?? fallback, width: raw.width ?? fallback };
}

function fromRaw(raw: RawPreprocessorConfig): ProcessorConfig {
    const size = normalizeSize(raw.size, 224);
    return {
        do_resize:      raw.do_resize      ?? true,
        size,
        resample:       PIL_RESAMPLE[raw.resample ?? 3] ?? "bicubic",
        do_center_crop: raw.do_center_crop ?? true,
        crop_size:      normalizeSize(raw.crop_size, size.height),
        do_rescale:     raw.do_rescale     ?? true,
        rescale_factor: raw.rescale_factor ?? 1 / 255,
        do_normalize:   raw.do_normalize   ?? true,
        image_mean:     raw.image_mean     ?? [0.5, 0.5, 0.5],
        image_std:      raw.image_std      ?? [0.5, 0.5, 0.5],
    };
}

export class ImageProcessor implements IImageProcessor {
    constructor(readonly config: ProcessorConfig) {}

    static async fromHub(modelId: string): Promise<ImageProcessor> {
        const raw = await fetchJSON<RawPreprocessorConfig>(modelId, "preprocessor_config.json");
        return new ImageProcessor(fromRaw(raw));
    }

    static fromConfig(config: ProcessorConfig): ImageProcessor {
        return new ImageProcessor(config);
    }

    /** Returns a CHW float32 tensor with a leading batch dim: [1, C, H, W]. */
    async preprocess(image: ImageData, override: Partial<ProcessorConfig> = {}): Promise<Float32Array> {
        const cfg = { ...this.config, ...override };

        let img = image;
        if (cfg.do_resize)      img = await resize(img, cfg.size, cfg.resample);
        if (cfg.do_center_crop) img = centerCrop(img, cfg.crop_size);
        if (cfg.do_rescale)     img = rescale(img, cfg.rescale_factor);
        if (cfg.do_normalize)   img = normalize(img, cfg.image_mean, cfg.image_std);

        return hwcToChw(img);
    }
}
