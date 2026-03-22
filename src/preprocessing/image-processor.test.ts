import { describe, it, expect, beforeAll } from "vitest";
import { ImageProcessor } from "./image-processor.js";
import { cpuResize } from "./resize/cpu.js";
import { setResizeImpl } from "./ops.js";
import type { ImageData } from "./ops.js";

// Use CPU resize in tests (no WebGPU in vitest)
beforeAll(() => setResizeImpl(cpuResize));

function solidRGB(width: number, height: number, r: number, g: number, b: number): ImageData {
    const data = new Float32Array(width * height * 3);
    for (let i = 0; i < width * height; i++) {
        data[i * 3 + 0] = r;
        data[i * 3 + 1] = g;
        data[i * 3 + 2] = b;
    }
    return { data, width, height, channels: 3 };
}

describe("ImageProcessor.preprocess", () => {
    it("returns a [1, C, H, W] CHW tensor", async () => {
        const proc = ImageProcessor.fromConfig({
            do_resize: true,
            size: { height: 224, width: 224 },
            resample: "bicubic",
            do_center_crop: false,
            crop_size: { height: 224, width: 224 },
            do_rescale: true,
            rescale_factor: 1 / 255,
            do_normalize: false,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
        });

        const img = solidRGB(64, 64, 128, 64, 32);
        const out = await proc.preprocess(img);

        expect(out.length).toBe(3 * 224 * 224);
    });

    it("rescales pixel values to [0, 1]", async () => {
        const proc = ImageProcessor.fromConfig({
            do_resize: false,
            size: { height: 4, width: 4 },
            resample: "bilinear",
            do_center_crop: false,
            crop_size: { height: 4, width: 4 },
            do_rescale: true,
            rescale_factor: 1 / 255,
            do_normalize: false,
            image_mean: [0, 0, 0],
            image_std: [1, 1, 1],
        });

        const img = solidRGB(4, 4, 255, 0, 128);
        const out = await proc.preprocess(img);

        for (let ch = 0; ch < 3 * 4 * 4; ch++) {
            expect(out[ch]).toBeGreaterThanOrEqual(0);
            expect(out[ch]).toBeLessThanOrEqual(1 + 1e-5);
        }
    });

    it("normalize shifts values by mean/std", async () => {
        const proc = ImageProcessor.fromConfig({
            do_resize: false,
            size: { height: 2, width: 2 },
            resample: "bilinear",
            do_center_crop: false,
            crop_size: { height: 2, width: 2 },
            do_rescale: false,
            rescale_factor: 1,
            do_normalize: true,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
        });

        // Input value 1.0 → (1.0 - 0.5) / 0.5 = 1.0
        const img = solidRGB(2, 2, 1.0, 1.0, 1.0);
        const out = await proc.preprocess(img);

        for (const v of out) {
            expect(v).toBeCloseTo(1.0, 5);
        }
    });

    it("normalize with ImageNet stats", async () => {
        const proc = ImageProcessor.fromConfig({
            do_resize: false,
            size: { height: 1, width: 1 },
            resample: "bilinear",
            do_center_crop: false,
            crop_size: { height: 1, width: 1 },
            do_rescale: true,
            rescale_factor: 1 / 255,
            do_normalize: true,
            image_mean: [0.485, 0.456, 0.406],
            image_std: [0.229, 0.224, 0.225],
        });

        // Pixel (128, 128, 128) → rescaled (0.502, 0.502, 0.502)
        // R: (0.502 - 0.485) / 0.229 ≈ 0.074
        const img = solidRGB(1, 1, 128, 128, 128);
        const out = await proc.preprocess(img);

        // CHW: out[0] = R channel pixel
        expect(out[0]).toBeCloseTo((128 / 255 - 0.485) / 0.229, 3);
        expect(out[1]).toBeCloseTo((128 / 255 - 0.456) / 0.224, 3);
        expect(out[2]).toBeCloseTo((128 / 255 - 0.406) / 0.225, 3);
    });

    it("center crop reduces spatial dimensions", async () => {
        const proc = ImageProcessor.fromConfig({
            do_resize: true,
            size: { height: 256, width: 256 },
            resample: "bilinear",
            do_center_crop: true,
            crop_size: { height: 224, width: 224 },
            do_rescale: false,
            rescale_factor: 1,
            do_normalize: false,
            image_mean: [0, 0, 0],
            image_std: [1, 1, 1],
        });

        const img = solidRGB(64, 64, 1, 1, 1);
        const out = await proc.preprocess(img);
        expect(out.length).toBe(3 * 224 * 224);
    });
});
