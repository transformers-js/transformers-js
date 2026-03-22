import { describe, it, expect } from "vitest";
import { cpuResize } from "./cpu.js";
import type { ImageData } from "../ops.js";

// ── Helpers ────────────────────────────────────────────────────────────────

function solidImage(width: number, height: number, channels: number, value: number): ImageData {
    return { data: new Float32Array(width * height * channels).fill(value), width, height, channels };
}

function gradientImage(width: number, height: number): ImageData {
    const data = new Float32Array(width * height * 3);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            data[(y * width + x) * 3 + 0] = x / (width - 1);  // R increases left→right
            data[(y * width + x) * 3 + 1] = y / (height - 1); // G increases top→bottom
            data[(y * width + x) * 3 + 2] = 0.5;
        }
    }
    return { data, width, height, channels: 3 };
}

function pixelAt(img: ImageData, x: number, y: number, ch: number): number {
    return img.data[(y * img.width + x) * img.channels + ch]!;
}

const TOLERANCE = 1e-5;

// ── Shared contract tests (all filters) ───────────────────────────────────

for (const filter of ["nearest", "bilinear", "bicubic"] as const) {
    describe(`cpuResize — ${filter}`, () => {
        it("produces correct output dimensions", async () => {
            const img = solidImage(64, 64, 3, 0.5);
            const out = await cpuResize(img, { width: 224, height: 224 }, filter);
            expect(out.width).toBe(224);
            expect(out.height).toBe(224);
            expect(out.channels).toBe(3);
            expect(out.data.length).toBe(224 * 224 * 3);
        });

        it("preserves solid color", async () => {
            const img = solidImage(32, 32, 3, 0.42);
            const out = await cpuResize(img, { width: 112, height: 112 }, filter);
            for (let i = 0; i < out.data.length; i++) {
                expect(out.data[i]).toBeCloseTo(0.42, 4);
            }
        });

        it("handles single-channel images", async () => {
            const img = solidImage(16, 16, 1, 0.8);
            const out = await cpuResize(img, { width: 32, height: 32 }, filter);
            expect(out.channels).toBe(1);
            expect(out.data.length).toBe(32 * 32 * 1);
        });

        it("handles 4-channel images", async () => {
            const img = solidImage(16, 16, 4, 0.5);
            const out = await cpuResize(img, { width: 8, height: 8 }, filter);
            expect(out.channels).toBe(4);
        });

        it("identity resize returns identical values", async () => {
            const img = gradientImage(32, 32);
            const out = await cpuResize(img, { width: 32, height: 32 }, filter);
            for (let i = 0; i < img.data.length; i++) {
                expect(out.data[i]).toBeCloseTo(img.data[i]!, 4);
            }
        });

        it("downscales correctly (output smaller than input)", async () => {
            const img = solidImage(256, 256, 3, 1.0);
            const out = await cpuResize(img, { width: 32, height: 32 }, filter);
            expect(out.width).toBe(32);
            expect(out.height).toBe(32);
        });
    });
}

// ── Bilinear-specific: interpolation accuracy ─────────────────────────────

describe("cpuResize — bilinear interpolation accuracy", () => {
    it("midpoint of a 2×1 image interpolates to the average", async () => {
        // Image: [0.0, 1.0] → resize to width 3 → [0.0, 0.5, 1.0] (approximately)
        const img: ImageData = {
            data: new Float32Array([0.0, 1.0]),
            width: 2, height: 1, channels: 1,
        };
        const out = await cpuResize(img, { width: 3, height: 1 }, "bilinear");
        expect(out.data[0]).toBeCloseTo(0.0, 4);
        expect(out.data[1]).toBeCloseTo(0.5, 4);
        expect(out.data[2]).toBeCloseTo(1.0, 4);
    });

    it("corner pixels of 2×2 → 4×4 upscale stay at original values", async () => {
        const img: ImageData = {
            data: new Float32Array([0, 1, 0, 1,  // row 0: left=0, right=1
                                    0, 1, 0, 1]), // (grayscale, wrong shape — use 1ch)
            width: 2, height: 2, channels: 1,
        };
        // Actually build it properly
        const src: ImageData = {
            data: new Float32Array([0, 1, 0, 1]),
            width: 2, height: 2, channels: 1,
        };
        const out = await cpuResize(src, { width: 4, height: 4 }, "bilinear");
        // Top-left corner should be close to 0
        expect(pixelAt(out, 0, 0, 0)).toBeCloseTo(0, 1);
        // Top-right corner should be close to 1
        expect(pixelAt(out, 3, 0, 0)).toBeCloseTo(1, 1);
    });
});

// ── Values stay in input range ─────────────────────────────────────────────

describe("cpuResize — no value overshoot", () => {
    it("bilinear stays within [min, max] of input", async () => {
        const img = gradientImage(16, 16);
        const out = await cpuResize(img, { width: 224, height: 224 }, "bilinear");
        const min = Math.min(...img.data);
        const max = Math.max(...img.data);
        for (const v of out.data) {
            expect(v).toBeGreaterThanOrEqual(min - TOLERANCE);
            expect(v).toBeLessThanOrEqual(max + TOLERANCE);
        }
    });

    it("bicubic may overshoot slightly but stays within 5% of range", async () => {
        const img = gradientImage(16, 16);
        const out = await cpuResize(img, { width: 224, height: 224 }, "bicubic");
        const min = Math.min(...img.data);
        const max = Math.max(...img.data);
        const range = max - min;
        for (const v of out.data) {
            expect(v).toBeGreaterThanOrEqual(min - range * 0.05);
            expect(v).toBeLessThanOrEqual(max + range * 0.05);
        }
    });
});
