import { describe, it, expect, beforeAll } from "vitest";
import { SAMImageProcessor, scalePoint, scaleBox, postProcessMask } from "./sam.js";
import { setResizeImpl } from "./ops.js";
import { cpuResize } from "./resize/cpu.js";
import type { ImageData } from "./ops.js";

beforeAll(() => setResizeImpl(cpuResize));

function solidImage(w: number, h: number, value: number): ImageData {
    return { data: new Float32Array(w * h * 3).fill(value), width: w, height: h, channels: 3 };
}

// ── SAMImageProcessor ──────────────────────────────────────────────────────────

describe("SAMImageProcessor.preprocess", () => {
    it("outputs a [3, 1024, 1024] CHW tensor", async () => {
        const proc = new SAMImageProcessor();
        const img  = solidImage(640, 480, 128);
        const { pixelValues } = await proc.preprocess(img);
        expect(pixelValues.length).toBe(3 * 1024 * 1024);
    });

    it("preserves originalSize", async () => {
        const proc = new SAMImageProcessor();
        const img  = solidImage(800, 600, 64);
        const { originalSize } = await proc.preprocess(img);
        expect(originalSize).toEqual({ width: 800, height: 600 });
    });

    it("scale makes longest edge = 1024", async () => {
        const proc = new SAMImageProcessor();
        const W = 640, H = 480;
        const { scale, resizedWidth, resizedHeight } = await proc.preprocess(solidImage(W, H, 0));
        expect(Math.round(W * scale)).toBe(resizedWidth);
        expect(Math.round(H * scale)).toBe(resizedHeight);
        expect(Math.max(resizedWidth, resizedHeight)).toBe(1024);
    });

    it("works for square images", async () => {
        const proc = new SAMImageProcessor();
        const { resizedWidth, resizedHeight } = await proc.preprocess(solidImage(512, 512, 0));
        expect(resizedWidth).toBe(1024);
        expect(resizedHeight).toBe(1024);
    });

    it("works for portrait images", async () => {
        const proc = new SAMImageProcessor();
        const { resizedWidth, resizedHeight } = await proc.preprocess(solidImage(480, 640, 0));
        expect(resizedHeight).toBe(1024);
        expect(resizedWidth).toBeLessThan(1024);
    });
});

// ── Coordinate transforms ──────────────────────────────────────────────────────

describe("scalePoint", () => {
    it("scales coordinates by scale factor", () => {
        const [sx, sy] = scalePoint(100, 200, 1.6);
        expect(sx).toBeCloseTo(160, 5);
        expect(sy).toBeCloseTo(320, 5);
    });

    it("identity scale is identity", () => {
        const [sx, sy] = scalePoint(42, 99, 1.0);
        expect(sx).toBe(42);
        expect(sy).toBe(99);
    });
});

describe("scaleBox", () => {
    it("scales all four coordinates", () => {
        const [x1, y1, x2, y2] = scaleBox(10, 20, 110, 120, 2.0);
        expect(x1).toBe(20);
        expect(y1).toBe(40);
        expect(x2).toBe(220);
        expect(y2).toBe(240);
    });
});

// ── postProcessMask ────────────────────────────────────────────────────────────

describe("postProcessMask", () => {
    it("returns correct output dimensions", async () => {
        const preprocessed = {
            pixelValues: new Float32Array(3 * 1024 * 1024),
            originalSize: { width: 640, height: 480 },
            scale: 1024 / 640,
            resizedWidth: 1024,
            resizedHeight: Math.round(480 * (1024 / 640)),
        };

        const logits = new Float32Array(256 * 256).fill(1); // all positive → all 1
        const mask   = await postProcessMask(logits, 256, 256, preprocessed);
        expect(mask.length).toBe(640 * 480);
    });

    it("positive logits → mask = 1, negative logits → mask = 0", async () => {
        const preprocessed = {
            pixelValues: new Float32Array(3 * 1024 * 1024),
            originalSize: { width: 32, height: 32 },
            scale: 1.0,
            resizedWidth: 32,
            resizedHeight: 32,
        };

        const posLogits = new Float32Array(256 * 256).fill(10);
        const posMask   = await postProcessMask(posLogits, 256, 256, preprocessed);
        expect(posMask.every((v) => v === 1)).toBe(true);

        const negLogits = new Float32Array(256 * 256).fill(-10);
        const negMask   = await postProcessMask(negLogits, 256, 256, preprocessed);
        expect(negMask.every((v) => v === 0)).toBe(true);
    });

    it("output contains only 0 and 1", async () => {
        const preprocessed = {
            pixelValues: new Float32Array(3 * 1024 * 1024),
            originalSize: { width: 64, height: 48 },
            scale: 1024 / 64,
            resizedWidth: 1024,
            resizedHeight: Math.round(48 * (1024 / 64)),
        };

        // Mixed logits
        const logits = Float32Array.from({ length: 256 * 256 }, (_, i) => (i % 2 === 0 ? 1 : -1));
        const mask   = await postProcessMask(logits, 256, 256, preprocessed);
        for (const v of mask) {
            expect(v === 0 || v === 1).toBe(true);
        }
    });
});
