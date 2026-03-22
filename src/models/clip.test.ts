import { describe, it, expect } from "vitest";
import { l2Normalize, cosineSimilarity } from "./clip.js";

describe("l2Normalize", () => {
    it("produces a unit vector", () => {
        const v = new Float32Array([3, 4]);
        const n = l2Normalize(v);
        const norm = Math.sqrt(n[0]! ** 2 + n[1]! ** 2);
        expect(norm).toBeCloseTo(1.0, 5);
    });

    it("preserves direction", () => {
        const v = new Float32Array([1, 0, 0]);
        const n = l2Normalize(v);
        expect(n[0]).toBeCloseTo(1, 5);
        expect(n[1]).toBeCloseTo(0, 5);
        expect(n[2]).toBeCloseTo(0, 5);
    });

    it("handles zero vector without NaN", () => {
        const v = new Float32Array([0, 0, 0]);
        const n = l2Normalize(v);
        for (const x of n) expect(isNaN(x)).toBe(false);
    });

    it("does not mutate input", () => {
        const v = new Float32Array([3, 4]);
        const original = [...v];
        l2Normalize(v);
        expect([...v]).toEqual(original);
    });
});

describe("cosineSimilarity (on normalized vectors)", () => {
    it("identical vectors → 1", () => {
        const v = l2Normalize(new Float32Array([1, 2, 3]));
        expect(cosineSimilarity(v, v)).toBeCloseTo(1.0, 5);
    });

    it("orthogonal vectors → 0", () => {
        const a = l2Normalize(new Float32Array([1, 0, 0]));
        const b = l2Normalize(new Float32Array([0, 1, 0]));
        expect(cosineSimilarity(a, b)).toBeCloseTo(0, 5);
    });

    it("opposite vectors → -1", () => {
        const a = l2Normalize(new Float32Array([1, 0, 0]));
        const b = l2Normalize(new Float32Array([-1, 0, 0]));
        expect(cosineSimilarity(a, b)).toBeCloseTo(-1.0, 5);
    });

    it("result is in [-1, 1]", () => {
        for (let i = 0; i < 20; i++) {
            const a = l2Normalize(Float32Array.from({ length: 512 }, () => Math.random() - 0.5));
            const b = l2Normalize(Float32Array.from({ length: 512 }, () => Math.random() - 0.5));
            const sim = cosineSimilarity(a, b);
            expect(sim).toBeGreaterThanOrEqual(-1 - 1e-4);
            expect(sim).toBeLessThanOrEqual(1 + 1e-4);
        }
    });
});
