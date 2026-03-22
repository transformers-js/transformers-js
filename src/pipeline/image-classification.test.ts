import { describe, it, expect } from "vitest";
import { softmax, topKSoftmax } from "./image-classification.js";

describe("softmax", () => {
    it("sums to 1", () => {
        const out = softmax(new Float32Array([1, 2, 3, 4]));
        expect(out.reduce((a, b) => a + b, 0)).toBeCloseTo(1.0, 5);
    });

    it("preserves order", () => {
        const out = softmax(new Float32Array([0.1, 5.0, 0.3]));
        expect(out[1]).toBeGreaterThan(out[2]!);
        expect(out[2]).toBeGreaterThan(out[0]!);
    });

    it("handles uniform logits (all equal probability)", () => {
        const out = softmax(new Float32Array([2, 2, 2, 2]));
        for (const v of out) expect(v).toBeCloseTo(0.25, 5);
    });

    it("is numerically stable with large logits", () => {
        const out = softmax(new Float32Array([1000, 1001, 1002]));
        expect(out.reduce((a, b) => a + b, 0)).toBeCloseTo(1.0, 5);
        expect(out[2]).toBeGreaterThan(out[1]!);
    });

    it("single logit returns probability 1", () => {
        const out = softmax(new Float32Array([42]));
        expect(out[0]).toBeCloseTo(1.0, 5);
    });
});

describe("topKSoftmax", () => {
    it("returns k results sorted by descending score", () => {
        const logits = new Float32Array([0.1, 5.0, 0.3, 2.0, 0.5]);
        const results = topKSoftmax(logits, 3, (i) => `CLASS_${i}`);

        expect(results).toHaveLength(3);
        expect(results[0]!.label).toBe("CLASS_1"); // highest logit
        expect(results[1]!.label).toBe("CLASS_3");
        expect(results[2]!.label).toBe("CLASS_4");
        expect(results[0]!.score).toBeGreaterThan(results[1]!.score);
    });

    it("scores sum to less than 1 when topK < total", () => {
        const logits = new Float32Array([1, 2, 3, 4, 5]);
        const results = topKSoftmax(logits, 3, (i) => `C${i}`);
        const total = results.reduce((a, r) => a + r.score, 0);
        expect(total).toBeLessThan(1.0);
    });

    it("all scores sum to 1 when topK equals total classes", () => {
        const logits = new Float32Array([1, 2, 3]);
        const results = topKSoftmax(logits, 3, (i) => `C${i}`);
        const total = results.reduce((a, r) => a + r.score, 0);
        expect(total).toBeCloseTo(1.0, 5);
    });

    it("scores are all in (0, 1)", () => {
        const logits = new Float32Array([0.1, 0.9, 0.5]);
        const results = topKSoftmax(logits, 3, (i) => `C${i}`);
        for (const r of results) {
            expect(r.score).toBeGreaterThan(0);
            expect(r.score).toBeLessThan(1);
        }
    });
});
