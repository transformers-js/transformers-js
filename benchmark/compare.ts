#!/usr/bin/env tsx
/**
 * Cross-validate our preprocessing against Python transformers.
 *
 * Requires benchmark/data/ to be populated first:
 *   python benchmark/generate.py
 *
 * Usage:
 *   npx tsx benchmark/compare.ts
 */
import { readFileSync } from "fs";
import { ImageProcessor } from "../src/preprocessing/image-processor.js";
import { setResizeImpl } from "../src/preprocessing/ops.js";
import { cpuResize } from "../src/preprocessing/resize/cpu.js";
import type { ImageData } from "../src/preprocessing/ops.js";

setResizeImpl(cpuResize);

const MODEL = "Xenova/vit-base-patch16-224";
// 256×256 so R=y, G=x, B=(y+x)//2 are all exact integers [0,255] — no modulo wrapping,
// no rounding, no discontinuities. Bicubic on smooth data causes no ringing.
const W = 256, H = 256;

// Generate same synthetic image as generate.py.
// All values are exact integers: no rounding, identical across languages.
const data = new Float32Array(W * H * 3);
for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
        const i = (y * W + x) * 3;
        data[i]     = y;               // R: vertical gradient 0-255
        data[i + 1] = x;               // G: horizontal gradient 0-255
        data[i + 2] = (y + x) >> 1;   // B: diagonal, 0-255, no wrap
    }
}
const image: ImageData = { data, width: W, height: H, channels: 3 };

console.log(`Model: ${MODEL}`);
console.log("Fetching preprocessor config from HF Hub...");
const processor = await ImageProcessor.fromHub(MODEL);
console.log(`  size: ${JSON.stringify(processor.config.size)}`);
console.log(`  resample: ${processor.config.resample}`);
console.log(`  do_center_crop: ${processor.config.do_center_crop}`);
console.log(`  image_mean: ${processor.config.image_mean}`);
console.log(`  image_std: ${processor.config.image_std}`);

console.log("\nRunning TypeScript preprocessing...");
const pixelValues = await processor.preprocess(image);

console.log("Loading Python reference output...");
const binPath = new URL("data/vit_pixel_values.bin", import.meta.url);
const expected = new Float32Array(readFileSync(binPath).buffer);

if (pixelValues.length !== expected.length) {
    console.error(`Shape mismatch: TS=${pixelValues.length} vs Python=${expected.length}`);
    process.exit(1);
}

// Compare
let maxDiff = 0;
let totalDiff = 0;
let maxIdx = 0;
for (let i = 0; i < pixelValues.length; i++) {
    const diff = Math.abs(pixelValues[i]! - expected[i]!);
    if (diff > maxDiff) { maxDiff = diff; maxIdx = i; }
    totalDiff += diff;
}
const meanDiff = totalDiff / pixelValues.length;

// Error budget from PIL's uint8 bicubic resize:
//   - PIL uses 14-bit fixed-point weights (weight error ≈ 1/(2^14))
//   - PIL output is uint8 (quantization ≈ 1 lsb)
//   - Combined budget ≈ 1 unit / 255 / min(std) after rescale+normalize
const minStd = Math.min(...processor.config.image_std);
const pilNoiseCeiling = 1.0 / 255 / minStd;

const N = pixelValues.length;
const C = 3, OH = Math.round(Math.sqrt(N / C)), OW = OH; // approx
console.log(`\nTensor elements: ${N} (${C}×${OH}×${OW} approx)`);
console.log(`Max   |Δ|: ${maxDiff.toExponential(4)}   (at index ${maxIdx})`);
console.log(`Mean  |Δ|: ${meanDiff.toExponential(4)}`);
console.log(`PIL quantization ceiling: ${pilNoiseCeiling.toExponential(4)}`);

const withinPilNoise = maxDiff <= pilNoiseCeiling * 1.05; // 5% margin for float rounding on top
console.log(`\nWithin PIL quantization noise: ${withinPilNoise ? "PASS ✓" : "FAIL ✗"}`);
console.log(`Within 1e-5 (strict):          ${maxDiff <= 1e-5 ? "PASS ✓" : "FAIL ✗"}`);

if (!withinPilNoise) process.exit(1);
