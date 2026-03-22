#!/usr/bin/env node
// Bundles benchmark/latency-entry.ts for the browser.
// onnxruntime-web and onnxruntime-node are kept external — the HTML page
// maps onnxruntime-web to the CDN build via importmap.
import * as esbuild from "esbuild";
import { mkdir } from "fs/promises";

await mkdir("benchmark/dist", { recursive: true });

await esbuild.build({
    entryPoints: ["benchmark/latency-entry.ts"],
    bundle: true,
    platform: "browser",
    format: "esm",
    // ORT packages stay external — importmap in the HTML resolves them.
    // onnxruntime-node is dead code in browser (isNode branch never taken)
    // but esbuild still sees the import; mark external to avoid bundling.
    external: ["onnxruntime-web", "onnxruntime-node"],
    outfile: "benchmark/dist/latency.js",
    minify: false,
    sourcemap: true,
});

console.log("Built benchmark/dist/latency.js");
