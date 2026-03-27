#!/usr/bin/env node
// Bundles benchmark entry points for the browser.
// onnxruntime-web and onnxruntime-node are kept external — the HTML page
// maps onnxruntime-web to the CDN build via importmap.
import * as esbuild from "esbuild";
import { mkdir } from "fs/promises";

await mkdir("benchmark/dist", { recursive: true });

const sharedConfig = {
    bundle: true,
    platform: "browser",
    format: "esm",
    external: ["onnxruntime-web", "onnxruntime-node"],
    minify: false,
    sourcemap: true,
};

await esbuild.build({
    ...sharedConfig,
    entryPoints: ["benchmark/playground-entry.ts"],
    outfile: "benchmark/dist/playground.js",
});
console.log("Built benchmark/dist/playground.js");

await esbuild.build({
    ...sharedConfig,
    entryPoints: ["benchmark/lfm2-entry.ts"],
    outfile: "benchmark/dist/lfm2.js",
});
console.log("Built benchmark/dist/lfm2.js");

await esbuild.build({
    ...sharedConfig,
    entryPoints: ["benchmark/lfm2-vl-entry.ts"],
    outfile: "benchmark/dist/lfm2-vl.js",
});
console.log("Built benchmark/dist/lfm2-vl.js");
