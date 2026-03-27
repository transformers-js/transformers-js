#!/usr/bin/env node
/**
 * CORS-enabled static file server for local model testing.
 * Usage: node scripts/serve-model.js <directory> [port]
 *
 * Example:
 *   node scripts/serve-model.js \
 *     ../onnx-export/exports/LFM2-350M-ONNX/flat 8001
 *
 * Then in benchmark/lfm2.html set:
 *   mirrorBaseUrl = http://localhost:8001
 */
import { createServer } from "node:http";
import { createReadStream, statSync } from "node:fs";
import { join, extname } from "node:path";

const dir  = process.argv[2];
const port = parseInt(process.argv[3] ?? "8001", 10);

if (!dir) {
    console.error("Usage: node scripts/serve-model.js <directory> [port]");
    process.exit(1);
}

const MIME = {
    ".onnx":      "application/octet-stream",
    ".json":      "application/json",
    ".bin":       "application/octet-stream",
    ".onnx_data": "application/octet-stream",
};

createServer((req, res) => {
    // Strip query string
    const pathname = req.url?.split("?")[0] ?? "/";
    const file = join(dir, pathname);

    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS");
    res.setHeader("Access-Control-Expose-Headers", "Content-Length");

    if (req.method === "OPTIONS") { res.writeHead(204); res.end(); return; }

    let stat;
    try { stat = statSync(file); } catch {
        res.writeHead(404); res.end("Not found"); return;
    }

    const ext = extname(file);
    const mime = MIME[ext] ?? "application/octet-stream";
    res.setHeader("Content-Type", mime);
    res.setHeader("Content-Length", stat.size);
    res.writeHead(200);
    createReadStream(file).pipe(res);
}).listen(port, () => {
    console.log(`Model server: http://localhost:${port}`);
    console.log(`Serving: ${dir}`);
});
