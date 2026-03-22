# transformers-js

JS inference runtime for transformer models. WebGPU-primary, WASM-fallback.
Not a Python API port. A production-grade JS runtime with automated preprocessing maintenance.

## Criteria (in priority order)
1. Preprocessing correctness: output tensors match Python transformers within 1e-5
2. Vision inference latency: ≤50ms for ViT-B/16 on WebGPU (Chrome, M-series Mac)
3. Time from Python transformers update to JS PR: automated, ≤24h
4. First-load cost: no WASM binary on the WebGPU path (lazy-load fallback only)

## What ONNX captures — never hand-maintain
- Model weights, architecture, attention, feed-forward ops
- Inference routing between encoder/decoder sessions

## What needs maintenance — the actual surface
- `src/preprocessing/` — image resize, normalize, pad (mirrors `image_processing_*.py`)
- `src/feature_extraction/` — audio mel filterbank, windowing (mirrors `feature_extraction_*.py`)
- `src/postprocessing/` — detection NMS, segmentation masks, depth map parsing

## Automated maintenance pipeline
A GitHub Actions workflow watches `huggingface/transformers` for changes to
`image_processing_*.py` and `feature_extraction_*.py`. On diff, it generates a JS
translation via Claude API (numpy → tensor ops) and opens a PR. Human reviews the
output — not the translation process.

Script lives in `codegen/sync-preprocessing.js`. Never manually edit files in
`src/preprocessing/` that are marked `@generated`.

## Device strategy
- Vision tasks: WebGPU default → WASM fallback
- Audio tasks: WASM default (WebGPU audio ops less mature)
- Node.js: `onnxruntime-node` with CUDA / CoreML / DirectML execution providers
- Image preprocessing: run on WebGPU compute shaders when device is `webgpu`

## Architecture
```
runtime/          ONNX session lifecycle, device routing, quantization
preprocessing/    Image and audio ops — generated from Python, @generated files never hand-edited
postprocessing/   Detection, segmentation, depth output parsing
generation/       Sampling, KV-cache, streaming, stopping criteria
pipeline/         Task registry and user-facing API
codegen/          Python→JS translator scripts and sync workflow
```

## Out of scope
- Model training
- Python API parity as a goal (functionally equivalent where it helps, ignored where it doesn't)
- Models without an available ONNX conversion
- Video temporal models (until core vision is stable)

## Build order
1. `codegen/` — automated PR workflow against HF Python repo (structural fix for maintenance)
2. WebGPU image preprocessing — GPU-accelerated resize/normalize
3. Three proof models: ViT (encoder-only), CLIP (vision+text), SAM (custom prompts)
4. WASM fallback — wire ONNX Runtime Web; adapt from existing HF codebase
