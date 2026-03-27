# transformers-js

Reference JS inference runtime for LFM (Liquid Foundation Models). WebGPU-primary, WASM-fallback.
Not a Python API port. Not a model zoo. The best way to run LFM models in the browser.

## Why LFM

LFM's hybrid architecture (attention + structured conv) has a property that matters specifically
for browser inference: the conv cache is fixed-size regardless of context length. The KV cache
still grows, but conv layers don't. This bounds memory for long conversations in a way pure
transformers cannot. It is the core architectural reason this runtime exists.

## Criteria (in priority order)

1. **Preprocessing correctness**: output tensors within PIL quantization noise of Python transformers.
   Measured bound: max |Δ| ≤ 1/(255 × min_std) ≈ 7.8e-3 (std=0.5).
   Note: 1e-5 is not achievable — PIL bicubic operates on uint8 while we use float32.
   Observed max |Δ|: 5.9e-3 on smooth test image. See `benchmark/`.

2. **LFM2-350M-q4 text generation latency** (WebGPU, Chrome, M-series Mac):
   TTFT:   median ≤27ms, p95 ≤28ms.
   Decode: median ≤16ms/token, p95 ≤18ms/token.
   Measured: TTFT medians 25.8/26.0/25.8ms, decode medians 15.7/14.7/15.0ms/token (3 sessions,
   8 runs each after 2 warmup, official Liquid4All export via local mirror). See `benchmark/lfm2.html`.
   Note: official export and community export perform identically — the 350M's decode cost is
   architectural, not export-quality-dependent. The 1.2B's faster decode (4ms/token) is due to
   better GPU workgroup utilisation at larger matrix sizes, not export differences.

3. **LFM2.5-1.2B-Instruct-q4 text generation latency** (WebGPU, Chrome, M-series Mac):
   TTFT:   median ≤50ms, p95 ≤52ms.
   Decode: median ≤5ms/token, p95 ≤5ms/token.
   Measured: TTFT medians 47.7/47.7/49.3ms, decode medians 3.8/3.9/4.1ms/token (3 sessions,
   8 runs each after 2 warmup, model `LiquidAI/LFM2.5-1.2B-Instruct-ONNX`). See `benchmark/lfm2.html`.
   Note: TTFT variance is negligible (~1ms across all runs) — unusually stable. Decode is faster than
   the 350M despite being 3.5× larger; the LFM2.5 ONNX export is better optimized than the community
   350M export.

4. **LFM2-2.6B-q4 text generation latency** (WebGPU, Chrome, M-series Mac):
   TTFT:   median ≤110ms, p95 ≤120ms.
   Decode: median ≤10ms/token, p95 ≤11ms/token.
   Measured: TTFT medians 98.1/105.0/99.2ms, decode medians 8.8/8.5/9.8ms/token (3 sessions,
   8 runs each after 2 warmup, model `onnx-community/LFM2-2.6B-ONNX`). See `benchmark/lfm2.html`.
   Scaling note: 1.2B→2.6B is 2.17× params, decode scales 4→9ms/token (~2.2×) — linear.

5. **LFM2-VL first-token latency**: target ≤ 200ms prefill for a single image + short prompt on WebGPU.
   TARGET — not yet measured.

6. **Preprocessing sync lag**: automated, ≤ 24h from Liquid AI model update to JS PR.

5. **First-load cost**: no WASM binary on the WebGPU path (lazy-load fallback only).

When a target is measured and validated, replace it with a bound and the measurement method.

## Supported models

ONNX exports come from two sources: `LiquidAI/` (official, optimized) and `onnx-community/`
(community, Transformers.js tooling). Liquid4All's export pipeline explicitly targets
interoperability with this runtime.

| Model | ONNX | Task | Status |
|-------|------|------|--------|
| LFM2.5-1.2B-Instruct | LiquidAI/LFM2.5-1.2B-Instruct-ONNX | Text generation | Working |
| LFM2-350M | onnx-community/LFM2-350M-ONNX | Text generation | Working |
| LFM2-2.6B | onnx-community/LFM2-2.6B-ONNX | Text generation | Untested |
| LFM2-VL-450M | onnx-community/LFM2-VL-450M-ONNX | Vision-language | Working |
| LFM2.5-VL-1.6B | LiquidAI/LFM2.5-VL-1.6B-ONNX | Vision-language | Untested |
| LFM2-VL-3B | onnx-community/LFM2-VL-3B-ONNX | Vision-language | Untested |
| LFM2-MoE-8B-A1B | onnx-community/LFM2-8B-A1B-ONNX | Text generation | Needs design |
| LFM2.5-Audio-1.5B | LiquidAI/LFM2.5-Audio-1.5B-ONNX | ASR/TTS | Not started |
| Specialized fine-tunes | onnx-community/LFM2-*-Tool/RAG/Math/Extract-ONNX | Text generation | Should work |

**MoE note**: 8B parameters, 1B active per token. Different routing/cache structure — needs
explicit support before use. Active param count makes it viable for browser inference.

**Audio note**: 5 ONNX sessions (decoder + audio_encoder + audio_embedding + audio_detokenizer
+ vocoder_depthformer). Significant lift. Deprioritised until text/VL path is complete.

**Specialized fine-tunes**: Same architecture as base models, just fine-tuned weights.
Should work with existing runtime without code changes — verify before claiming support.

## What ONNX captures — never hand-maintain

- Model weights, architecture, attention, conv layers, feed-forward ops
- The KV + conv cache tensor shapes and routing
- Inference routing between encoder/decoder/embed sessions (VL path)

## What needs maintenance — the actual surface

- `src/preprocessing/` — image resize, normalize, pad (mirrors `image_processing_*.py`)
- `src/generation/` — sampling, KV+conv cache management, stopping criteria
- `src/models/` — session wiring, input name detection, VL embed splicing

## Automated maintenance pipeline

A GitHub Actions workflow watches `huggingface/transformers` for changes to
`image_processing_*.py` files used by LFM-compatible processors. On diff, it generates a JS
translation via Claude API (numpy → tensor ops) and opens a PR. Human reviews the output —
not the translation process.

Scope: only processors relevant to LFM models. Do not expand scope to unrelated model families.

Script lives in `codegen/sync.ts`. Never manually edit files in
`src/preprocessing/` that are marked `@generated`.

## LFM architecture properties — know these

- **Hybrid layers**: each transformer block is either an attention layer or a conv layer.
  Layer type is determined by the ONNX input names — `past_key_values.N.*` for attention,
  `past_conv.N` for conv. `initCache` in `generation/loop.ts` reads this from input names.

- **Conv cache**: fixed size `[1, hidden_size, conv_L_cache]` regardless of sequence length.
  Updated every step. Never grows. This is the memory advantage.

- **KV cache**: grows with context as in standard transformers. Still the memory bottleneck
  for long conversations, but bounded by the absence of conv growth.

- **VL path**: three ONNX sessions — embed_images, embed_tokens, decoder.
  Image embeddings splice into the token embedding sequence before prefill.
  Each image_token_id placeholder expands to `numTiles × TOKENS_PER_TILE` rows.

- **num_logits_to_keep**: some LFM exports include this input to avoid materialising the full
  `[1, seqLen, vocabSize]` logit tensor during prefill. Always check input names; both
  the text and VL decode paths handle it.

## Device strategy

- Text generation: WebGPU default → WASM fallback
- Image preprocessing: WebGPU compute shaders when device is `webgpu`
- Node.js: `onnxruntime-node` with CUDA / CoreML / DirectML execution providers
- Audio: not in scope for LFM models

## Architecture

```
runtime/          ONNX session lifecycle, device routing
preprocessing/    Image ops — generated from Python, @generated files never hand-edited
generation/       Sampling, KV+conv cache, streaming, stopping criteria
models/           LFM2, LFM2-VL session wiring and input construction
pipeline/         User-facing API
codegen/          Python→JS translator and HF sync workflow
```

## Distribution

- Published to GitHub Packages as `@transformers-js/transformers-js` (single consumer for now)
- Migrate to npm when ready for external consumers — one `publishConfig` change

## Out of scope

- Model training
- Python API parity
- Models other than LFM family
- Generic transformer zoo (ViT, CLIP, SAM were proofs of concept — do not expand)
- Video temporal models
- Audio models (LFM does not have a public audio model)
- Models without ONNX exports

## Next priorities

1. **LFM2-VL latency benchmark** — VL TTFT target ≤200ms not yet measured. Prefill path differs
   (image embeddings splice into token sequence). Needs its own benchmark. Model:
   `onnx-community/LFM2-VL-450M-ONNX` (Working status, smallest VL model).
2. **Verify specialized fine-tunes** — same architecture as base, should work without code changes.
   Test: `LiquidAI/LFM2-1.2B-Tool-ONNX` (or RAG/Math/Extract), run a chat, confirm output.
3. **Speculative decoding** — 350M draft + 2.6B verifier. High UX leverage: 2.6B quality at near-350M
   cost. Requires new generation loop design.
4. **IOBinding / GPU buffer optimization** — keep KV+conv cache tensors on GPU between decode steps.
   Requires refactoring updateCache and run() to pass raw ORT tensor objects. True decode speedup path.
5. **Audio** — deprioritised; revisit after text/VL path is stable.
