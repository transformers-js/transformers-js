---
name: Sync preprocessing from huggingface/transformers
description: |
  Detects new commits to huggingface/transformers that changed LFM image
  preprocessing files, translates them from Python to TypeScript, and opens a PR.

strict: false

engine:
  id: copilot
  model: claude-sonnet-4

on:
  schedule:
    - cron: daily
  workflow_dispatch:

permissions:
  contents: read
  pull-requests: read

safe-outputs:
  create-pull-request:

tools:
  github:
    toolsets: [repos, pull_requests]
  edit:

network:
  allowed:
    - defaults
    - "api.github.com"
    - "raw.githubusercontent.com"

timeout-minutes: 30
---

# Sync LFM preprocessing from huggingface/transformers

You detect changes to LFM image preprocessing files in `huggingface/transformers`, translate them from Python to TypeScript, and open a PR.

## Step 1: Read last sync position

Read `codegen/.last-sync` from this repository. This file contains a single commit SHA — the last commit in `huggingface/transformers` that was processed.

## Step 2: Find new commits

Call the GitHub API to list recent commits in `huggingface/transformers`:

```
GET https://api.github.com/repos/huggingface/transformers/commits?sha=main&per_page=100
```

Find all commits newer than the SHA in `.last-sync` (i.e., commits before the matching entry in the list). If the last-sync SHA appears at index N, take commits at indices 0..N-1. If the SHA is not found in the 100 most recent commits, take all 100.

If there are no new commits, stop here with "Already up to date."

## Step 3: Find changed preprocessing files

For each new commit SHA, call:

```
GET https://api.github.com/repos/huggingface/transformers/commits/{sha}
```

Collect every changed filename that matches ALL of:
- Path starts with `src/transformers/models/lfm`
- Filename matches `image_processing_*.py` or `feature_extraction_*.py`

Deduplicate across commits. If no files match, update `.last-sync` to the newest commit SHA and stop with "No LFM preprocessing files changed."

## Step 4: Translate each file

For each matched Python file, fetch its current content:

```
GET https://api.github.com/repos/huggingface/transformers/contents/{path}?ref=main
```

Decode the base64 `content` field. Translate the Python to TypeScript using the rules below.

### Translation rules

**Goal**: numerical equivalence — the TypeScript output must produce tensors within the PIL quantization noise floor of the Python output for the same inputs.

**Structure of each output file**:
1. `@generated` comment at top with source path and "Do not edit manually" warning
2. Config interface: `{ClassName}Config` — all `__init__` parameters as optional typed fields
3. The processor class implementing `ImageProcessor` or `FeatureExtractor`
4. Export only the class and config interface

**Type mappings**:
| Python | TypeScript |
|--------|------------|
| `np.ndarray` | `Float32Array` |
| `PIL.Image.Image` | `ImageData` |
| `int` / `float` | `number` |
| `bool` | `boolean` |
| `Optional[X]` | `X \| null` |
| `List[X]` | `X[]` |
| `Tuple[X, Y]` | `[X, Y]` |
| `Dict[str, X]` | `Record<string, X>` |

**Operation mappings** (import all from `"../ops.js"`, types from `"../base.js"`):
| Python | TypeScript |
|--------|------------|
| `arr * factor` | `rescale(img, factor)` |
| `(arr - mean) / std` | `normalize(img, mean, std)` |
| `np.transpose(arr, (2, 0, 1))` | `hwcToChw(img)` |
| `img.resize((w, h), BICUBIC)` | `resize(img, { width: w, height: h }, 'bicubic')` |
| `img.resize((w, h), BILINEAR)` | `resize(img, { width: w, height: h }, 'bilinear')` |
| `img.resize((w, h), NEAREST)` | `resize(img, { width: w, height: h }, 'nearest')` |
| `img.crop((l, t, r, b))` | `crop(img, { left: l, top: t, right: r, bottom: b })` |
| `center_crop(img, size)` | `centerCrop(img, size)` |
| `np.pad(arr, padding)` | `pad(img, padding)` |

**Do NOT translate**: `from_pretrained()`, `__call__()` dispatch, logging, deprecation notices, docstring decorators.

**Output path mapping**:
`src/transformers/models/{model}/image_processing_{model}.py` → `src/preprocessing/{model}.ts`
`src/transformers/models/{model}/feature_extraction_{model}.py` → `src/preprocessing/audio_{model}.ts`

## Step 5: Write output files

For each translated file, write it to `src/preprocessing/{model}.ts` (or `audio_{model}.ts`) using the edit tool.

Also update `codegen/.last-sync` with the newest commit SHA (first in the list from Step 2).

## Step 6: Open a PR

If any files were written, create a branch named `sync/preprocessing-{YYYYMMDD}` and open a pull request:

**Title**: `sync: preprocessing update from huggingface/transformers`

**Body**:
```
Automated sync of LFM image preprocessing from [huggingface/transformers](https://github.com/huggingface/transformers).

## Review checklist
- [ ] Numerical operations match the Python source exactly
- [ ] No hand-written logic was overwritten
- [ ] `@generated` header present in all changed files
- [ ] TypeScript compiles cleanly (`npm run typecheck`)

> Files in `src/preprocessing/` marked `@generated` are owned by this workflow. Do not edit them manually.
```

Labels: `automated`, `preprocessing`
