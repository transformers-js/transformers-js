import type { ImageData, ResampleFilter } from "../ops.js";

// Pixel alignment matches PyTorch / PIL: half-pixel offset (align_corners=false).
// sx = (dx + 0.5) * (src_w / dst_w) - 0.5

function clamp(v: number, lo: number, hi: number): number {
    return v < lo ? lo : v > hi ? hi : v;
}

function idx(image: ImageData, x: number, y: number, ch: number): number {
    return (y * image.width + x) * image.channels + ch;
}

// ── Nearest neighbour ──────────────────────────────────────────────────────

function nearest(image: ImageData, size: { width: number; height: number }): ImageData {
    const { data, width: sw, height: sh, channels: c } = image;
    const { width: dw, height: dh } = size;
    const out = new Float32Array(dw * dh * c);

    for (let dy = 0; dy < dh; dy++) {
        for (let dx = 0; dx < dw; dx++) {
            const sx = clamp(Math.floor((dx + 0.5) * (sw / dw)), 0, sw - 1);
            const sy = clamp(Math.floor((dy + 0.5) * (sh / dh)), 0, sh - 1);
            for (let ch = 0; ch < c; ch++) {
                out[(dy * dw + dx) * c + ch] = data[idx(image, sx, sy, ch)]!;
            }
        }
    }

    return { data: out, width: dw, height: dh, channels: c };
}

// ── Bilinear ───────────────────────────────────────────────────────────────

function bilinear(image: ImageData, size: { width: number; height: number }): ImageData {
    const { data, width: sw, height: sh, channels: c } = image;
    const { width: dw, height: dh } = size;
    const out = new Float32Array(dw * dh * c);

    const scaleX = sw / dw;
    const scaleY = sh / dh;

    for (let dy = 0; dy < dh; dy++) {
        for (let dx = 0; dx < dw; dx++) {
            const sx = (dx + 0.5) * scaleX - 0.5;
            const sy = (dy + 0.5) * scaleY - 0.5;

            const x0 = Math.floor(sx);
            const y0 = Math.floor(sy);
            const fx = sx - x0;
            const fy = sy - y0;

            const cx0 = clamp(x0,     0, sw - 1);
            const cx1 = clamp(x0 + 1, 0, sw - 1);
            const cy0 = clamp(y0,     0, sh - 1);
            const cy1 = clamp(y0 + 1, 0, sh - 1);

            const w00 = (1 - fx) * (1 - fy);
            const w10 =      fx  * (1 - fy);
            const w01 = (1 - fx) *      fy;
            const w11 =      fx  *      fy;

            for (let ch = 0; ch < c; ch++) {
                out[(dy * dw + dx) * c + ch] =
                    w00 * data[idx(image, cx0, cy0, ch)]! +
                    w10 * data[idx(image, cx1, cy0, ch)]! +
                    w01 * data[idx(image, cx0, cy1, ch)]! +
                    w11 * data[idx(image, cx1, cy1, ch)]!;
            }
        }
    }

    return { data: out, width: dw, height: dh, channels: c };
}

// ── Bicubic (Keys cubic, a = -0.5 — matches PIL BICUBIC) ───────────────────
// Note: OpenCV and PyTorch use a = -0.75; PIL uses a = -0.5.
// HuggingFace transformers calls PIL for resize, so we match PIL.

function cubicWeight(t: number): number {
    const a = -0.5;
    const at = Math.abs(t);
    if (at <= 1) return (a + 2) * at ** 3 - (a + 3) * at ** 2 + 1;
    if (at < 2)  return a * at ** 3 - 5 * a * at ** 2 + 8 * a * at - 4 * a;
    return 0;
}

function bicubic(image: ImageData, size: { width: number; height: number }): ImageData {
    const { data, width: sw, height: sh, channels: c } = image;
    const { width: dw, height: dh } = size;
    const out = new Float32Array(dw * dh * c);

    const scaleX = sw / dw;
    const scaleY = sh / dh;

    for (let dy = 0; dy < dh; dy++) {
        for (let dx = 0; dx < dw; dx++) {
            const sx = (dx + 0.5) * scaleX - 0.5;
            const sy = (dy + 0.5) * scaleY - 0.5;

            const x0 = Math.floor(sx);
            const y0 = Math.floor(sy);

            // Precompute horizontal and vertical weights for the 4-tap kernel
            const wx: number[] = [];
            const wy: number[] = [];
            for (let k = -1; k <= 2; k++) {
                wx.push(cubicWeight(sx - (x0 + k)));
                wy.push(cubicWeight(sy - (y0 + k)));
            }

            for (let ch = 0; ch < c; ch++) {
                let val = 0;
                for (let ky = 0; ky < 4; ky++) {
                    const py = clamp(y0 + ky - 1, 0, sh - 1);
                    for (let kx = 0; kx < 4; kx++) {
                        const px = clamp(x0 + kx - 1, 0, sw - 1);
                        val += wy[ky]! * wx[kx]! * data[idx(image, px, py, ch)]!;
                    }
                }
                out[(dy * dw + dx) * c + ch] = val;
            }
        }
    }

    return { data: out, width: dw, height: dh, channels: c };
}

// ── Public API ─────────────────────────────────────────────────────────────

export async function cpuResize(
    image: ImageData,
    size: { width: number; height: number },
    filter: ResampleFilter = "bilinear",
): Promise<ImageData> {
    switch (filter) {
        case "nearest":  return nearest(image, size);
        case "bilinear": return bilinear(image, size);
        case "bicubic":  return bicubic(image, size);
        case "lanczos":  return bicubic(image, size); // lanczos deferred; bicubic is close
    }
}
