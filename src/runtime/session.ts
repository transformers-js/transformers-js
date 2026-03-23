import type { Device } from "./index.js";

export interface TensorInput {
    data: Float32Array | BigInt64Array;
    dims: readonly number[];
}

export interface TensorOutput {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    data: any;
    dims: readonly number[];
}

// ORT is loaded once and reused. Dynamic import keeps onnxruntime-node
// out of the browser bundle and onnxruntime-web out of the Node bundle.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let _ort: any = null;
const WASM_CDN = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/";

async function getORT() {
    if (_ort) return _ort;

    const isNode = typeof process !== "undefined" && !!process.versions?.node;
    if (isNode) {
        _ort = await import("onnxruntime-node");
    } else {
        _ort = await import("onnxruntime-web");
    }

    return _ort;
}

function ensureWasmPaths(ort: unknown): void {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const o = ort as any;
    if (o?.env?.wasm && !o.env.wasm.wasmPaths) {
        o.env.wasm.wasmPaths = WASM_CDN;
    }
}

export interface ExternalDataFile {
    /** Filename as referenced inside the .onnx proto (e.g. "model_q8.onnx_data"). */
    path: string;
    data: ArrayBuffer;
}

export class ONNXSession {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    private constructor(private readonly session: any, private readonly ort: any) {}

    static async load(
        modelBuffer: ArrayBuffer,
        device: Device,
        externalData?: ExternalDataFile[],
    ): Promise<ONNXSession> {
        const ort = await getORT();

        // Protobuf first-byte sanity check. HTML (gated model redirect) and
        // other text responses start with '<' (0x3c) — catch before ORT does.
        const firstByte = new Uint8Array(modelBuffer, 0, 1)[0];
        if (firstByte === 0x3c /* '<' */) {
            throw new Error(
                "Model buffer starts with '<' — received HTML instead of ONNX binary. " +
                "The model may be gated; accept its license on huggingface.co.",
            );
        }

        const opts = externalData ? { externalData } : {};

        // Try preferred EP first; if WebGPU session creation fails, fall back to
        // WASM so the model can still run (slower, but beats a crash).
        const candidates: string[][] = device === "webgpu" ? [["webgpu"], ["wasm"]] : [["wasm"]];

        let lastErr: unknown;
        for (const eps of candidates) {
            if (eps[0] === "wasm") ensureWasmPaths(ort);
            try {
                const session = await ort.InferenceSession.create(modelBuffer, {
                    executionProviders: eps,
                    ...opts,
                });
                if (device === "webgpu" && eps[0] === "wasm") {
                    console.warn("[transformers-js] WebGPU EP failed, fell back to WASM EP.");
                }
                return new ONNXSession(session, ort);
            } catch (err) {
                lastErr = err;
                if (eps !== candidates[candidates.length - 1]) {
                    console.warn(`[transformers-js] ${eps[0]} EP failed (${err}), trying wasm…`);
                }
            }
        }

        // Convert raw WASM exception pointers to readable Errors.
        if (typeof lastErr === "number") {
            throw new Error(
                `ORT session creation failed with native exception (code ${lastErr}). ` +
                `Check the browser console above for ORT error details.`,
            );
        }
        throw lastErr;
    }

    async run(inputs: Record<string, TensorInput>): Promise<Record<string, TensorOutput>> {
        const ort = this.ort;
        const feeds: Record<string, unknown> = {};
        // Filter to inputs the model actually declares; some ONNX exports omit
        // optional inputs (e.g. attention_mask in CLIP text models) and ORT
        // errors on any key not in inputNames.
        const validNames = new Set<string>(this.session.inputNames ?? []);
        for (const [name, { data, dims }] of Object.entries(inputs)) {
            if (validNames.size > 0 && !validNames.has(name)) continue;
            const dtype = data instanceof BigInt64Array ? "int64" : "float32";
            feeds[name] = new ort.Tensor(dtype, data, dims);
        }

        const results = await this.session.run(feeds) as Record<string, { data: Float32Array; dims: readonly number[] }>;
        const out: Record<string, TensorOutput> = {};
        for (const [name, tensor] of Object.entries(results)) {
            out[name] = { data: tensor.data, dims: tensor.dims };
        }
        return out;
    }

    dispose(): void {
        this.session.release?.();
    }
}
