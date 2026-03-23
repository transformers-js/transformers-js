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

// ORT WebGPU EP only supports one session creation at a time. Serialize all
// WebGPU session creations through this promise chain to prevent the
// "another WebGPU EP inference session is being created" error that would
// otherwise cause a spurious WASM fallback when multiple models load together.
let _webgpuQueue: Promise<unknown> = Promise.resolve();

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

        if (device === "webgpu") {
            // Serialize WebGPU session creation: wait for any in-progress
            // creation to finish, then run ours. If WebGPU truly fails (not
            // just a concurrency race), fall back to WASM.
            let release!: () => void;
            const ticket = new Promise<void>(r => { release = r; });
            const prev = _webgpuQueue;
            _webgpuQueue = ticket;

            try {
                await prev;
                const session = await ort.InferenceSession.create(modelBuffer, {
                    executionProviders: ["webgpu"],
                    ...opts,
                });
                return new ONNXSession(session, ort);
            } catch (err) {
                console.warn(`[transformers-js] WebGPU EP failed (${err}), falling back to WASM.`);
                // Fall through to WASM below.
            } finally {
                release();
            }
        }

        // WASM path (either device="wasm" or WebGPU fallback).
        ensureWasmPaths(ort);
        try {
            const session = await ort.InferenceSession.create(modelBuffer, {
                executionProviders: ["wasm"],
                ...opts,
            });
            return new ONNXSession(session, ort);
        } catch (err) {
            if (typeof err === "number") {
                throw new Error(
                    `ORT session creation failed with native exception (code ${err}). ` +
                    `Check the browser console for ORT error details.`,
                );
            }
            throw err;
        }
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
