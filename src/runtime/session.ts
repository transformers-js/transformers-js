import type { Device } from "./index.js";

export interface TensorInput {
    data: Float32Array | BigInt64Array;
    dims: readonly number[];
}

export interface TensorOutput {
    data: Float32Array;
    dims: readonly number[];
}

// ORT is loaded once and reused. Dynamic import keeps onnxruntime-node
// out of the browser bundle and onnxruntime-web out of the Node bundle.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let _ort: any = null;

async function getORT(device: Device) {
    if (_ort) return _ort;

    const isNode = typeof process !== "undefined" && !!process.versions?.node;
    if (isNode) {
        _ort = await import("onnxruntime-node");
    } else {
        _ort = await import("onnxruntime-web");
        if (device !== "webgpu") {
            // Point WASM loader at the CDN — keeps the binary out of the bundle
            // and off the WebGPU path entirely (loaded lazily only when needed).
            _ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
        }
    }

    return _ort;
}

export class ONNXSession {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    private constructor(private readonly session: any, private readonly ort: any) {}

    static async load(modelBuffer: ArrayBuffer, device: Device): Promise<ONNXSession> {
        const ort = await getORT(device);
        // WebGPU path: WASM EP excluded — binary never loaded on the WebGPU path.
        // CPU path: WASM EP only.
        const eps = device === "webgpu" ? ["webgpu"] : ["wasm"];
        const session = await ort.InferenceSession.create(modelBuffer, {
            executionProviders: eps,
        });
        return new ONNXSession(session, ort);
    }

    async run(inputs: Record<string, TensorInput>): Promise<Record<string, TensorOutput>> {
        const ort = this.ort;
        const feeds: Record<string, unknown> = {};
        for (const [name, { data, dims }] of Object.entries(inputs)) {
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
