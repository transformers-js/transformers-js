import type { Device } from "./index.js";

export interface TensorInput {
    data: Float32Array;
    dims: readonly number[];
}

// ORT is loaded once and reused. Dynamic import keeps onnxruntime-node
// out of the browser bundle and onnxruntime-web out of the Node bundle.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let _ort: any = null;

async function getORT() {
    if (_ort) return _ort;

    const isNode = typeof process !== "undefined" && !!process.versions?.node;
    if (isNode) {
        _ort = await import("onnxruntime-node");
    } else {
        _ort = await import("onnxruntime-web");
        // Point WASM loader at the CDN so the browser doesn't need a local copy
        _ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
    }

    return _ort;
}

export class ONNXSession {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    private constructor(private readonly session: any) {}

    static async load(modelBuffer: ArrayBuffer, device: Device): Promise<ONNXSession> {
        const ort = await getORT();
        const eps = device === "webgpu" ? ["webgpu", "wasm"] : ["wasm"];
        const session = await ort.InferenceSession.create(modelBuffer, {
            executionProviders: eps,
        });
        return new ONNXSession(session);
    }

    async run(inputs: Record<string, TensorInput>): Promise<Record<string, Float32Array>> {
        const ort = await getORT();
        const feeds: Record<string, unknown> = {};
        for (const [name, { data, dims }] of Object.entries(inputs)) {
            feeds[name] = new ort.Tensor("float32", data, dims);
        }

        const results = await this.session.run(feeds) as Record<string, { data: Float32Array }>;
        const out: Record<string, Float32Array> = {};
        for (const [name, tensor] of Object.entries(results)) {
            out[name] = tensor.data;
        }
        return out;
    }

    dispose(): void {
        this.session.release?.();
    }
}
