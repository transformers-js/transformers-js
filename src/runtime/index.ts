import { setResizeImpl } from "../preprocessing/ops.js";
import { cpuResize } from "../preprocessing/resize/cpu.js";

export type Device = "webgpu" | "cpu";

export interface RuntimeInfo {
    device: Device;
    gpuAdapter?: GPUAdapter;
}

/** Initialize the runtime. Must be called before any preprocessing.
 *  Automatically selects WebGPU if available, falls back to CPU. */
export async function initRuntime(preferred: Device = "webgpu"): Promise<RuntimeInfo> {
    if (preferred === "webgpu" && typeof navigator !== "undefined" && navigator.gpu) {
        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (adapter) {
                const gpuDevice = await adapter.requestDevice();
                const { makeWebGPUResize } = await import("../preprocessing/resize/webgpu.js");
                setResizeImpl(makeWebGPUResize(gpuDevice));

                gpuDevice.addEventListener("uncapturederror", (e: Event) => {
                    console.error("[transformers-js] WebGPU device error:", e);
                });

                return { device: "webgpu", gpuAdapter: adapter };
            }
        } catch (err) {
            console.warn("[transformers-js] WebGPU unavailable, falling back to CPU:", err);
        }
    }

    setResizeImpl(cpuResize);
    return { device: "cpu" };
}
