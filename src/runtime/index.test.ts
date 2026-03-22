import { describe, it, expect, vi, afterEach } from "vitest";
import { setResizeImpl } from "../preprocessing/ops.js";
import { cpuResize } from "../preprocessing/resize/cpu.js";

// Keep a reference to the real resize implementation so we can detect what
// initRuntime() installs without importing the function under test directly.
afterEach(() => {
    // Reset to CPU after each test so other tests are unaffected
    setResizeImpl(cpuResize);
    vi.unstubAllGlobals();
});

describe("initRuntime", () => {
    it("returns cpu when navigator.gpu is absent", async () => {
        vi.stubGlobal("navigator", {});
        const { initRuntime } = await import("./index.js");
        const info = await initRuntime("webgpu");
        expect(info.device).toBe("cpu");
        expect(info.gpuAdapter).toBeUndefined();
    });

    it("returns cpu when preferred is cpu", async () => {
        const { initRuntime } = await import("./index.js");
        const info = await initRuntime("cpu");
        expect(info.device).toBe("cpu");
        expect(info.gpuAdapter).toBeUndefined();
    });

    it("returns cpu when requestAdapter returns null", async () => {
        vi.stubGlobal("navigator", {
            gpu: { requestAdapter: async () => null },
        });
        const { initRuntime } = await import("./index.js");
        const info = await initRuntime("webgpu");
        expect(info.device).toBe("cpu");
    });
});
