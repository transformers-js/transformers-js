import type { ImageData, ResampleFilter } from "../ops.js";

// Bilinear resize as a WebGPU compute shader.
// Uses float32 storage buffers throughout — no uint8 quantization.
// Bicubic/nearest fall back to CPU (imported lazily to avoid bundling CPU code on GPU path).

const SHADER = /* wgsl */ `
struct Params {
    src_w:   u32,
    src_h:   u32,
    dst_w:   u32,
    dst_h:   u32,
    channels: u32,
}

@group(0) @binding(0) var<storage, read>       src:    array<f32>;
@group(0) @binding(1) var<storage, read_write> dst:    array<f32>;
@group(0) @binding(2) var<uniform>             params: Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let dx = id.x;
    let dy = id.y;
    if (dx >= params.dst_w || dy >= params.dst_h) { return; }

    let scale_x = f32(params.src_w) / f32(params.dst_w);
    let scale_y = f32(params.src_h) / f32(params.dst_h);

    // Half-pixel offset — matches PyTorch align_corners=false
    let sx = (f32(dx) + 0.5) * scale_x - 0.5;
    let sy = (f32(dy) + 0.5) * scale_y - 0.5;

    let x0 = i32(floor(sx));
    let y0 = i32(floor(sy));
    let fx = sx - floor(sx);
    let fy = sy - floor(sy);

    let cx0 = u32(clamp(x0,     0, i32(params.src_w) - 1));
    let cx1 = u32(clamp(x0 + 1, 0, i32(params.src_w) - 1));
    let cy0 = u32(clamp(y0,     0, i32(params.src_h) - 1));
    let cy1 = u32(clamp(y0 + 1, 0, i32(params.src_h) - 1));

    let w00 = (1.0 - fx) * (1.0 - fy);
    let w10 =        fx  * (1.0 - fy);
    let w01 = (1.0 - fx) *        fy;
    let w11 =        fx  *        fy;

    let c = params.channels;
    for (var ch = 0u; ch < c; ch++) {
        let tl = src[(cy0 * params.src_w + cx0) * c + ch];
        let tr = src[(cy0 * params.src_w + cx1) * c + ch];
        let bl = src[(cy1 * params.src_w + cx0) * c + ch];
        let br = src[(cy1 * params.src_w + cx1) * c + ch];
        dst[(dy * params.dst_w + dx) * c + ch] = w00*tl + w10*tr + w01*bl + w11*br;
    }
}
`;

export function makeWebGPUResize(device: GPUDevice) {
    // Compile the shader once and reuse across calls
    const module = device.createShaderModule({ code: SHADER });
    const pipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module, entryPoint: "main" },
    });

    return async function webgpuResize(
        image: ImageData,
        size: { width: number; height: number },
        filter: ResampleFilter = "bilinear",
    ): Promise<ImageData> {
        // Non-bilinear filters fall back to CPU — import lazily so this module
        // is self-contained (no circular dep on ops.ts)
        if (filter !== "bilinear") {
            const { cpuResize } = await import("./cpu.js");
            return cpuResize(image, size, filter);
        }

        const { width: dw, height: dh } = size;
        const { width: sw, height: sh, channels: c } = image;

        const srcBytes = image.data.byteLength;
        const dstBytes = dw * dh * c * 4; // float32

        // Upload input
        const srcBuf = device.createBuffer({
            size: srcBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(srcBuf, 0, image.data.buffer as ArrayBuffer, image.data.byteOffset, image.data.byteLength);

        // Output buffer
        const dstBuf = device.createBuffer({
            size: dstBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        // Uniform params
        const paramsBuf = device.createBuffer({
            size: 20, // 5 x u32
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(paramsBuf, 0, new Uint32Array([sw, sh, dw, dh, c]));

        const bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: srcBuf } },
                { binding: 1, resource: { buffer: dstBuf } },
                { binding: 2, resource: { buffer: paramsBuf } },
            ],
        });

        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(dw / 8), Math.ceil(dh / 8));
        pass.end();

        // Readback
        const readBuf = device.createBuffer({
            size: dstBytes,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        enc.copyBufferToBuffer(dstBuf, 0, readBuf, 0, dstBytes);
        device.queue.submit([enc.finish()]);

        await readBuf.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(readBuf.getMappedRange().slice(0));
        readBuf.unmap();

        // Cleanup
        srcBuf.destroy();
        dstBuf.destroy();
        paramsBuf.destroy();
        readBuf.destroy();

        return { data: result, width: dw, height: dh, channels: c };
    };
}
