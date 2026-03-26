import type { ImageData, ResampleFilter } from "../ops.js";

// WebGPU compute shaders for image resize.
// Both share the same Params struct and bind group layout so gpuDispatch is reused.
// Float32 storage buffers throughout — no uint8 quantization at any stage.

const PARAMS_STRUCT = /* wgsl */ `
struct Params {
    src_w:    u32,
    src_h:    u32,
    dst_w:    u32,
    dst_h:    u32,
    channels: u32,
}
@group(0) @binding(0) var<storage, read>       src:    array<f32>;
@group(0) @binding(1) var<storage, read_write> dst:    array<f32>;
@group(0) @binding(2) var<uniform>             params: Params;
`;

const BILINEAR_SHADER = PARAMS_STRUCT + /* wgsl */ `
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let dx = id.x;
    let dy = id.y;
    if (dx >= params.dst_w || dy >= params.dst_h) { return; }

    let scale_x = f32(params.src_w) / f32(params.dst_w);
    let scale_y = f32(params.src_h) / f32(params.dst_h);

    // Half-pixel offset — matches PyTorch align_corners=false and PIL
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

// Keys cubic kernel, a = -0.5 — matches PIL BICUBIC (used by HuggingFace transformers).
// (PyTorch/OpenCV use a = -0.75 — different convention.)
const BICUBIC_SHADER = PARAMS_STRUCT + /* wgsl */ `
fn cubic(t: f32) -> f32 {
    let a = abs(t);
    if (a <= 1.0) { return (1.5*a - 2.5)*a*a + 1.0; }
    if (a <  2.0) { return ((-0.5*a + 2.5)*a - 4.0)*a + 2.0; }
    return 0.0;
}

fn px(x: i32, y: i32, ch: u32) -> f32 {
    let cx = u32(clamp(x, 0, i32(params.src_w) - 1));
    let cy = u32(clamp(y, 0, i32(params.src_h) - 1));
    return src[(cy * params.src_w + cx) * params.channels + ch];
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let dx = id.x;
    let dy = id.y;
    if (dx >= params.dst_w || dy >= params.dst_h) { return; }

    let sx = (f32(dx) + 0.5) * f32(params.src_w) / f32(params.dst_w) - 0.5;
    let sy = (f32(dy) + 0.5) * f32(params.src_h) / f32(params.dst_h) - 0.5;

    let x0 = i32(floor(sx));
    let y0 = i32(floor(sy));
    let fx = sx - floor(sx);
    let fy = sy - floor(sy);

    // 4-tap Keys cubic weights per axis — use named components to avoid
    // dynamic vec4 indexing (wider WGSL compatibility)
    let wx = vec4<f32>(cubic(fx+1.0), cubic(fx), cubic(fx-1.0), cubic(fx-2.0));
    let wy = vec4<f32>(cubic(fy+1.0), cubic(fy), cubic(fy-1.0), cubic(fy-2.0));

    let c = params.channels;
    for (var ch = 0u; ch < c; ch++) {
        let v = wy.x * (wx.x*px(x0-1,y0-1,ch) + wx.y*px(x0,y0-1,ch) + wx.z*px(x0+1,y0-1,ch) + wx.w*px(x0+2,y0-1,ch))
              + wy.y * (wx.x*px(x0-1,y0,  ch) + wx.y*px(x0,y0,  ch) + wx.z*px(x0+1,y0,  ch) + wx.w*px(x0+2,y0,  ch))
              + wy.z * (wx.x*px(x0-1,y0+1,ch) + wx.y*px(x0,y0+1,ch) + wx.z*px(x0+1,y0+1,ch) + wx.w*px(x0+2,y0+1,ch))
              + wy.w * (wx.x*px(x0-1,y0+2,ch) + wx.y*px(x0,y0+2,ch) + wx.z*px(x0+1,y0+2,ch) + wx.w*px(x0+2,y0+2,ch));
        dst[(dy * params.dst_w + dx) * c + ch] = v;
    }
}
`;

export function makeWebGPUResize(device: GPUDevice) {
    const bilinearPipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module: device.createShaderModule({ code: BILINEAR_SHADER }), entryPoint: "main" },
    });
    const bicubicPipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module: device.createShaderModule({ code: BICUBIC_SHADER }), entryPoint: "main" },
    });

    async function gpuDispatch(
        pipeline: GPUComputePipeline,
        image: ImageData,
        size: { width: number; height: number },
    ): Promise<ImageData> {
        const { width: dw, height: dh } = size;
        const { width: sw, height: sh, channels: c } = image;
        const dstBytes = dw * dh * c * 4;

        const srcBuf = device.createBuffer({
            size: image.data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(srcBuf, 0, image.data.buffer as ArrayBuffer, image.data.byteOffset, image.data.byteLength);

        const dstBuf = device.createBuffer({
            size: dstBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const paramsBuf = device.createBuffer({
            size: 20, // 5 × u32
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

        const readBuf = device.createBuffer({
            size: dstBytes,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        enc.copyBufferToBuffer(dstBuf, 0, readBuf, 0, dstBytes);
        device.queue.submit([enc.finish()]);

        await readBuf.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(readBuf.getMappedRange().slice(0));
        readBuf.unmap();

        srcBuf.destroy();
        dstBuf.destroy();
        paramsBuf.destroy();
        readBuf.destroy();

        return { data: result, width: dw, height: dh, channels: c };
    }

    return async function webgpuResize(
        image: ImageData,
        size: { width: number; height: number },
        filter: ResampleFilter = "bilinear",
    ): Promise<ImageData> {
        if (filter === "bilinear") return gpuDispatch(bilinearPipeline, image, size);
        if (filter === "bicubic") return gpuDispatch(bicubicPipeline, image, size);
        // Nearest: fast enough on CPU, not worth a GPU round-trip for most images.
        const { cpuResize } = await import("./cpu.js");
        return cpuResize(image, size, filter);
    };
}
