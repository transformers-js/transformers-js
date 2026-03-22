export type PipelineTask = "image-classification" | "object-detection" | "image-segmentation";

export interface PipelineOptions {
    model: string;
    device?: "webgpu" | "wasm";
}

export async function pipeline(_task: PipelineTask, _options: PipelineOptions): Promise<never> {
    throw new Error("Not implemented yet");
}
