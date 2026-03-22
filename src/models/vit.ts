import { fetchRaw, fetchJSON } from "../runtime/hub.js";
import { ONNXSession } from "../runtime/session.js";
import { ImageProcessor } from "../preprocessing/image-processor.js";
import type { Device } from "../runtime/index.js";
import type { ImageData } from "../preprocessing/ops.js";

interface ModelConfig {
    id2label?: Record<string, string>;
    num_labels?: number;
}

export class ViTForImageClassification {
    private constructor(
        private readonly session: ONNXSession,
        private readonly processor: ImageProcessor,
        private readonly id2label: Record<number, string>,
    ) {}

    static async fromHub(modelId: string, device: Device = "webgpu"): Promise<ViTForImageClassification> {
        const [modelBuffer, processor, config] = await Promise.all([
            fetchRaw(modelId, "onnx/model.onnx"),
            ImageProcessor.fromHub(modelId),
            fetchJSON<ModelConfig>(modelId, "config.json"),
        ]);

        const session = await ONNXSession.load(modelBuffer, device);

        // id2label keys are strings in JSON — convert to numbers
        const id2label: Record<number, string> = {};
        for (const [k, v] of Object.entries(config.id2label ?? {})) {
            id2label[Number(k)] = v;
        }

        return new ViTForImageClassification(session, processor, id2label);
    }

    async run(image: ImageData): Promise<Float32Array> {
        const { config } = this.processor;
        const pixelValues = await this.processor.preprocess(image);
        const dims = [1, 3, config.size.height, config.size.width] as const;

        const outputs = await this.session.run({
            pixel_values: { data: pixelValues, dims },
        });

        return outputs["logits"]!;
    }

    label(classIndex: number): string {
        return this.id2label[classIndex] ?? `LABEL_${classIndex}`;
    }

    dispose(): void {
        this.session.dispose();
    }
}
