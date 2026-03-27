import type { ONNXSession, TensorInput } from "../runtime/session.js";
import { argmax, sampleTopP, type SamplingOptions } from "./sampling.js";

export interface GenerationConfig {
    eosTokenId: number;
    maxNewTokens?: number;
    sampling?: SamplingOptions;
}

export interface LFM2ModelConfig {
    num_key_value_heads: number;
    hidden_size: number;
    conv_L_cache: number;
    /** head_dim = hidden_size / num_attention_heads */
    num_attention_heads: number;
}

/**
 * Build the initial (empty) KV+conv cache directly from the session's input
 * names. This is the only reliable source — inferring from config fields like
 * layer_types leads to fragile index arithmetic when naming conventions differ.
 *
 * Recognised patterns:
 *   past_key_values.N.key / past_key_values.N.value → [1, num_kv_heads, 0, head_dim]
 *   past_conv.N                                      → [1, hidden_size, conv_L_cache]
 */
export function initCache(inputNames: string[], cfg: LFM2ModelConfig): Record<string, TensorInput> {
    const cache: Record<string, TensorInput> = {};
    const headDim = cfg.hidden_size / cfg.num_attention_heads;

    for (const name of inputNames) {
        if (name.endsWith(".key") || name.endsWith(".value")) {
            cache[name] = { data: new Float32Array(0), dims: [1, cfg.num_key_value_heads, 0, headDim] };
        } else if (name.startsWith("past_conv.")) {
            cache[name] = {
                data: new Float32Array(cfg.hidden_size * cfg.conv_L_cache),
                dims: [1, cfg.hidden_size, cfg.conv_L_cache],
            };
        }
    }

    return cache;
}

/**
 * Copy present-state outputs back into the cache under their past-state names.
 * present.N.key/value  → past_key_values.N.key/value
 * present_conv.N       → past_conv.N
 */
export function updateCache(
    cache: Record<string, TensorInput>,
    outputs: Record<string, { data: unknown; dims: readonly number[] }>,
): void {
    for (const [name, tensor] of Object.entries(outputs)) {
        if (name === "logits") continue;
        const cacheKey = name
            .replace("present_conv.", "past_conv.")
            .replace(/^present\./, "past_key_values.");
        cache[cacheKey] = { data: tensor.data as Float32Array, dims: tensor.dims };
    }
}

/**
 * Autoregressive generation loop for LFM2-style ONNX models.
 *
 * @param onToken - Called with each generated token id as it is produced,
 *                  including the first token from prefill. Enables streaming.
 */
export async function generate(
    session: ONNXSession,
    promptIds: number[],
    modelCfg: LFM2ModelConfig,
    genCfg: GenerationConfig,
    hasPositionIds: boolean,
    inputNames: string[],
    onToken?: (tokenId: number) => void,
): Promise<number[]> {
    const { eosTokenId, maxNewTokens = 512, sampling } = genCfg;
    const generated: number[] = [];
    const cache = initCache(inputNames, modelCfg);

    // ── Prefill ────────────────────────────────────────────────────────────
    const seqLen = promptIds.length;
    const inputIds = new BigInt64Array(promptIds.map(BigInt));
    const attentionMask = new BigInt64Array(seqLen).fill(1n);

    const hasNumLogitsToKeep = inputNames.includes("num_logits_to_keep");

    const prefillInputs: Record<string, TensorInput> = {
        input_ids: { data: inputIds, dims: [1, seqLen] },
        attention_mask: { data: attentionMask, dims: [1, seqLen] },
        ...cache,
    };
    if (hasPositionIds) {
        const posIds = new BigInt64Array(seqLen).map((_, i) => BigInt(i));
        prefillInputs["position_ids"] = { data: posIds, dims: [1, seqLen] };
    }
    if (hasNumLogitsToKeep) {
        prefillInputs["num_logits_to_keep"] = { data: new BigInt64Array([1n]), dims: [1] };
    }

    const prefillOut = await session.run(prefillInputs);
    updateCache(cache, prefillOut);

    const logitsDims = prefillOut["logits"]!.dims;
    // vocab size is always the last dimension regardless of output shape.
    const vocabSize = logitsDims[logitsDims.length - 1]!;
    // Always slice the last vocabSize elements: works for [1,seqLen,V],
    // [1,1,V] (when num_logits_to_keep=1), and [1,V] (2-D exports).
    const logitsData = prefillOut["logits"]!.data as Float32Array;
    const lastLogits = logitsData.subarray(logitsData.length - vocabSize);
    let nextToken = sampling ? sampleTopP(lastLogits, sampling) : argmax(lastLogits);
    generated.push(nextToken);
    onToken?.(nextToken);

    // ── Decode loop ────────────────────────────────────────────────────────
    let pastLen = seqLen;

    while (nextToken !== eosTokenId && generated.length < maxNewTokens) {
        const decodeInputs: Record<string, TensorInput> = {
            input_ids: { data: new BigInt64Array([BigInt(nextToken)]), dims: [1, 1] },
            attention_mask: { data: new BigInt64Array(pastLen + 1).fill(1n), dims: [1, pastLen + 1] },
            ...cache,
        };
        if (hasPositionIds) {
            decodeInputs["position_ids"] = {
                data: new BigInt64Array([BigInt(pastLen)]),
                dims: [1, 1],
            };
        }
        if (hasNumLogitsToKeep) {
            decodeInputs["num_logits_to_keep"] = { data: new BigInt64Array([1n]), dims: [1] };
        }

        const out = await session.run(decodeInputs);
        updateCache(cache, out);
        pastLen++;

        const logits = out["logits"]!.data as Float32Array;
        nextToken = sampling ? sampleTopP(logits, sampling) : argmax(logits);
        generated.push(nextToken);
        onToken?.(nextToken);
    }

    // Strip trailing EOS if present
    if (generated[generated.length - 1] === eosTokenId) generated.pop();

    return generated;
}
