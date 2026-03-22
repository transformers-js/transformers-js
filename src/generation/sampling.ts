/**
 * Token sampling strategies.
 * All operate on a raw logits slice (Float32Array of length vocab_size).
 */

export interface SamplingOptions {
    /** temperature=1 (default) is unmodified; <1 sharpens, >1 flattens. */
    temperature?: number;
    /** Top-p (nucleus) sampling threshold. 1.0 = disabled. */
    topP?: number;
}

/** Greedy — always pick the highest-logit token. */
export function argmax(logits: Float32Array): number {
    let best = 0;
    for (let i = 1; i < logits.length; i++) {
        if (logits[i]! > logits[best]!) best = i;
    }
    return best;
}

/** Top-p nucleus sampling with optional temperature. */
export function sampleTopP(logits: Float32Array, opts: SamplingOptions = {}): number {
    const { temperature = 1.0, topP = 1.0 } = opts;

    // Apply temperature
    const scaled = temperature === 1.0 ? logits : logits.map((v) => v / temperature);

    // Softmax
    const max = Math.max(...scaled);
    const exps = scaled.map((v) => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map((v) => v / sum);

    if (topP >= 1.0) return sampleFromProbs(probs);

    // Sort by descending probability, accumulate until cumulative >= topP
    const indexed = Array.from(probs).map((p, i) => [i, p] as [number, number]);
    indexed.sort((a, b) => b[1] - a[1]);

    let cumulative = 0;
    const nucleus: [number, number][] = [];
    for (const [i, p] of indexed) {
        nucleus.push([i, p]);
        cumulative += p;
        if (cumulative >= topP) break;
    }

    // Renormalize within nucleus
    const nucleusSum = nucleus.reduce((a, [, p]) => a + p, 0);
    const renorm = nucleus.map(([i, p]) => [i, p / nucleusSum] as [number, number]);

    // Sample
    const r = Math.random();
    let acc = 0;
    for (const [i, p] of renorm) {
        acc += p;
        if (r <= acc) return i;
    }
    return renorm[renorm.length - 1]![0];
}

function sampleFromProbs(probs: Float32Array): number {
    const r = Math.random();
    let acc = 0;
    for (let i = 0; i < probs.length; i++) {
        acc += probs[i]!;
        if (r <= acc) return i;
    }
    return probs.length - 1;
}
