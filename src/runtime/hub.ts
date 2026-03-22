const HF_ENDPOINT = "https://huggingface.co";

export async function fetchRaw(modelId: string, filename: string): Promise<ArrayBuffer> {
    const url = `${HF_ENDPOINT}/${modelId}/resolve/main/${filename}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Hub fetch failed (${res.status}): ${url}`);
    return res.arrayBuffer();
}

export async function fetchJSON<T>(modelId: string, filename: string): Promise<T> {
    const buf = await fetchRaw(modelId, filename);
    return JSON.parse(new TextDecoder().decode(buf)) as T;
}
