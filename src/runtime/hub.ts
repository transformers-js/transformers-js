const HF_ENDPOINT = "https://huggingface.co";

export async function fetchRaw(modelId: string, filename: string): Promise<ArrayBuffer> {
    const url = `${HF_ENDPOINT}/${modelId}/resolve/main/${filename}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Hub fetch failed (${res.status}): ${url}`);
    // HuggingFace gated models redirect to a login page (200 HTML) instead of
    // returning a 401. Detect this so ORT doesn't get HTML bytes as model data.
    const ct = res.headers.get("content-type") ?? "";
    if (ct.includes("text/html")) {
        throw new Error(
            `Expected binary data but got HTML for ${url}. ` +
            `The model may be gated — accept its license on huggingface.co first.`,
        );
    }
    return res.arrayBuffer();
}

export async function fetchJSON<T>(modelId: string, filename: string): Promise<T> {
    const buf = await fetchRaw(modelId, filename);
    return JSON.parse(new TextDecoder().decode(buf)) as T;
}
