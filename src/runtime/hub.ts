const HF_ENDPOINT = "https://huggingface.co";

let _hfToken: string | null = null;

/** Set a HuggingFace access token for downloading gated models. */
export function setHFToken(token: string): void {
    _hfToken = token.trim() || null;
}

export async function fetchRaw(modelId: string, filename: string): Promise<ArrayBuffer> {
    const url = `${HF_ENDPOINT}/${modelId}/resolve/main/${filename}`;
    const headers: HeadersInit = _hfToken ? { Authorization: `Bearer ${_hfToken}` } : {};
    const res = await fetch(url, { headers });
    if (!res.ok) {
        const hint = res.status === 401
            ? " (gated model — accept the license on huggingface.co and provide your access token)"
            : "";
        throw new Error(`Hub fetch failed (${res.status})${hint}: ${url}`);
    }
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
