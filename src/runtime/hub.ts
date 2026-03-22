const HF_ENDPOINT = "https://huggingface.co";

let _hfToken: string | null = null;

/** Set a HuggingFace access token for downloading gated models. */
export function setHFToken(token: string): void {
    _hfToken = token.trim() || null;
}

/**
 * Fetch a raw file either from HuggingFace Hub or a mirror base URL.
 *
 * When `mirrorBaseUrl` is provided the file is fetched as:
 *   `${mirrorBaseUrl}/${basename(filename)}`
 * Mirror files are assumed public — no HF token is sent.
 */
export async function fetchRaw(
    modelId: string,
    filename: string,
    mirrorBaseUrl?: string,
): Promise<ArrayBuffer> {
    const url = mirrorBaseUrl
        ? `${mirrorBaseUrl}/${filename.split("/").pop()!}`
        : `${HF_ENDPOINT}/${modelId}/resolve/main/${filename}`;

    const headers: HeadersInit =
        !mirrorBaseUrl && _hfToken ? { Authorization: `Bearer ${_hfToken}` } : {};

    const res = await fetch(url, { headers });
    if (!res.ok) {
        const hint = !mirrorBaseUrl && res.status === 401
            ? " (gated model — accept the license on huggingface.co and provide your access token)"
            : "";
        throw new Error(`Hub fetch failed (${res.status})${hint}: ${url}`);
    }
    const ct = res.headers.get("content-type") ?? "";
    if (ct.includes("text/html")) {
        throw new Error(
            `Expected binary data but got HTML for ${url}. ` +
            `The model may be gated — accept its license on huggingface.co first.`,
        );
    }
    return res.arrayBuffer();
}

export async function fetchJSON<T>(
    modelId: string,
    filename: string,
    mirrorBaseUrl?: string,
): Promise<T> {
    const buf = await fetchRaw(modelId, filename, mirrorBaseUrl);
    return JSON.parse(new TextDecoder().decode(buf)) as T;
}
