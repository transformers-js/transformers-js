/**
 * Sync preprocessing code from huggingface/transformers.
 *
 * Usage:
 *   npm run sync               — check for new commits and translate changed files
 *   npm run sync:init <sha>    — set the starting commit SHA (run once to bootstrap)
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { translate } from "./translate.js";

const HF_REPO = "huggingface/transformers";
const LAST_SYNC_FILE = join(import.meta.dirname, ".last-sync");
const OUT_DIR = join(import.meta.dirname, "../src/preprocessing");

const WATCHED_PATTERNS = [
    /^src\/transformers\/models\/[^/]+\/image_processing_[^/]+\.py$/,
    /^src\/transformers\/models\/[^/]+\/feature_extraction_[^/]+\.py$/,
];

// Registry / routing files — not processor implementations, skip them.
const EXCLUDED_FILES = new Set([
    "src/transformers/models/auto/image_processing_auto.py",
    "src/transformers/models/auto/feature_extraction_auto.py",
]);

// ── GitHub API ─────────────────────────────────────────────────────────────

async function ghFetch(path: string): Promise<Response> {
    const token = process.env.GITHUB_TOKEN ?? process.env.GH_TOKEN;
    const res = await fetch(`https://api.github.com${path}`, {
        headers: {
            Accept: "application/vnd.github.v3+json",
            "User-Agent": "transformers-js-sync",
            ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
    });
    if (!res.ok) throw new Error(`GitHub API ${path}: ${res.status} ${res.statusText}`);
    return res;
}

async function getCommitsSince(lastSha: string): Promise<string[]> {
    const res = await ghFetch(`/repos/${HF_REPO}/commits?sha=main&per_page=100`);
    const commits = (await res.json()) as { sha: string }[];
    const sinceIdx = commits.findIndex((c) => c.sha.startsWith(lastSha));
    return sinceIdx === -1
        ? commits.map((c) => c.sha)
        : commits.slice(0, sinceIdx).map((c) => c.sha);
}

async function getChangedFiles(commitSha: string): Promise<string[]> {
    try {
        const res = await ghFetch(`/repos/${HF_REPO}/commits/${commitSha}`);
        const data = (await res.json()) as { files?: { filename: string }[] };
        return (data.files ?? []).map((f) => f.filename);
    } catch {
        return []; // non-fatal: skip commits that fail
    }
}

async function getFileContent(path: string): Promise<string> {
    const res = await ghFetch(`/repos/${HF_REPO}/contents/${path}?ref=main`);
    const data = (await res.json()) as { content: string };
    return Buffer.from(data.content, "base64").toString("utf-8");
}

// ── Path helpers ────────────────────────────────────────────────────────────

function outputPath(pythonPath: string): string {
    // src/transformers/models/clip/image_processing_clip.py → src/preprocessing/clip.ts
    const match = pythonPath.match(
        /models\/([^/]+)\/(image_processing|feature_extraction)_[^/]+\.py$/,
    );
    if (!match) throw new Error(`Unexpected path: ${pythonPath}`);
    const [, modelName, kind] = match;
    const prefix = kind === "feature_extraction" ? "audio_" : "";
    return join(OUT_DIR, `${prefix}${modelName}.ts`);
}

// ── Main ────────────────────────────────────────────────────────────────────

async function main() {
    // Bootstrap: npm run sync:init <sha>
    if (process.argv.includes("--init")) {
        const sha = process.argv[process.argv.indexOf("--init") + 1];
        if (!sha) {
            console.error("Usage: npm run sync:init <commit-sha>");
            process.exit(1);
        }
        writeFileSync(LAST_SYNC_FILE, sha.trim());
        console.log(`Initialized .last-sync to ${sha.slice(0, 8)}`);
        process.exit(0);
    }

    if (!existsSync(LAST_SYNC_FILE)) {
        console.error(
            "No .last-sync found. Bootstrap with: npm run sync:init <commit-sha>\n" +
            "Get the current HEAD SHA from: https://github.com/huggingface/transformers/commits/main",
        );
        process.exit(1);
    }

    const lastSync = readFileSync(LAST_SYNC_FILE, "utf-8").trim();
    console.log(`Checking commits since ${lastSync.slice(0, 8)}...`);

    const newCommits = await getCommitsSince(lastSync);
    if (newCommits.length === 0) {
        console.log("Already up to date.");
        process.exit(0);
    }
    console.log(`${newCommits.length} new commit(s).`);

    // Collect unique changed preprocessing files across all new commits
    const changedFiles = new Set<string>();
    for (const sha of newCommits) {
        const files = await getChangedFiles(sha);
        for (const file of files) {
            if (WATCHED_PATTERNS.some((p) => p.test(file)) && !EXCLUDED_FILES.has(file)) {
                changedFiles.add(file);
            }
        }
    }

    if (changedFiles.size === 0) {
        console.log("No preprocessing files changed.");
        writeFileSync(LAST_SYNC_FILE, newCommits[0]!);
        process.exit(0);
    }

    console.log(`\nTranslating ${changedFiles.size} file(s):`);
    mkdirSync(OUT_DIR, { recursive: true });

    const outputs: { dest: string; content: string; src: string }[] = [];
    for (const src of changedFiles) {
        process.stdout.write(`  ${src} ... `);
        const pythonContent = await getFileContent(src);
        const tsContent = await translate(pythonContent, src);
        const dest = outputPath(src);
        outputs.push({ dest, content: tsContent, src });
        console.log("done");
    }

    for (const { dest, content } of outputs) {
        writeFileSync(dest, content);
    }

    writeFileSync(LAST_SYNC_FILE, newCommits[0]!);

    console.log(`\nWritten:`);
    for (const { dest } of outputs) console.log(`  ${dest}`);
    console.log(`\nUpdated .last-sync → ${newCommits[0]!.slice(0, 8)}`);
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
