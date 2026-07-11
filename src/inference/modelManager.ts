import { createWriteStream, existsSync } from "node:fs";
import { readFile, writeFile, mkdir, unlink } from "node:fs/promises";
import path from "node:path";
import { createHash } from "node:crypto";
import { getPantherDir } from "../config/index.js";
import { type ModelTier } from "../config/types.js";

/**
 * Manifest entry for a model tier.
 */
export interface ModelManifestEntry {
  name: string;
  tier: "small" | "large";
  huggingFaceRepo: string;
  filename: string;
  contextWindow: number;
  expectedHash: string;
}

/**
 * The built-in model manifest mapping tiers to downloadable GGUF files.
 * References Qwen coding-model family per the tech stack decision.
 */
const MODEL_MANIFEST: Record<string, ModelManifestEntry> = {
  small: {
    name: "Qwen3-Coder-4B-Instruct-Q4_K_M",
    tier: "small",
    huggingFaceRepo: "Qwen/Qwen3-Coder-4B-Instruct-GGUF",
    filename: "Qwen3-Coder-4B-Instruct-Q4_K_M.gguf",
    contextWindow: 32768,
    expectedHash: "", // populated after download verification
  },
  large: {
    name: "Qwen3-Coder-80B-A3B-Q4_K_M",
    tier: "large",
    huggingFaceRepo: "Qwen/Qwen3-Coder-80B-A3B-GGUF",
    filename: "Qwen3-Coder-80B-A3B-Q4_K_M.gguf",
    contextWindow: 32768,
    expectedHash: "", // populated after download verification
  },
};

/**
 * Get the path to the models directory.
 */
function getModelsDir(): string {
  return path.join(getPantherDir(), "models");
}

/**
 * Get the path to a model file for a given tier.
 */
function getModelPath(tier: ModelTier): string {
  const entry = MODEL_MANIFEST[tier];
  return path.join(getModelsDir(), entry.filename);
}

/**
 * Get the path to the manifest.json file.
 */
function getManifestPath(): string {
  return path.join(getModelsDir(), "manifest.json");
}

/**
 * Get the download URL for a model tier from Hugging Face.
 */
function getDownloadUrl(entry: ModelManifestEntry): string {
  return `https://huggingface.co/${entry.huggingFaceRepo}/resolve/main/${entry.filename}`;
}

/**
 * Download a model file with progress bar and resume support.
 */
async function downloadModel(
  entry: ModelManifestEntry,
  onProgress?: (downloaded: number, total: number) => void,
): Promise<string> {
  const modelDir = getModelsDir();
  if (!existsSync(modelDir)) {
    await mkdir(modelDir, { recursive: true });
  }

  const modelPath = path.join(modelDir, entry.filename);
  const tempPath = modelPath + ".download";

  // Check if a partial download exists for resume
  let downloadedBytes = 0;
  if (existsSync(tempPath)) {
    const stat = await import("node:fs/promises").then((fs) => fs.stat(tempPath));
    downloadedBytes = stat.size;
  }

  const url = getDownloadUrl(entry);
  const response = await fetch(url, {
    headers: downloadedBytes > 0 ? { Range: `bytes=${downloadedBytes}-` } : undefined,
  });

  if (!response.ok && response.status !== 206) {
    throw new Error(`Failed to download model: ${response.status} ${response.statusText}`);
  }

  const total = parseInt(response.headers.get("content-range")?.split("/")[1] ?? "0", 10)
    || parseInt(response.headers.get("content-length") ?? "0", 10)
    || 0;

  const writeStream = createWriteStream(tempPath, {
    flags: downloadedBytes > 0 ? "a" : "w",
  });

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("No response body stream available");
  }

  let downloaded = downloadedBytes;
  const decoder = new TextDecoder();

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      downloaded += value.length;
      writeStream.write(Buffer.from(value));

      if (onProgress && total > 0) {
        onProgress(downloaded, total);
      } else if (onProgress) {
        onProgress(downloaded, downloaded);
      }
    }
  } finally {
    writeStream.end();
    reader.releaseLock();
  }

  // Verify hash if we have one
  if (entry.expectedHash) {
    const fileBuffer = await readFile(tempPath);
    const actualHash = createHash("sha256").update(fileBuffer).digest("hex");
    if (actualHash !== entry.expectedHash) {
      await unlink(tempPath).catch(() => {});
      throw new Error(
        `Hash mismatch for ${entry.filename}: expected ${entry.expectedHash}, got ${actualHash}`,
      );
    }
  }

  // Rename temp file to final path
  await import("node:fs/promises").then((fs) => fs.rename(tempPath, modelPath));

  return modelPath;
}

/**
 * Ensure a model for the given tier is downloaded and cached.
 * Returns the path to the model file.
 */
export async function ensureModel(
  tier: ModelTier,
  onProgress?: (downloaded: number, total: number) => void,
): Promise<string> {
  const modelPath = getModelPath(tier);

  if (existsSync(modelPath)) {
    return modelPath;
  }

  const entry = MODEL_MANIFEST[tier];
  if (!entry) {
    throw new Error(`Unknown model tier: ${tier}`);
  }

  console.log(`Downloading ${entry.name} (${tier} tier)...`);
  const downloadedPath = await downloadModel(entry, onProgress);

  // Write manifest
  const manifest = {
    name: entry.name,
    tier: entry.tier,
    contextWindow: entry.contextWindow,
    path: downloadedPath,
    downloadedAt: new Date().toISOString(),
  };
  const manifestDir = path.dirname(getManifestPath());
  if (!existsSync(manifestDir)) {
    await mkdir(manifestDir, { recursive: true });
  }
  await writeFile(getManifestPath(), JSON.stringify(manifest, null, 2));

  return downloadedPath;
}

/**
 * Get the cached model path for a tier, or null if not downloaded.
 */
export async function getCachedModel(tier: ModelTier): Promise<string | null> {
  const modelPath = getModelPath(tier);
  if (existsSync(modelPath)) {
    return modelPath;
  }
  return null;
}

/**
 * Switch the active model tier, downloading if necessary and cleaning up the old model.
 */
export async function switchModel(
  tier: ModelTier,
  onProgress?: (downloaded: number, total: number) => void,
): Promise<string> {
  // Get the new model
  const modelPath = await ensureModel(tier, onProgress);

  // Remove old model files for the other tier
  const otherTier: ModelTier = tier === "small" ? "large" : "small";
  const oldPath = getModelPath(otherTier);
  if (existsSync(oldPath)) {
    await unlink(oldPath).catch(() => {});
  }

  return modelPath;
}

/**
 * List all model tiers with their download status.
 */
export async function listModels(): Promise<Array<{ tier: string; name: string; downloaded: boolean }>> {
  const results: Array<{ tier: string; name: string; downloaded: boolean }> = [];

  for (const [tier, entry] of Object.entries(MODEL_MANIFEST)) {
    const modelPath = getModelPath(tier as ModelTier);
    results.push({
      tier,
      name: entry.name,
      downloaded: existsSync(modelPath),
    });
  }

  return results;
}

/**
 * Get the built-in model manifest (for bundling).
 */
export function getModelManifest(): Record<string, ModelManifestEntry> {
  return { ...MODEL_MANIFEST };
}
