import { readFile, writeFile, mkdir, stat, readdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { chunkFile } from "./chunker.js";
import { type VectorStore } from "./store.js";
import { type ChunkWithEmbedding } from "./types.js";

interface FileIndex {
  filePath: string;
  mtime: number;
}

/**
 * Simple file-based mtime tracking for incremental indexing.
 */
class FileTracker {
  private index: Map<string, number> = new Map();
  private indexPath: string;

  constructor(projectRoot: string) {
    this.indexPath = path.join(projectRoot, ".panther", "file-index.json");
  }

  async load(): Promise<void> {
    try {
      const raw = await readFile(this.indexPath, "utf-8");
      const data: FileIndex[] = JSON.parse(raw);
      for (const entry of data) {
        this.index.set(entry.filePath, entry.mtime);
      }
    } catch {
      // No existing index
    }
  }

  async save(): Promise<void> {
    const data: FileIndex[] = [];
    for (const [filePath, mtime] of this.index.entries()) {
      data.push({ filePath, mtime });
    }
    const dir = path.dirname(this.indexPath);
    if (!existsSync(dir)) {
      await mkdir(dir, { recursive: true });
    }
    await writeFile(this.indexPath, JSON.stringify(data, null, 2));
  }

  isChanged(filePath: string, mtime: number): boolean {
    const stored = this.index.get(filePath);
    return stored === undefined || stored < mtime;
  }

  markIndexed(filePath: string, mtime: number): void {
    this.index.set(filePath, mtime);
  }
}

/**
 * Generate a mock embedding vector for demonstration/testing.
 * In production, this would use node-llama-cpp's embedding API.
 */
function generateMockEmbedding(text: string, dimensions: number = 128): Float32Array {
  const embedding = new Float32Array(dimensions);
  // Simple hash-based pseudo-embedding for testing
  let hash = 0;
  for (let i = 0; i < text.length; i++) {
    hash = ((hash << 5) - hash) + text.charCodeAt(i);
    hash |= 0;
    embedding[i % dimensions] += hash % 1000 / 1000;
  }
  // Normalize
  let norm = 0;
  for (let i = 0; i < dimensions; i++) norm += embedding[i] * embedding[i];
  norm = Math.sqrt(norm);
  if (norm > 0) {
    for (let i = 0; i < dimensions; i++) embedding[i] /= norm;
  }
  return embedding;
}

/**
 * Incremental project indexer.
 */
export class ProjectIndexer {
  private store: VectorStore;
  private fileTracker: FileTracker;
  private projectRoot: string;
  private useRealEmbeddings: boolean;

  constructor(
    store: VectorStore,
    projectRoot: string,
    useRealEmbeddings: boolean = false,
  ) {
    this.store = store;
    this.projectRoot = projectRoot;
    this.fileTracker = new FileTracker(projectRoot);
    this.useRealEmbeddings = useRealEmbeddings;
  }

  /**
   * Recursively find source files in the project directory.
   * Compatible with Node.js 20+ (avoids glob from node:fs/promises, which requires Node 22+).
   */
  private async *findFiles(
    dirPath: string,
    relativePath: string,
    extensions: Set<string>,
  ): AsyncGenerator<string> {
    let entries: string[] = [];
    try {
      entries = await readdir(dirPath, { withFileTypes: false });
    } catch {
      return;
    }

    for (const entry of entries) {
      const fullPath = path.join(dirPath, entry);
      const relPath = relativePath ? `${relativePath}/${entry}` : entry;

      try {
        const stats = await stat(fullPath);
        if (stats.isDirectory()) {
          const dirName = entry.toLowerCase();
          if (dirName === "node_modules" || dirName === ".panther" || dirName === "dist" || dirName === ".git" || dirName === "__pycache__" || dirName === ".next" || dirName === "build") {
            continue;
          }
          yield* this.findFiles(fullPath, relPath, extensions);
        } else if (stats.isFile() && extensions.has(path.extname(entry).toLowerCase())) {
          yield relPath;
        }
      } catch {
        // Skip files/dirs we can't read
      }
    }
  }

  /**
   * Index the project, processing only files that have changed since the last index.
   */
  async indexProject(): Promise<{ filesIndexed: number; chunksCreated: number }> {
    await this.fileTracker.load();
    await this.store.initialize();

    let filesIndexed = 0;
    let chunksCreated = 0;

    // Find all source files
    const extensions = new Set([".ts", ".tsx", ".js", ".jsx", ".py", ".go", ".rs", ".java", ".cs", ".md"]);
    const files: string[] = [];

    for await (const file of this.findFiles(this.projectRoot, "", extensions)) {
      files.push(file);
    }

    for (const file of files) {
      const fullPath = path.join(this.projectRoot, file);
      try {
        const stats = await stat(fullPath);

        if (!this.fileTracker.isChanged(file, stats.mtimeMs)) {
          continue; // Skip unchanged files
        }

        // Chunk the file
        const chunks = await chunkFile(fullPath);
        chunksCreated += chunks.length;

        // Delete old chunks for this file
        await this.store.deleteChunksForFile(file);

        // Generate embeddings and store chunks
        for (const chunk of chunks) {
          const embedding = this.useRealEmbeddings
            ? new Float32Array(128) // placeholder for real embedding
            : generateMockEmbedding(chunk.content);

          await this.store.upsertChunk({ ...chunk, embedding });
        }

        this.fileTracker.markIndexed(file, stats.mtimeMs);
        filesIndexed++;
      } catch (err) {
        console.warn(`Warning: Could not index ${file}: ${(err as Error).message}`);
      }
    }

    await this.fileTracker.save();
    return { filesIndexed, chunksCreated };
  }
}
