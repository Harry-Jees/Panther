import { type Chunk, type ChunkWithEmbedding, type SearchResult } from "./types.js";

/**
 * Abstract vector store interface.
 * Implementations can use SQLite + sqlite-vec or a lightweight in-memory store.
 */
export interface VectorStore {
  initialize(): Promise<void>;
  upsertChunk(chunk: ChunkWithEmbedding): Promise<void>;
  deleteChunksForFile(filePath: string): Promise<void>;
  search(embedding: number[], k: number): Promise<SearchResult[]>;
  close(): Promise<void>;
}

/**
 * Simple in-memory vector store using cosine similarity.
 * Useful for testing and environments without sqlite-vec.
 * In production, swap for SQLite + sqlite-vec.
 */
export class InMemoryVectorStore implements VectorStore {
  private chunks: ChunkWithEmbedding[] = [];

  async initialize(): Promise<void> {
    this.chunks = [];
  }

  async upsertChunk(chunk: ChunkWithEmbedding): Promise<void> {
    const existingIndex = this.chunks.findIndex(
      (c) => c.filePath === chunk.filePath && c.startLine === chunk.startLine,
    );
    if (existingIndex >= 0) {
      this.chunks[existingIndex] = chunk;
    } else {
      this.chunks.push(chunk);
    }
  }

  async deleteChunksForFile(filePath: string): Promise<void> {
    this.chunks = this.chunks.filter((c) => c.filePath !== filePath);
  }

  async search(embedding: number[], k: number): Promise<SearchResult[]> {
    const queryVector = new Float32Array(embedding);

    const scored = this.chunks.map((chunk) => ({
      chunk,
      score: cosineSimilarity(queryVector, chunk.embedding),
    }));

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, k).map((s) => ({
      chunk: {
        id: s.chunk.id,
        filePath: s.chunk.filePath,
        startLine: s.chunk.startLine,
        endLine: s.chunk.endLine,
        content: s.chunk.content,
      },
      score: s.score,
    }));
  }

  async close(): Promise<void> {
    this.chunks = [];
  }
}

/**
 * Compute cosine similarity between two vectors.
 */
function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}
