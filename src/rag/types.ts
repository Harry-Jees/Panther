/**
 * A code chunk extracted from a source file.
 */
export interface Chunk {
  id?: number;
  filePath: string;
  startLine: number;
  endLine: number;
  content: string;
}

/**
 * A chunk with its embedding vector.
 */
export interface ChunkWithEmbedding extends Chunk {
  embedding: Float32Array;
}

/**
 * A search result with similarity score.
 */
export interface SearchResult {
  chunk: Chunk;
  score: number;
}

/**
 * Configuration for the RAG system.
 */
export interface RAGConfig {
  projectRoot: string;
  indexDbPath: string;
  embeddingModel: string;
  chunkSize: number;
  chunkOverlap: number;
  topK: number;
  maxContextTokens: number;
  contextWindowFraction: number;
}

export const DEFAULT_RAG_CONFIG: RAGConfig = {
  projectRoot: "",
  indexDbPath: "",
  embeddingModel: "embedding",
  chunkSize: 512,
  chunkOverlap: 32,
  topK: 5,
  maxContextTokens: 4096,
  contextWindowFraction: 0.3,
};
