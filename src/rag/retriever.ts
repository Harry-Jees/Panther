import { type VectorStore } from "./store.js";
import { type SearchResult, DEFAULT_RAG_CONFIG } from "./types.js";

/**
 * Generate a simple embedding for a query string.
 * In production, this would use node-llama-cpp's embedding API.
 */
function computeQueryEmbedding(query: string, dimensions: number = 128): number[] {
  const embedding = new Array(dimensions).fill(0);
  let hash = 0;
  for (let i = 0; i < query.length; i++) {
    hash = ((hash << 5) - hash) + query.charCodeAt(i);
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
 * Estimate token count for a context string (rough: 4 chars ≈ 1 token).
 */
function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

/**
 * Retrieve relevant context chunks for a task.
 */
export async function retrieveContext(
  task: string,
  store: VectorStore,
  modelContextWindow: number,
): Promise<string> {
  const maxTokens = Math.floor(modelContextWindow * DEFAULT_RAG_CONFIG.contextWindowFraction);
  const k = DEFAULT_RAG_CONFIG.topK;

  // Embed the query
  const queryEmbedding = computeQueryEmbedding(task);

  // Search the vector store
  const results = await store.search(queryEmbedding, k);

  if (results.length === 0) {
    return "";
  }

  // Format as context block, respecting token budget
  const contextParts: string[] = ["<context>"];
  let totalTokens = estimateTokens(contextParts[0]);

  for (const result of results) {
    const part = formatResult(result);
    const tokens = estimateTokens(part);

    if (totalTokens + tokens <= maxTokens) {
      contextParts.push(part);
      totalTokens += tokens;
    } else {
      break;
    }
  }

  contextParts.push("</context>");
  return contextParts.join("\n\n");
}

/**
 * Format a search result as a context snippet.
 */
function formatResult(result: SearchResult): string {
  return [
    `File: ${result.chunk.filePath} (lines ${result.chunk.startLine}-${result.chunk.endLine})`,
    "```",
    result.chunk.content,
    "```",
  ].join("\n");
}
