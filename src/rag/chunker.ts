import { readFile } from "node:fs/promises";
import { type Chunk } from "./types.js";

export interface ChunkerOptions {
  maxChunkSize: number;
  overlap: number;
}

const DEFAULT_CHUNKER_OPTIONS: ChunkerOptions = {
  maxChunkSize: 512,
  overlap: 32,
};

/**
 * Regex patterns for detecting function/class/method boundaries in common languages.
 * In production, these would be replaced with tree-sitter grammars.
 */
const BOUNDARY_PATTERNS = [
  // TypeScript/JavaScript functions
  /^(export\s+)?(async\s+)?function\s+\w+/m,
  /^(export\s+)?(async\s+)?\w+\s*\([^)]*\)\s*{/m,
  /^(export\s+)?class\s+\w+/m,
  /^(export\s+)?interface\s+\w+/m,
  /^(export\s+)?type\s+\w+/m,
  /^(export\s+)?enum\s+\w+/m,
  /^(export\s+)?const\s+\w+\s*[:=]/m,
  /^\s*(public|private|protected)\s+(static\s+)?(async\s+)?\w+\s*\(/m,
  /^\s*(public|private|protected)\s+(static\s+)?(readonly\s+)?\w+/m,

  // Python
  /^def\s+\w+/m,
  /^class\s+\w+/m,
  /^@\w+/m,

  // Go
  /^func\s+\w+/m,
  /^type\s+\w+\s+struct/m,
  /^type\s+\w+\s+interface/m,

  // Rust
  /^fn\s+\w+/m,
  /^struct\s+\w+/m,
  /^impl\s+\w+/m,
  /^trait\s+\w+/m,

  // Java/C#
  /^\s*(public|private|protected)\s+(static\s+)?\w+\s+\w+\s*\(/m,
  /^\s*(public|private|protected)\s+class\s+\w+/m,
  /^\s*(public|private|protected)\s+interface\s+\w+/m,
];

/**
 * Detect if a line is a structural boundary (function/class/interface start).
 */
function isBoundary(line: string): boolean {
  return BOUNDARY_PATTERNS.some((pattern) => pattern.test(line));
}

/**
 * Chunk a file into function/class-boundary-aware segments.
 */
export async function chunkFile(
  filePath: string,
  options: ChunkerOptions = DEFAULT_CHUNKER_OPTIONS,
): Promise<Chunk[]> {
  const content = await readFile(filePath, "utf-8");
  return chunkText(content, filePath, options);
}

/**
 * Chunk text content into segments.
 */
export function chunkText(
  content: string,
  filePath: string,
  options: ChunkerOptions = DEFAULT_CHUNKER_OPTIONS,
): Chunk[] {
  const lines = content.split("\n");
  const chunks: Chunk[] = [];
  let chunkStart = 0;
  let lastBoundary = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const chunkSize = i - chunkStart;

    // Check if this line is a structural boundary
    if (isBoundary(line)) {
      lastBoundary = i;
    }

    // If we've exceeded the max chunk size, split at the last boundary
    if (chunkSize >= options.maxChunkSize && lastBoundary > chunkStart) {
      const endLine = Math.min(lastBoundary + 1, lines.length);
      chunks.push({
        filePath,
        startLine: chunkStart,
        endLine,
        content: lines.slice(chunkStart, endLine).join("\n"),
      });
      chunkStart = Math.max(0, endLine - options.overlap);
    }
  }

  // Add the last chunk
  if (chunkStart < lines.length) {
    chunks.push({
      filePath,
      startLine: chunkStart,
      endLine: lines.length,
      content: lines.slice(chunkStart).join("\n"),
    });
  }

  return chunks;
}
