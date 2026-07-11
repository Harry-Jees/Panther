import { readFile, writeFile, appendFile, mkdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";

const DEFAULT_MEMORY_FILE = "memory.md";
const DEFAULT_TOKEN_BUDGET = 2048;

/**
 * Rough token count estimation (4 chars ≈ 1 token).
 */
function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

/**
 * Memory system for per-project agent context persistence.
 */
export class MemorySystem {
  private memoryDir: string;
  private tokenBudget: number;

  constructor(projectRoot: string, tokenBudget?: number) {
    this.memoryDir = path.join(projectRoot, ".panther");
    this.tokenBudget = tokenBudget ?? DEFAULT_TOKEN_BUDGET;
  }

  private get memoryPath(): string {
    return path.join(this.memoryDir, DEFAULT_MEMORY_FILE);
  }

  private async ensureDir(): Promise<void> {
    if (!existsSync(this.memoryDir)) {
      await mkdir(this.memoryDir, { recursive: true });
    }
  }

  /**
   * Read the memory file, truncated to fit within the token budget.
   * Returns the most recent entries that fit.
   */
  async readMemory(): Promise<string> {
    try {
      const content = await readFile(this.memoryPath, "utf-8");
      return this.truncateToBudget(content);
    } catch {
      return "";
    }
  }

  /**
   * Append a dated session log entry to memory.md.
   */
  async appendSessionLog(entry: {
    date?: string;
    summary: string;
    filesTouched: string[];
    decisions: string[];
    followUps: string[];
  }): Promise<void> {
    await this.ensureDir();

    const date = entry.date ?? new Date().toISOString().split("T")[0];
    const lines: string[] = [
      "",
      `### ${date} — ${entry.summary}`,
      "",
      "- **Files touched:** " + entry.filesTouched.join(", "),
      ...entry.decisions.map((d) => `- **Decision:** ${d}`),
      ...entry.followUps.map((f) => `- **Follow-up:** ${f}`),
      "",
    ];

    // Append to the end of the file
    await appendFile(this.memoryPath, lines.join("\n"), "utf-8");
  }

  /**
   * Write initial memory structure (called on first use).
   */
  async initialize(): Promise<void> {
    await this.ensureDir();
    if (!existsSync(this.memoryPath)) {
      const header = [
        "# Project memory",
        "",
        "## Architecture notes",
        "(agent-maintained summary of the codebase's structure, updated as it learns)",
        "",
        "## Session log",
        "",
      ].join("\n");
      await writeFile(this.memoryPath, header, "utf-8");
    }
  }

  /**
   * Truncate content to fit within the token budget.
   * Keeps the header and the most recent session log entries.
   */
  private truncateToBudget(content: string): string {
    if (estimateTokens(content) <= this.tokenBudget) {
      return content;
    }

    const sections = content.split("\n## ");
    const header = sections[0]; // Keep the header always

    // Keep sections from the end (most recent) until we hit the budget
    const keptSections: string[] = [];
    let tokenCount = estimateTokens(header);

    for (let i = sections.length - 1; i >= 1; i--) {
      const section = "## " + sections[i];
      const sectionTokens = estimateTokens(section);
      if (tokenCount + sectionTokens <= this.tokenBudget) {
        keptSections.unshift(section);
        tokenCount += sectionTokens;
      } else {
        break;
      }
    }

    return [header, ...keptSections].join("\n");
  }
}
