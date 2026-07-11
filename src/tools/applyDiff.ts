import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { type ToolExecutor, type ToolResult, type SessionContext, type JSONSchema } from "./types.js";

/**
 * Parse a unified diff and apply it to the target file.
 * This is a simplified diff application — for production use, consider
 * using the `patch-package` or `diff` npm modules.
 */
function parseAndApplyDiff(original: string, diff: string): string | { error: string } {
  const lines = diff.split("\n");
  const fileLines = original.split("\n");
  let result: string[] = [];
  let i = 0; // index into fileLines

  let inHunk = false;
  let hunkStartOld = 0;
  let hunkRemoveCount = 0;
  let hunkAddCount = 0;
  let removed = 0;
  let added = 0;

  for (const line of lines) {
    if (line.startsWith("--- ") || line.startsWith("+++ ") || line.startsWith("diff ")) {
      continue; // skip header lines
    }

    if (line.startsWith("@@ ")) {
      // New hunk: flush remaining lines and parse hunk header
      if (inHunk) {
        // Apply any pending removals/additions
      }

      const match = line.match(/@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@/);
      if (!match) continue;

      hunkStartOld = parseInt(match[1], 10) - 1; // 0-indexed
      hunkRemoveCount = match[2] ? parseInt(match[2], 10) : 1;
      hunkAddCount = match[4] ? parseInt(match[4], 10) : 1;
      removed = 0;
      added = 0;
      inHunk = true;

      // Copy lines up to the hunk start
      while (i < hunkStartOld) {
        result.push(fileLines[i]);
        i++;
      }
      continue;
    }

    if (!inHunk) continue;

    if (line.startsWith("-")) {
      // Remove line — skip it in the original
      removed++;
      const expectedContent = line.substring(1).trimEnd();
      if (fileLines[i].trimEnd() !== expectedContent) {
        return { error: `Context mismatch at line ${i + 1}: expected "${fileLines[i]}", got "${expectedContent}"` };
      }
      i++;
    } else if (line.startsWith("+")) {
      // Add line
      added++;
      result.push(line.substring(1));
    } else if (line.startsWith(" ")) {
      // Context line
      result.push(line.substring(1));
      i++;
    }

    // End of hunk
    if (removed >= hunkRemoveCount && added >= hunkAddCount) {
      inHunk = false;
    }
  }

  // Append any remaining lines
  while (i < fileLines.length) {
    result.push(fileLines[i]);
    i++;
  }

  return result.join("\n");
}

export const applyDiffTool: ToolExecutor = {
  name: "apply_diff",
  description: "Apply a unified diff/patch to a file",
  riskLevel: "risky",
  schema: {
    type: "object",
    properties: {
      filePath: {
        type: "string",
        description: "Path to the file to patch, relative to the project root",
      },
      diff: {
        type: "string",
        description: "Unified diff content to apply",
      },
    },
    required: ["filePath", "diff"],
  } as JSONSchema,

  async execute(args: Record<string, unknown>, ctx: SessionContext): Promise<ToolResult> {
    const filePath = args.filePath as string | undefined;
    const diff = args.diff as string | undefined;

    if (!filePath) return { success: false, error: "Missing required argument: filePath" };
    if (!diff) return { success: false, error: "Missing required argument: diff" };

    try {
      const resolvedPath = path.resolve(ctx.projectRoot, filePath);

      // Security: prevent path traversal
      if (!resolvedPath.startsWith(path.resolve(ctx.projectRoot))) {
        return { success: false, error: "Path traversal detected: file must be within the project root" };
      }

      // Read original file
      let original: string;
      try {
        original = await readFile(resolvedPath, "utf-8");
      } catch {
        // File might not exist yet (new file patch)
        original = "";
      }

      // Apply the diff
      const result = parseAndApplyDiff(original, diff);
      if (typeof result === "object" && "error" in result) {
        return { success: false, error: result.error };
      }

      await writeFile(resolvedPath, result, "utf-8");
      return { success: true, data: `Diff applied to: ${filePath}` };
    } catch (err) {
      return { success: false, error: `Failed to apply diff: ${(err as Error).message}` };
    }
  },
};
