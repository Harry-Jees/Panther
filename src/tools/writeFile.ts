import { writeFile, mkdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { type ToolExecutor, type ToolResult, type SessionContext, type JSONSchema } from "./types.js";

export const writeFileTool: ToolExecutor = {
  name: "write_file",
  description: "Write content to a file, creating parent directories if needed",
  riskLevel: "risky",
  schema: {
    type: "object",
    properties: {
      filePath: {
        type: "string",
        description: "Path to the file, relative to the project root",
      },
      content: {
        type: "string",
        description: "Content to write to the file",
      },
    },
    required: ["filePath", "content"],
  } as JSONSchema,

  async execute(args: Record<string, unknown>, ctx: SessionContext): Promise<ToolResult> {
    const filePath = args.filePath as string | undefined;
    const content = args.content as string | undefined;

    if (!filePath) return { success: false, error: "Missing required argument: filePath" };
    if (content === undefined) return { success: false, error: "Missing required argument: content" };

    try {
      const resolvedPath = path.resolve(ctx.projectRoot, filePath);

      // Security: prevent path traversal outside project root
      if (!resolvedPath.startsWith(path.resolve(ctx.projectRoot))) {
        return { success: false, error: "Path traversal detected: file must be within the project root" };
      }

      // Create parent directories if they don't exist
      const parentDir = path.dirname(resolvedPath);
      if (!existsSync(parentDir)) {
        await mkdir(parentDir, { recursive: true });
      }

      await writeFile(resolvedPath, content, "utf-8");
      return { success: true, data: `File written: ${filePath}` };
    } catch (err) {
      return { success: false, error: `Failed to write file: ${(err as Error).message}` };
    }
  },
};
