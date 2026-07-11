import { readdir } from "node:fs/promises";
import path from "node:path";
import { type ToolExecutor, type ToolResult, type SessionContext, type JSONSchema } from "./types.js";

export const listDirTool: ToolExecutor = {
  name: "list_dir",
  description: "List files and directories in a given path",
  riskLevel: "safe",
  schema: {
    type: "object",
    properties: {
      path: {
        type: "string",
        description: "Directory path, relative to the project root (default: '.')",
      },
    },
  } as JSONSchema,

  async execute(args: Record<string, unknown>, ctx: SessionContext): Promise<ToolResult> {
    const dirPath = (args.path as string) || ".";
    try {
      const resolvedPath = path.isAbsolute(dirPath)
        ? dirPath
        : path.join(ctx.projectRoot, dirPath);

      const entries = await readdir(resolvedPath, { withFileTypes: true });
      const files = entries
        .filter((e) => e.isFile())
        .map((e) => e.name);
      const dirs = entries
        .filter((e) => e.isDirectory())
        .map((e) => e.name + "/");

      const listing = [...dirs, ...files].join("\n");
      return { success: true, data: listing, meta: { fileCount: files.length, dirCount: dirs.length } };
    } catch (err) {
      return { success: false, error: `Failed to list directory: ${(err as Error).message}` };
    }
  },
};
