import { readFile } from "node:fs/promises";
import path from "node:path";
import { type ToolExecutor, type ToolResult, type SessionContext, type JSONSchema } from "./types.js";

export const readFileTool: ToolExecutor = {
  name: "read_file",
  description: "Read the contents of a file from the project",
  riskLevel: "safe",
  schema: {
    type: "object",
    properties: {
      filePath: {
        type: "string",
        description: "Path to the file, relative to the project root",
      },
    },
    required: ["filePath"],
  } as JSONSchema,

  async execute(args: Record<string, unknown>, ctx: SessionContext): Promise<ToolResult> {
    const filePath = args.filePath as string | undefined;
    if (!filePath) {
      return { success: false, error: "Missing required argument: filePath" };
    }

    try {
      // Resolve relative to project root
      const resolvedPath = path.isAbsolute(filePath)
        ? filePath
        : path.join(ctx.projectRoot, filePath);

      const content = await readFile(resolvedPath, "utf-8");
      return { success: true, data: content };
    } catch (err) {
      const nodeErr = err as NodeJS.ErrnoException;
      if (nodeErr.code === "ENOENT") {
        return { success: false, error: `File not found: ${filePath}` };
      }
      return { success: false, error: `Failed to read file: ${(err as Error).message}` };
    }
  },
};
