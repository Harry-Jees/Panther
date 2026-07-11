import { type ToolExecutor, type ToolResult, type SessionContext, type JSONSchema } from "./types.js";
import { execSync } from "node:child_process";

export const grepTool: ToolExecutor = {
  name: "grep",
  description: "Search for a pattern in project files using ripgrep (rg) or fallback to grep/findstr",
  riskLevel: "safe",
  schema: {
    type: "object",
    properties: {
      pattern: {
        type: "string",
        description: "Search pattern (plain text or regex)",
      },
      path: {
        type: "string",
        description: "Directory or file to search within (default: '.')",
      },
      glob: {
        type: "string",
        description: "Optional file glob pattern (e.g., '*.ts')",
      },
    },
    required: ["pattern"],
  } as JSONSchema,

  async execute(args: Record<string, unknown>, ctx: SessionContext): Promise<ToolResult> {
    const pattern = args.pattern as string;
    const searchPath = (args.path as string) || ".";
    const glob = args.glob as string | undefined;

    if (!pattern) {
      return { success: false, error: "Missing required argument: pattern" };
    }

    try {
      const resolvedPath = searchPath.startsWith("/")
        ? searchPath
        : `${ctx.projectRoot}/${searchPath}`;

      // Try rg first, fall back to grep
      let cmd: string;
      try {
        execSync("rg --version", { stdio: "ignore" });
        cmd = `rg -n --no-heading ${glob ? `-g '${glob}'` : ""} '${pattern.replace(/'/g, "'\\''")}' '${resolvedPath}'`;
      } catch {
        // fallback to grep
        if (process.platform === "win32") {
          cmd = `findstr /s /n /c:"${pattern}" "${resolvedPath}\\*"`;
          if (glob) cmd += ` "${glob}"`;
        } else {
          cmd = `grep -rn '${pattern.replace(/'/g, "'\\''")}' '${resolvedPath}'`;
          if (glob) cmd = `grep -rn --include='${glob}' '${pattern.replace(/'/g, "'\\''")}' '${resolvedPath}'`;
        }
      }

      const output = execSync(cmd, {
        cwd: ctx.projectRoot,
        encoding: "utf-8",
        maxBuffer: 1024 * 1024, // 1MB
        stdio: ["ignore", "pipe", "pipe"],
      });

      const lines = output.trim().split("\n").filter(Boolean);
      if (lines.length === 0) {
        return { success: true, data: "No matches found." };
      }

      return {
        success: true,
        data: lines.slice(0, 100).join("\n"),
        meta: { totalMatches: lines.length, truncated: lines.length > 100 },
      };
    } catch (err) {
      // grep exits with code 1 when no matches found
      const execErr = err as { status?: number; stdout?: string; stderr?: string };
      if (execErr.status === 1) {
        return { success: true, data: "No matches found." };
      }
      return { success: false, error: `Grep failed: ${(err as Error).message}` };
    }
  },
};
