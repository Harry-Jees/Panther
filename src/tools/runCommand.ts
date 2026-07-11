import { execSync } from "node:child_process";
import { type ToolExecutor, type ToolResult, type SessionContext, type JSONSchema } from "./types.js";

export const runCommandTool: ToolExecutor = {
  name: "run_command",
  description: "Execute a shell command in the project root directory",
  riskLevel: "risky",
  schema: {
    type: "object",
    properties: {
      command: {
        type: "string",
        description: "Shell command to execute",
      },
      timeout: {
        type: "number",
        description: "Timeout in milliseconds (default: 30000)",
      },
    },
    required: ["command"],
  } as JSONSchema,

  async execute(args: Record<string, unknown>, ctx: SessionContext): Promise<ToolResult> {
    const command = args.command as string | undefined;
    const timeout = (args.timeout as number) || 30_000;

    if (!command) {
      return { success: false, error: "Missing required argument: command" };
    }

    // Security: reject commands that try to escape the project root
    const dangerousPatterns = ["..", "~", "$HOME", "&& cd", "|| cd"];
    for (const pattern of dangerousPatterns) {
      if (command.includes(pattern)) {
        return {
          success: false,
          error: `Command rejected: contains potentially dangerous pattern "${pattern}"`,
        };
      }
    }

    try {
      const output = execSync(command, {
        cwd: ctx.projectRoot,
        encoding: "utf-8",
        maxBuffer: 5 * 1024 * 1024, // 5MB
        stdio: ["ignore", "pipe", "pipe"],
        timeout,
        env: {
          PATH: process.env.PATH || "",
          NODE_ENV: process.env.NODE_ENV || "development",
          HOME: process.env.HOME || "",
        },
      });

      return {
        success: true,
        data: output.trim() || "(command completed with no output)",
        meta: { exitCode: 0 },
      };
    } catch (err) {
      const execErr = err as { status?: number; stdout?: string; stderr?: string; message: string };
      return {
        success: false,
        data: execErr.stdout?.trim() || execErr.stderr?.trim() || execErr.message,
        meta: { command, exitCode: execErr.status ?? -1 },
      };
    }
  },
};
