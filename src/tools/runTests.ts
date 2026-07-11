import { execSync } from "node:child_process";
import { type ToolExecutor, type ToolResult, type SessionContext, type JSONSchema } from "./types.js";

export const runTestsTool: ToolExecutor = {
  name: "run_tests",
  description: "Run tests in the project using npm test or a specified command",
  riskLevel: "safe", // Running tests is read-only for source code
  schema: {
    type: "object",
    properties: {
      command: {
        type: "string",
        description: "Test command to run (default: 'npm test')",
      },
      path: {
        type: "string",
        description: "Subdirectory to run tests in (default: project root)",
      },
    },
  } as JSONSchema,

  async execute(args: Record<string, unknown>, ctx: SessionContext): Promise<ToolResult> {
    const testCmd = (args.command as string) || "npm test";
    const testPath = args.path ? `${ctx.projectRoot}/${args.path}` : ctx.projectRoot;

    try {
      const output = execSync(testCmd, {
        cwd: testPath,
        encoding: "utf-8",
        maxBuffer: 5 * 1024 * 1024, // 5MB
        stdio: ["ignore", "pipe", "pipe"],
        timeout: 120_000, // 2 minute timeout
      });

      return {
        success: true,
        data: output.trim(),
        meta: { command: testCmd, exitCode: 0 },
      };
    } catch (err) {
      const execErr = err as { status?: number; stdout?: string; stderr?: string; message: string };
      return {
        success: false,
        data: execErr.stdout?.trim() || execErr.stderr?.trim() || execErr.message,
        meta: { command: testCmd, exitCode: execErr.status ?? -1 },
      };
    }
  },
};
