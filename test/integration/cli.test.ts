import { describe, it, expect, beforeAll } from "vitest";
import { execSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const cliPath = path.resolve(__dirname, "../../dist/cli/index.js");

// Guard: skip these tests if the CLI hasn't been built
beforeAll(() => {
  if (!existsSync(cliPath)) {
    throw new Error(
      "CLI not built — run 'npm run build' first. " +
      "Alternatively, run: npm run build && npm test",
    );
  }
});

function runCLI(args: string): { stdout: string; stderr: string; exitCode: number } {
  try {
    const stdout = execSync(`node "${cliPath}" ${args}`, {
      encoding: "utf-8",
      timeout: 10_000,
    });
    return { stdout: stdout.trim(), stderr: "", exitCode: 0 };
  } catch (err) {
    const execErr = err as { stdout?: Buffer | string; stderr?: Buffer | string; status?: number };
    return {
      stdout: (execErr.stdout ?? "").toString().trim(),
      stderr: (execErr.stderr ?? "").toString().trim(),
      exitCode: execErr.status ?? 1,
    };
  }
}

describe("CLI integration tests", () => {
  it("should display help output with all commands", () => {
    const { stdout } = runCLI("--help");
    expect(stdout).toContain("panther");
    expect(stdout).toContain("run");
    expect(stdout).toContain("model");
    expect(stdout).toContain("config");
    expect(stdout).toContain("help");
  });

  it("should display version number", () => {
    const { stdout } = runCLI("--version");
    expect(stdout).toContain("0.1.0");
  });

  it("should list available models", () => {
    const { stdout } = runCLI("model list");
    expect(stdout).toContain("small");
    expect(stdout).toContain("large");
    expect(stdout).toContain("Qwen3-Coder");
    expect(stdout).toContain("not downloaded");
  });

  it("should display configuration", () => {
    const { stdout } = runCLI("config");
    expect(stdout).toContain("Panther Configuration");
    expect(stdout).toContain("permissionMode");
    expect(stdout).toContain("modelTier");
    expect(stdout).toContain("activeProvider");
  });

  it("should exit with non-zero code for invalid model tier", () => {
    const { stderr, exitCode } = runCLI("model switch invalid");
    expect(exitCode).not.toBe(0);
    expect(stderr).toContain("specify a tier");
  });


});
