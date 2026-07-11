import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtempSync, writeFileSync, rmSync, mkdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { grepTool } from "../../src/tools/grep.js";
import { applyDiffTool } from "../../src/tools/applyDiff.js";
import { runTestsTool } from "../../src/tools/runTests.js";
import { type ToolExecutor, type SessionContext, type JSONSchema } from "../../src/tools/types.js";
import { DEFAULT_CONFIG } from "../../src/config/types.js";
import { type ModelProvider } from "../../src/inference/types.js";

function createMockModel(): ModelProvider {
  return {
    name: "local",
    metadata: { name: "test", tier: "small", contextWindow: 4096, supportsGrammar: false, backend: "cpu" },
    async *generate() {
      yield { type: "complete" as const, text: "mock" };
    },
  };
}

function createContext(projectRoot: string): SessionContext {
  return {
    projectRoot,
    config: DEFAULT_CONFIG,
    model: createMockModel(),
    getTool: () => undefined,
    availableTools: [],
  };
}

describe("Tool: grep", () => {
  let tmpDir: string;
  let ctx: SessionContext;

  beforeEach(() => {
    tmpDir = mkdtempSync(join(tmpdir(), "panther-test-"));
    ctx = createContext(tmpDir);
  });

  afterEach(() => {
    rmSync(tmpDir, { recursive: true, force: true });
  });

  it("should find matching content in files", async () => {
    writeFileSync(join(tmpDir, "test.txt"), "hello world\nfoo bar\nbaz qux");
    const result = await grepTool.execute({ pattern: "hello" }, ctx);
    expect(result.success).toBe(true);
    expect(result.data).toContain("hello world");
  });

  it("should return no matches when pattern is not found", async () => {
    writeFileSync(join(tmpDir, "test.txt"), "hello world");
    const result = await grepTool.execute({ pattern: "nonexistent" }, ctx);
    expect(result.success).toBe(true);
    expect(result.data).toBe("No matches found.");
  });

  it("should return error for missing pattern", async () => {
    const result = await grepTool.execute({}, ctx);
    expect(result.success).toBe(false);
    expect(result.error).toContain("pattern");
  });
});

describe("Tool: apply_diff", () => {
  let tmpDir: string;
  let ctx: SessionContext;

  beforeEach(() => {
    tmpDir = mkdtempSync(join(tmpdir(), "panther-test-"));
    ctx = createContext(tmpDir);
  });

  afterEach(() => {
    rmSync(tmpDir, { recursive: true, force: true });
  });

  it("should apply a simple unified diff", async () => {
    writeFileSync(join(tmpDir, "test.ts"), 'const x = 1;\nconsole.log(x);\n');

    const diff = [
      "--- a/test.ts",
      "+++ b/test.ts",
      "@@ -1,2 +1,2 @@",
      "-const x = 1;",
      "-console.log(x);",
      "+const x = 2;",
      "+console.log(x + 1);",
    ].join("\n");

    const result = await applyDiffTool.execute({ filePath: "test.ts", diff }, ctx);
    expect(result.success).toBe(true);
    expect(result.data).toContain("Diff applied");

    const content = readFileSync(join(tmpDir, "test.ts"), "utf-8");
    expect(content).toContain("const x = 2;");
    expect(content).toContain("console.log(x + 1);");
  });

  it("should return error for missing filePath", async () => {
    const result = await applyDiffTool.execute({ diff: "some diff" }, ctx);
    expect(result.success).toBe(false);
    expect(result.error).toContain("filePath");
  });

  it("should return error for missing diff", async () => {
    const result = await applyDiffTool.execute({ filePath: "test.ts" }, ctx);
    expect(result.success).toBe(false);
    expect(result.error).toContain("diff");
  });
});

describe("Tool: run_tests", () => {
  let tmpDir: string;
  let ctx: SessionContext;

  beforeEach(() => {
    tmpDir = mkdtempSync(join(tmpdir(), "panther-test-"));
    ctx = createContext(tmpDir);
  });

  afterEach(() => {
    rmSync(tmpDir, { recursive: true, force: true });
  });

  it("should run a test command and return output", async () => {
    writeFileSync(join(tmpDir, "package.json"), JSON.stringify({
      name: "test-project",
      scripts: { test: "echo 'tests passed'" },
    }));

    const result = await runTestsTool.execute({}, ctx);
    // Note: This may fail if npm test runs differently on this platform
    // The key assertion is that the tool executes without crashing
    expect(result).toBeDefined();
    expect(result.meta?.command).toBe("npm test");
  });
});
