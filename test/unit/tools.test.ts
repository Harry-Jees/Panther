import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtempSync, writeFileSync, rmSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { readFileTool } from "../../src/tools/readFile.js";
import { listDirTool } from "../../src/tools/listDir.js";
import { writeFileTool } from "../../src/tools/writeFile.js";
import { runCommandTool } from "../../src/tools/runCommand.js";
import { type ToolExecutor, type SessionContext } from "../../src/tools/types.js";
import { type ModelProvider, type ModelOutput } from "../../src/inference/types.js";
import { type PantherConfig, DEFAULT_CONFIG } from "../../src/config/types.js";

function createMockModel(): ModelProvider {
  return {
    name: "local",
    metadata: { name: "test", tier: "small", contextWindow: 4096, supportsGrammar: false, backend: "cpu" },
    async *generate() {
      yield { type: "complete" as const, text: "mock response" };
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

describe("Tool: read_file", () => {
  let tmpDir: string;
  let ctx: SessionContext;

  beforeEach(() => {
    tmpDir = mkdtempSync(join(tmpdir(), "panther-test-"));
    ctx = createContext(tmpDir);
  });

  afterEach(() => {
    rmSync(tmpDir, { recursive: true, force: true });
  });

  it("should read an existing file", async () => {
    writeFileSync(join(tmpDir, "test.txt"), "hello world");
    const result = await readFileTool.execute({ filePath: "test.txt" }, ctx);
    expect(result.success).toBe(true);
    expect(result.data).toBe("hello world");
  });

  it("should return error for missing file", async () => {
    const result = await readFileTool.execute({ filePath: "nonexistent.txt" }, ctx);
    expect(result.success).toBe(false);
    expect(result.error).toContain("not found");
  });

  it("should return error for missing argument", async () => {
    const result = await readFileTool.execute({}, ctx);
    expect(result.success).toBe(false);
    expect(result.error).toContain("filePath");
  });
});

describe("Tool: list_dir", () => {
  let tmpDir: string;
  let ctx: SessionContext;

  beforeEach(() => {
    tmpDir = mkdtempSync(join(tmpdir(), "panther-test-"));
    ctx = createContext(tmpDir);
  });

  afterEach(() => {
    rmSync(tmpDir, { recursive: true, force: true });
  });

  it("should list files and directories", async () => {
    writeFileSync(join(tmpDir, "file1.ts"), "");
    writeFileSync(join(tmpDir, "file2.ts"), "");
    mkdirSync(join(tmpDir, "subdir"));

    const result = await listDirTool.execute({}, ctx);
    expect(result.success).toBe(true);
    expect(result.data).toContain("subdir/");
    expect(result.data).toContain("file1.ts");
    expect(result.data).toContain("file2.ts");
  });
});

describe("Tool: write_file", () => {
  let tmpDir: string;
  let ctx: SessionContext;

  beforeEach(() => {
    tmpDir = mkdtempSync(join(tmpdir(), "panther-test-"));
    ctx = createContext(tmpDir);
  });

  afterEach(() => {
    rmSync(tmpDir, { recursive: true, force: true });
  });

  it("should write a file", async () => {
    const result = await writeFileTool.execute({ filePath: "new.txt", content: "test content" }, ctx);
    expect(result.success).toBe(true);

    const { readFileSync } = await import("node:fs");
    const content = readFileSync(join(tmpDir, "new.txt"), "utf-8");
    expect(content).toBe("test content");
  });

  it("should reject path traversal", async () => {
    const result = await writeFileTool.execute({ filePath: "../escape.txt", content: "bad" }, ctx);
    expect(result.success).toBe(false);
    expect(result.error).toContain("Path traversal");
  });

  it("should create parent directories", async () => {
    const result = await writeFileTool.execute({ filePath: "nested/deep/file.txt", content: "deep" }, ctx);
    expect(result.success).toBe(true);

    const { existsSync } = await import("node:fs");
    expect(existsSync(join(tmpDir, "nested", "deep", "file.txt"))).toBe(true);
  });
});

describe("Tool: run_command", () => {
  let tmpDir: string;
  let ctx: SessionContext;

  beforeEach(() => {
    tmpDir = mkdtempSync(join(tmpdir(), "panther-test-"));
    ctx = createContext(tmpDir);
  });

  afterEach(() => {
    rmSync(tmpDir, { recursive: true, force: true });
  });

  it("should execute a simple command", async () => {
    // Use platform-appropriate command
    const cmd = process.platform === "win32" ? "echo hello" : "echo hello";
    const result = await runCommandTool.execute({ command: cmd }, ctx);
    expect(result.success).toBe(true);
    expect(result.data).toContain("hello");
  });

  it("should reject dangerous patterns", async () => {
    const result = await runCommandTool.execute({ command: "cd .. && cat /etc/passwd" }, ctx);
    expect(result.success).toBe(false);
    expect(result.error).toContain("dangerous");
  });

  it("should reject path traversal attempts", async () => {
    const result = await runCommandTool.execute({ command: "ls ~/../" }, ctx);
    expect(result.success).toBe(false);
    expect(result.error).toContain("dangerous");
  });
});
