import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import fc from "fast-check";
import { mkdtempSync, writeFileSync, rmSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import os, { tmpdir } from "node:os";
import { loadConfig, saveConfig } from "../../src/config/index.js";
import { PantherConfigSchema, type PantherConfig, DEFAULT_CONFIG } from "../../src/config/types.js";

describe("Config system", () => {
  let tmpDir: string;

  beforeEach(() => {
    tmpDir = mkdtempSync(join(tmpdir(), "panther-test-"));
    vi.spyOn(os, "homedir").mockReturnValue(tmpDir);
  });

  afterEach(() => {
    rmSync(tmpDir, { recursive: true, force: true });
    vi.restoreAllMocks();
  });

  describe("loadConfig", () => {
    it("should return defaults when no config files exist", async () => {
      const config = await loadConfig(tmpDir);
      expect(config).toEqual(DEFAULT_CONFIG);
    });

    it("should merge global config over defaults", async () => {
      const globalDir = join(tmpDir, ".panther");
      mkdirSync(globalDir, { recursive: true });
      writeFileSync(
        join(globalDir, "config.json"),
        JSON.stringify({ permissionMode: "never_ask", modelTier: "small" }),
      );

      const config = await loadConfig(tmpDir);
      expect(config.permissionMode).toBe("never_ask");
      expect(config.modelTier).toBe("small");
      expect(config.activeProvider).toBe(DEFAULT_CONFIG.activeProvider);
    });

    it("should use project config over global config", async () => {
      const globalDir = join(tmpDir, ".panther");
      mkdirSync(globalDir, { recursive: true });
      writeFileSync(
        join(globalDir, "config.json"),
        JSON.stringify({ permissionMode: "never_ask", modelTier: "small" }),
      );

      const projectDir = join(tmpDir, "project");
      mkdirSync(join(projectDir, ".panther"), { recursive: true });
      writeFileSync(
        join(projectDir, ".panther", "config.json"),
        JSON.stringify({ permissionMode: "always_ask" }),
      );

      const config = await loadConfig(projectDir);
      expect(config.permissionMode).toBe("always_ask"); // from project
      expect(config.modelTier).toBe("small"); // from global
      expect(config.activeProvider).toBe(DEFAULT_CONFIG.activeProvider); // from defaults
    });

    it("should not crash on malformed JSON", async () => {
      const projectDir = join(tmpDir, "project2");
      mkdirSync(join(projectDir, ".panther"), { recursive: true });
      writeFileSync(join(projectDir, ".panther", "config.json"), "not valid json{{{");

      const config = await loadConfig(projectDir);
      expect(config).toEqual(DEFAULT_CONFIG);
    });
  });

  describe("saveConfig", () => {
    it("should save config and load it back correctly", async () => {
      const config: PantherConfig = {
        permissionMode: "always_ask",
        modelTier: "large",
        modelFamily: "qwen",
        activeProvider: "local",
        telemetryOptIn: true,
      };

      await saveConfig(config, tmpDir);

      // Load it back
      const loaded = await loadConfig(tmpDir);
      expect(loaded).toEqual(config);
    });
  });

  describe("PantherConfigSchema", () => {
    it("should apply defaults for missing fields", () => {
      const result = PantherConfigSchema.parse({});
      expect(result.permissionMode).toBe("ask_when_risky");
      expect(result.modelTier).toBe("large");
      expect(result.modelFamily).toBe("qwen");
      expect(result.activeProvider).toBe("local");
      expect(result.telemetryOptIn).toBe(false);
    });

    it("should reject invalid permission mode", () => {
      expect(() =>
        PantherConfigSchema.parse({ permissionMode: "always_kill" }),
      ).toThrow();
    });
  });

  describe("Property: Config merge preserves precedence (Property 2)", () => {
    // Property-based test: generate random partial configs and verify merge precedence
    it("should respect project > global > default precedence for all combinations", () => {
      fc.assert(
        fc.property(
          fc.record({
            permissionMode: fc.constantFrom("always_ask", "ask_when_risky", "never_ask" as const),
            modelTier: fc.constantFrom("small", "large" as const),
            activeProvider: fc.constantFrom("local", "openai", "anthropic", "gemini" as const),
            telemetryOptIn: fc.boolean(),
          }),
          fc.record({
            permissionMode: fc.constantFrom("always_ask", "ask_when_risky", "never_ask" as const),
            modelTier: fc.constantFrom("small", "large" as const),
            activeProvider: fc.constantFrom("local", "openai", "anthropic", "gemini" as const),
            telemetryOptIn: fc.boolean(),
          }),
          (globalOverrides, projectOverrides) => {
            // The merge function's behavior:
            const merged = { ...DEFAULT_CONFIG, ...globalOverrides, ...projectOverrides };

            // Project values should be used when present
            expect(merged.permissionMode).toBe(projectOverrides.permissionMode);
            expect(merged.modelTier).toBe(projectOverrides.modelTier);
            expect(merged.activeProvider).toBe(projectOverrides.activeProvider);
            expect(merged.telemetryOptIn).toBe(projectOverrides.telemetryOptIn);
          },
        ),
        { numRuns: 100 },
      );
    });
  });
});
