import { describe, it, expect } from "vitest";
import { getModelManifest, listModels } from "../../src/inference/modelManager.js";

describe("Model manager", () => {
  describe("getModelManifest", () => {
    it("should return entries for small and large tiers", () => {
      const manifest = getModelManifest();
      expect(manifest.small).toBeDefined();
      expect(manifest.large).toBeDefined();
      expect(manifest.small.tier).toBe("small");
      expect(manifest.large.tier).toBe("large");
    });

    it("should have valid HuggingFace repo URLs", () => {
      const manifest = getModelManifest();
      for (const entry of Object.values(manifest)) {
        expect(entry.huggingFaceRepo).toMatch(/^[\w-]+\/[\w-]+$/);
        expect(entry.filename).toMatch(/\.gguf$/);
        expect(entry.contextWindow).toBeGreaterThan(0);
      }
    });
  });

  describe("listModels", () => {
    it("should return model status entries", async () => {
      const models = await listModels();
      expect(models.length).toBe(2);
      expect(models.map((m) => m.tier).sort()).toEqual(["large", "small"]);
    });
  });
});
