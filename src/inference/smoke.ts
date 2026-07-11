#!/usr/bin/env node
/**
 * Smoke test for node-llama-cpp integration.
 *
 * Run: npx tsx src/inference/smoke.ts [path-to-model.gguf]
 *
 * If no model path is given, it will attempt to load the default cached model.
 */

import { getLlama, LlamaChatSession } from "node-llama-cpp";
import { getCachedModel } from "./modelManager.js";
import { loadConfig } from "../config/index.js";

async function main() {
  const modelPath = process.argv[2];

  let resolvedPath: string;
  if (modelPath) {
    resolvedPath = modelPath;
  } else {
    const config = await loadConfig(process.cwd());
    const cached = await getCachedModel(config.modelTier);
    if (!cached) {
      console.error(
        "No model path provided and no cached model found.\n" +
          "Usage: npx tsx src/inference/smoke.ts [path-to-model.gguf]\n" +
          "Or run 'panther run' first to download a model.",
      );
      process.exit(1);
    }
    resolvedPath = cached;
  }

  console.log(`\n🔧 Loading model from: ${resolvedPath}`);
  console.log("⏳ This may take a moment for large models...\n");

  const startTime = Date.now();

  const llama = await getLlama();

  console.log(`  GPU backend: ${llama.gpu || "cpu"}`);

  const model = await llama.loadModel({
    modelPath: resolvedPath,
  });

  const context = await model.createContext();
  const session = new LlamaChatSession({
    contextSequence: context.getSequence(),
  });

  const loadTime = Date.now() - startTime;
  console.log(`  Loaded in ${loadTime}ms\n`);

  // Test prompt
  const prompt = "Write a one-line hello world function in TypeScript.";
  console.log(`📝 Prompt: "${prompt}"\n`);
  console.log("💬 Response:");

  const responseStart = Date.now();
  const response = await session.prompt(prompt);
  const responseTime = Date.now() - responseStart;

  console.log(`  ${response}`);
  console.log(`\n  (${responseTime}ms, ${response.length} chars)`);

  // Print model info
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const modelAny = model as any;
  console.log(`\n📊 Model info:`);
  console.log(`  Context size: ${modelAny.contextSize ?? "N/A"}`);
  console.log(`  Train context size: ${modelAny.trainContextSize ?? "N/A"}`);

  console.log("\n✅ Smoke test passed!\n");
}

main().catch((err) => {
  console.error("\n❌ Smoke test failed:", err.message);
  console.error(err.stack);
  process.exit(1);
});
