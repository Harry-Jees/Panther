import { type PantherConfig } from "../config/types.js";
import { type ModelProvider } from "./types.js";
import { LocalProvider } from "./local.js";

// Remote providers require SDK installation:
//   OpenAI:    npm install openai
//   Anthropic: npm install @anthropic-ai/sdk
//   Gemini:    npm install @google/genai
// Uncomment the appropriate imports and cases when SDKs are installed.
// import { OpenAIProvider } from "./openai.js";
// import { AnthropicProvider } from "./anthropic.js";
// import { GeminiProvider } from "./gemini.js";

/**
 * Resolve a ModelProvider based on config and optional override.
 */
export async function resolveProvider(
  config: PantherConfig,
  modelOverride?: string,
  modelPath?: string,
): Promise<ModelProvider> {
  const requested = modelOverride ?? config.activeProvider;

  if (requested === "local") {
    if (!modelPath) {
      throw new Error("Local model path is required for local provider");
    }
    return LocalProvider.create(modelPath);
  }

  throw new Error(
    `Remote provider "${requested}" is not available in this build. ` +
    `Install the SDK and uncomment the provider in src/providers/index.ts. ` +
    `Local provider is always available.`,
  );
}

export { type ModelProvider } from "./types.js";
export { LocalProvider } from "./local.js";
