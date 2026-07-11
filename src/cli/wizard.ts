import { createInterface } from "node:readline";
import { loadConfig, saveConfig } from "../config/index.js";
import { type PantherConfig, type ModelTier, type PermissionMode } from "../config/types.js";
import { ensureModel } from "../inference/modelManager.js";

/**
 * Print a simple text-based progress bar for model downloads.
 */
function printProgress(downloaded: number, total: number): void {
  const width = 40;
  const percent = total > 0 ? Math.min(Math.round((downloaded / total) * 100), 100) : 0;
  const filled = Math.round((width * percent) / 100);
  const empty = width - filled;
  const bar = "█".repeat(filled) + "░".repeat(empty);
  const downloadedMB = (downloaded / (1024 * 1024)).toFixed(1);
  const totalMB = total > 0 ? (total / (1024 * 1024)).toFixed(1) : "???";
  process.stdout.write(`\r  ${bar} ${percent}% (${downloadedMB}MB / ${totalMB}MB)`);
}

/**
 * Prompt user with a question and return the answer.
 */
async function prompt(question: string, defaultValue?: string): Promise<string> {
  const rl = createInterface({ input: process.stdin, output: process.stdout });
  return new Promise((resolve) => {
    rl.question(question + " ", (answer) => {
      rl.close();
      resolve(answer.trim() || defaultValue || "");
    });
  });
}

/**
 * Prompt user to select from a list of options.
 */
async function select(message: string, options: string[]): Promise<string> {
  console.log(`\n${message}`);
  for (let i = 0; i < options.length; i++) {
    console.log(`  ${i + 1}. ${options[i]}`);
  }
  const answer = await prompt(`Enter a number (1-${options.length}):`);
  const index = parseInt(answer, 10) - 1;
  if (index >= 0 && index < options.length) {
    return options[index];
  }
  return options[0];
}

/**
 * Run the first-time setup wizard.
 * Returns true if setup completed successfully, false if the user cancelled.
 */
export async function runSetupWizard(): Promise<boolean> {
  console.log("\n🐆 Panther Setup Wizard");
  console.log("━━━━━━━━━━━━━━━━━━━━━━\n");
  console.log("Welcome! Let's get you set up.\n");

  // Step 1: Model tier selection with hardware disclosure
  const modelTier = (await select(
    "Select a model tier:",
    [
      "small  - Qwen3-Coder-4B (fast, ~2.5GB RAM) - Recommended for most users",
      "large  - Qwen3-Coder-80B-A3B (capable, ~8GB RAM) - Better results, needs more RAM",
    ],
  )) as string;

  const isLarge = modelTier.startsWith("large");

  if (isLarge) {
    console.log("\n⚠️  Large model requires ~8GB of available RAM and may take");
    console.log("   longer to download (~4-5GB). Make sure you have:");
    console.log("   • 8GB+ free RAM");
    console.log("   • Stable internet connection");
    console.log("   • 15-30 minutes for first download\n");

    const confirm = await prompt("Continue with large model? (Y/n):", "y");
    if (confirm.toLowerCase() !== "y" && confirm !== "") {
      console.log("Setup cancelled.");
      return false;
    }
  }

  const tier: ModelTier = isLarge ? "large" : "small";

  // Step 2: Permission mode
  const permissionMode = (await select(
    "Select permission mode:",
    [
      "always_ask     - Confirm every action (safest)",
      "ask_when_risky - Only confirm file writes and commands (recommended)",
      "never_ask      - Full autonomy (fastest, least safe)",
    ],
  )) as string;

  let mode: PermissionMode = "ask_when_risky";
  if (permissionMode.startsWith("always")) mode = "always_ask";
  else if (permissionMode.startsWith("never")) mode = "never_ask";

  // Step 3: Telemetry opt-in
  const telemetry = await prompt("Allow anonymous usage telemetry? (y/N):", "n");
  const telemetryOptIn = telemetry.toLowerCase() === "y";

  // Step 4: Optional BYOK setup
  const addKey = await prompt("Configure a remote AI provider API key? (y/N):", "n");

  if (addKey.toLowerCase() === "y") {
    const provider = await select("Select provider:", [
      "openai",
      "anthropic",
      "gemini",
    ]);
    const key = await prompt(`Enter your ${provider} API key:`);
    if (key) {
      try {
        const { addCredential } = await import("../config/credentials.js");
        await addCredential(provider as "openai" | "anthropic" | "gemini", key);
        console.log(`✅ ${provider} API key stored securely in OS keychain.`);
      } catch {
        console.log(`⚠️  Could not store API key. You can add it later with: panther auth add ${provider}`);
      }
    }
  }

  // Build config
  const config: PantherConfig = {
    permissionMode: mode,
    modelTier: tier,
    modelFamily: "qwen",
    activeProvider: "local",
    telemetryOptIn,
  };

  // Save config
  await saveConfig(config);
  console.log("\n✅ Configuration saved to ~/.panther/config.json\n");

  // Step 5: Download model
  console.log(`Downloading ${tier} model...`);
  try {
    await ensureModel(tier, printProgress);
    console.log("\n\n✅ Model downloaded successfully!");
  } catch (err) {
    console.log(`\n⚠️  Model download failed: ${(err as Error).message}`);
    console.log("  You can retry later with: panther model switch <tier>");
  }

  console.log("\n━━━━━━━━━━━━━━━━━━━━━━");
  console.log("🎉 Setup complete! Run 'panther' to start.\n");

  return true;
}
