import { createInterface } from "node:readline";
import { loadConfig, saveConfig } from "../config/index.js";
import { type PermissionMode } from "../config/types.js";

/**
 * Interactive permission config UI.
 * Prompts the user to select a permission mode and saves it.
 */
export async function configurePermissions(): Promise<void> {
  const config = await loadConfig(process.cwd());

  console.log("\nCurrent permission mode:", config.permissionMode);
  console.log("\nSelect a permission mode:");
  console.log("  1. always_ask     — Confirm every action (safest)");
  console.log("  2. ask_when_risky — Only confirm file writes and commands (recommended)");
  console.log("  3. never_ask      — Full autonomy (fastest, least safe)");

  const rl = createInterface({ input: process.stdin, output: process.stdout });
  const answer = await new Promise<string>((resolve) => {
    rl.question("Enter a number (1-3): ", resolve);
  });
  rl.close();

  const index = parseInt(answer, 10);
  let mode: PermissionMode;

  switch (index) {
    case 1:
      mode = "always_ask";
      break;
    case 2:
      mode = "ask_when_risky";
      break;
    case 3:
      mode = "never_ask";
      break;
    default:
      console.log("Invalid selection. Keeping current mode.");
      return;
  }

  config.permissionMode = mode;
  await saveConfig(config);
  console.log(`\n✅ Permission mode updated to: ${mode}`);
}
