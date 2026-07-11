#!/usr/bin/env node
import { Command } from "commander";
import { readFileSync, existsSync } from "node:fs";
import { createInterface } from "node:readline";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadConfig, saveConfig } from "../config/index.js";
import { ensureModel, listModels, switchModel } from "../inference/modelManager.js";
import { LocalProvider } from "../inference/localProvider.js";
import { type ModelTier } from "../config/types.js";
import { SessionManager } from "../session/manager.js";
import { MemorySystem } from "../memory/index.js";
import { ProjectIndexer, InMemoryVectorStore, retrieveContext } from "../rag/index.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const pkgPath = path.join(__dirname, "..", "..", "package.json");
let pkgVersion = "0.1.0";
try {
  const pkg = JSON.parse(readFileSync(pkgPath, "utf-8"));
  pkgVersion = pkg.version;
} catch { /* use default */ }

function printBanner(): void {
  const yellow = "\x1b[33m";
  const reset = "\x1b[0m";
  console.log("");
  console.log(`${yellow}  ██████  █████  ███    ██ ████████ ██   ██ ███████ ██████  ${reset}`);
  console.log(`${yellow}  ██      ██   ██ ████   ██    ██    ██   ██ ██      ██   ██ ${reset}`);
  console.log(`${yellow}  ██      ███████ ██ ██  ██    ██    ███████ █████   ██████  ${reset}`);
  console.log(`${yellow}  ██      ██   ██ ██  ██ ██    ██    ██   ██ ██      ██   ██ ${reset}`);
  console.log(`${yellow}  ██████  ██   ██ ██   ████    ██    ██   ██ ███████ ██   ██ ${reset}`);
  console.log(`${yellow}                                                                 ${reset}`);
  console.log(`${yellow}  A self-contained, spec-driven local coding agent${reset}`);
  console.log(`${yellow}  v${pkgVersion}${reset}`);
  console.log("");
}

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

async function runRepl(sessionId?: string): Promise<void> {
  printBanner();
  const config = await loadConfig(process.cwd());
  const projectRoot = process.cwd();
  const manager = new SessionManager({ sessionsDir: path.join(projectRoot, ".panther", "sessions") });
  const memory = new MemorySystem(projectRoot);
  await memory.initialize();

  let session = sessionId ? (await manager.loadSession(sessionId)) ?? (await manager.createSession()) : await manager.createSession();

  console.log(`Session: ${session.id}`);
  console.log(`Model tier: ${config.modelTier}`);
  console.log(`Permission mode: ${config.permissionMode}`);
  console.log("\nType 'exit' or press Ctrl+C to quit.\n");

  // Load memory context
  const memoryContext = await memory.readMemory();
  if (memoryContext) {
    console.log("📝 Previous session memory loaded.\n");
  }

  const rl = createInterface({ input: process.stdin, output: process.stdout });

  rl.on("line", async (answer: string) => {
    const trimmed = answer.trim().toLowerCase();
    if (trimmed === "exit" || trimmed === "quit") {
      rl.close();
      return;
    }

    try {
      const modelPath = await ensureModel(config.modelTier, printProgress);
      console.log("\n");

      const provider = await LocalProvider.create(modelPath);
      session.history.push({ role: "user", content: answer });

      // Retrieve RAG context if vector store has data
      const store = new InMemoryVectorStore();
      const context = await retrieveContext(answer, store, provider.metadata.contextWindow);
      if (context) {
        session.history.push({ role: "system", content: context });
      }

      for await (const chunk of provider.generate(session.history)) {
        if (chunk.type === "complete") {
          process.stdout.write(chunk.text);
          session.history.push({ role: "assistant", content: chunk.text });
        }
      }

      // Save session and memory
      await manager.saveSession(session);
      await memory.appendSessionLog({
        summary: answer.slice(0, 100),
        filesTouched: [],
        decisions: [],
        followUps: [],
      });

      await provider.dispose();
      console.log("\n\n");
    } catch (err) {
      console.error(`\n❌ Error: ${(err as Error).message}`);
    }

    rl.prompt();
  });

  rl.on("close", () => {
    console.log("\nGoodbye!");
    process.exit(0);
  });

  rl.setPrompt("🐾 ");
  rl.prompt();
}

async function main(): Promise<void> {
  const program = new Command();

  program
    .name("panther")
    .description("A self-contained, spec-driven local coding agent")
    .version(pkgVersion);

  // ── run command ──────────────────────────────────────────────
  program
    .command("run")
    .description("Execute a one-shot task")
    .argument("[task]", "The task description to execute")
    .option("--json", "Output as newline-delimited JSON")
    .option("--session <id>", "Resume an existing session")
    .option("--model <provider>", "Use a specific model provider (e.g., gpt-4o-mini)")
    .action(async (task: string | undefined, options: { json?: boolean; session?: string; model?: string }) => {
      if (options.json) console.log(JSON.stringify({ type: "start", task, session: options.session }));

      const config = await loadConfig(process.cwd());

      if (!options.json) printBanner();

      // Load or create session
      const projectRoot = process.cwd();
      const manager = new SessionManager({ sessionsDir: path.join(projectRoot, ".panther", "sessions") });
      let session = options.session
        ? (await manager.loadSession(options.session)) ?? (await manager.createSession())
        : await manager.createSession();

      if (!options.json) console.log(`\nSession: ${session.id}`);

      if (!task) {
        const rl = createInterface({ input: process.stdin, output: process.stdout });
        task = await new Promise<string>((resolve) => {
          rl.question("🐾 Task: ", (answer) => { rl.close(); resolve(answer); });
        });
      }

      if (!task || task.trim().length === 0) {
        console.log("No task provided. Exiting.");
        process.exit(0);
      }

      try {
        if (!options.json) console.log("\nChecking for local model...");
        const modelPath = await ensureModel(config.modelTier, (d, t) => {
          if (!options.json) printProgress(d, t);
        });
        if (!options.json) console.log("\n\nLoading model...");
        const provider = await LocalProvider.create(modelPath);
        if (!options.json) console.log("\nGenerating response...\n");

        session.history.push({ role: "user", content: task });
        let fullResponse = "";

        for await (const chunk of provider.generate(session.history)) {
          if (chunk.type === "complete") {
            fullResponse += chunk.text;
            if (!options.json) process.stdout.write(chunk.text);
          }
        }

        session.history.push({ role: "assistant", content: fullResponse });
        await manager.saveSession(session);

        if (!options.json) {
          console.log("\n\n✅ Task complete.");
        } else {
          console.log(JSON.stringify({ type: "complete", result: fullResponse }));
        }

        await provider.dispose();
      } catch (err) {
        const message = (err as Error).message;
        if (options.json) console.log(JSON.stringify({ type: "error", message }));
        else console.error(`\n❌ Error: ${message}`);
        process.exit(1);
      }
    });

  // ── model command ────────────────────────────────────────────
  program
    .command("model")
    .description("Manage AI models")
    .argument("<action>", "Action: switch, list")
    .argument("[tier]", "Model tier: small or large")
    .action(async (action: string, tier?: string) => {
      if (action === "list") {
        const models = await listModels();
        console.log("\nInstalled models:");
        for (const m of models) {
          console.log(`  ${m.tier === "large" ? "🐘" : "🐁"} ${m.tier}: ${m.name} ${m.downloaded ? "✅" : "❌ not downloaded"}`);
        }
        return;
      }

      if (action === "switch") {
        if (!tier || !["small", "large"].includes(tier)) {
          console.error("Please specify a tier: small or large");
          process.exit(1);
        }

        const rl = createInterface({ input: process.stdin, output: process.stdout });
        const answer = await new Promise<string>((resolve) => {
          rl.question(`Switch to ${tier} model? This will delete the current model. (y/N): `, resolve);
        });
        rl.close();
        if (answer.toLowerCase() !== "y") { console.log("Cancelled."); return; }

        console.log(`\nSwitching to ${tier} model tier...`);
        const modelPath = await switchModel(tier as ModelTier, printProgress);
        const config = await loadConfig(process.cwd());
        config.modelTier = tier as ModelTier;
        await saveConfig(config);
        console.log(`\n\n✅ Switched to ${tier} model at: ${modelPath}`);
      }
    });

  // ── config command ───────────────────────────────────────────
  program
    .command("config")
    .description("View or update configuration")
    .option("--permissions", "Configure permission mode interactively")
    .action(async (options: { permissions?: boolean }) => {
      if (options.permissions) {
        const { configurePermissions } = await import("../permissions/configUI.js");
        await configurePermissions();
        return;
      }
      const config = await loadConfig(process.cwd());
      console.log("\nPanther Configuration:");
      console.log(JSON.stringify(config, null, 2));
    });

  // ── session command ──────────────────────────────────────────
  program
    .command("session")
    .description("Manage sessions")
    .argument("[action]", "Action: list, show <id>, delete <id>")
    .argument("[id]", "Session ID")
    .action(async (action?: string, id?: string) => {
      const manager = new SessionManager({ sessionsDir: path.join(process.cwd(), ".panther", "sessions") });

      if (action === "list" || !action) {
        const sessions = await manager.listSessions();
        console.log("\nSessions:");
        for (const s of sessions.slice(0, 20)) {
          const lastMsg = s.history[s.history.length - 1]?.content.slice(0, 80) ?? "(empty)";
          console.log(`  ${s.id}  ${s.updatedAt.slice(0, 10)}  "${lastMsg}"`);
        }
        return;
      }

      if (action === "show" && id) {
        const session = await manager.loadSession(id);
        if (!session) { console.log("Session not found."); return; }
        console.log(`\nSession: ${session.id}`);
        console.log(`Created: ${session.createdAt}`);
        console.log(`Updated: ${session.updatedAt}`);
        console.log(`Messages: ${session.history.length}`);
        for (const msg of session.history.slice(-5)) {
          console.log(`  [${msg.role}] ${msg.content.slice(0, 100)}`);
        }
        return;
      }

      if (action === "delete" && id) {
        const deleted = await manager.deleteSession(id);
        console.log(deleted ? `✅ Deleted session: ${id}` : "Session not found.");
        return;
      }
    });

  // ── auth command ─────────────────────────────────────────────
  program
    .command("auth")
    .description("Manage API keys for remote providers")
    .argument("[action]", "Action: add, remove, list")
    .argument("[provider]", "Provider name: openai, anthropic, gemini")
    .argument("[key]", "API key value")
    .action(async (action?: string, provider?: string, key?: string) => {
      const { addCredential, removeCredential, listCredentials } = await import("../config/credentials.js");

      if (action === "list" || !action) {
        const providers = await listCredentials();
        console.log("\nConfigured providers:");
        if (providers.length === 0) console.log("  (none)");
        for (const p of providers) console.log(`  ✅ ${p}`);
        return;
      }

      if (action === "add" && provider) {
        if (!key) {
          const rl = createInterface({ input: process.stdin, output: process.stdout });
          key = await new Promise<string>((resolve) => {
            rl.question(`Enter API key for ${provider}: `, (answer) => { rl.close(); resolve(answer); });
          });
        }
        if (!key) { console.log("No key provided."); return; }
        try {
          await addCredential(provider as "openai" | "anthropic" | "gemini", key);
          console.log(`✅ API key for ${provider} stored securely in OS keychain.`);
        } catch (err) {
          console.error(`❌ Failed to store key: ${(err as Error).message}`);
        }
        return;
      }

      if (action === "remove" && provider) {
        await removeCredential(provider as "openai" | "anthropic" | "gemini");
        console.log(`✅ Removed API key for ${provider}.`);
        return;
      }
    });

  // ── index command ────────────────────────────────────────────
  program
    .command("index")
    .description("Index the project for RAG context retrieval")
    .action(async () => {
      const projectRoot = process.cwd();
      console.log("\n🔍 Indexing project...");
      const store = new InMemoryVectorStore();
      const indexer = new ProjectIndexer(store, projectRoot);
      const result = await indexer.indexProject();
      console.log(`✅ Indexed ${result.filesIndexed} files, ${result.chunksCreated} chunks created.`);
    });

  // ── skills command ───────────────────────────────────────────
  program
    .command("skills")
    .description("List available skills")
    .argument("[action]", "Action: list")
    .action(async () => {
      const { discoverSkills } = await import("../skills/index.js");
      const skills = await discoverSkills(process.cwd());
      console.log("\nAvailable skills:");
      if (skills.length === 0) console.log("  (none)");
      for (const skill of skills) {
        console.log(`  📋 ${skill.name} — ${skill.description}`);
        if (skill.triggers.length > 0) console.log(`     Triggers: ${skill.triggers.join(", ")}`);
      }
    });

  // ── Determine mode ───────────────────────────────────────────
  const hasSubcommand = process.argv.length > 2;

  if (hasSubcommand) {
    await program.parseAsync(process.argv);
  } else {
    // No args → check if first run, else REPL
    const { homedir } = await import("node:os");
    const globalConfigPath = path.join(homedir(), ".panther", "config.json");
    if (!existsSync(globalConfigPath)) {
      const { runSetupWizard } = await import("./wizard.js");
      const completed = await runSetupWizard();
      if (!completed) process.exit(0);
    }
    await runRepl();
  }
}

main().catch((err) => {
  console.error("Fatal error:", err.message);
  process.exit(1);
});
