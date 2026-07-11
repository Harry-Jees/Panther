# Implementation Document
## Panther — build plan, folder structure, core interfaces

---

## 1. Repo structure

```
panther/
├── src/
│   ├── cli/                 # command definitions (commander/oclif)
│   ├── session/              # session manager, REPL (ink) + one-shot runner
│   ├── spec/                 # spec workflow state machine
│   ├── agent/                # agent loop, tool-call dispatch, GBNF grammars
│   ├── tools/                # read_file, write_file, apply_diff, run_command, etc.
│   ├── inference/             # node-llama-cpp wrapper, model manager
│   ├── providers/               # ModelProvider interface: local + BYOK remote (OpenAI, Anthropic, Gemini, NVIDIA)
│   ├── rag/                   # chunker (tree-sitter), embeddings, vector store
│   ├── memory/                 # memory.md read/write
│   ├── skills/                 # skill discovery, loading, tool registration
│   ├── permissions/             # permission gate logic
│   └── config/                   # config file read/write, defaults
├── test/
│   ├── unit/
│   └── integration/            # mocked-model agent loop tests
└── .panther/                        # (created at runtime in user projects, not in this repo)
```

## 2. Phased build plan

**Phase 0 — Foundations**
- `tsconfig.json` with ESM config (`module: NodeNext`, `moduleResolution: NodeNext`, strict mode, source maps, `types: ["node"]`).
- CLI skeleton (commander), config system (Zod validation with project > global > default merge), model manager (resumable download with progress bar + SHA-256 hash verification).
- Goal: prove `node-llama-cpp` works end-to-end on all three target OSes before building anything else on top.

**Phase 1 — Core agent loop**
- Tool executors (`read_file`, `write_file`, `apply_diff`, `list_dir`, `grep`, `run_command`).
- GBNF grammar-constrained tool-calling.
- Permission gate wired to a hardcoded `ask_when_risky` default.
- Goal: agent can complete a simple single-file bug fix end to end, with approval prompts working.

**Phase 2 — Sessions**
- REPL mode (ink UI), one-shot mode, shared session state file format.
- `memory.md` read at session start, write at session end.
- Goal: a REPL session and a `panther run` one-shot invocation can both resume the same in-progress task.

**Phase 3 — RAG**
- tree-sitter chunking, embedding model integration, SQLite + sqlite-vec index, incremental re-indexing.
- Goal: agent's file edits are informed by retrieved related code, not just the single file in view.

**Phase 4 — Spec-driven workflow**
- State machine (requirements → design → tasks → execution), markdown file generation per stage, user review/edit/approve loop at each gate.
- Goal: a non-trivial feature request produces reviewable requirements.md/design.md/tasks.md before any code is touched.

**Phase 5 — Skills**
- `SKILL.md` discovery/loading (project-local and global), lazy-loading by description match, skill-registered tools.
- Goal: a hand-authored example skill (e.g., "React component conventions") measurably changes agent output on a matching task.

**Phase 6 — BYOK model providers**
- `ModelProvider` interface, `LocalProvider` refactored to implement it, `RemoteProvider` implementations for OpenAI-compatible/Anthropic/Gemini APIs, secure key storage via OS keychain, per-session/per-project provider selection.
- Goal: `panther run --model gpt-4o-mini "..."` (with a key already configured) completes a task through a remote API, using the exact same agent loop and tool set as the local path.

**Phase 7 — Permissions & polish**
- Full dropdown config (always ask / ask when risky / never ask), per-skill tool pre-approval, setup wizard (model tier picker, permission default, telemetry opt-in).
- Goal: first-run experience is a complete, guided flow from `npm install` to first completed task.

**Phase 8 — Packaging & release**
- Cross-platform native addon verification, GitHub Actions CI across OSes, npm publish, versioned model-manifest for `panther model switch`.

## 3. Core interfaces (pseudocode)

**Tool executor interface**
```ts
interface ToolExecutor {
  name: string;
  riskLevel: "safe" | "risky";
  schema: JSONSchema;           // used to build the GBNF grammar
  execute(args: unknown, ctx: SessionContext): Promise<ToolResult>;
}
```

**Model provider interface**
```ts
interface ModelProvider {
  readonly name: "local" | "openai" | "anthropic" | "gemini" | "nvidia";
  readonly contextWindow: number;
  readonly supportsGrammar: boolean;
  generate(history: Message[], tools: ToolSchema[]): AsyncIterable<TokenChunk | ToolCall>;
  embedText?(text: string): Promise<number[]>;
}

// Selected once per session, then the agent loop is provider-agnostic.
function resolveProvider(config: PantherConfig, override?: string): ModelProvider {
  const requested = override ?? config.activeProvider;
  if (requested !== "local" && !credentialStore.has(requested)) {
    throw new Error(`No API key configured for ${requested}. Run: panther auth add ${requested}`);
  }
  return providerRegistry.get(requested);
}
```

**Agent loop**
```ts
async function runAgentLoop(task: string, ctx: SessionContext) {
  let history = ctx.loadHistory();
  history.push({ role: "user", content: task });

  while (true) {
    // The ModelProvider abstraction hides provider-specific details.
    // For the local provider (node-llama-cpp v3), the implementation:
    //   1. Converts ToolSchema[] → session.prompt() `functions` array
    //   2. Calls session.prompt(userMessage, { functions })
    //      — node-llama-cpp handles GBNF grammar internally
    //   3. Returns normalized ModelOutput via AsyncIterable
    // For remote providers, each SDK's native function-calling is used.
    const response = await ctx.model.generate(history, ctx.availableTools);

    if (response.type === "tool_call") {
      const tool = ctx.getTool(response.toolName);
      const decision = permissionGate.check(tool, ctx.config);

      if (decision === "denied") {
        history.push(userDeclined(tool));
        continue;
      }
      if (decision === "needs_prompt") {
        const approved = await promptUserApproval(tool, response.args);
        if (!approved) { history.push(userDeclined(tool)); continue; }
      }

      const result = await tool.execute(response.args, ctx);
      history.push(toolResult(result));
      continue;
    }

    if (response.type === "complete") {
      // Normal text — either continue interactive loop or complete
      if (ctx.isInteractive) {
        history.push({ role: "assistant", content: response.text });
        const userInput = await getNextUserInput(ctx);
        if (userInput === null) return;
        history.push({ role: "user", content: userInput });
      } else {
        await ctx.memory.appendSessionLog(summarize(history));
        return response.text;
      }
    }
  }
}
```

> **Note:** The `ModelProvider` interface normalizes all provider outputs into the same `AsyncIterable<ModelOutput>` shape. For the local provider (node-llama-cpp v3), this means:
> - `session.prompt()` is called with a `functions` array (no manual GBNF grammar building)
> - The plain string response is parsed and wrapped into `ToolCall` or `TokenChunk`/ `CompletionSignal` objects
> - GPU backend logging happens at initialization time
> 
> For BYOK remote providers (Phase 6), each SDK's streaming function-calling is normalized to the same `ModelOutput` shape.

**Skill loading (lazy)**
```ts
async function loadRelevantSkills(task: string, ctx: SessionContext) {
  const allSkills = await discoverSkills(ctx.projectRoot); // metadata only
  const matched = allSkills.filter(s => matchesDescription(s, task));
  return Promise.all(matched.map(s => loadSkillBody(s)));  // full content only for matches
}
```

## 4. File format specs

**`.panther/memory.md`** — see design doc §5 for structure. Parsed/written via a small markdown-section utility, not a full markdown AST library — sections are simple enough for line-based parsing.

**`SKILL.md`** frontmatter:
```yaml
---
name: react-component-conventions
description: Use when creating or editing React components in this project
triggers: ["create component", "new component", "*.tsx"]
tools: []  # optional additional tools this skill registers
---
```

**`.panther/config.json`**
```json
{
  "permissionMode": "ask_when_risky",
  "modelTier": "large",
  "modelFamily": "qwen",
  "activeProvider": "local",
  "telemetryOptIn": false
}
```
Note: no API keys live in this file — `activeProvider` only names which credential (stored via OS keychain, see design doc §10) to use. `panther auth add <provider>` is the only command that ever writes a key, straight to the keychain.

## 5. Milestone sequencing note

Phases 0–2 form the smallest useful product (a local agent that can do a supervised bug fix across sessions) — worth treating as an internal alpha checkpoint before investing in Phases 3–6, since Phase 0's cross-platform native-addon risk is the biggest unknown in the whole plan and should be de-risked first.
