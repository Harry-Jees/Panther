# Implementation Plan: Panther CLI

## Overview

This plan builds Panther in eight phases, each ending at a testable milestone. Phases 0–2 form the internal alpha (a local agent that can do a supervised bug fix across sessions). Phases 3–8 layer in RAG, spec-driven workflow, skills, BYOK providers, permissions polish, and release packaging.

---

## Tasks

- [ ] 1. Repo scaffold and TypeScript setup
  - Run `npm init` and install `typescript`, `vitest`, `fast-check`, `@types/node`, `eslint`, `prettier`
  - Create `tsconfig.json` with:
    - `target: ES2022`, `module: NodeNext`, `moduleResolution: NodeNext`
    - `strict: true`, `esModuleInterop: true`, `skipLibCheck: true`
    - `outDir: dist`, `rootDir: src`
    - `declaration: true`, `declarationMap: true`, `sourceMap: true`
    - `resolveJsonModule: true`, `isolatedModules: true`
    - `types: ["node"]` for Node.js API types
  - Create `eslint.config.js` with flat config (ESLint 10.x) using `@typescript-eslint/parser` and `@typescript-eslint/eslint-plugin`
  - Create `.prettierrc` with project formatting conventions
  - Create the full `src/` folder tree: `cli/`, `session/`, `spec/`, `agent/`, `tools/`, `inference/`, `providers/`, `rag/`, `memory/`, `skills/`, `permissions/`, `config/`
  - Create `test/unit/` and `test/integration/` folders
  - Add `.gitignore` including `.panther/`, `dist/`, and `node_modules/`
  - Configure `vitest.config.ts` (include `fast-check` as a dependency for property tests)
  - Add npm scripts: `build`, `test`, `typecheck`, `lint`
  - _Requirements: 14.1_
  - [ ]* 1.1 Verify `npm test` exits 0 on empty test suite

- [ ] 2. Config system
  - [ ] 2.1 Implement config load and save
    - Create `src/config/index.ts` with `loadConfig(projectRoot)` and `saveConfig(projectRoot, config)`
    - Define config shape with `zod`: `permissionMode`, `modelTier`, `modelFamily`, `activeProvider`, `telemetryOptIn`
    - Implement merge logic: project `.panther/config.json` overrides `~/.panther/config.json` overrides built-in defaults
    - _Requirements: 9.2, 9.3_
  - [ ]* 2.2 Write property test for config merge precedence
    - **Property 2: Config merge preserves precedence**
    - **Validates: Requirements 9.2, 9.3**
  - [ ]* 2.3 Write unit tests for config edge cases
    - Missing file fallback returns defaults; invalid JSON falls back with warning and does not crash
    - _Requirements: 9.2, 9.3_

- [ ] 3. Model manager — download and cache GGUF
  - [ ] 3.1 Implement model download with progress and resume
    - Create `src/inference/modelManager.ts`
    - Implement `downloadModel(tier)` with resumable fetch and `readline` progress bar
    - Write `manifest.json` to `~/.panther/models/` with model name, tier, context window, and SHA-256 hash
    - Implement hash verification after download completion
    - Add `model-manifest.json` mapping tier names to HuggingFace repo, filename, and expected hash
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
  - [ ] 3.2 Implement `panther model switch <tier>` command
    - Extend `modelManager.ts` with `switchModel(tier)` function
    - Prompt user for confirmation before deleting the old model file
    - Update active model in config after successful switch
    - _Requirements: 3.6, 3.7_
  - [ ]* 3.3 Write unit tests for model manager (mocked HTTP)
    - Confirm resume logic, manifest write, hash verification failure path, and old-model deletion confirmation
    - _Requirements: 3.3, 3.4, 3.7_

- [ ] 4. Local inference provider
  - [ ] 4.1 Implement LocalProvider
    - Create `src/inference/localProvider.ts` implementing the `ModelProvider` interface
    - Use `getLlama()` → `llama.loadModel()` → `model.createContext()` → `LlamaChatSession`
    - Expose `generate(history, tools): AsyncIterable<TokenChunk | ToolCall>`
    - Log active GPU backend (Metal / CUDA / Vulkan / CPU) at load time
    - _Requirements: 1.1, 1.2, 1.3_
  - [ ] 4.2 Create `src/inference/smoke.ts` manual test script
    - Loads model, generates one completion, prints output — used for real-hardware validation
    - _Requirements: 1.1_
  - [ ]* 4.3 Write unit tests for LocalProvider (mocked node-llama-cpp)
    - Verify generate stream, GPU log output, CPU fallback notification
    - _Requirements: 1.2, 1.3, 1.4_

- [ ] 5. CLI skeleton and `panther run` command
  - [ ] 5.1 Implement CLI entry point
    - Create `src/cli/index.ts` using `commander`
    - Register `run` subcommand wired to `localProvider.generate()`
    - Add `--json` flag for newline-delimited JSON output
    - Add `--session <id>` flag
    - Add `bin` entry in `package.json`: `{ "panther": "dist/cli/index.js" }`
    - Add build script compiling TypeScript to `dist/`
    - _Requirements: 4.1, 4.4, 4.6_
  - [ ]* 5.2 Write unit tests for CLI argument parsing
    - `run` command parses `--json`, `--session`, and task string; passes correct arguments to provider (provider mocked)
    - _Requirements: 4.1, 4.4_

- [ ] 6. GitHub Actions CI
  - Create `.github/workflows/ci.yml`
  - Configure matrix: `[ubuntu-latest, macos-latest, windows-latest]` × `[node 18, node 20]`
  - Steps: checkout, `npm ci`, `npm run typecheck`, `npm run lint`, `npm test`
  - Add CI status badge to `README.md`
  - _Requirements: 14.2_

- [ ] 7. Checkpoint — Phase 0 complete
  - Ensure all tests pass, `npm run build` succeeds, and CI is green on all matrix targets. Ask the user if questions arise.

- [ ] 8. Tool executor interface and safe tools
  - [ ] 8.1 Define ToolExecutor interface
    - Create `src/tools/types.ts` with `ToolExecutor`, `ToolResult`, `SessionContext` interfaces
    - _Requirements: 10.1_
  - [ ] 8.2 Implement safe tools
    - Implement `read_file` in `src/tools/readFile.ts` (`riskLevel: "safe"`)
    - Implement `list_dir` in `src/tools/listDir.ts` (`riskLevel: "safe"`)
    - Implement `grep` in `src/tools/grep.ts` (`riskLevel: "safe"`)
    - Implement `run_tests` stub in `src/tools/runTests.ts` (`riskLevel: "safe"`)
    - _Requirements: 10.1, 10.3_
  - [ ]* 8.3 Write unit tests for safe tools
    - Each tool tested against temp files/dirs; error results returned as structured `ToolResult` (not thrown)
    - _Requirements: 10.3, 10.6_

- [ ] 9. Risky tool implementations
  - [ ] 9.1 Implement risky tools
    - Implement `write_file` in `src/tools/writeFile.ts` (`riskLevel: "risky"`)
    - Implement `apply_diff` in `src/tools/applyDiff.ts` (`riskLevel: "risky"`)
    - Implement `run_command` in `src/tools/runCommand.ts` with cwd locked to project root, env allowlist, and path traversal rejection (`riskLevel: "risky"`)
    - _Requirements: 10.1, 10.2, 10.4, 10.5_
  - [ ]* 9.2 Write unit tests for risky tools
    - `apply_diff` patches temp file correctly; `run_command` rejects path traversal; all tools return structured errors on failure
    - _Requirements: 10.2, 10.4, 10.5, 10.6_

- [ ] 10. Permission gate
  - [ ] 10.1 Implement PermissionGate
    - Create `src/permissions/gate.ts`
    - Implement `permissionGate.check(tool, config)` returning `"approved" | "denied" | "needs_prompt"`
    - In one-shot mode: `needs_prompt` auto-denies with log message
    - _Requirements: 9.1, 9.4, 9.5, 9.6, 4.5_
  - [ ]* 10.2 Write property test for permission gate
    - **Property 1: Permission gate covers all tool × mode combinations**
    - **Validates: Requirements 9.4, 9.5, 9.6**
  - [ ]* 10.3 Write unit tests for one-shot auto-denial
    - **Property 8: Risky tool one-shot auto-denial**
    - **Validates: Requirements 4.5**

- [ ] 11. Agent loop
  - [ ] 11.1 Implement core agent loop
    - Create `src/agent/loop.ts`
    - Implement `runAgentLoop(task, ctx)` using `node-llama-cpp` native JSON schema function-calling
    - Define `task_complete` pseudo-tool the model calls to signal completion
    - Wire permission gate: check before executing risky tools; append denial result to history on block
    - Append tool results to conversation history after each execution
    - _Requirements: 4.1, 4.2, 4.3, 4.5, 9.4, 9.5, 9.6_
  - [ ]* 11.2 Write property test for agent loop termination
    - **Property 7: Agent loop task_complete terminates loop**
    - **Validates: Requirements 4.1, 4.2**
  - [ ]* 11.3 Write integration tests for agent loop (mocked model)
    - `read_file → write_file → task_complete` sequence completes successfully
    - Denied permission path: tool blocked, history updated, loop continues
    - One-shot auto-denial path
    - _Requirements: 4.1, 4.2, 4.5_

- [ ] 12. Checkpoint — Phase 1 complete
  - Ensure all tests pass. Agent can complete a simple single-file task end-to-end with mocked model. Ask the user if questions arise.

- [ ] 13. Session state format and manager
  - [ ] 13.1 Implement session manager
    - Define session schema: `id`, `createdAt`, `updatedAt`, `history`, `specId`, `taskId`
    - Create `src/session/manager.ts` with `createSession()`, `loadSession(id)`, `saveSession(session)`, `listSessions()`
    - Sessions stored in `.panther/sessions/<id>.json`
    - Wire `--session <id>` flag to `panther run` and `panther` REPL commands
    - _Requirements: 4.6, 5.5, 5.6_
  - [ ]* 13.2 Write property test for session round-trip
    - **Property 3: Session round-trip consistency**
    - **Validates: Requirements 4.6, 5.6**

- [ ] 14. memory.md read/write
  - [ ] 14.1 Implement memory system
    - Create `src/memory/index.ts`
    - Implement `readMemory(projectRoot)`: reads `.panther/memory.md`, truncates to configurable token budget
    - Implement `appendSessionLog(projectRoot, entry)`: adds a dated section to `memory.md`
    - Load memory into agent context at session start
    - Write memory entry at session end (both REPL and one-shot modes)
    - Add `.panther/memory.md` to project `.gitignore` on first write if not already present
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_
  - [ ]* 14.2 Write property test for memory append monotonicity
    - **Property 4: Memory append monotonicity**
    - **Validates: Requirements 7.3**
  - [ ]* 14.3 Write unit tests for memory truncation
    - `readMemory` truncates correctly at token budget boundary
    - _Requirements: 7.4_

- [ ] 15. REPL mode with Ink UI
  - [ ] 15.1 Implement REPL components
    - Create `src/session/repl.tsx`
    - Implement `<Prompt>` component for user text input
    - Implement `<StreamingOutput>` component for real-time token display
    - Implement `<ApprovalGate>` component for risky tool `y/N` prompts
    - Wire `PermissionGate.needs_prompt` to `<ApprovalGate>` in REPL mode
    - Persist session state after every turn
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_
  - [ ]* 15.2 Write Ink render tests for REPL
    - Simulate user input, assert streaming output rendered correctly with mocked agent loop
    - Simulate approval prompt, assert `y` approves and `N` denies
    - _Requirements: 5.3, 5.4_

- [ ] 16. Startup banner
  - [ ] 16.1 Implement branded banner
    - Create `src/cli/banner.tsx`
    - Render ASCII art "Panther" wordmark using Ink `<Text color="yellow">`
    - Display banner at start of REPL session
    - Display banner before first output in one-shot mode
    - _Requirements: 13.1, 13.2_
  - [ ]* 16.2 Write snapshot test for banner render output
    - _Requirements: 13.1_

- [ ] 17. Checkpoint — Phase 2 / Internal Alpha complete
  - Ensure all tests pass. REPL and one-shot modes work end-to-end with session persistence and memory. Ask the user if questions arise.

- [ ] 18. SQLite + sqlite-vec store
  - [ ] 18.1 Implement vector store
    - Install `better-sqlite3` and `sqlite-vec`
    - Create `src/rag/store.ts`
    - Initialise `.panther/index.db` with `sqlite-vec` extension loaded
    - Define schema: `chunks(id, file_path, start_line, end_line, content, embedding BLOB)` and `file_index(file_path, mtime)`
    - Implement `upsertChunk()`, `deleteChunksForFile()`, `search(queryEmbedding, k)` using cosine distance
    - _Requirements: 11.3_
  - [ ]* 18.2 Write unit tests for vector store
    - Insert 3 chunks; search returns nearest by cosine distance; upsert replaces existing chunk
    - _Requirements: 11.3_

- [ ] 19. Tree-sitter chunker
  - [ ] 19.1 Implement file chunker
    - Install `tree-sitter` and grammars for TypeScript, JavaScript, Python, Go
    - Create `src/rag/chunker.ts`
    - Implement `chunkFile(filePath)`: splits file into function/class-boundary-aware chunks, each with `{ filePath, startLine, endLine, content }`
    - _Requirements: 11.1_
  - [ ]* 19.2 Write unit tests for chunker
    - Parse sample TypeScript file; assert chunks align with function and class boundaries
    - _Requirements: 11.1_

- [ ] 20. Embedding model and incremental indexer
  - [ ] 20.1 Implement incremental project indexer
    - Create `src/rag/indexer.ts`
    - Load embedding GGUF model separately from chat model via `node-llama-cpp` embedding API
    - Implement `indexProject(projectRoot)`: chunks and embeds only files with newer mtime than `file_index` record
    - Add `panther index` CLI command
    - _Requirements: 11.2, 11.4, 11.7_
  - [ ]* 20.2 Write unit tests for incremental indexer
    - Index 3 files; modify 1 file; re-index → only modified file's chunks replaced, others unchanged
    - _Requirements: 11.4_

- [ ] 21. RAG retrieval wired into agent context
  - [ ] 21.1 Implement RAG retriever
    - Create `src/rag/retriever.ts`
    - Implement `retrieveContext(task, ctx)`: embed task string, query vector store, return top-k chunks
    - Format retrieved chunks as `<context>` block in system prompt
    - Cap total retrieved tokens to configurable fraction of model `contextWindow`
    - Inject retrieved context into agent loop before each iteration
    - _Requirements: 11.5, 11.6_
  - [ ]* 21.2 Write property test for RAG token budget
    - **Property 6: RAG retrieval is capped within context budget**
    - **Validates: Requirements 11.6**
  - [ ]* 21.3 Write unit tests for retriever (mocked vector store)
    - Assert retrieved chunks appear in prompt; assert cap enforced when chunks exceed budget
    - _Requirements: 11.5, 11.6_

- [ ] 22. Checkpoint — Phase 3 / RAG complete
  - Ensure all tests pass. `panther index` indexes the project; agent prompts include retrieved context. Ask the user if questions arise.

- [ ] 23. Spec state machine and file format
  - [ ] 23.1 Implement SpecWorkflow state machine
    - Create `src/spec/stateMachine.ts`
    - Implement states: `idle → requirements → design → tasks → execution → done`
    - Require explicit `approve()` call to advance between states; reject out-of-order transitions
    - Write each stage's output to `.panther/specs/<spec-id>/<stage>.md` before requesting approval
    - _Requirements: 6.1, 6.2, 6.3_
  - [ ]* 23.2 Write property test for spec state machine
    - **Property 10: SpecWorkflow state transitions are strictly ordered**
    - **Validates: Requirements 6.1, 6.3**
  - [ ]* 23.3 Write unit tests for state machine
    - All valid transitions write the correct markdown file; invalid transitions are rejected
    - _Requirements: 6.2, 6.3_

- [ ] 24. Requirements stage
  - [ ] 24.1 Implement requirements generation
    - Create `src/spec/requirements.ts`
    - Implement system prompt that produces numbered EARS-formatted requirements
    - Present draft to user as editable block (open `$EDITOR` or inline Ink editing)
    - User types `approve` or edits inline to proceed; edited version used as next-stage input
    - _Requirements: 6.1, 6.4, 6.6_
  - [ ]* 24.2 Write golden-file test for requirements stage
    - Fixed task input + mocked model → output parses as structurally valid EARS requirements
    - _Requirements: 6.4_

- [ ] 25. Design and tasks stages
  - [ ] 25.1 Implement design generation
    - Create `src/spec/design.ts`
    - Design prompt includes `requirements.md` content plus relevant RAG context
    - Stage writes `design.md` and awaits user approval
    - _Requirements: 6.1, 6.2, 6.3, 6.6_
  - [ ] 25.2 Implement task list generation
    - Create `src/spec/tasks.ts`
    - Tasks prompt includes `design.md` and `requirements.md` content
    - Stage writes `tasks.md` with checkbox items and awaits user approval
    - _Requirements: 6.1, 6.2, 6.3, 6.6_
  - [ ]* 25.3 Write golden-file tests for design and tasks stages
    - Fixed inputs + mocked model → `design.md` contains required section headings; `tasks.md` contains checkbox items
    - _Requirements: 6.2_

- [ ] 26. Execution stage
  - [ ] 26.1 Implement spec execution
    - Create `src/spec/execution.ts`
    - Parse `tasks.md` for unchecked checkbox items
    - For each unchecked task, run agent loop with that task as the objective
    - After each task completes, mark its checkbox in `tasks.md` and append a memory entry
    - _Requirements: 6.1, 6.5, 7.5_
  - [ ]* 26.2 Write integration test for execution stage
    - 3-task `tasks.md` with mocked model completing each task; assert all boxes ticked and memory has 3 entries
    - _Requirements: 6.5, 7.5_

- [ ] 27. Checkpoint — Phase 4 / Spec-driven workflow complete
  - Ensure all tests pass. Full requirements → design → tasks → execution cycle works end-to-end with mocked model. Ask the user if questions arise.

- [ ] 28. Skill discovery and metadata loading
  - [ ] 28.1 Implement skill discovery
    - Create `src/skills/discovery.ts`
    - Scan `.panther/skills/` and `~/.panther/skills/` for `SKILL.md` files at session start
    - Parse YAML frontmatter: `name`, `description`, `triggers`, `tools`, `preApproved`
    - Return array of `SkillMeta` without loading full body
    - Add `panther skills list` CLI command displaying name and description per skill
    - _Requirements: 8.1, 8.2, 8.5_
  - [ ]* 28.2 Write unit tests for skill discovery
    - 3 temp `SKILL.md` files → all 3 discovered with correct metadata; full body not read
    - _Requirements: 8.1, 8.2_

- [ ] 29. Lazy skill loading and context injection
  - [ ] 29.1 Implement lazy skill loader
    - Create `src/skills/loader.ts`
    - Implement `loadRelevantSkills(task, ctx)`: substring/glob match against `description` and `triggers`
    - Load full `SKILL.md` body only for matches
    - Inject matched skill bodies into system prompt ahead of task instruction
    - _Requirements: 8.3_
  - [ ]* 29.2 Write property test for lazy skill loading
    - **Property 5: Skill body loaded only on match**
    - **Validates: Requirements 8.3**

- [ ] 30. Skill-registered tools
  - [ ] 30.1 Implement skill tool registrar
    - Create `src/skills/toolRegistrar.ts`
    - Parse `tools` array from `SKILL.md` frontmatter on skill load
    - Reference scripts in skill folder and register each as a `ToolExecutor` in the session tool registry
    - Implement `preApproved: true` frontmatter bypass in the permission gate (pre-approved skill tools skip prompt)
    - _Requirements: 8.4_
  - [ ]* 30.2 Write unit tests for skill tool registration
    - Skill with `run_linter` in `tools` → tool available after load, absent without it
    - Pre-approved skill tool bypasses permission prompt
    - _Requirements: 8.4_

- [ ] 31. Checkpoint — Phase 5 / Skills complete
  - Ensure all tests pass. Skills are discovered, lazy-loaded, and inject context + tools. Ask the user if questions arise.

- [ ] 32. ModelProvider interface and LocalProvider refactor
  - [ ] 32.1 Formalize ModelProvider interface
    - Create `src/providers/types.ts` with `ModelProvider` interface
    - Create `src/providers/local.ts` implementing `ModelProvider` (refactored from `src/inference/localProvider.ts`)
    - Create `src/providers/index.ts` with `resolveProvider(config, override?)` factory
    - Refactor agent loop to import `ModelProvider`, not `LocalProvider` directly
    - _Requirements: 12.1, 12.2_
  - [ ]* 32.2 Verify existing agent loop tests still pass after refactor
    - No test changes required; interface is behaviorally identical

- [ ] 33. Secure credential store
  - [ ] 33.1 Implement keychain credential store
    - Install `cross-keychain` (replaces deprecated `keytar`; wraps `@napi-rs/keyring` with a CLI fallback)
    - Create `src/config/credentials.ts`
    - Implement `addCredential(provider, key)`, `removeCredential(provider)`, `hasCredential(provider)`
    - Service name: `panther-cli`; account name: provider name
    - Add `panther auth add <provider>`, `panther auth remove <provider>`, `panther auth list` CLI commands
    - `panther auth list` shows configured providers without revealing key values
    - _Requirements: 12.3, 12.4, 12.5_
  - [ ]* 33.2 Write property test for credential isolation
    - **Property 9: Credential never written to config file**
    - **Validates: Requirements 12.3**
  - [ ]* 33.3 Write unit tests for credential store (mocked keyring)
    - `addCredential` writes to keychain; `removeCredential` deletes; `auth list` omits key values
    - _Requirements: 12.3, 12.4, 12.5_

- [ ] 34. OpenAI-compatible remote provider
  - [ ] 34.1 Implement OpenAI provider
    - Install `openai` SDK
    - Create `src/providers/openai.ts` implementing `ModelProvider`
    - Implement streaming chat completions with tool/function calling
    - Normalise tool calls to internal `ToolCall` shape
    - Surface token usage back to `SessionContext`
    - _Requirements: 12.1, 12.6_
  - [ ]* 34.2 Write unit tests for OpenAI provider (mocked SDK)
    - Tool call normalised correctly; token usage captured; auth error returns descriptive message
    - _Requirements: 12.6, 12.7_

- [ ] 35. Anthropic and Gemini providers
  - [ ] 35.1 Implement Anthropic provider
    - Install `@anthropic-ai/sdk`
    - Create `src/providers/anthropic.ts` implementing `ModelProvider`
    - Normalise tool calls and surface token usage
    - _Requirements: 12.1, 12.6_
  - [ ] 35.2 Implement Gemini provider
    - Install `@google/genai`
    - Create `src/providers/gemini.ts` implementing `ModelProvider`
    - Normalise tool calls and surface token usage
    - _Requirements: 12.1, 12.6_
  - [ ]* 35.3 Write unit tests for Anthropic and Gemini providers (mocked SDKs)
    - Tool calls normalised correctly; token usage captured for both providers
    - _Requirements: 12.6_

- [ ] 36. Checkpoint — Phase 6 / BYOK complete
  - Ensure all tests pass. `panther run --model gpt-4o-mini` (with mocked SDK) completes a task through the OpenAI provider. Ask the user if questions arise.

- [ ] 37. Full permission config UI and per-skill pre-approval
  - [ ] 37.1 Implement permission config command
    - Create `src/permissions/configUI.tsx` with Ink select component for `permissionMode`
    - Add `panther config permissions` CLI command
    - _Requirements: 9.7_
  - [ ]* 37.2 Write Ink render test for permission config UI
    - Simulate each mode selection; assert resulting config has correct value
    - _Requirements: 9.7_

- [ ] 38. Setup wizard
  - [ ] 38.1 Implement first-run setup wizard
    - Create `src/cli/wizard.tsx`
    - Auto-trigger wizard when no config exists on first `panther` run
    - Ink UI steps: model tier select (with hardware disclosure for large tier), permission mode select, telemetry opt-in, optional BYOK setup
    - On completion: download selected model with progress bar, write `config.json`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7_
  - [ ]* 38.2 Write unit tests for setup wizard
    - Simulate each selection combination; assert resulting `config.json` has correct values for all fields
    - Hardware disclosure shown when large tier selected
    - _Requirements: 2.2, 2.5, 2.7_

- [ ] 39. Checkpoint — Phase 7 / Permissions and polish complete
  - Ensure all tests pass. First-run wizard works end-to-end; permission config command updates config correctly. Ask the user if questions arise.

- [ ] 40. Cross-platform native addon verification and npm publish prep
  - Add `postinstall` script validating native addon loading with a clear, actionable error on failure (not a raw stack trace)
  - Pin all native addon versions exactly in `package.json`
  - Set `engines: { "node": ">=18" }` in `package.json`
  - Verify `bin` entry: `{ "panther": "dist/cli/index.js" }`
  - Expand CI matrix to include Node 18 + Node 20 on all three OSes
  - Add `postinstall` smoke test step to CI
  - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5_

- [ ] 41. Nightly CI and release workflow
  - Create `.github/workflows/nightly.yml` running mocked integration suite and smoke tests
  - Create `.github/workflows/release.yml` triggered on `v*` tags: full CI then `npm publish`
  - Add `npm publish --dry-run` step in CI to verify package shape
  - Create `CHANGELOG.md` and configure conventional commits
  - Create `README.md` with full usage documentation and install instructions
  - _Requirements: 14.1, 14.2_

- [ ] 42. Final checkpoint — Release ready
  - Ensure all tests pass on all matrix targets. `npm publish --dry-run` succeeds. Ask the user if questions arise before publishing.

---

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP path
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at phase boundaries
- Property tests (fast-check, 100+ iterations) validate universal correctness properties
- Unit tests validate specific examples, edge cases, and error conditions
- Integration tests use a mocked `ModelProvider` — no real model download required in CI
- Real-model smoke tests run on a self-hosted GPU runner in the nightly workflow
