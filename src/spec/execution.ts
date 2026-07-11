import { readFile, writeFile } from "node:fs/promises";
import { type ModelProvider, type Message } from "../inference/types.js";
import { type ToolExecutor, type SessionContext } from "../tools/types.js";
import { MemorySystem } from "../memory/index.js";
import { SpecWorkflow } from "./stateMachine.js";
import { runAgentLoop } from "../agent/loop.js";
import { defaultPermissionGate } from "../permissions/gate.js";

/**
 * Execute all unchecked tasks in the tasks.md file sequentially.
 */
export async function executeSpec(
  workflow: SpecWorkflow,
  model: ModelProvider,
  tools: ToolExecutor[],
  ctx: SessionContext,
): Promise<void> {
  const tasksContent = await workflow.getStageContent("tasks");
  if (!tasksContent) {
    throw new Error("No tasks.md found — run the tasks stage first");
  }

  const lines = tasksContent.split("\n");
  const uncheckedTasks: Array<{ index: number; line: string; description: string }> = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.trimStart().startsWith("- [ ]")) {
      uncheckedTasks.push({
        index: i,
        line,
        description: line.replace("- [ ]", "").trim(),
      });
    }
  }

  if (uncheckedTasks.length === 0) {
    throw new Error("No unchecked tasks found in tasks.md");
  }

  const memory = new MemorySystem(ctx.projectRoot);
  await memory.initialize();

  for (const task of uncheckedTasks) {
    console.log(`\n▶ Executing: ${task.description}`);

    const result = await runAgentLoop(
      task.description,
      model,
      tools,
      {
        isOneShot: false,
        permissionMode: ctx.config.permissionMode,
        permissionGate: defaultPermissionGate,
      },
      ctx,
    );

    // Mark task as complete
    lines[task.index] = lines[task.index].replace("- [ ]", "- [x]");

    // Log to memory
    await memory.appendSessionLog({
      summary: `Completed task: ${task.description}`,
      filesTouched: [],
      decisions: [`Task completed with result: ${result.slice(0, 200)}`],
      followUps: [],
    });

    console.log(`✅ Completed: ${task.description}`);
  }

  // Write updated tasks.md
  const updatedContent = lines.join("\n");
  const tasksPath = workflow.getStagePath("tasks");
  await writeFile(tasksPath, updatedContent, "utf-8");

  // Advance to done
  await workflow.completeStage("All tasks completed");
}
