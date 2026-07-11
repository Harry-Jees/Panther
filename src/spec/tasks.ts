import { type ModelProvider, type Message } from "../inference/types.js";
import { SpecWorkflow } from "./stateMachine.js";

const TASKS_SYSTEM_PROMPT = `You are a project manager breaking down a design into ordered, checkable tasks.
Output tasks as a markdown checklist:
- [ ] Task description (with file paths and expected outcome)
Each task should be independently verifiable and reference specific files.`;

/**
 * Generate a task list from the design document using the model.
 */
export async function generateTasks(
  design: string,
  model: ModelProvider,
  workflow: SpecWorkflow,
): Promise<string> {
  const history: Message[] = [
    { role: "system", content: TASKS_SYSTEM_PROMPT },
    { role: "user", content: `Design:\n\n${design}\n\nGenerate a task list.` },
  ];

  let tasks = "";
  for await (const chunk of model.generate(history)) {
    if (chunk.type === "complete") {
      tasks = chunk.text;
    }
  }

  const content = [
    `# Implementation Tasks — ${workflow.specId}`,
    "",
    tasks,
    "",
  ].join("\n");

  await workflow.completeStage(content);
  return content;
}
