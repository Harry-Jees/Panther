import { type ModelProvider, type Message } from "../inference/types.js";
import { SpecWorkflow } from "./stateMachine.js";

const REQUIREMENTS_SYSTEM_PROMPT = `You are a product manager creating requirements in EARS (Easy Approach to Requirements Syntax) notation.

Given a task description, produce numbered requirements in EARS format using:
- "THE System SHALL ..." for mandatory behavior
- "WHEN ... THEN ..." for conditional behavior
- "WHERE ... THE System SHALL ..." for state-based behavior
- "IF ... THEN ..." for exception behavior
- "WHILE ... THE System SHALL ..." for continuous behavior

Output only the requirements, one per line, numbered.`;

/**
 * Generate requirements for a task using the model.
 */
export async function generateRequirements(
  task: string,
  model: ModelProvider,
  workflow: SpecWorkflow,
): Promise<string> {
  const history: Message[] = [
    { role: "system", content: REQUIREMENTS_SYSTEM_PROMPT },
    { role: "user", content: `Task: ${task}\n\nGenerate EARS requirements for this task.` },
  ];

  let requirements = "";
  for await (const chunk of model.generate(history)) {
    if (chunk.type === "complete") {
      requirements = chunk.text;
    }
  }

  // Format as markdown
  const content = [
    `# Requirements Document — ${workflow.specId}`,
    "",
    `## Task`,
    `\`\`\``,
    task,
    `\`\`\``,
    "",
    `## Requirements`,
    "",
    requirements,
    "",
  ].join("\n");

  await workflow.completeStage(content);
  return content;
}
