import { type ModelProvider, type Message } from "../inference/types.js";
import { SpecWorkflow } from "./stateMachine.js";

const DESIGN_SYSTEM_PROMPT = `You are a software architect. Given requirements, produce a design document with:
1. Overview / approach
2. Architecture diagram (ASCII)
3. Component breakdown
4. Data flow
5. Key interfaces
6. Open questions`;

/**
 * Generate a design document from requirements using the model.
 */
export async function generateDesign(
  requirements: string,
  model: ModelProvider,
  workflow: SpecWorkflow,
): Promise<string> {
  const history: Message[] = [
    { role: "system", content: DESIGN_SYSTEM_PROMPT },
    { role: "user", content: `Requirements:\n\n${requirements}\n\nGenerate a design document.` },
  ];

  let design = "";
  for await (const chunk of model.generate(history)) {
    if (chunk.type === "complete") {
      design = chunk.text;
    }
  }

  const content = [
    `# Design Document — ${workflow.specId}`,
    "",
    design,
    "",
  ].join("\n");

  await workflow.completeStage(content);
  return content;
}
