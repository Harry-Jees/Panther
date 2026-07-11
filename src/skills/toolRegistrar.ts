import { execSync } from "node:child_process";
import { type ToolExecutor } from "../tools/types.js";
import { type SkillMeta } from "./types.js";

/**
 * Parse the tools array from a skill's frontmatter and register them as ToolExecutors.
 * Each tool is a shell command that the skill folder provides a script for.
 */
export function registerSkillTools(
  skills: Array<{ meta: SkillMeta; body: string }>,
  existingTools: ToolExecutor[],
): ToolExecutor[] {
  const toolMap = new Map<string, ToolExecutor>();
  for (const tool of existingTools) toolMap.set(tool.name, tool);

  for (const skill of skills) {
    const skillDir = skill.meta.filePath.replace(/SKILL\.md$/, "");

    for (const toolName of skill.meta.tools) {
      if (!toolMap.has(toolName)) {
        const skillTool: ToolExecutor = {
          name: toolName,
          description: `Custom tool from skill: ${skill.meta.name}`,
          riskLevel: skill.meta.preApproved ? "safe" : "risky",
          schema: {
            type: "object",
            properties: {
              args: {
                type: "string",
                description: "Arguments to pass to the skill's tool script",
              },
            },
          },
          async execute(args) {
            try {
              const scriptPath = `${skillDir}${toolName}`;
              const scriptArgs = (args.args as string) ?? "";
              const output = execSync(`${scriptPath} ${scriptArgs}`, {
                encoding: "utf-8",
                timeout: 30_000,
              });
              return { success: true, data: output.trim() };
            } catch (err) {
              return { success: false, error: `Skill tool failed: ${(err as Error).message}` };
            }
          },
        };
        toolMap.set(toolName, skillTool);
      }
    }
  }

  return Array.from(toolMap.values());
}
