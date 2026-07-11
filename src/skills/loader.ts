import { readFile } from "node:fs/promises";
import { type SkillMeta, type LoadedSkill } from "./types.js";

/**
 * Check if a skill's description or triggers match the current task.
 */
function matchesTask(skill: SkillMeta, task: string): boolean {
  const taskLower = task.toLowerCase();

  // Check description
  if (skill.description && taskLower.includes(skill.description.toLowerCase())) {
    return true;
  }

  // Check triggers
  for (const trigger of skill.triggers) {
    if (trigger.startsWith("*.")) {
      // Glob pattern - check if task references files with this extension
      const ext = trigger.slice(1);
      if (taskLower.includes(ext)) return true;
    } else if (taskLower.includes(trigger.toLowerCase())) {
      return true;
    }
  }

  return false;
}

/**
 * Load the full body of a SKILL.md file.
 */
async function loadSkillBody(skill: SkillMeta): Promise<LoadedSkill> {
  const content = await readFile(skill.filePath, "utf-8");

  // Remove frontmatter
  const body = content.replace(/^---\n[\s\S]*?\n---\n/, "").trim();

  return { meta: skill, body };
}

/**
 * Load skills relevant to the current task.
 * Only loads full body for skills whose description or triggers match.
 */
export async function loadRelevantSkills(
  skills: SkillMeta[],
  task: string,
): Promise<LoadedSkill[]> {
  const matched = skills.filter((s) => matchesTask(s, task));
  return Promise.all(matched.map(loadSkillBody));
}

/**
 * Format loaded skills as context for the system prompt.
 */
export function formatSkillContext(loadedSkills: LoadedSkill[]): string {
  if (loadedSkills.length === 0) return "";

  return [
    "<skills>",
    ...loadedSkills.map((skill) => [
      `## Skill: ${skill.meta.name}`,
      skill.body,
    ].join("\n")),
    "</skills>",
  ].join("\n\n");
}
