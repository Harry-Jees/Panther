import { readdir, readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import os from "node:os";
import { type SkillMeta } from "./types.js";

/**
 * Parse YAML-like frontmatter from a SKILL.md file.
 * Returns null if no valid frontmatter is found.
 */
function parseFrontmatter(content: string): Partial<SkillMeta> | null {
  const match = content.match(/^---\n([\s\S]*?)\n---\n/);
  if (!match) return null;

  const yaml = match[1];
  const result: Partial<SkillMeta> = {};

  for (const line of yaml.split("\n")) {
    const [key, ...rest] = line.split(":");
    const value = rest.join(":").trim();

    switch (key.trim()) {
      case "name":
        result.name = value;
        break;
      case "description":
        result.description = value;
        break;
      case "triggers":
        try {
          result.triggers = JSON.parse(value.replace(/'/g, '"'));
        } catch {
          result.triggers = [value];
        }
        break;
      case "tools":
        try {
          result.tools = JSON.parse(value.replace(/'/g, '"'));
        } catch {
          result.tools = value ? [value] : [];
        }
        break;
      case "preApproved":
        result.preApproved = value === "true";
        break;
    }
  }

  return Object.keys(result).length > 0 ? result : null;
}

/**
 * Discover SKILL.md files in a given directory (metadata only, no body loading).
 */
async function discoverInDir(dir: string): Promise<SkillMeta[]> {
  const skills: SkillMeta[] = [];

  try {
    const entries = await readdir(dir, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.isDirectory()) {
        const skillDir = path.join(dir, entry.name);
        const skillFile = path.join(skillDir, "SKILL.md");
        if (existsSync(skillFile)) {
          const content = await readFile(skillFile, "utf-8");
          const frontmatter = parseFrontmatter(content);
          if (frontmatter && frontmatter.name) {
            skills.push({
              name: frontmatter.name,
              description: frontmatter.description ?? "",
              triggers: frontmatter.triggers ?? [],
              tools: frontmatter.tools ?? [],
              preApproved: frontmatter.preApproved ?? false,
              filePath: skillFile,
            });
          }
        }
      }
    }
  } catch {
    // Directory doesn't exist or can't be read
  }

  return skills;
}

/**
 * Discover all available skills from project-local and global directories.
 * Returns metadata only (no body content loaded).
 */
export async function discoverSkills(projectRoot: string): Promise<SkillMeta[]> {
  const globalDir = path.join(os.homedir(), ".panther", "skills");
  const projectDir = path.join(projectRoot, ".panther", "skills");

  const [globalSkills, projectSkills] = await Promise.all([
    discoverInDir(globalDir),
    discoverInDir(projectDir),
  ]);

  // Project skills override global skills with the same name
  const skillMap = new Map<string, SkillMeta>();
  for (const skill of globalSkills) skillMap.set(skill.name, skill);
  for (const skill of projectSkills) skillMap.set(skill.name, skill);

  return Array.from(skillMap.values());
}
