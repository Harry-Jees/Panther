/**
 * Parsed metadata from a SKILL.md frontmatter.
 */
export interface SkillMeta {
  name: string;
  description: string;
  triggers: string[];
  tools: string[];
  preApproved: boolean;
  filePath: string;
}

/**
 * A fully loaded skill with its body content.
 */
export interface LoadedSkill {
  meta: SkillMeta;
  body: string;
}
