import { readFile, writeFile, mkdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import os from "node:os";
import { PantherConfigSchema, type PantherConfig, DEFAULT_CONFIG } from "./types.js";

interface FileError {
  code?: string;
}

/**
 * Get the global config directory (~/.panther).
 */
function getGlobalConfigDir(): string {
  return path.join(os.homedir(), ".panther");
}

/**
 * Get the global config file path.
 */
function getGlobalConfigPath(): string {
  return path.join(getGlobalConfigDir(), "config.json");
}

/**
 * Get the project-level config file path.
 */
function getProjectConfigPath(projectRoot: string): string {
  return path.join(projectRoot, ".panther", "config.json");
}

/**
 * Load config from a JSON file path. Returns null if file doesn't exist or is malformed.
 */
async function loadConfigFile(filePath: string): Promise<Partial<PantherConfig> | null> {
  try {
    const raw = await readFile(filePath, "utf-8");
    const parsed = JSON.parse(raw);
    const result = PantherConfigSchema.partial().safeParse(parsed);
    if (!result.success) {
      console.warn(`Warning: Config file ${filePath} has invalid fields, using defaults for those fields`);
      return null;
    }
    return result.data;
  } catch (err) {
    const nodeError = err as FileError;
    if (nodeError.code === "ENOENT") {
      return null;
    }
    console.warn(`Warning: Could not parse config file ${filePath}, falling back to defaults`);
    return null;
  }
}

/**
 * Merge global config, project config, and defaults.
 * Project config values take precedence over global, which take precedence over defaults.
 */
function mergeConfig(
  defaults: PantherConfig,
  global?: Partial<PantherConfig> | null,
  project?: Partial<PantherConfig> | null,
): PantherConfig {
  return { ...defaults, ...global, ...project };
}

/**
 * Load and merge config from global and project-level config files.
 * Returns the merged config with built-in defaults for any missing fields.
 */
export async function loadConfig(projectRoot?: string): Promise<PantherConfig> {
  const globalConfig = await loadConfigFile(getGlobalConfigPath());

  let projectConfig: Partial<PantherConfig> | null = null;
  if (projectRoot) {
    projectConfig = await loadConfigFile(getProjectConfigPath(projectRoot));
  }

  return mergeConfig(DEFAULT_CONFIG, globalConfig, projectConfig);
}

/**
 * Save config to a project-level or global config file.
 */
export async function saveConfig(
  config: PantherConfig,
  projectRoot?: string,
): Promise<void> {
  const configDir = projectRoot ? getProjectConfigPath(projectRoot) : getGlobalConfigPath();
  const dir = path.dirname(configDir);

  if (!existsSync(dir)) {
    await mkdir(dir, { recursive: true });
  }

  // Serialize only fields that differ from defaults to keep the config file clean.
  // If all values are defaults, write an empty object to stay minimal.
  const validated = PantherConfigSchema.parse(config);
  const overrides: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(validated)) {
    if (value !== DEFAULT_CONFIG[key as keyof PantherConfig]) {
      overrides[key] = value;
    }
  }
  const serialized = Object.keys(overrides).length > 0 ? overrides : {};
  await writeFile(configDir, JSON.stringify(serialized, null, 2), "utf-8");
}

/**
 * Synchronous utility to get the global config directory path.
 */
export function getPantherDir(): string {
  return getGlobalConfigDir();
}
