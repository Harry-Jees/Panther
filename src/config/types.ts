import { z } from "zod";

export const PermissionMode = z.enum(["always_ask", "ask_when_risky", "never_ask"]);
export type PermissionMode = z.infer<typeof PermissionMode>;

export const ModelTier = z.enum(["small", "large"]);
export type ModelTier = z.infer<typeof ModelTier>;

export const ModelFamily = z.enum(["qwen"]);
export type ModelFamily = z.infer<typeof ModelFamily>;

export const ProviderName = z.enum(["local", "openai", "anthropic", "gemini"]);
export type ProviderName = z.infer<typeof ProviderName>;

export const PantherConfigSchema = z.object({
  permissionMode: PermissionMode.default("ask_when_risky"),
  modelTier: ModelTier.default("large"),
  modelFamily: ModelFamily.default("qwen"),
  activeProvider: ProviderName.default("local"),
  telemetryOptIn: z.boolean().default(false),
});

export type PantherConfig = z.infer<typeof PantherConfigSchema>;

export const DEFAULT_CONFIG: PantherConfig = {
  permissionMode: "ask_when_risky",
  modelTier: "large",
  modelFamily: "qwen",
  activeProvider: "local",
  telemetryOptIn: false,
};
