import { type ModelProvider } from "../inference/types.js";
import { type PantherConfig } from "../config/types.js";

/**
 * Result of a tool execution.
 */
export interface ToolResult {
  success: boolean;
  data?: string;
  error?: string;
  meta?: Record<string, unknown>;
}

/**
 * JSON Schema for a tool's parameters.
 */
export interface JSONSchema {
  type: string;
  properties?: Record<string, unknown>;
  required?: string[];
  [key: string]: unknown;
}

/**
 * A discrete, independently-testable tool that the agent can call.
 */
export interface ToolExecutor {
  readonly name: string;
  readonly description: string;
  readonly riskLevel: "safe" | "risky";
  readonly schema: JSONSchema;
  execute(args: Record<string, unknown>, ctx: SessionContext): Promise<ToolResult>;
}

/**
 * Runtime container for the session's tools, config, and model.
 */
export interface SessionContext {
  projectRoot: string;
  config: PantherConfig;
  model: ModelProvider;
  getTool(name: string): ToolExecutor | undefined;
  availableTools: ToolExecutor[];
}
