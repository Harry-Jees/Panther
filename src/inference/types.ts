import { type PantherConfig } from "../config/types.js";

/**
 * A single message in the conversation history.
 */
export interface Message {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  toolCallId?: string;
  toolName?: string;
}

/**
 * Schema for a tool that the model can call.
 */
export interface ToolSchema {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
}

/**
 * A chunk of generated tokens.
 */
export interface TokenChunk {
  type: "token";
  text: string;
}

/**
 * A tool call requested by the model.
 */
export interface ToolCall {
  type: "tool_call";
  toolName: string;
  args: Record<string, unknown>;
  id: string;
}

/**
 * A completion signal from the model.
 */
export interface CompletionSignal {
  type: "complete";
  text: string;
}

/**
 * Union type for model output chunks.
 */
export type ModelOutput = TokenChunk | ToolCall | CompletionSignal;

/**
 * Metadata about a loaded model.
 */
export interface ModelMetadata {
  name: string;
  tier: "small" | "large";
  contextWindow: number;
  supportsGrammar: boolean;
  backend: "metal" | "cuda" | "vulkan" | "cpu";
}

/**
 * The single abstraction between the agent loop and the inference backend.
 */
export interface ModelProvider {
  readonly name: "local" | "openai" | "anthropic" | "gemini";
  readonly metadata: ModelMetadata;
  generate(
    history: Message[],
    tools?: ToolSchema[],
  ): AsyncIterable<ModelOutput>;
  embedText?(text: string): Promise<number[]>;
}

/**
 * Resolve a ModelProvider based on config and optional override.
 * For Phase 0, this always returns "local".
 */
export function resolveProvider(
  config: PantherConfig,
  _override?: string,
): string {
  const requested = _override ?? config.activeProvider;
  if (requested !== "local") {
    throw new Error(
      `No API key configured for ${requested}. Run: panther auth add ${requested}`,
    );
  }
  return "local";
}
