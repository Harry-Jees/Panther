import { type Message, type ToolSchema, type ModelOutput, type ModelMetadata } from "../inference/types.js";

/**
 * Provider-agnostic ModelProvider interface.
 * Each implementation handles its own API calls and normalizes output.
 */
export interface ModelProvider {
  readonly name: "local" | "openai" | "anthropic" | "gemini";
  readonly metadata: ModelMetadata;
  generate(history: Message[], tools?: ToolSchema[]): AsyncIterable<ModelOutput>;
  embedText?(text: string): Promise<number[]>;
}
