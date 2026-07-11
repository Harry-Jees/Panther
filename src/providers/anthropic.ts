// @ts-nocheck — Requires: npm install @anthropic-ai/sdk
import Anthropic from "@anthropic-ai/sdk";
import { type ModelProvider } from "./types.js";
import { type Message, type ToolSchema, type ModelOutput, type ModelMetadata } from "../inference/types.js";

export class AnthropicProvider implements ModelProvider {
  readonly name = "anthropic" as const;
  readonly metadata: ModelMetadata;
  private client: Anthropic;
  private modelName: string;

  constructor(apiKey: string, modelName: string = "claude-sonnet-4-20250514") {
    this.client = new Anthropic({ apiKey });
    this.modelName = modelName;
    this.metadata = {
      name: modelName,
      tier: "large",
      contextWindow: 200000,
      supportsGrammar: false,
      backend: "cpu",
    };
  }

  async *generate(history: Message[], tools?: ToolSchema[]): AsyncIterable<ModelOutput> {
    const systemPrompt = history.find((m) => m.role === "system")?.content ?? "";
    const messages = history
      .filter((m) => m.role !== "system")
      .map((m) => ({
        role: m.role === "tool" ? "assistant" as const : m.role as "user" | "assistant",
        content: m.content,
      }));

    const stream = await this.client.messages.create({
      model: this.modelName,
      system: systemPrompt,
      max_tokens: 4096,
      messages,
      ...(tools && tools.length > 0
        ? {
            tools: tools.map((t) => ({
              name: t.name,
              description: t.description,
              input_schema: t.parameters as Record<string, unknown>,
            })),
          }
        : {}),
      stream: true,
    });

    let fullText = "";
    for await (const event of stream) {
      if (event.type === "content_block_delta") {
        if (event.delta.type === "text_delta") {
          fullText += event.delta.text;
          yield { type: "token" as const, text: event.delta.text };
        }
      }
      if (event.type === "content_block_start" && event.content_block.type === "tool_use") {
        yield {
          type: "tool_call" as const,
          toolName: event.content_block.name,
          args: event.content_block.input as Record<string, unknown>,
          id: event.content_block.id,
        };
      }
    }

    yield { type: "complete" as const, text: fullText };
  }
}
