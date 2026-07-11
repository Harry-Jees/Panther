// @ts-nocheck — Requires: npm install openai
import OpenAI from "openai";
import { type ModelProvider } from "./types.js";
import { type Message, type ToolSchema, type ModelOutput, type ModelMetadata } from "../inference/types.js";

export class OpenAIProvider implements ModelProvider {
  readonly name = "openai" as const;
  readonly metadata: ModelMetadata;
  private client: OpenAI;
  private modelName: string;

  constructor(apiKey: string, modelName: string = "gpt-4o-mini", baseURL?: string) {
    this.client = new OpenAI({
      apiKey,
      ...(baseURL ? { baseURL } : {}),
    });
    this.modelName = modelName;
    this.metadata = {
      name: modelName,
      tier: "large",
      contextWindow: 128000,
      supportsGrammar: false,
      backend: "cpu",
    };
  }

  async *generate(history: Message[], tools?: ToolSchema[]): AsyncIterable<ModelOutput> {
    const messages = history.map((m) => ({
      role: m.role as "system" | "user" | "assistant" | "tool",
      content: m.content,
    }));

    const stream = await this.client.chat.completions.create({
      model: this.modelName,
      messages,
      ...(tools && tools.length > 0
        ? {
            tools: tools.map((t) => ({
              type: "function" as const,
              function: {
                name: t.name,
                description: t.description,
                parameters: t.parameters as Record<string, unknown>,
              },
            })),
            tool_choice: "auto" as const,
          }
        : {}),
      stream: true,
    });

    let fullText = "";
    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta;
      if (delta?.content) {
        fullText += delta.content;
        yield { type: "token" as const, text: delta.content };
      }
      if (delta?.tool_calls) {
        for (const toolCall of delta.tool_calls) {
          if (toolCall.function?.name) {
            yield {
              type: "tool_call" as const,
              toolName: toolCall.function.name,
              args: JSON.parse(toolCall.function.arguments || "{}"),
              id: toolCall.id ?? "",
            };
          }
        }
      }
    }

    yield { type: "complete" as const, text: fullText };
  }
}
