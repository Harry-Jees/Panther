// @ts-nocheck — Requires: npm install @google/genai
import { GoogleGenAI } from "@google/genai";
import { type ModelProvider } from "./types.js";
import { type Message, type ToolSchema, type ModelOutput, type ModelMetadata } from "../inference/types.js";

export class GeminiProvider implements ModelProvider {
  readonly name = "gemini" as const;
  readonly metadata: ModelMetadata;
  private client: GoogleGenAI;
  private modelName: string;

  constructor(apiKey: string, modelName: string = "gemini-2.5-flash") {
    this.client = new GoogleGenAI({ apiKey });
    this.modelName = modelName;
    this.metadata = {
      name: modelName,
      tier: "large",
      contextWindow: 1048576,
      supportsGrammar: false,
      backend: "cpu",
    };
  }

  async *generate(history: Message[], tools?: ToolSchema[]): AsyncIterable<ModelOutput> {
    const systemPrompt = history.find((m) => m.role === "system")?.content ?? "";
    const contents = history
      .filter((m) => m.role !== "system")
      .map((m) => ({
        role: m.role === "assistant" ? "model" as const : "user" as const,
        parts: [{ text: m.content }],
      }));

    const result = await this.client.models.generateContent({
      model: this.modelName,
      contents,
      config: {
        systemInstruction: systemPrompt ? { parts: [{ text: systemPrompt }] } : undefined,
        tools: tools?.map((t) => ({
          functionDeclarations: [{
            name: t.name,
            description: t.description,
            parameters: t.parameters as Record<string, unknown>,
          }],
        })),
      },
    });

    const text = result.text ?? "";
    yield { type: "complete" as const, text };
  }
}
