import { getLlama, type LlamaModel, type LlamaContext, LlamaChatSession } from "node-llama-cpp";
import { type ModelProvider, type Message, type ToolSchema, type ModelOutput, type ModelMetadata } from "./types.js";

/**
 * Local inference provider using node-llama-cpp in-process bindings.
 * Loads a GGUF model and generates responses with optional tool/function calling.
 */
export class LocalProvider implements ModelProvider {
  readonly name = "local" as const;
  readonly metadata: ModelMetadata;
  private model: LlamaModel | null = null;
  private context: LlamaContext | null = null;
  private session: LlamaChatSession | null = null;
  private systemPrompt: string = "";

  private constructor(metadata: ModelMetadata) {
    this.metadata = metadata;
  }

  /**
   * Create a LocalProvider instance, loading the model at the given path.
   */
  static async create(modelPath: string, systemPrompt?: string): Promise<LocalProvider> {
    const startTime = Date.now();
    const llama = await getLlama();

    let backend: ModelMetadata["backend"] = "cpu";
    if (llama.gpu === "metal") backend = "metal";
    else if (llama.gpu === "cuda") backend = "cuda";
    else if (llama.gpu === "vulkan") backend = "vulkan";

    console.log(`[Panther] GPU backend: ${backend}${backend === "cpu" ? " (inference will be slower)" : ""}`);

    const model = await llama.loadModel({
      modelPath,
    });

    const context = await model.createContext();

    // Pass system prompt as part of session construction
    const session = new LlamaChatSession({
      contextSequence: context.getSequence(),
      ...(systemPrompt ? { systemPrompt } : {}),
    });

    const fileName = modelPath.split(/[/\\]/).pop() ?? "unknown";

    const provider = new LocalProvider({
      name: fileName,
      tier: fileName.includes("80B") ? "large" : "small",
      contextWindow: 32768,
      supportsGrammar: true,
      backend,
    });

    provider.model = model;
    provider.context = context;
    provider.session = session;
    provider.systemPrompt = systemPrompt ?? "";

    const loadTime = Date.now() - startTime;
    console.log(`[Panther] Model loaded in ${loadTime}ms`);

    return provider;
  }

  /**
   * Generate a response from the local model.
   * Supports streaming token output and function/tool calling via node-llama-cpp's built-in JSON schema support.
   */
  async *generate(history: Message[], tools?: ToolSchema[]): AsyncIterable<ModelOutput> {
    if (!this.session) {
      throw new Error("LocalProvider not initialized. Call LocalProvider.create() first.");
    }

    const chatHistory = history.filter((m) => m.role !== "system").map((m) => ({
      role: m.role === "tool" ? "assistant" as const : m.role as "user" | "assistant",
      content: m.content,
    }));

    const userPrompt = chatHistory.find((m) => m.role === "user")?.content ?? "";

    try {
      const promptOptions: Record<string, unknown> = {};

      if (tools && tools.length > 0) {
        promptOptions.functions = tools.map((t) => ({
          name: t.name,
          description: t.description,
          parameters: t.parameters,
        }));
      }

      const response = await this.session.prompt(userPrompt, promptOptions);

      yield {
        type: "complete",
        text: response,
      };
    } catch (err) {
      throw new Error(`Model generation failed: ${(err as Error).message}`);
    }
  }

  /**
   * Clean up resources held by the provider.
   */
  async dispose(): Promise<void> {
    this.model = null;
    this.context = null;
    this.session = null;
  }
}
