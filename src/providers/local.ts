import { LocalProvider as CoreLocalProvider } from "../inference/localProvider.js";
import { type ModelProvider } from "./types.js";
import { type Message, type ToolSchema, type ModelOutput, type ModelMetadata } from "../inference/types.js";

/**
 * Wraps the core LocalProvider into the ModelProvider interface.
 */
export class LocalProvider implements ModelProvider {
  readonly name = "local" as const;
  readonly metadata: ModelMetadata;
  private core: CoreLocalProvider;

  constructor(core: CoreLocalProvider, metadata: ModelMetadata) {
    this.core = core;
    this.metadata = metadata;
  }

  static async create(modelPath: string): Promise<LocalProvider> {
    const core = await CoreLocalProvider.create(modelPath);
    return new LocalProvider(core, core.metadata);
  }

  async *generate(history: Message[], tools?: ToolSchema[]): AsyncIterable<ModelOutput> {
    yield* this.core.generate(history, tools);
  }

  async dispose(): Promise<void> {
    await this.core.dispose();
  }
}
