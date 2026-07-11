import { type ModelProvider, type Message, type ToolSchema } from "../inference/types.js";
import { type ToolExecutor, type ToolResult, type SessionContext } from "../tools/types.js";
import { type PermissionGate, checkOneShot } from "../permissions/gate.js";
import { type PermissionMode } from "../config/types.js";

/**
 * Configuration for the agent loop.
 */
export interface AgentLoopConfig {
  isOneShot: boolean;
  permissionMode: PermissionMode;
  permissionGate: PermissionGate;
  onToolCall?: (tool: ToolExecutor, args: Record<string, unknown>) => void;
  onToolResult?: (result: ToolResult) => void;
}

/**
 * Run the agent loop: model → tool call → execute → repeat.
 *
 * @param task - The user's task description
 * @param model - The model provider to use
 * @param tools - Available tools
 * @param config - Loop configuration
 * @param ctx - Session context
 * @returns The final response text
 */
export async function runAgentLoop(
  task: string,
  model: ModelProvider,
  tools: ToolExecutor[],
  config: AgentLoopConfig,
  ctx: SessionContext,
): Promise<string> {
  const history: Message[] = [
    { role: "system", content: "You are Panther, a helpful coding assistant." },
    { role: "user", content: task },
  ];

  const toolSchemas: ToolSchema[] = tools.map((t) => ({
    name: t.name,
    description: t.description,
    parameters: t.schema as unknown as Record<string, unknown>,
  }));

  const maxIterations = 50; // safety limit to prevent infinite loops
  for (let iteration = 0; iteration < maxIterations; iteration++) {
    // Generate response from model
    let modelResponse = "";
    for await (const chunk of model.generate(history, toolSchemas)) {
      if (chunk.type === "complete") {
        modelResponse = chunk.text;
        history.push({ role: "assistant", content: chunk.text });
      }
    }

    // Try to parse a function call from the response
    const functionCall = tryParseFunctionCall(modelResponse, tools);
    if (!functionCall) {
      // No function call — this is the final response
      return modelResponse;
    }

    // Check permissions
    const decision = config.isOneShot
      ? checkOneShot(config.permissionGate, functionCall.tool, config.permissionMode)
      : config.permissionGate.check(functionCall.tool, config.permissionMode);

    config.onToolCall?.(functionCall.tool, functionCall.args);

    if (decision === "denied") {
      history.push({
        role: "system",
        content: `Tool "${functionCall.tool.name}" was denied by the permission gate. Propose an alternative approach.`,
      });
      continue;
    }

    if (decision === "needs_prompt") {
      // In one-shot mode this won't happen (checkOneShot returns denied)
      // In interactive mode, we'd show a prompt here
      history.push({
        role: "system",
        content: `Tool "${functionCall.tool.name}" requires approval. Please confirm or reject:`,
      });
      return `[NEEDS_APPROVAL] Tool: ${functionCall.tool.name}\nArgs: ${JSON.stringify(functionCall.args, null, 2)}`;
    }

    // Execute the tool
    try {
      const result = await functionCall.tool.execute(functionCall.args, ctx);
      config.onToolResult?.(result);

      history.push({
        role: "tool",
        content: result.success ? result.data ?? "(completed)" : `Error: ${result.error}`,
        toolName: functionCall.tool.name,
      });
    } catch (err) {
      history.push({
        role: "tool",
        content: `Unexpected error: ${(err as Error).message}`,
        toolName: functionCall.tool.name,
      });
    }
  }

  return "Agent loop reached maximum iterations without completing the task.";
}

/**
 * Try to parse a function call from the model's response text.
 * Looks for a function call pattern in the text.
 */
function tryParseFunctionCall(
  text: string,
  tools: ToolExecutor[],
): { tool: ToolExecutor; args: Record<string, unknown> } | null {
  // Try to parse JSON function call format: {"name": "...", "arguments": {...}}
  try {
    const parsed = JSON.parse(text);
    if (parsed.name && tools.some((t) => t.name === parsed.name)) {
      const tool = tools.find((t) => t.name === parsed.name)!;
      return { tool, args: typeof parsed.arguments === "object" && parsed.arguments ? parsed.arguments as Record<string, unknown> : {} };
    }
  } catch {
    // Not JSON, try other formats
  }

  // Try to find function call in code blocks: <function=name>{"arg": "value"}</function>
  const funcMatch = text.match(/<function=(\w+)>([\s\S]*?)<\/function>/);
  if (funcMatch) {
    const toolName = funcMatch[1];
    const tool = tools.find((t) => t.name === toolName);
    if (tool) {
      try {
        const args = JSON.parse(funcMatch[2]);
        return { tool, args: args as Record<string, unknown> };
      } catch {
        return { tool, args: {} };
      }
    }
  }

  // Try to find function call in markdown format: - tool_name: {"arg": "value"}
  const mdMatch = text.match(/^- (\w+):\s*(\{.*\})$/m);
  if (mdMatch) {
    const toolName = mdMatch[1];
    const tool = tools.find((t) => t.name === toolName);
    if (tool) {
      try {
        const args = JSON.parse(mdMatch[2]);
        return { tool, args: args as Record<string, unknown> };
      } catch {
        return { tool, args: {} };
      }
    }
  }

  return null;
}
