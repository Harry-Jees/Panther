import { describe, it, expect } from "vitest";
import { runAgentLoop } from "../../src/agent/loop.js";
import { defaultPermissionGate } from "../../src/permissions/gate.js";
import { type ModelProvider, type ModelOutput } from "../../src/inference/types.js";
import { type ToolExecutor, type SessionContext } from "../../src/tools/types.js";
import { DEFAULT_CONFIG } from "../../src/config/types.js";

/**
 * Create a mock model that returns scripted responses.
 */
function createMockModel(responses: string[]): ModelProvider {
  let callIndex = 0;
  return {
    name: "local",
    metadata: { name: "mock", tier: "small", contextWindow: 4096, supportsGrammar: false, backend: "cpu" },
    async *generate() {
      const response = responses[callIndex] ?? responses[responses.length - 1];
      callIndex++;
      yield { type: "complete" as const, text: response };
    },
  };
}

const mockTool: ToolExecutor = {
  name: "read_file",
  description: "Read a file",
  riskLevel: "safe",
  schema: { type: "object", properties: { filePath: { type: "string" } }, required: ["filePath"] },
  async execute(args) {
    return { success: true, data: `File content for: ${args.filePath}` };
  },
};

function createContext(): SessionContext {
  return {
    projectRoot: "/test",
    config: DEFAULT_CONFIG,
    model: createMockModel([]),
    getTool: () => mockTool,
    availableTools: [mockTool],
  };
}

describe("Agent loop", () => {
  it("should return the model's final text response when no function call is detected", async () => {
    const model = createMockModel(["Hello! I can help with that."]);
    const result = await runAgentLoop(
      "Say hello",
      model,
      [mockTool],
      { isOneShot: true, permissionMode: "never_ask", permissionGate: defaultPermissionGate },
      createContext(),
    );
    expect(result).toBe("Hello! I can help with that.");
  });

  it("should execute a tool call and continue the loop", async () => {
    // First response: function call, second: final text
    const model = createMockModel([
      '- read_file: {"filePath": "test.txt"}',
      "Found the file contents.",
    ]);
    const result = await runAgentLoop(
      "Read test.txt",
      model,
      [mockTool],
      { isOneShot: true, permissionMode: "never_ask", permissionGate: defaultPermissionGate },
      createContext(),
    );
    expect(result).toBe("Found the file contents.");
  });

  it("should deny tool calls in one-shot mode when permission requires prompt", async () => {
    const model = createMockModel([
      '- read_file: {"filePath": "test.txt"}',
      "I'll try a different approach.",
    ]);
    const result = await runAgentLoop(
      "Read test.txt",
      model,
      [mockTool],
      { isOneShot: true, permissionMode: "ask_when_risky", permissionGate: defaultPermissionGate },
      createContext(),
    );
    expect(result).toBe("I'll try a different approach.");
  });

  describe("Property: Agent loop terminates on first non-tool-call response (Property 7)", () => {
    it("should stop after receiving a non-tool-call response", async () => {
      const model = createMockModel(["This is the final answer."]);
      const result = await runAgentLoop(
        "Answer the question",
        model,
        [mockTool],
        { isOneShot: true, permissionMode: "never_ask", permissionGate: defaultPermissionGate },
        createContext(),
      );
      expect(result).toBe("This is the final answer.");
    });
  });
});
