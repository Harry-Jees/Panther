import { describe, it, expect } from "vitest";
import fc from "fast-check";
import { defaultPermissionGate, checkOneShot, type PermissionDecision } from "../../src/permissions/gate.js";
import { type PermissionMode } from "../../src/config/types.js";
import { type ToolExecutor } from "../../src/tools/types.js";

const safeTool: ToolExecutor = {
  name: "test_safe",
  description: "A safe tool for testing",
  riskLevel: "safe",
  schema: { type: "object", properties: {} },
  async execute() {
    return { success: true, data: "ok" };
  },
};

const riskyTool: ToolExecutor = {
  name: "test_risky",
  description: "A risky tool for testing",
  riskLevel: "risky",
  schema: { type: "object", properties: {} },
  async execute() {
    return { success: true, data: "ok" };
  },
};

describe("PermissionGate", () => {
  describe("Decision matrix", () => {
    const testCases: Array<{ mode: PermissionMode; risk: "safe" | "risky"; expected: PermissionDecision }> = [
      { mode: "always_ask", risk: "safe", expected: "needs_prompt" },
      { mode: "always_ask", risk: "risky", expected: "needs_prompt" },
      { mode: "ask_when_risky", risk: "safe", expected: "approved" },
      { mode: "ask_when_risky", risk: "risky", expected: "needs_prompt" },
      { mode: "never_ask", risk: "safe", expected: "approved" },
      { mode: "never_ask", risk: "risky", expected: "approved" },
    ];

    for (const { mode, risk, expected } of testCases) {
      it(`should return ${expected} for mode=${mode} risk=${risk}`, () => {
        const tool = risk === "safe" ? safeTool : riskyTool;
        const decision = defaultPermissionGate.check(tool, mode);
        expect(decision).toBe(expected);
      });
    }
  });

  describe("One-shot mode auto-denial", () => {
    it("should deny risky tools in ask_when_risky mode", () => {
      const decision = checkOneShot(defaultPermissionGate, riskyTool, "ask_when_risky");
      expect(decision).toBe("denied");
    });

    it("should deny all tools in always_ask mode", () => {
      const decision = checkOneShot(defaultPermissionGate, safeTool, "always_ask");
      expect(decision).toBe("denied");
    });

    it("should approve all tools in never_ask mode", () => {
      const decision = checkOneShot(defaultPermissionGate, riskyTool, "never_ask");
      expect(decision).toBe("approved");
    });
  });

  describe("Property: Permission gate covers all mode × risk combinations (Property 1)", () => {
    it("should return a valid decision for every combination", () => {
      fc.assert(
        fc.property(
          fc.constantFrom(...(["always_ask", "ask_when_risky", "never_ask"] as PermissionMode[])),
          fc.constantFrom(...(["safe", "risky"] as Array<"safe" | "risky">)),
          (mode, riskLevel) => {
            const tool = riskLevel === "safe" ? safeTool : riskyTool;
            const decision = defaultPermissionGate.check(tool, mode);
            const valid: PermissionDecision[] = ["approved", "denied", "needs_prompt"];
            return valid.includes(decision);
          },
        ),
        { numRuns: 100 },
      );
    });
  });
});
