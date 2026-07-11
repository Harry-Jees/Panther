import { type ToolExecutor } from "../tools/types.js";
import { type PermissionMode } from "../config/types.js";

export type PermissionDecision = "approved" | "denied" | "needs_prompt";

/**
 * The PermissionGate decides whether a tool execution requires user confirmation
 * based on the active permission mode and the tool's risk level.
 */
export interface PermissionGate {
  check(tool: ToolExecutor, mode: PermissionMode): PermissionDecision;
}

/**
 * Decision matrix:
 *
 * | Mode           | Safe tool     | Risky tool    |
 * |----------------|---------------|---------------|
 * | always_ask     | needs_prompt  | needs_prompt  |
 * | ask_when_risky | approved      | needs_prompt  |
 * | never_ask      | approved      | approved      |
 */
const DECISION_MATRIX: Record<PermissionMode, { safe: PermissionDecision; risky: PermissionDecision }> = {
  always_ask: { safe: "needs_prompt", risky: "needs_prompt" },
  ask_when_risky: { safe: "approved", risky: "needs_prompt" },
  never_ask: { safe: "approved", risky: "approved" },
};

export const defaultPermissionGate: PermissionGate = {
  check(tool: ToolExecutor, mode: PermissionMode): PermissionDecision {
    const row = DECISION_MATRIX[mode];
    if (!row) {
      // Unknown mode — default to asking
      return "needs_prompt";
    }
    return tool.riskLevel === "risky" ? row.risky : row.safe;
  },
};

/**
 * Determine the effective permission decision for one-shot mode.
 * In one-shot mode, any "needs_prompt" result is automatically denied.
 */
export function checkOneShot(
  gate: PermissionGate,
  tool: ToolExecutor,
  mode: PermissionMode,
): PermissionDecision {
  const decision = gate.check(tool, mode);
  if (decision === "needs_prompt") {
    return "denied";
  }
  return decision;
}
