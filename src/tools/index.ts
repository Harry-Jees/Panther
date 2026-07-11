import { type ToolExecutor } from "./types.js";
import { readFileTool } from "./readFile.js";
import { listDirTool } from "./listDir.js";
import { grepTool } from "./grep.js";
import { runTestsTool } from "./runTests.js";
import { writeFileTool } from "./writeFile.js";
import { applyDiffTool } from "./applyDiff.js";
import { runCommandTool } from "./runCommand.js";

/**
 * All built-in tools available to the agent.
 */
export const ALL_TOOLS: ToolExecutor[] = [
  readFileTool,
  listDirTool,
  grepTool,
  runTestsTool,
  writeFileTool,
  applyDiffTool,
  runCommandTool,
];

export { type ToolExecutor, type ToolResult, type SessionContext, type JSONSchema } from "./types.js";
export { readFileTool } from "./readFile.js";
export { listDirTool } from "./listDir.js";
export { grepTool } from "./grep.js";
export { runTestsTool } from "./runTests.js";
export { writeFileTool } from "./writeFile.js";
export { applyDiffTool } from "./applyDiff.js";
export { runCommandTool } from "./runCommand.js";
