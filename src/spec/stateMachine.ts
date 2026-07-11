import { writeFile, mkdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";

export type SpecStage = "idle" | "requirements" | "design" | "tasks" | "execution" | "done";

const STAGE_ORDER: SpecStage[] = ["idle", "requirements", "design", "tasks", "execution", "done"];

export interface SpecWorkflowState {
  specId: string;
  currentStage: SpecStage;
  projectRoot: string;
}

export class SpecWorkflowError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SpecWorkflowError";
  }
}

/**
 * SpecWorkflow state machine: guides a task through requirements → design → tasks → execution.
 */
export class SpecWorkflow {
  private state: SpecWorkflowState;

  constructor(state: SpecWorkflowState) {
    this.state = state;
  }

  get currentStage(): SpecStage {
    return this.state.currentStage;
  }

  get specId(): string {
    return this.state.specId;
  }

  getStagePath(stage: SpecStage): string {
    return path.join(this.state.projectRoot, ".panther", "specs", this.state.specId, `${stage}.md`);
  }

  /**
   * Advance to the next stage if the current stage is approved.
   */
  async approve(): Promise<SpecStage> {
    const currentIndex = STAGE_ORDER.indexOf(this.state.currentStage);
    if (currentIndex >= STAGE_ORDER.length - 1) {
      throw new SpecWorkflowError("Already at final stage (done)");
    }

    const nextStage = STAGE_ORDER[currentIndex + 1];
    this.state.currentStage = nextStage;
    return nextStage;
  }

  /**
   * Write stage output to disk and advance.
   */
  async completeStage(content: string): Promise<SpecStage> {
    const stage = this.state.currentStage;
    if (stage === "idle" || stage === "done") {
      throw new SpecWorkflowError(`Cannot complete stage: ${stage}`);
    }

    const stagePath = this.getStagePath(stage);
    const dir = path.dirname(stagePath);
    if (!existsSync(dir)) {
      await mkdir(dir, { recursive: true });
    }

    await writeFile(stagePath, content, "utf-8");
    return this.approve();
  }

  /**
   * Get the content of a previous stage's output file.
   */
  async getStageContent(stage: SpecStage): Promise<string | null> {
    try {
      const { readFile } = await import("node:fs/promises");
      return await readFile(this.getStagePath(stage), "utf-8");
    } catch {
      return null;
    }
  }

  /**
   * Check if a transition would be valid (for trivial-task shortcut).
   */
  canSkipTo(target: SpecStage): boolean {
    const currentIndex = STAGE_ORDER.indexOf(this.state.currentStage);
    const targetIndex = STAGE_ORDER.indexOf(target);
    return targetIndex > currentIndex;
  }

  /**
   * Create a new spec workflow starting at the requirements stage.
   */
  static create(specId: string, projectRoot: string): SpecWorkflow {
    return new SpecWorkflow({
      specId,
      currentStage: "requirements",
      projectRoot,
    });
  }
}
