import { readFile, writeFile, mkdir, readdir, unlink } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { z } from "zod";


export const MessageSchema = z.object({
  role: z.enum(["system", "user", "assistant", "tool"]),
  content: z.string(),
  toolCallId: z.string().optional(),
  toolName: z.string().optional(),
});

export const SessionSchema = z.object({
  id: z.string(),
  createdAt: z.string(),
  updatedAt: z.string(),
  history: z.array(MessageSchema).default([]),
  specId: z.string().optional(),
  taskId: z.string().optional(),
});

export type Session = z.infer<typeof SessionSchema>;

export interface SessionManagerConfig {
  sessionsDir: string;
}

function generateId(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 9);
}

export class SessionManager {
  private sessionsDir: string;

  constructor(config: SessionManagerConfig) {
    this.sessionsDir = config.sessionsDir;
  }

  private getSessionPath(id: string): string {
    return path.join(this.sessionsDir, `${id}.json`);
  }

  async ensureDir(): Promise<void> {
    if (!existsSync(this.sessionsDir)) {
      await mkdir(this.sessionsDir, { recursive: true });
    }
  }

  async createSession(initialHistory: Session["history"] = []): Promise<Session> {
    await this.ensureDir();
    const now = new Date().toISOString();
    const session: Session = {
      id: generateId(),
      createdAt: now,
      updatedAt: now,
      history: initialHistory,
    };
    await this.saveSession(session);
    return session;
  }

  async saveSession(session: Session): Promise<void> {
    await this.ensureDir();
    session.updatedAt = new Date().toISOString();
    const validated = SessionSchema.parse(session);
    await writeFile(this.getSessionPath(session.id), JSON.stringify(validated, null, 2), "utf-8");
  }

  async loadSession(id: string): Promise<Session | null> {
    try {
      const raw = await readFile(this.getSessionPath(id), "utf-8");
      const parsed = JSON.parse(raw);
      const result = SessionSchema.safeParse(parsed);
      if (!result.success) return null;
      return result.data;
    } catch {
      return null;
    }
  }

  async listSessions(): Promise<Session[]> {
    await this.ensureDir();
    const files = await readdir(this.sessionsDir);
    const jsonFiles = files.filter((f) => f.endsWith(".json"));
    const sessions: Session[] = [];
    for (const file of jsonFiles) {
      const id = file.replace(".json", "");
      const session = await this.loadSession(id);
      if (session) sessions.push(session);
    }
    sessions.sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
    return sessions;
  }

  async deleteSession(id: string): Promise<boolean> {
    try {
      await unlink(this.getSessionPath(id));
      return true;
    } catch {
      return false;
    }
  }
}
