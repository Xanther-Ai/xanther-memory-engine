/**
 * Xanther Memory SDK
 *
 * TypeScript client for the XME MCP Server.
 * Wraps all MCP tools into a typed, ergonomic API.
 */

// --- Types ---

export type MemoryKind =
  | "decision"
  | "bug"
  | "insight"
  | "pattern"
  | "question"
  | "preference"
  | "state";

export interface Memory {
  id: string;
  session_id: string;
  repo_id: string;
  kind: MemoryKind;
  summary: string;
  reasoning: string | null;
  confidence: number;
  priority: number;
  refs: string[];
  created_at: string;
}

export interface RememberParams {
  kind: MemoryKind;
  summary: string;
  reasoning?: string;
  references?: string[];
  repo_id: string;
  session_id: string;
  user_id: string;
}

export interface RecallParams {
  query: string;
  user_id: string;
  kind?: MemoryKind;
  repo_id?: string;
  since?: string;
  limit?: number;
}

export interface SessionStateParams {
  session_id: string;
  user_id: string;
  key: string;
}

export interface PreferenceParams {
  user_id: string;
  key?: string;
  value?: unknown;
  repo_id?: string;
}

export interface HistoryParams {
  user_id: string;
  repo_id?: string;
  since?: string;
  limit?: number;
}

export interface RememberResult {
  stored: boolean;
  memory_id: string;
  kind: MemoryKind;
  summary: string;
}

export interface RecallResult {
  memories: Memory[];
  count: number;
  query: string;
}

export interface SessionStateResult {
  key: string;
  value: unknown;
  updated_at?: string;
}

export interface PreferencesListResult {
  preferences: Record<string, string>;
  count: number;
}

export interface HistoryResult {
  decisions: Memory[];
  count: number;
}

export interface XMEClientOptions {
  /** Base URL of the XME MCP Server (e.g., http://localhost:8100) */
  baseUrl: string;
  /** Request timeout in ms (default: 30000) */
  timeout?: number;
}

// --- MCP SSE Client ---

interface JsonRpcRequest {
  jsonrpc: "2.0";
  id: number;
  method: string;
  params?: Record<string, unknown>;
}

interface JsonRpcResponse {
  jsonrpc: "2.0";
  id: number;
  result?: {
    content?: Array<{ type: string; text: string }>;
    isError?: boolean;
    tools?: unknown[];
    [key: string]: unknown;
  };
  error?: { code: number; message: string };
}

/**
 * XME Client — wraps the MCP SSE transport for the Xanther Memory Engine.
 *
 * Usage:
 * ```ts
 * const client = new XMEClient({ baseUrl: "http://localhost:8100" });
 * await client.connect();
 *
 * const result = await client.remember({
 *   kind: "decision",
 *   summary: "Chose PostgreSQL over DynamoDB for relational queries",
 *   repo_id: "my-repo",
 *   session_id: "sess-123",
 *   user_id: "user-1",
 * });
 * ```
 */
export class XMEClient {
  private baseUrl: string;
  private timeout: number;
  private sessionId: string | null = null;
  private messagesUrl: string | null = null;
  private eventSource: EventSource | null = null;
  private requestId = 0;
  private pendingRequests = new Map<
    number,
    { resolve: (v: JsonRpcResponse) => void; reject: (e: Error) => void }
  >();

  constructor(options: XMEClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/$/, "");
    this.timeout = options.timeout ?? 30000;
  }

  /**
   * Connect to the MCP SSE server.
   * Must be called before any tool invocations.
   */
  async connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      const sseUrl = `${this.baseUrl}/sse`;
      this.eventSource = new EventSource(sseUrl);

      const timeout = setTimeout(() => {
        reject(new Error("SSE connection timeout"));
        this.eventSource?.close();
      }, this.timeout);

      this.eventSource.addEventListener("endpoint", (event: MessageEvent) => {
        clearTimeout(timeout);
        this.messagesUrl = `${this.baseUrl}${event.data}`;
        this.sessionId = new URL(this.messagesUrl).searchParams.get("session_id");
        resolve();
      });

      this.eventSource.addEventListener("message", (event: MessageEvent) => {
        try {
          const response: JsonRpcResponse = JSON.parse(event.data);
          const pending = this.pendingRequests.get(response.id);
          if (pending) {
            this.pendingRequests.delete(response.id);
            pending.resolve(response);
          }
        } catch {
          // Ignore malformed messages
        }
      });

      this.eventSource.onerror = () => {
        clearTimeout(timeout);
        if (!this.sessionId) {
          reject(new Error("SSE connection failed"));
        }
      };
    });
  }

  /** Disconnect from the server. */
  disconnect(): void {
    this.eventSource?.close();
    this.eventSource = null;
    this.sessionId = null;
    this.messagesUrl = null;
    this.pendingRequests.clear();
  }

  /** Send a JSON-RPC request and wait for the response via SSE. */
  private async sendRequest(
    method: string,
    params?: Record<string, unknown>
  ): Promise<JsonRpcResponse> {
    if (!this.messagesUrl) {
      throw new Error("Not connected. Call connect() first.");
    }

    const id = ++this.requestId;
    const request: JsonRpcRequest = { jsonrpc: "2.0", id, method, params };

    const responsePromise = new Promise<JsonRpcResponse>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pendingRequests.delete(id);
        reject(new Error(`Request timeout (method: ${method})`));
      }, this.timeout);

      this.pendingRequests.set(id, {
        resolve: (v) => {
          clearTimeout(timer);
          resolve(v);
        },
        reject: (e) => {
          clearTimeout(timer);
          reject(e);
        },
      });
    });

    const res = await fetch(this.messagesUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });

    if (!res.ok) {
      this.pendingRequests.delete(id);
      throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    }

    return responsePromise;
  }

  /** Call an MCP tool and parse the result. */
  private async callTool<T>(name: string, args: Record<string, unknown>): Promise<T> {
    const response = await this.sendRequest("tools/call", { name, arguments: args });

    if (response.error) {
      throw new Error(`RPC Error: ${response.error.message}`);
    }

    const content = response.result?.content;
    if (!content || content.length === 0) {
      throw new Error("Empty response from server");
    }

    const text = content[0].text;
    const parsed = JSON.parse(text);

    if (response.result?.isError) {
      throw new Error(parsed.error || "Tool execution failed");
    }

    return parsed as T;
  }

  // --- Public API ---

  /**
   * Store a memory directly.
   */
  async remember(params: RememberParams): Promise<RememberResult> {
    return this.callTool<RememberResult>("xme_remember", params as unknown as Record<string, unknown>);
  }

  /**
   * Query memories by topic with optional filters.
   */
  async recall(params: RecallParams): Promise<RecallResult> {
    return this.callTool<RecallResult>("xme_recall", params as unknown as Record<string, unknown>);
  }

  /**
   * Get session state by key.
   */
  async getSessionState(params: SessionStateParams): Promise<SessionStateResult> {
    return this.callTool<SessionStateResult>("xme_session_state", {
      action: "get",
      ...params,
    });
  }

  /**
   * Set session state.
   */
  async setSessionState(
    params: SessionStateParams & { value: unknown }
  ): Promise<{ key: string; stored: boolean }> {
    return this.callTool("xme_session_state", {
      action: "set",
      ...params,
    });
  }

  /**
   * Get a user preference.
   */
  async getPreferences(params: PreferenceParams): Promise<SessionStateResult> {
    return this.callTool<SessionStateResult>("xme_preferences", {
      action: params.key ? "get" : "list",
      ...params,
    });
  }

  /**
   * Set a user preference.
   */
  async setPreferences(
    params: PreferenceParams & { key: string; value: unknown }
  ): Promise<{ key: string; stored: boolean }> {
    return this.callTool("xme_preferences", {
      action: "set",
      ...params,
    });
  }

  /**
   * Query decision history.
   */
  async getHistory(params: HistoryParams): Promise<HistoryResult> {
    return this.callTool<HistoryResult>("xme_history", params as unknown as Record<string, unknown>);
  }
}

export default XMEClient;
