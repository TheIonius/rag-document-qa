// Cortex Intelligence - Real-Time Workspace Event Stream Service (SSE)
import { getAuthToken, getActiveWorkspaceId } from "./api.js";

class WorkspaceEventManager {
  constructor() {
    this.eventSource = null;
    this.currentWorkspaceId = null;
    this.listeners = new Map();
    this.reconnectTimeout = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 10;
  }

  on(eventType, handler) {
    if (!this.listeners.has(eventType)) {
      this.listeners.set(eventType, new Set());
    }
    this.listeners.get(eventType).add(handler);
    return () => this.off(eventType, handler);
  }

  off(eventType, handler) {
    if (this.listeners.has(eventType)) {
      this.listeners.get(eventType).delete(handler);
    }
  }

  emit(eventType, data) {
    const handlers = this.listeners.get(eventType);
    if (handlers) {
      handlers.forEach((fn) => {
        try {
          fn(data);
        } catch (err) {
          console.error(`Error in event listener for ${eventType}:`, err);
        }
      });
    }
  }

  connect(workspaceId) {
    const targetWs = workspaceId || getActiveWorkspaceId();
    if (this.eventSource && this.currentWorkspaceId === targetWs) {
      return;
    }

    this.disconnect();
    this.currentWorkspaceId = targetWs;

    const token = getAuthToken();
    let url = `/api/v1/workspaces/${encodeURIComponent(targetWs)}/events`;
    if (token) {
      url += `?token=${encodeURIComponent(token)}`;
    }

    try {
      this.eventSource = new EventSource(url);

      this.eventSource.addEventListener("connected", (e) => {
        this.reconnectAttempts = 0;
        try {
          const payload = JSON.parse(e.data);
          this.emit("connected", payload);
        } catch {
          this.emit("connected", { workspace_id: targetWs });
        }
      });

      this.eventSource.addEventListener("document_indexed", (e) => {
        try {
          const payload = JSON.parse(e.data);
          this.emit("document_indexed", payload);
        } catch (err) {
          console.error("Failed to parse document_indexed event payload:", err);
        }
      });

      this.eventSource.addEventListener("document_deleted", (e) => {
        try {
          const payload = JSON.parse(e.data);
          this.emit("document_deleted", payload);
        } catch (err) {
          console.error("Failed to parse document_deleted event payload:", err);
        }
      });

      this.eventSource.addEventListener("member_added", (e) => {
        try {
          const payload = JSON.parse(e.data);
          this.emit("member_added", payload);
        } catch (err) {
          console.error("Failed to parse member_added event payload:", err);
        }
      });

      this.eventSource.addEventListener("member_removed", (e) => {
        try {
          const payload = JSON.parse(e.data);
          this.emit("member_removed", payload);
        } catch (err) {
          console.error("Failed to parse member_removed event payload:", err);
        }
      });

      this.eventSource.onerror = () => {
        if (this.eventSource) {
          this.eventSource.close();
          this.eventSource = null;
        }
        this.scheduleReconnect();
      };
    } catch (err) {
      console.warn("Unable to initiate EventSource connection:", err);
      this.scheduleReconnect();
    }
  }

  scheduleReconnect() {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
    }
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.warn("Max event stream reconnection attempts reached.");
      return;
    }
    const delay = Math.min(1000 * Math.pow(1.5, this.reconnectAttempts), 15000);
    this.reconnectAttempts += 1;
    this.reconnectTimeout = setTimeout(() => {
      this.connect(this.currentWorkspaceId);
    }, delay);
  }

  disconnect() {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }
    this.currentWorkspaceId = null;
  }
}

export const workspaceEvents = new WorkspaceEventManager();
