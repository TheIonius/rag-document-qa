// Cortex Intelligence - Modular API & Client Storage Service

export const STORAGE_KEYS = {
  AUTH_TOKEN: "cortex_auth_token",
  WORKSPACE_ID: "cortex_workspace_id",
  RETRIEVAL_CONFIG: "cortex_retrieval_config"
};

export function getAuthToken() {
  return localStorage.getItem(STORAGE_KEYS.AUTH_TOKEN);
}

export function setAuthToken(token) {
  if (token) {
    localStorage.setItem(STORAGE_KEYS.AUTH_TOKEN, token);
  } else {
    localStorage.removeItem(STORAGE_KEYS.AUTH_TOKEN);
  }
}

export function clearAuthToken() {
  localStorage.removeItem(STORAGE_KEYS.AUTH_TOKEN);
}

export function getActiveWorkspaceId() {
  return localStorage.getItem(STORAGE_KEYS.WORKSPACE_ID) || "ws_default";
}

export function setActiveWorkspaceId(workspaceId) {
  if (workspaceId) {
    localStorage.setItem(STORAGE_KEYS.WORKSPACE_ID, workspaceId);
  }
}

export async function apiFetch(endpoint, options = {}) {
  const token = getAuthToken();
  const currentWorkspaceId = getActiveWorkspaceId();
  const headers = options.headers ? { ...options.headers } : {};

  if (token && !headers["Authorization"]) {
    headers["Authorization"] = `Bearer ${token}`;
  }
  if (!headers["X-Workspace-Id"]) {
    headers["X-Workspace-Id"] = currentWorkspaceId;
  }

  let url = endpoint;
  if (url.startsWith("/api/") && !url.startsWith("/api/v1/")) {
    url = url.replace("/api/", "/api/v1/");
  }

  const res = await fetch(url, { ...options, headers });
  if (res.status === 401 && token && !endpoint.includes("/auth/login") && !endpoint.includes("/auth/sso/callback")) {
    clearAuthToken();
    if (window.updateUserAuthUI) {
      window.currentUser = null;
      window.updateUserAuthUI();
    }
  }
  return res;
}
