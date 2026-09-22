// Cortex Intelligence - Enterprise SSO / OIDC Client Service
import { apiFetch, setAuthToken } from "./api.js";

export async function fetchSSOProviders() {
  try {
    const res = await apiFetch("/api/v1/auth/sso/providers");
    if (res.ok) {
      return await res.json();
    }
  } catch (err) {
    console.error("Failed to load SSO identity providers:", err);
  }
  return [];
}

export async function executeSSOCallback(payload) {
  try {
    const res = await apiFetch("/api/v1/auth/sso/callback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Single Sign-On authentication failed.");
    }
    const data = await res.json();
    setAuthToken(data.access_token);
    return data;
  } catch (err) {
    throw err;
  }
}
