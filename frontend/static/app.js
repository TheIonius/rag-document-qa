// Cortex Intelligence - Enterprise Document Intelligence Platform Logic

let currentThreadId = null;
let activeThreadMessages = [];
let allThreads = [];
let allDocuments = [];
let activeCollectionFilter = "all";
let activeSplitReaderDoc = null;
let currentHighlightedChunkId = null;

let currentWorkspaceId = localStorage.getItem("cortex_workspace_id") || "ws_default";
let currentUser = null;
let currentAuthTab = "login";

let retrievalConfig = {
  topK: 4,
  denseWeight: 0.50,
  bm25Weight: 0.50,
  refusalThreshold: 0.20,
  multiHop: false
};

async function apiFetch(endpoint, options = {}) {
  const token = localStorage.getItem("cortex_auth_token");
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
  if (res.status === 401 && token && !endpoint.includes("/auth/login")) {
    localStorage.removeItem("cortex_auth_token");
    currentUser = null;
    updateUserAuthUI();
  }
  return res;
}

document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  checkAuthStatus();
  loadWorkspaces();
  loadDocuments();
  loadThreads();
  updateHistoryBadge();
});

// Event listeners

function setupEventListeners() {
  // Query Form
  const form = document.getElementById("query-form");
  const queryInput = document.getElementById("query-input");

  if (form && queryInput) {
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      const q = queryInput.value.trim();
      if (q) {
        submitQuery(q);
      }
    });
  }

  // Keyboard Shortcuts
  document.addEventListener("keydown", (e) => {
    // Ctrl+\ / Cmd+\ -> Toggle Sidebar
    if ((e.ctrlKey || e.metaKey) && e.key === "\\") {
      e.preventDefault();
      toggleSidebar();
    }
    // Ctrl+N / Cmd+N -> New Investigation
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "n") {
      e.preventDefault();
      startNewInvestigation();
    }
    // "/" or Cmd+K -> Focus Query Input (if not already focused in an input)
    if ((e.key === "/" || ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k")) &&
        document.activeElement.tagName !== "INPUT" &&
        document.activeElement.tagName !== "TEXTAREA") {
      e.preventDefault();
      const input = document.getElementById("query-input");
      if (input) input.focus();
    }
    // Esc -> Close reader, mobile sidebar drawer, or modals
    if (e.key === "Escape") {
      toggleSidebar(false);
      closeSplitReader();
      closeEngineModal();
      closeAuditModal();
      closeInspectorModal();
    }
  });

  // Restore Desktop Sidebar Preference
  if (window.innerWidth > 960) {
    try {
      if (localStorage.getItem("cortex_sidebar_collapsed") === "true") {
        const layout = document.getElementById("workspace-layout");
        if (layout) layout.classList.add("sidebar-collapsed");
      }
    } catch (e) {}
  }

  // Responsive resize cleanup
  window.addEventListener("resize", () => {
    const layout = document.getElementById("workspace-layout");
    if (!layout) return;
    if (window.innerWidth > 960) {
      layout.classList.remove("sidebar-open");
    }
  });

  // Upload Dropzone
  const uploadZone = document.getElementById("upload-zone");
  const fileInput = document.getElementById("file-upload-input");

  if (uploadZone && fileInput) {
    uploadZone.addEventListener("click", () => fileInput.click());
    uploadZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      uploadZone.classList.add("drag-over");
    });
    uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("drag-over"));
    uploadZone.addEventListener("drop", (e) => {
      e.preventDefault();
      uploadZone.classList.remove("drag-over");
      if (e.dataTransfer.files.length > 0) {
        handleFileUpload(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener("change", (e) => {
      if (e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
      }
    });
  }

  // Engine Diagnostics Modal Controls
  const btnEngineSettings = document.getElementById("btn-open-engine-settings");
  if (btnEngineSettings) {
    btnEngineSettings.addEventListener("click", openEngineModal);
  }

  // Audit Ledger Modal Controls
  const btnAuditLog = document.getElementById("btn-open-audit-log");
  if (btnAuditLog) {
    btnAuditLog.addEventListener("click", openAuditModal);
  }

  // Top-K Depth Selector
  document.querySelectorAll(".btn-param-k").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".btn-param-k").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const k = parseInt(btn.getAttribute("data-k"), 10);
      retrievalConfig.topK = k;
      const valEl = document.getElementById("top-k-val");
      if (valEl) valEl.textContent = `${k} chunks`;
    });
  });

  // Hybrid Slider
  const hybridSlider = document.getElementById("slider-hybrid-ratio");
  if (hybridSlider) {
    hybridSlider.addEventListener("input", (e) => {
      const densePct = parseInt(e.target.value, 10);
      const bm25Pct = 100 - densePct;
      retrievalConfig.denseWeight = densePct / 100.0;
      retrievalConfig.bm25Weight = bm25Pct / 100.0;
      const badge = document.getElementById("hybrid-ratio-val");
      if (badge) badge.textContent = `${densePct}% Dense / ${bm25Pct}% BM25`;
      document.querySelectorAll(".btn-preset").forEach(b => b.classList.remove("active"));
    });
  }

  // Hybrid Presets
  document.querySelectorAll(".btn-preset").forEach(btn => {
    btn.addEventListener("click", () => {
      const denseVal = parseInt(btn.getAttribute("data-dense"), 10);
      if (hybridSlider) hybridSlider.value = denseVal;
      const bm25Val = 100 - denseVal;
      retrievalConfig.denseWeight = denseVal / 100.0;
      retrievalConfig.bm25Weight = bm25Val / 100.0;
      const badge = document.getElementById("hybrid-ratio-val");
      if (badge) badge.textContent = `${denseVal}% Dense / ${bm25Val}% BM25`;
      document.querySelectorAll(".btn-preset").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
    });
  });

  // Refusal Threshold Slider
  const refusalSlider = document.getElementById("slider-refusal-thresh");
  if (refusalSlider) {
    refusalSlider.addEventListener("input", (e) => {
      const val = parseInt(e.target.value, 10) / 100.0;
      retrievalConfig.refusalThreshold = val;
      const badge = document.getElementById("refusal-thresh-val");
      if (badge) badge.textContent = `Cutoff: ${val.toFixed(2)}`;
    });
  }
}

// Auth and workspaces

async function checkAuthStatus() {
  const token = localStorage.getItem("cortex_auth_token");
  if (!token) {
    updateUserAuthUI();
    return;
  }

  try {
    const res = await apiFetch("/api/v1/auth/me");
    if (res.ok) {
      currentUser = await res.json();
    } else {
      currentUser = null;
      localStorage.removeItem("cortex_auth_token");
    }
  } catch (e) {
    currentUser = null;
  }
  updateUserAuthUI();
}

function updateUserAuthUI() {
  const displayBtn = document.getElementById("user-display-name");
  const profileSection = document.getElementById("user-profile-section");
  const authForm = document.getElementById("auth-form");
  const authTabsRow = document.getElementById("auth-tabs-row");
  const envBadge = document.querySelector(".env-badge");
  const uploadSub = document.querySelector(".upload-compact-sub");

  if (currentUser) {
    if (displayBtn) displayBtn.textContent = currentUser.full_name || currentUser.email;
    if (profileSection) profileSection.style.display = "block";
    if (authForm) authForm.style.display = "none";
    if (authTabsRow) authTabsRow.style.display = "none";
    if (envBadge) {
      envBadge.textContent = currentUser.is_superuser ? "Enterprise Superuser" : "Authenticated Tenant";
      envBadge.style.background = "#064e3b";
      envBadge.style.color = "#a7f3d0";
      envBadge.style.borderColor = "#059669";
    }
    if (uploadSub) {
      uploadSub.textContent = "Upload to current workspace";
    }

    const pName = document.getElementById("profile-name");
    const pEmail = document.getElementById("profile-email");
    const pRole = document.getElementById("profile-role");
    const pAvatar = document.getElementById("profile-avatar");

    if (pName) pName.textContent = currentUser.full_name;
    if (pEmail) pEmail.textContent = currentUser.email;
    if (pRole) pRole.textContent = currentUser.is_superuser ? "Superuser / Owner" : "Workspace Member";
    if (pAvatar) {
      const initials = currentUser.full_name
        ? currentUser.full_name.split(" ").map(n => n[0]).join("").toUpperCase().slice(0, 2)
        : "US";
      pAvatar.textContent = initials;
    }
  } else {
    if (displayBtn) displayBtn.textContent = "Sign In";
    if (profileSection) profileSection.style.display = "none";
    if (authForm) authForm.style.display = "block";
    if (authTabsRow) authTabsRow.style.display = "flex";
    if (envBadge) {
      envBadge.textContent = "Guest Mode (Demo Sandbox)";
      envBadge.style.background = "var(--bg-tertiary)";
      envBadge.style.color = "var(--text-muted)";
      envBadge.style.borderColor = "var(--border-color)";
    }
    if (uploadSub) {
      uploadSub.textContent = "Sign in for private workspace";
    }
  }
}

async function loadWorkspaces() {
  const selector = document.getElementById("workspace-select");
  if (!selector) return;

  try {
    const res = await apiFetch("/api/v1/workspaces");
    if (!res.ok) return;
    const workspaces = await res.json();

    if (workspaces.length > 0) {
      if (!currentWorkspaceId || !workspaces.some(w => w.id === currentWorkspaceId)) {
        currentWorkspaceId = workspaces[0].id;
        localStorage.setItem("cortex_workspace_id", currentWorkspaceId);
      }
      selector.innerHTML = workspaces.map(w => `
        <option value="${w.id}" ${w.id === currentWorkspaceId ? 'selected' : ''}>
          ${escapeHtml(w.name)} (${w.role || 'member'})
        </option>
      `).join("");
    }
  } catch (e) {
    // Keep default option
  }
}

function changeActiveWorkspace(wsId) {
  currentWorkspaceId = wsId;
  localStorage.setItem("cortex_workspace_id", wsId);
  loadDocuments();
  loadThreads();
  updateHistoryBadge();
}

function openAuthModal() {
  const modal = document.getElementById("auth-modal");
  if (modal) {
    modal.style.display = "flex";
    const errBanner = document.getElementById("auth-error-banner");
    if (errBanner) errBanner.style.display = "none";
  }
}

function closeAuthModal() {
  const modal = document.getElementById("auth-modal");
  if (modal) modal.style.display = "none";
}

function closeAuthModalOnBackdrop(e) {
  if (e.target.id === "auth-modal") closeAuthModal();
}

function switchAuthTab(tab) {
  currentAuthTab = tab;
  const loginTab = document.getElementById("tab-login");
  const regTab = document.getElementById("tab-register");
  const nameGroup = document.getElementById("register-name-group");
  const submitBtn = document.getElementById("btn-auth-submit");
  const errBanner = document.getElementById("auth-error-banner");

  if (errBanner) errBanner.style.display = "none";

  if (tab === "login") {
    if (loginTab) loginTab.classList.add("active");
    if (regTab) regTab.classList.remove("active");
    if (nameGroup) nameGroup.style.display = "none";
    if (submitBtn) submitBtn.textContent = "Sign In";
  } else {
    if (loginTab) loginTab.classList.remove("active");
    if (regTab) regTab.classList.add("active");
    if (nameGroup) nameGroup.style.display = "block";
    if (submitBtn) submitBtn.textContent = "Create Account";
  }
}

async function handleAuthSubmit(e) {
  e.preventDefault();
  const emailInput = document.getElementById("auth-email");
  const passwordInput = document.getElementById("auth-password");
  const nameInput = document.getElementById("auth-fullname");
  const errBanner = document.getElementById("auth-error-banner");

  const email = emailInput ? emailInput.value.trim() : "";
  const password = passwordInput ? passwordInput.value : "";
  const fullName = nameInput ? nameInput.value.trim() : "";

  if (errBanner) {
    errBanner.textContent = "";
    errBanner.style.display = "none";
  }

  try {
    let res;
    if (currentAuthTab === "register") {
      res = await fetch("/api/v1/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password, full_name: fullName || "Enterprise Member" })
      });
    } else {
      res = await fetch("/api/v1/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password })
      });
    }

    const data = await res.json();
    if (!res.ok) {
      let msg = "Authentication failed.";
      if (typeof data.detail === "string") {
        msg = data.detail;
      } else if (Array.isArray(data.detail)) {
        msg = data.detail.map(d => (typeof d === "string" ? d : (d.msg || JSON.stringify(d)))).join("; ");
      } else if (data.error && data.error.message) {
        msg = data.error.message;
      }
      if (errBanner) {
        errBanner.textContent = msg;
        errBanner.style.display = "block";
      }
      return;
    }

    localStorage.setItem("cortex_auth_token", data.access_token);
    currentUser = data.user;
    if (passwordInput) passwordInput.value = "";
    if (errBanner) errBanner.style.display = "none";
    updateUserAuthUI();
    await loadWorkspaces();
    loadDocuments();
    loadThreads();
    closeAuthModal();
  } catch (err) {
    if (errBanner) {
      errBanner.textContent = err.message || "Network error. Please try again.";
      errBanner.style.display = "block";
    }
  }
}

function handleSignOut() {
  localStorage.removeItem("cortex_auth_token");
  currentUser = null;
  currentWorkspaceId = "ws_default";
  localStorage.setItem("cortex_workspace_id", "ws_default");
  updateUserAuthUI();
  closeAuthModal();
  loadWorkspaces();
  loadDocuments();
  loadThreads();
}

async function sendFeedback(btn, isPositive, queryLogId) {
  try {
    await apiFetch("/api/v1/query/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        is_positive: isPositive,
        query_log_id: queryLogId || null,
        notes: isPositive ? "Helpful verified response" : "User marked unhelpful"
      })
    });
    if (btn && btn.parentElement) {
      btn.parentElement.innerHTML = `<span class="feedback-submitted-pill">${isPositive ? 'Helpful' : 'Feedback Recorded'}</span>`;
    }
  } catch (e) {
    if (btn) btn.disabled = true;
  }
}

// Document collections

async function loadDocuments() {
  const container = document.getElementById("collections-container");
  const countBadge = document.getElementById("docs-count");
  const statusPillText = document.getElementById("corpus-status-text");

  try {
    const res = await apiFetch("/api/v1/documents");
    if (!res.ok) throw new Error("Failed to load documents");
    allDocuments = await res.json();

    if (countBadge) countBadge.textContent = `${allDocuments.length} docs`;

    const totalChunks = allDocuments.reduce((sum, d) => sum + d.chunk_count, 0);
    if (statusPillText) statusPillText.textContent = `${totalChunks} Partitions Active`;

    // Group documents by collection
    const collectionsMap = {};
    allDocuments.forEach(doc => {
      const col = doc.collection || "General Documentation";
      if (!collectionsMap[col]) collectionsMap[col] = [];
      collectionsMap[col].push(doc);
    });

    // Populate Scope Filter Pills in canvas header
    populateScopeFilter(collectionsMap);

    // Render Collections in sidebar
    if (container) {
      if (allDocuments.length === 0) {
        container.innerHTML = `<div class="empty-threads-notice">No documents indexed yet.</div>`;
        return;
      }

      container.innerHTML = Object.keys(collectionsMap).map(colName => {
        const docs = collectionsMap[colName];
        return `
          <div class="collection-group">
            <div class="collection-group-title">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
              </svg>
              <span>${escapeHtml(colName)}</span>
            </div>
            ${docs.map(d => `
              <div class="collection-doc-item" onclick="openSplitReader('${d.id}', null, null)" title="${escapeHtml(d.title)} (Click to view full text in reader)">
                <span class="collection-doc-name">${escapeHtml(d.title)}</span>
                <span class="collection-doc-chunks">${d.chunk_count}p</span>
              </div>
            `).join("")}
          </div>
        `;
      }).join("");
    }
  } catch (err) {
    if (container) {
      container.innerHTML = `<div class="empty-threads-notice">Error loading library: ${escapeHtml(err.message)}</div>`;
    }
  }
}

const STARTER_INQUIRIES_BY_SCOPE = {
  "all": [
    {
      tag: "Security & Infrastructure",
      title: "Production MFA Policy & Approved Hardware Protocols",
      query: "What are the MFA requirements and prohibited methods for production?"
    },
    {
      tag: "People Operations",
      title: "401(k) Company Matching Formula & Vesting Schedule",
      query: "What is the 401(k) company matching formula and vesting schedule?"
    },
    {
      tag: "API & System Architecture",
      title: "API Rate Limits, HTTP 429 Codes & Webhook Retries",
      query: "What are the API rate limits and webhook retry backoff schedule?"
    },
    {
      tag: "People Operations",
      title: "Paid Caregiver & Parental Leave Eligibility",
      query: "How many weeks of paid parental leave are offered to caregivers?"
    }
  ],
  "Security & Infrastructure": [
    {
      tag: "Security & Infrastructure",
      title: "Production MFA Policy & SMS Authentication Prohibition",
      query: "What are the MFA requirements and prohibited methods for production?"
    },
    {
      tag: "Security & Infrastructure",
      title: "Security Incident Severity Classification (P1 to P4 SLA)",
      query: "How are security incident escalation severities P1 through P4 classified and what are their response SLAs?"
    },
    {
      tag: "Security & Infrastructure",
      title: "Password Rotation, Complexity & Session Timeout Rules",
      query: "What is the password rotation schedule, minimum length, and idle session timeout policy?"
    },
    {
      tag: "Security & Infrastructure",
      title: "Encryption Standards for Data-at-Rest & In-Transit",
      query: "What are the mandatory TLS and AES encryption standards for data-at-rest and data-in-transit?"
    }
  ],
  "People Operations & Policies": [
    {
      tag: "People Operations",
      title: "Paid Parental Leave Duration & Primary Caregiver Rules",
      query: "How many weeks of paid parental leave are offered to caregivers?"
    },
    {
      tag: "People Operations",
      title: "401(k) Contribution Matching & 4-Year Vesting Schedule",
      query: "What is the 401(k) company matching formula and vesting schedule?"
    },
    {
      tag: "People Operations",
      title: "Annual Education Stipend & Conference Approval Guidelines",
      query: "What is the annual continuous education and conference reimbursement stipend?"
    },
    {
      tag: "People Operations",
      title: "Remote Work Equipment Allowance & Health Benefit Tiers",
      query: "What are the remote work equipment allowances and health insurance contribution tiers?"
    }
  ],
  "Developer & API Specifications": [
    {
      tag: "API & System Architecture",
      title: "API Rate Limits, Token Expiration & Header Requirements",
      query: "What are the API rate limits, authentication headers, and token expiration rules?"
    },
    {
      tag: "API & System Architecture",
      title: "Asynchronous Webhook Retries & Exponential Backoff",
      query: "What are the API rate limits and webhook retry backoff schedule?"
    },
    {
      tag: "API & System Architecture",
      title: "Idempotency Keys on Payment & Transfer Mutations",
      query: "How do idempotency keys prevent duplicate transaction execution on POST requests?"
    },
    {
      tag: "API & System Architecture",
      title: "Pagination Limits & Cursor-based Query Headers",
      query: "What are the default and maximum pagination limit parameters for collection endpoints?"
    }
  ]
};

function renderStarterInquiries(colName) {
  const grid = document.getElementById("starters-grid");
  const label = document.getElementById("starters-label");
  if (!grid) return;

  const items = STARTER_INQUIRIES_BY_SCOPE[colName] || STARTER_INQUIRIES_BY_SCOPE["all"];

  if (label) {
    if (colName === "all") {
      label.textContent = "Suggested Research Inquiries (All Collections):";
    } else {
      label.textContent = `Suggested Research Inquiries: ${colName}`;
    }
  }

  grid.innerHTML = items.map(item => `
    <button type="button" class="starter-card" data-query="${escapeHtml(item.query)}" onclick="executeStarterInquiry(this)">
      <div class="starter-tag">${escapeHtml(item.tag)}</div>
      <div class="starter-title">${escapeHtml(item.title)}</div>
    </button>
  `).join("");
}

function populateScopeFilter(collectionsMap) {
  const scopeRow = document.getElementById("scope-chips");
  if (!scopeRow) return;

  collectionsMap = collectionsMap || {};
  const totalChunks = allDocuments.reduce((sum, d) => sum + (d.chunk_count || 0), 0);
  const keys = Object.keys(collectionsMap);

  scopeRow.innerHTML = `
    <button type="button" class="scope-pill ${activeCollectionFilter === 'all' ? 'active' : ''}" onclick="setCollectionFilter('all')">
      <span>All Collections</span>
      <span class="scope-pill-count">${totalChunks}p</span>
    </button>
    ${keys.map(col => {
      const docs = collectionsMap[col] || [];
      const colChunks = docs.reduce((sum, d) => sum + (d.chunk_count || 0), 0);
      return `
        <button type="button" class="scope-pill ${activeCollectionFilter === col ? 'active' : ''}" onclick="setCollectionFilter('${escapeHtml(col)}')">
          <span>${escapeHtml(col)}</span>
          <span class="scope-pill-count">${colChunks}p</span>
        </button>
      `;
    }).join("")}
  `;
}

function setCollectionFilter(colName) {
  activeCollectionFilter = colName;

  // 1. Update pills active state
  document.querySelectorAll(".scope-pill").forEach(p => {
    const textSpan = p.querySelector("span");
    const label = textSpan ? textSpan.textContent.trim() : p.textContent.trim();
    const isTarget = colName === "all" ? label === "All Collections" : label === colName;
    p.classList.toggle("active", isTarget);
  });

  // 2. Update Active Scope Status Banner
  const banner = document.getElementById("active-scope-banner");
  const bannerText = document.getElementById("active-scope-banner-text");
  if (banner && bannerText) {
    if (colName === "all") {
      banner.style.display = "none";
    } else {
      banner.style.display = "flex";
      const filtered = allDocuments.filter(d => (d.collection || "General Documentation") === colName);
      const chunkCount = filtered.reduce((sum, d) => sum + (d.chunk_count || 0), 0);
      bannerText.textContent = `Isolated to "${colName}" (${filtered.length} doc, ${chunkCount} partitions) · Inquiries will query this collection exclusively`;
    }
  }

  // 3. Update Starter Inquiries dynamically
  renderStarterInquiries(colName);

  // 4. Update Input Placeholder
  const queryInput = document.getElementById("query-input");
  if (queryInput) {
    if (colName === "all") {
      queryInput.placeholder = "Ask an enterprise inquiry across all collections...";
    } else {
      queryInput.placeholder = `Ask an inquiry scoped strictly to ${colName}...`;
    }
  }

  // 5. Highlight matching Collection in Sidebar
  document.querySelectorAll(".collection-group").forEach(group => {
    const titleSpan = group.querySelector(".collection-group-title span");
    const groupTitle = titleSpan ? titleSpan.textContent.trim() : "";
    group.classList.toggle("is-scoped", colName !== "all" && groupTitle === colName);
  });
}

function getActiveDocFilter() {
  if (activeCollectionFilter === "all") return null;
  const filtered = allDocuments.filter(d => (d.collection || "General Documentation") === activeCollectionFilter);
  return filtered.length > 0 ? filtered.map(d => d.id) : null;
}

async function handleFileUpload(file) {
  const statusEl = document.getElementById("upload-status");
  statusEl.style.color = "var(--text-muted)";
  statusEl.textContent = `Indexing ${file.name}...`;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await apiFetch("/api/v1/documents/upload", {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || "Upload failed");
    }

    const doc = await res.json();
    const shortSha = doc.sha256_checksum ? doc.sha256_checksum.substring(0, 8) : "";
    statusEl.style.color = "#065f46";
    statusEl.textContent = `Indexed: ${doc.title} (v${doc.version || 1}, ${doc.chunk_count} partitions, SHA: ${shortSha})`;
    loadDocuments();
    setTimeout(() => { statusEl.textContent = ""; }, 4000);
  } catch (err) {
    statusEl.style.color = "#991b1b";
    statusEl.textContent = `Error: ${err.message}`;
  }
}

// Threads

async function loadThreads() {
  const listContainer = document.getElementById("threads-list");
  const countBadge = document.getElementById("threads-count-badge");

  try {
    const res = await apiFetch("/api/v1/threads");
    if (!res.ok) throw new Error("Failed to load threads");
    allThreads = await res.json();

    if (countBadge) countBadge.textContent = allThreads.length;

    if (!listContainer) return;
    if (allThreads.length === 0) {
      listContainer.innerHTML = `<div class="empty-threads-notice">No past investigations. Start a query below.</div>`;
      return;
    }

    listContainer.innerHTML = allThreads.map(th => {
      const isActive = th.id === currentThreadId;
      const dateStr = formatRelativeTime(th.updated_at);
      return `
        <div class="thread-item ${isActive ? 'active' : ''}" onclick="selectThread('${th.id}')">
          <div class="thread-item-meta">
            <span class="thread-item-title">${escapeHtml(th.title)}</span>
            <span class="thread-item-sub">
              <span>${th.message_count} msg${th.message_count === 1 ? '' : 's'}</span>
              <span class="meta-dot">&bull;</span>
              <span>${dateStr}</span>
            </span>
          </div>
          <button type="button" class="thread-del-btn" onclick="deleteThread(event, '${th.id}')" title="Delete investigation session">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <line x1="18" y1="6" x2="6" y2="18"/>
              <line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
          </button>
        </div>
      `;
    }).join("");
  } catch (err) {
    if (listContainer) {
      listContainer.innerHTML = `<div class="empty-threads-notice">Error loading threads: ${escapeHtml(err.message)}</div>`;
    }
  }
}

async function selectThread(threadId) {
  if (currentThreadId === threadId) {
    if (window.innerWidth <= 960) toggleSidebar(false);
    return;
  }
  currentThreadId = threadId;

  if (window.innerWidth <= 960) {
    toggleSidebar(false);
  }

  // Highlight in sidebar
  document.querySelectorAll(".thread-item").forEach(item => {
    item.classList.toggle("active", item.getAttribute("onclick")?.includes(threadId));
  });

  const emptyState = document.getElementById("empty-investigation-state");
  const feed = document.getElementById("thread-messages-feed");
  const titleEl = document.getElementById("active-thread-title");
  const exportBtn = document.getElementById("btn-export-investigation-brief");

  try {
    const res = await apiFetch(`/api/v1/threads/${threadId}`);
    if (!res.ok) throw new Error("Could not load thread messages");
    const data = await res.json();

    if (titleEl) titleEl.textContent = data.thread.title;
    if (exportBtn) exportBtn.style.display = "inline-flex";

    activeThreadMessages = data.messages || [];

    if (emptyState) emptyState.style.display = "none";
    if (feed) {
      feed.innerHTML = "";
      activeThreadMessages.forEach(msg => {
        if (msg.role === "user") {
          feed.appendChild(createUserMessageElement(msg.content, msg.created_at));
        } else {
          feed.appendChild(createAssistantMessageElement(msg.content, msg.citations, msg.metadata, msg.created_at));
        }
      });
      scrollCanvasToBottom();
    }
  } catch (err) {
    if (feed) {
      feed.innerHTML = `<div class="synthesis-paragraph" style="color:#dc2626;">Error opening thread: ${escapeHtml(err.message)}</div>`;
    }
  }
}

function startNewInvestigation() {
  currentThreadId = null;
  activeThreadMessages = [];

  if (window.innerWidth <= 960) {
    toggleSidebar(false);
  }

  document.querySelectorAll(".thread-item").forEach(i => i.classList.remove("active"));

  const emptyState = document.getElementById("empty-investigation-state");
  const feed = document.getElementById("thread-messages-feed");
  const titleEl = document.getElementById("active-thread-title");
  const exportBtn = document.getElementById("btn-export-investigation-brief");
  const input = document.getElementById("query-input");

  if (emptyState) emptyState.style.display = "flex";
  if (feed) feed.innerHTML = "";
  if (titleEl) titleEl.textContent = "Enterprise Document Intelligence";
  if (exportBtn) exportBtn.style.display = "none";

  if (input) {
    input.value = "";
    input.focus();
  }
}

async function deleteThread(e, threadId) {
  e.stopPropagation();
  try {
    await apiFetch(`/api/v1/threads/${threadId}`, { method: "DELETE" });
    if (currentThreadId === threadId) {
      startNewInvestigation();
    }
    loadThreads();
  } catch (err) {
    console.error("Failed to delete thread", err);
  }
}

function executeStarterInquiry(btn) {
  const query = btn.getAttribute("data-query");
  if (!query) return;
  const input = document.getElementById("query-input");
  if (input) input.value = query;
  submitQuery(query);
}

// Query submission

async function submitQuery(queryText) {
  const input = document.getElementById("query-input");
  const submitBtn = document.getElementById("btn-submit-query");
  const emptyState = document.getElementById("empty-investigation-state");
  const feed = document.getElementById("thread-messages-feed");
  const exportBtn = document.getElementById("btn-export-investigation-brief");

  if (input) input.value = "";
  if (emptyState) emptyState.style.display = "none";
  if (exportBtn) exportBtn.style.display = "inline-flex";

  // Append user message immediately
  activeThreadMessages.push({
    role: "user",
    content: queryText,
    created_at: new Date().toISOString()
  });
  if (feed) {
    feed.appendChild(createUserMessageElement(queryText, new Date().toISOString()));
  }

  // Append loading state placeholder
  const loadingId = `loading-${Date.now()}`;
  const loadingEl = document.createElement("div");
  loadingEl.id = loadingId;
  loadingEl.className = "msg-asst-card";
  loadingEl.innerHTML = `
    <div class="benchmark-loading">
      <div class="loading-spinner"></div>
      <span>Retrieving semantic chunks &amp; verifying facts...</span>
    </div>
  `;
  if (feed) {
    feed.appendChild(loadingEl);
    scrollCanvasToBottom();
  }

  if (submitBtn) {
    submitBtn.disabled = true;
    submitBtn.innerHTML = `<span>Searching...</span>`;
  }

  try {
    const res = await apiFetch("/api/v1/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: queryText,
        top_k: retrievalConfig.topK,
        dense_weight: retrievalConfig.denseWeight,
        bm25_weight: retrievalConfig.bm25Weight,
        refusal_threshold: retrievalConfig.refusalThreshold,
        thread_id: currentThreadId,
        document_filter: getActiveDocFilter(),
        multi_hop: retrievalConfig.multiHop
      })
    });

    if (!res.ok) throw new Error("Query execution failed");
    const data = await res.json();

    activeThreadMessages.push({
      role: "assistant",
      content: data.answer,
      citations: data.citations || [],
      created_at: new Date().toISOString()
    });

    // Set thread ID and refresh thread ledger in sidebar
    currentThreadId = data.thread_id;
    loadThreads();

    // Update active title if first inquiry
    const titleEl = document.getElementById("active-thread-title");
    if (titleEl && (titleEl.textContent === "Enterprise Document Intelligence" || !titleEl.textContent)) {
      titleEl.textContent = queryText.length > 55 ? queryText.substring(0, 52) + "..." : queryText;
    }

    // Replace loading card with assistant response
    const asstEl = createAssistantMessageElement(
      data.answer,
      data.citations,
      {
        confidence_score: data.confidence_score,
        groundedness_score: data.groundedness_score,
        processing_time_ms: data.processing_time_ms,
        model_used: data.model_used,
        is_out_of_scope: data.is_out_of_scope,
        query_log_id: data.query_log_id
      },
      new Date().toISOString()
    );

    const placeholder = document.getElementById(loadingId);
    if (placeholder && placeholder.parentNode) {
      placeholder.parentNode.replaceChild(asstEl, placeholder);
    }

    updateHistoryBadge();
    scrollCanvasToBottom();
  } catch (err) {
    const placeholder = document.getElementById(loadingId);
    if (placeholder) {
      placeholder.innerHTML = `<div class="synthesis-paragraph" style="color:#dc2626;">Error executing query: ${escapeHtml(err.message)}</div>`;
    }
  } finally {
    if (submitBtn) {
      submitBtn.disabled = false;
      submitBtn.innerHTML = `
        <span class="inquiry-btn-text">Run Inquiry</span>
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-left: 4px;">
          <line x1="5" y1="12" x2="19" y2="12"/>
          <polyline points="12 5 19 12 12 19"/>
        </svg>
      `;
    }
  }
}

function createUserMessageElement(text, timestamp) {
  const card = document.createElement("div");
  card.className = "msg-user-card";
  card.innerHTML = `
    <div class="msg-user-header">
      <span class="msg-user-badge">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="4"/>
          <path d="M16 8v5a3 3 0 0 0 6 0v-1a10 10 0 1 0-4 8"/>
        </svg>
        Inquiry
      </span>
      <span class="msg-timestamp">${formatTimeOnly(timestamp)}</span>
    </div>
    <div class="msg-user-text">${escapeHtml(text)}</div>
  `;
  return card;
}

function createAssistantMessageElement(answerText, citations, meta, timestamp) {
  const card = document.createElement("div");
  card.className = "msg-asst-card";

  citations = citations || [];
  meta = meta || {};

  const confidencePct = meta.confidence_score ? Math.round(meta.confidence_score * 100) : 95;
  const latency = meta.processing_time_ms ? `${meta.processing_time_ms}ms` : "Fast";
  const isOutOfScope = meta.is_out_of_scope;

  const formattedHtml = formatSynthesizedAnswer(answerText, citations);

  card.innerHTML = `
    <div class="asst-card-header">
      <div class="asst-header-left">
        ${isOutOfScope ? `
          <span class="asst-badge-verified" style="background:#fef2f2; color:#991b1b; border-color:#fecaca;">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="12" cy="12" r="10"/>
              <line x1="12" y1="8" x2="12" y2="12"/>
              <line x1="12" y1="16" x2="12.01" y2="16"/>
            </svg>
            Out-of-Scope (Guardrail Triggered)
          </span>
        ` : `
          <span class="asst-badge-verified">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <polyline points="20 6 9 17 4 12"/>
            </svg>
            Verified Grounded (${confidencePct}% Confidence)
          </span>
        `}
        <span class="asst-latency-pill">${latency}</span>
      </div>
      <div class="asst-header-actions">
        <div class="feedback-actions-group">
          <button type="button" class="btn-feedback" onclick="sendFeedback(this, true, '${meta.query_log_id || ''}')" title="Helpful response">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/>
            </svg>
            <span>Helpful</span>
          </button>
          <button type="button" class="btn-feedback" onclick="sendFeedback(this, false, '${meta.query_log_id || ''}')" title="Unhelpful or inaccurate">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3"/>
            </svg>
            <span>Unhelpful</span>
          </button>
        </div>
        <button type="button" class="btn btn-sm btn-ghost" onclick="copyCardAnswer(this)" title="Copy text to clipboard">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect width="14" height="14" x="8" y="8" rx="2" ry="2"/>
            <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>
          </svg>
          <span>Copy</span>
        </button>
      </div>
    </div>

    <div class="synthesis-content">
      ${formattedHtml}
    </div>

    ${citations.length > 0 ? `
      <div class="asst-card-footer">
        <div class="citations-badge-group">
          <span style="font-size:11px; font-weight:600; color:var(--text-muted); margin-right:4px;">Verified Sources:</span>
          ${citations.map(c => `
            <button type="button" class="cite-source-pill" onclick="openSplitReader('${c.document_id}', '${c.chunk_id}', ${c.citation_index})" title="Inspect chunk in context inside Split Document Reader">
              <span style="font-weight:700; color:var(--accent-blue);">[${c.citation_index}]</span>
              <span>${escapeHtml(c.document_title)}</span>
              ${c.section_title ? `<span style="color:var(--text-muted);">&bull; ${escapeHtml(c.section_title)}</span>` : ''}
            </button>
          `).join("")}
        </div>
      </div>
    ` : ''}
  `;

  return card;
}

// Synthesis and citations

function formatSynthesizedAnswer(rawAnswer, citations) {
  if (!rawAnswer) return "";

  // 1. Structured Extractive Format: Check for "According to **<Doc>** in <Sec> [<idx>]:"
  const structuredRegex = /According to \*\*([^\*]+?)\*\*(?: in ([^\[\n]+?))? \[(\d+)\]:\s*"?([\s\S]*?)"?(?=\n\nAccording to|$)/g;
  let matches = [];
  let match;
  while ((match = structuredRegex.exec(rawAnswer)) !== null) {
    matches.push({
      docTitle: match[1].trim(),
      sectionTitle: match[2] ? match[2].trim() : "",
      citationIndex: match[3],
      content: match[4].trim()
    });
  }

  if (matches.length > 0) {
    const firstIdx = rawAnswer.indexOf("According to");
    let leadText = firstIdx > 0 ? rawAnswer.substring(0, firstIdx).trim() : "";
    
    let html = '<div class="synthesis-container">';
    
    if (leadText) {
      html += `
        <div class="synthesis-lead-banner">
          <div class="lead-text">${escapeHtml(leadText)}</div>
        </div>
      `;
    }

    matches.forEach(item => {
      const citeObj = (citations || []).find(c => String(c.citation_index) === String(item.citationIndex));
      const docId = citeObj ? citeObj.document_id : "";
      const chunkId = citeObj ? citeObj.chunk_id : "";

      const cleanContent = renderMarkdownSnippet(item.content, item.sectionTitle, docId, chunkId, item.citationIndex);
      html += `
        <article class="evidence-block">
          <header class="evidence-header">
            <div class="evidence-doc-info">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color:var(--accent-blue);">
                <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/>
                <polyline points="14 2 14 8 20 8"/>
              </svg>
              <span class="evidence-doc-title">${escapeHtml(item.docTitle)}</span>
              ${item.sectionTitle ? `<span class="evidence-section-tag">${escapeHtml(item.sectionTitle)}</span>` : ''}
            </div>
            <button type="button" class="citation-pill-btn" onclick="openSplitReader('${docId}', '${chunkId}', ${item.citationIndex})" title="Open native document sheet and inspect Citation [${item.citationIndex}] in context">
              <span>View Source [${item.citationIndex}]</span>
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <line x1="7" y1="17" x2="17" y2="7"/>
                <polyline points="7 7 17 7 17 17"/>
              </svg>
            </button>
          </header>
          <div class="evidence-content">
            ${cleanContent}
          </div>
        </article>
      `;
    });

    html += '</div>';
    return html;
  }

  // Fallback: General Markdown response with inline [1] badge transforms
  return `<div class="synthesis-general-content">${transformInlineFootnotes(renderGeneralMarkdown(rawAnswer), citations)}</div>`;
}

function renderMarkdownSnippet(text, sectionTitle, docId, chunkId, citationIndex) {
  if (!text) return "";

  let cleaned = text.replace(/^"+|"+$/g, '').replace(/---/g, '').trim();
  const rawLines = cleaned.split("\n");
  let html = "";
  let inUl = false;
  let inOl = false;

  for (let i = 0; i < rawLines.length; i++) {
    let line = rawLines[i].trim();
    if (!line) {
      if (inUl) { html += "</ul>"; inUl = false; }
      if (inOl) { html += "</ol>"; inOl = false; }
      continue;
    }

    if (line.startsWith("### ") || line.startsWith("## ") || line.startsWith("# ")) {
      const headingText = line.replace(/^#+\s*/, '').trim();
      if (sectionTitle && headingText.toLowerCase() === sectionTitle.toLowerCase()) {
        continue;
      }
      if (inUl) { html += "</ul>"; inUl = false; }
      if (inOl) { html += "</ol>"; inOl = false; }
      html += `<h4 class="evidence-subheading" style="font-size:12.5px; font-weight:700; margin:10px 0 6px;">${formatInlineMarkdown(headingText)}</h4>`;
      continue;
    }

    if (line.startsWith("- ") || line.startsWith("* ")) {
      if (inOl) { html += "</ol>"; inOl = false; }
      if (!inUl) { html += '<ul class="synthesis-list">'; inUl = true; }
      const itemText = line.substring(2).trim();
      html += `<li>${formatInlineMarkdown(itemText)}</li>`;
      continue;
    }

    const numMatch = line.match(/^(\d+)\.\s+(.*)/);
    if (numMatch) {
      if (inUl) { html += "</ul>"; inUl = false; }
      if (!inOl) { html += '<ol class="synthesis-list" style="list-style-type:decimal;">'; inOl = true; }
      html += `<li>${formatInlineMarkdown(numMatch[2].trim())}</li>`;
      continue;
    }

    if (inUl) { html += "</ul>"; inUl = false; }
    if (inOl) { html += "</ol>"; inOl = false; }
    html += `<p class="synthesis-paragraph">${formatInlineMarkdown(line)}</p>`;
  }

  if (inUl) html += "</ul>";
  if (inOl) html += "</ol>";

  return html;
}

function renderGeneralMarkdown(text) {
  if (!text) return "";
  const rawLines = text.split("\n");
  let html = "";
  let inUl = false;

  for (let i = 0; i < rawLines.length; i++) {
    let line = rawLines[i].trim();
    if (!line) {
      if (inUl) { html += "</ul>"; inUl = false; }
      continue;
    }
    if (line.startsWith("- ") || line.startsWith("* ")) {
      if (!inUl) { html += '<ul class="synthesis-list">'; inUl = true; }
      html += `<li>${formatInlineMarkdown(line.substring(2).trim())}</li>`;
      continue;
    }
    if (inUl) { html += "</ul>"; inUl = false; }
    html += `<p class="synthesis-paragraph">${formatInlineMarkdown(line)}</p>`;
  }
  if (inUl) html += "</ul>";
  return html;
}

function transformInlineFootnotes(html, citations) {
  if (!citations || citations.length === 0) return html;
  return html.replace(/\[(\d+)\]/g, (match, idx) => {
    const cite = citations.find(c => String(c.citation_index) === String(idx));
    if (cite) {
      return `<button type="button" class="cite-tag-inline" onclick="openSplitReader('${cite.document_id}', '${cite.chunk_id}', ${cite.citation_index})" title="Inspect Source [${idx}] in Split Reader">[${idx}]</button>`;
    }
    return match;
  });
}

function formatInlineMarkdown(text) {
  if (!text) return "";
  // Strictly escape all HTML entities first to prevent Stored XSS
  const safeText = escapeHtml(text);
  return safeText
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code style="font-family:\'JetBrains Mono\',monospace; font-size:12px; background:#e2e8f0; padding:1px 4px; border-radius:3px;">$1</code>');
}

// Document reader

async function openSplitReader(docId, highlightChunkId, citationIndex) {
  const pane = document.getElementById("split-reader-pane");
  const titleEl = document.getElementById("reader-doc-title");
  const colTag = document.getElementById("reader-collection-tag");
  const citeBadge = document.getElementById("reader-active-cite-badge");
  const secName = document.getElementById("reader-section-name");
  const jumpPillsEl = document.getElementById("reader-jump-pills");
  const bodyEl = document.getElementById("reader-body");

  if (!pane) return;

  if (window.innerWidth <= 960) {
    toggleSidebar(false);
  }

  pane.style.display = "flex";
  activeSplitReaderDoc = docId;
  currentHighlightedChunkId = highlightChunkId;

  if (titleEl) titleEl.textContent = "Loading source document...";
  if (bodyEl) {
    bodyEl.innerHTML = `
      <div class="benchmark-loading">
        <div class="loading-spinner"></div>
        <span>Retrieving full document and mapping cited chunks...</span>
      </div>
    `;
  }

  try {
    const res = await apiFetch(`/api/v1/documents/${docId}`);
    if (!res.ok) throw new Error("Failed to load source document");
    const data = await res.json();
    const doc = data.document;
    const chunks = data.chunks || [];

    if (titleEl) titleEl.textContent = doc.title;
    if (colTag) colTag.textContent = doc.collection || "Enterprise Library";
    if (citeBadge) {
      citeBadge.style.display = citationIndex ? "inline-flex" : "none";
      if (citationIndex) citeBadge.textContent = `Citation [${citationIndex}] in Context`;
    }

    // Default target chunk if none provided
    if (!highlightChunkId && chunks.length > 0) {
      highlightChunkId = chunks[0].id;
      currentHighlightedChunkId = highlightChunkId;
    }

    // Determine target section
    const targetChunk = chunks.find(c => c.id === highlightChunkId);
    if (secName) {
      secName.textContent = targetChunk ? `Section: ${targetChunk.section_title || 'General'}` : "";
    }

    // Render jump pills
    if (jumpPillsEl) {
      jumpPillsEl.innerHTML = chunks.map((c, i) => {
        const isTarget = c.id === highlightChunkId;
        return `
          <button type="button" class="reader-jump-pill ${isTarget ? 'active' : ''}" onclick="jumpToChunkInReader('${c.id}')" title="Partition #${c.chunk_index}: ${escapeHtml(c.section_title || 'General')}">
            <span>Chunk ${c.chunk_index + 1}</span>
          </button>
        `;
      }).join("");
    }

    // Render full chunks body
    if (bodyEl) {
      bodyEl.innerHTML = chunks.map(c => {
        const isTarget = c.id === highlightChunkId;
        return `
          <div id="reader-chunk-${c.id}" class="reader-chunk-block ${isTarget ? 'is-target' : ''}">
            <div class="reader-chunk-header">
              <span>Partition #${c.chunk_index + 1} &bull; ${escapeHtml(c.section_title || 'Overview')}</span>
              <span>${c.word_count} words</span>
            </div>
            <div class="reader-chunk-text">${escapeHtml(c.text)}</div>
          </div>
        `;
      }).join("");

      // Smooth scroll to target chunk
      setTimeout(() => {
        const targetEl = document.getElementById(`reader-chunk-${highlightChunkId}`);
        if (targetEl) {
          targetEl.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      }, 50);
    }
  } catch (err) {
    if (bodyEl) {
      bodyEl.innerHTML = `<div class="synthesis-paragraph" style="color:#dc2626;">Error loading document: ${escapeHtml(err.message)}</div>`;
    }
  }
}

function jumpToChunkInReader(chunkId) {
  currentHighlightedChunkId = chunkId;
  document.querySelectorAll(".reader-chunk-block").forEach(b => {
    b.classList.toggle("is-target", b.id === `reader-chunk-${chunkId}`);
  });
  document.querySelectorAll(".reader-jump-pill").forEach(p => {
    p.classList.toggle("active", p.getAttribute("onclick")?.includes(chunkId));
  });
  const targetEl = document.getElementById(`reader-chunk-${chunkId}`);
  if (targetEl) {
    targetEl.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

function closeSplitReader() {
  const pane = document.getElementById("split-reader-pane");
  if (pane) pane.style.display = "none";
  activeSplitReaderDoc = null;
  currentHighlightedChunkId = null;
}

// Sidebar navigation

function toggleSidebar(forceState) {
  const layout = document.getElementById("workspace-layout");
  const toggleBtn = document.getElementById("btn-toggle-sidebar");
  if (!layout) return;

  const isMobile = window.innerWidth <= 960;

  if (isMobile) {
    const shouldOpen = typeof forceState === "boolean" ? forceState : !layout.classList.contains("sidebar-open");
    layout.classList.toggle("sidebar-open", shouldOpen);
    if (toggleBtn) {
      toggleBtn.setAttribute("aria-expanded", String(shouldOpen));
      toggleBtn.classList.toggle("active", shouldOpen);
    }
  } else {
    const shouldCollapse = typeof forceState === "boolean" ? !forceState : !layout.classList.contains("sidebar-collapsed");
    layout.classList.toggle("sidebar-collapsed", shouldCollapse);
    if (toggleBtn) {
      toggleBtn.setAttribute("aria-expanded", String(!shouldCollapse));
      toggleBtn.classList.toggle("active", !shouldCollapse);
    }
    try {
      localStorage.setItem("cortex_sidebar_collapsed", shouldCollapse ? "true" : "false");
    } catch (e) {}
  }
}

// export investigation brief

function formatReportMarkdown(text) {
  if (!text) return "";
  let clean = escapeHtml(text);
  clean = clean.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  clean = clean.replace(/\[(\d+)\]/g, '<span class="citation-ref">[$1]</span>');
  clean = clean.replace(/^### (.*$)/gim, '<h4 style="margin:12px 0 6px 0; font-size:13px; color:#0f172a;">$1</h4>');
  clean = clean.replace(/^## (.*$)/gim, '<h3 style="margin:14px 0 8px 0; font-size:14px; color:#0f172a;">$1</h3>');
  clean = clean.replace(/^\- (.*$)/gim, '<li>$1</li>');
  clean = clean.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
  const paragraphs = clean.split(/\n\n+/).map(p => {
    p = p.trim();
    if (p.startsWith('<h') || p.startsWith('<ul') || p.startsWith('<ol')) return p;
    return `<p style="margin:0 0 10px 0;">${p.replace(/\n/g, '<br>')}</p>`;
  });
  return paragraphs.join("");
}

function generateInvestigationReportHtml(threadTitle, messages, workspaceName, userName) {
  const now = new Date();
  const dateStr = now.toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit"
  });

  const totalInquiries = messages.filter(m => m.role === "user").length;
  const allCitations = messages.flatMap(m => m.citations || []);
  const verifiedCitations = allCitations.filter(c => c.verification_status === "verified").length;
  const verifiedPct = allCitations.length > 0 ? Math.round((verifiedCitations / allCitations.length) * 100) : 100;

  let contentHtml = "";
  messages.forEach((msg) => {
    if (msg.role === "user") {
      contentHtml += `
        <div class="report-section inquiry-section">
          <div class="inquiry-header">
            <span class="inquiry-badge">Inquiry</span>
            <span class="inquiry-timestamp">${escapeHtml(formatRelativeTime(msg.created_at || new Date().toISOString()))}</span>
          </div>
          <h2 class="inquiry-text">${escapeHtml(msg.content)}</h2>
        </div>
      `;
    } else {
      const citations = msg.citations || [];
      contentHtml += `
        <div class="report-section response-section">
          <div class="section-label">Synthesized Findings</div>
          <div class="synthesis-body">${formatReportMarkdown(msg.content)}</div>
      `;

      if (citations.length > 0) {
        contentHtml += `
          <div class="citations-container">
            <div class="citations-header">
              <span class="citations-title">Verified Source Citations (${citations.length})</span>
            </div>
            <table class="citation-table">
              <thead>
                <tr>
                  <th style="width: 48px; text-align: center;">Ref</th>
                  <th style="width: 230px; text-align: left;">Source Document &amp; Section</th>
                  <th style="width: 95px; text-align: center;">Status</th>
                  <th style="text-align: left;">Exact Source Excerpt</th>
                </tr>
              </thead>
              <tbody>
                ${citations.map(c => `
                  <tr>
                    <td style="text-align: center;">
                      <span class="citation-ref">[${c.citation_index}]</span>
                    </td>
                    <td>
                      <div class="source-doc-title">${escapeHtml(c.document_title)}</div>
                      <div class="source-doc-meta">${escapeHtml(c.section_title || 'General')} &bull; Page ${c.page_number || 1}</div>
                    </td>
                    <td style="text-align: center;">
                      <span class="status-pill status-${c.verification_status === 'verified' ? 'verified' : 'partial'}">
                        ${c.verification_status === 'verified' ? 'Verified' : 'Partial'}
                      </span>
                    </td>
                    <td>
                      <blockquote class="excerpt-quote">
                        &ldquo;${escapeHtml(c.exact_quote || '')}&rdquo;
                      </blockquote>
                    </td>
                  </tr>
                `).join("")}
              </tbody>
            </table>
          </div>
        `;
      }
      contentHtml += `</div>`;
    }
  });

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Investigation Brief - ${escapeHtml(threadTitle)}</title>
  <style>
    @page {
      size: letter portrait;
      margin: 16mm 20mm;
    }
    *, *::before, *::after {
      box-sizing: border-box;
    }
    body {
      margin: 0;
      padding: 0;
      background: #f8fafc;
      color: #0f172a;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      font-size: 13.5px;
      line-height: 1.6;
      -webkit-font-smoothing: antialiased;
    }
    .toolbar-bar {
      position: sticky;
      top: 0;
      z-index: 100;
      background: #ffffff;
      border-bottom: 1px solid #e2e8f0;
      padding: 12px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .toolbar-title {
      font-weight: 600;
      font-size: 14px;
      color: #334155;
    }
    .toolbar-actions {
      display: flex;
      gap: 8px;
    }
    .btn {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 7px 14px;
      font-size: 12.5px;
      font-weight: 600;
      border-radius: 6px;
      cursor: pointer;
      border: 1px solid transparent;
      text-decoration: none;
      transition: all 0.15s ease;
    }
    .btn-primary {
      background: #0284c7;
      color: #ffffff;
    }
    .btn-primary:hover {
      background: #0369a1;
    }
    .btn-secondary {
      background: #ffffff;
      color: #334155;
      border-color: #cbd5e1;
    }
    .btn-secondary:hover {
      background: #f1f5f9;
    }
    .report-page {
      max-width: 820px;
      margin: 32px auto 60px auto;
      background: #ffffff;
      padding: 48px 52px;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.04);
    }
    .report-header {
      border-bottom: 2px solid #0f172a;
      padding-bottom: 20px;
      margin-bottom: 28px;
    }
    .report-brand-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
    }
    .report-brand-name {
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.8px;
      text-transform: uppercase;
      color: #0284c7;
    }
    .report-classification {
      font-size: 10.5px;
      font-weight: 700;
      letter-spacing: 1px;
      text-transform: uppercase;
      padding: 3px 8px;
      background: #f1f5f9;
      color: #475569;
      border-radius: 4px;
      border: 1px solid #e2e8f0;
    }
    .report-title {
      font-size: 24px;
      font-weight: 700;
      color: #0f172a;
      margin: 0 0 16px 0;
      line-height: 1.25;
    }
    .meta-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      background: #f8fafc;
      padding: 12px 16px;
      border-radius: 6px;
      border: 1px solid #e2e8f0;
      font-size: 11.5px;
    }
    .meta-item-label {
      color: #64748b;
      margin-bottom: 2px;
      text-transform: uppercase;
      font-size: 9.5px;
      font-weight: 600;
      letter-spacing: 0.5px;
    }
    .meta-item-val {
      color: #1e293b;
      font-weight: 600;
    }
    .report-section {
      margin-bottom: 28px;
      break-inside: avoid;
      page-break-inside: avoid;
    }
    .inquiry-section {
      background: #f8fafc;
      border-left: 3px solid #0284c7;
      padding: 14px 18px;
      border-radius: 0 6px 6px 0;
    }
    .inquiry-header {
      display: flex;
      justify-content: space-between;
      margin-bottom: 4px;
    }
    .inquiry-badge {
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: #0284c7;
    }
    .inquiry-timestamp {
      font-size: 11px;
      color: #94a3b8;
    }
    .inquiry-text {
      font-size: 15px;
      font-weight: 600;
      color: #0f172a;
      margin: 0;
      line-height: 1.4;
    }
    .section-label {
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      color: #475569;
      margin-bottom: 10px;
    }
    .synthesis-body {
      font-size: 13.5px;
      color: #1e293b;
      line-height: 1.7;
    }
    .citations-container {
      margin-top: 18px;
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      overflow: hidden;
      break-inside: avoid;
      page-break-inside: avoid;
    }
    .citations-header {
      background: #f8fafc;
      padding: 8px 14px;
      border-bottom: 1px solid #e2e8f0;
      font-size: 11.5px;
      font-weight: 700;
      color: #334155;
    }
    .citation-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      table-layout: fixed;
    }
    .citation-table th {
      background: #f1f5f9;
      color: #475569;
      font-weight: 600;
      font-size: 10.5px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      padding: 8px 12px;
      border-bottom: 1px solid #cbd5e1;
    }
    .citation-table td {
      padding: 10px 12px;
      border-bottom: 1px solid #f1f5f9;
      vertical-align: top;
      line-height: 1.45;
    }
    .citation-table tr:last-child td {
      border-bottom: none;
    }
    .citation-ref {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-weight: 700;
      color: #0284c7;
      background: #e0f2fe;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 11px;
    }
    .source-doc-title {
      font-weight: 600;
      color: #0f172a;
      margin-bottom: 2px;
    }
    .source-doc-meta {
      font-size: 11px;
      color: #64748b;
    }
    .status-pill {
      font-size: 10.5px;
      font-weight: 600;
      padding: 2px 7px;
      border-radius: 4px;
      display: inline-block;
    }
    .status-verified {
      background: #dcfce7;
      color: #15803d;
    }
    .status-partial {
      background: #fef3c7;
      color: #b45309;
    }
    .excerpt-quote {
      margin: 0;
      font-style: italic;
      color: #334155;
      background: #f8fafc;
      padding: 6px 10px;
      border-left: 2px solid #cbd5e1;
      border-radius: 0 4px 4px 0;
      word-break: break-word;
    }
    .report-footer {
      margin-top: 36px;
      padding-top: 14px;
      border-top: 1px solid #e2e8f0;
      display: flex;
      justify-content: space-between;
      color: #94a3b8;
      font-size: 10.5px;
    }
    @media print {
      body {
        background: #ffffff !important;
        font-size: 11pt;
      }
      .toolbar-bar {
        display: none !important;
      }
      .report-page {
        margin: 0 !important;
        padding: 0 !important;
        border: none !important;
        box-shadow: none !important;
        max-width: 100% !important;
      }
      .report-section, .citations-container, .citation-table {
        break-inside: avoid !important;
        page-break-inside: avoid !important;
      }
    }
  </style>
</head>
<body>
  <div class="toolbar-bar no-print">
    <div class="toolbar-title">
      <span>Investigation Memo Preview</span>
    </div>
    <div class="toolbar-actions">
      <button type="button" class="btn btn-primary" onclick="window.print()">
        Print / Save as PDF
      </button>
      <button type="button" class="btn btn-secondary" onclick="downloadHtmlReport()">
        Download HTML
      </button>
      <button type="button" class="btn btn-secondary" onclick="window.close()">
        Close
      </button>
    </div>
  </div>

  <div class="report-page">
    <div class="report-header">
      <div class="report-brand-row">
        <span class="report-brand-name">Cortex Intelligence</span>
        <span class="report-classification">Internal Audit Memo</span>
      </div>
      <h1 class="report-title">${escapeHtml(threadTitle)}</h1>
      <div class="meta-grid">
        <div>
          <div class="meta-item-label">Date Generated</div>
          <div class="meta-item-val">${escapeHtml(dateStr)}</div>
        </div>
        <div>
          <div class="meta-item-label">Workspace</div>
          <div class="meta-item-val">${escapeHtml(workspaceName)}</div>
        </div>
        <div>
          <div class="meta-item-label">Analyst</div>
          <div class="meta-item-val">${escapeHtml(userName)}</div>
        </div>
        <div>
          <div class="meta-item-label">Citation Grounding</div>
          <div class="meta-item-val">${verifiedPct}% Verified (${allCitations.length} cited)</div>
        </div>
      </div>
    </div>

    ${contentHtml}

    <div class="report-footer">
      <span>Enterprise Document Intelligence Platform &bull; Verifiable Hybrid Retrieval</span>
      <span>Investigation Brief</span>
    </div>
  </div>

  <script>
    function downloadHtmlReport() {
      const clone = document.documentElement.cloneNode(true);
      const tb = clone.querySelector('.toolbar-bar');
      if (tb) tb.remove();
      const html = '<!DOCTYPE html>' + clone.outerHTML;
      const blob = new Blob([html], { type: 'text/html;charset=utf-8' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'investigation_brief_' + Date.now() + '.html';
      a.click();
    }
  </script>
</body>
</html>`;
}

function exportInvestigationBrief() {
  if (activeThreadMessages.length === 0) return;

  const titleEl = document.getElementById("active-thread-title");
  const threadTitle = titleEl ? titleEl.textContent.trim() : "Investigation Brief";
  const wsSelector = document.getElementById("workspace-select");
  const wsName = wsSelector && wsSelector.selectedOptions.length > 0 ? wsSelector.selectedOptions[0].textContent.trim() : "Production Workspace";
  const userName = currentUser ? (currentUser.full_name || currentUser.email) : "Guest Analyst";

  const reportHtml = generateInvestigationReportHtml(threadTitle, activeThreadMessages, wsName, userName);

  const printWin = window.open("", "_blank");
  if (printWin) {
    printWin.document.open();
    printWin.document.write(reportHtml);
    printWin.document.close();
  }
}

function copyCardAnswer(btn) {
  const card = btn.closest(".msg-asst-card");
  if (!card) return;
  const content = card.querySelector(".synthesis-content");
  if (!content) return;

  navigator.clipboard.writeText(content.innerText).then(() => {
    const origHtml = btn.innerHTML;
    btn.innerHTML = `<span>Copied!</span>`;
    setTimeout(() => { btn.innerHTML = origHtml; }, 2000);
  });
}

// Modals and benchmarks

function openEngineModal() {
  const modal = document.getElementById("engine-modal-backdrop");
  if (modal) modal.style.display = "flex";
}

function closeEngineModal() {
  const modal = document.getElementById("engine-modal-backdrop");
  if (modal) modal.style.display = "none";
}

function closeEngineModalOnBackdrop(e) {
  if (e.target.id === "engine-modal-backdrop") closeEngineModal();
}

function openAuditModal() {
  const modal = document.getElementById("audit-modal-backdrop");
  if (modal) {
    modal.style.display = "flex";
    loadAuditLedger();
  }
}

function closeAuditModal() {
  const modal = document.getElementById("audit-modal-backdrop");
  if (modal) modal.style.display = "none";
}

function closeAuditModalOnBackdrop(e) {
  if (e.target.id === "audit-modal-backdrop") closeAuditModal();
}

async function loadAuditLedger() {
  const listEl = document.getElementById("history-ledger-list");
  if (!listEl) return;
  listEl.innerHTML = `
    <div class="benchmark-loading">
      <div class="loading-spinner"></div>
      <span>Loading compliance & query execution ledger...</span>
    </div>
  `;

  try {
    let auditEvents = [];
    if (currentUser) {
      try {
        const aRes = await apiFetch("/api/v1/audit?limit=25");
        if (aRes.ok) {
          auditEvents = await aRes.json();
        }
      } catch (e) {}
    }

    const res = await apiFetch("/api/v1/query/history");
    const queryLogs = res.ok ? await res.json() : [];

    if (auditEvents.length === 0 && queryLogs.length === 0) {
      listEl.innerHTML = `<div class="empty-threads-notice">No audit logs or queries recorded yet.</div>`;
      return;
    }

    let html = "";
    if (auditEvents.length > 0) {
      html += `
        <div style="margin-bottom:12px; font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.5px; color:var(--text-muted);">
          Immutable Governance Ledger (Identity-Attributed Events)
        </div>
      `;
      html += auditEvents.slice(0, 10).map(e => `
        <div class="history-card" style="cursor:default; border-left: 3px solid #3b82f6;">
          <div class="history-card-header">
            <span class="history-card-query" style="font-family:monospace; font-size:12px;">${escapeHtml(e.action)} &bull; ${escapeHtml(e.resource_type)}</span>
            <span style="font-size:11px; font-weight:600; color:#3b82f6; background:rgba(59,130,246,0.1); padding:2px 6px; border-radius:4px;">
              ${escapeHtml(e.actor_email || 'System')}
            </span>
          </div>
          <div class="history-card-meta">
            <span>Workspace: ${escapeHtml(e.workspace_id || 'Global')}</span>
            <span class="meta-dot">&bull;</span>
            <span>${formatRelativeTime(e.created_at)}</span>
          </div>
        </div>
      `).join("");
      html += `
        <div style="margin:16px 0 12px 0; font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.5px; color:var(--text-muted);">
          Query Execution Ledger (Click to replay inquiry)
        </div>
      `;
    }

    if (queryLogs.length > 0) {
      html += queryLogs.map(l => `
        <div class="history-card" onclick="startInquiryFromHistory('${escapeHtml(l.query)}')">
          <div class="history-card-header">
            <span class="history-card-query">${escapeHtml(l.query)}</span>
            <span style="font-size:11px; font-weight:600; color:${l.is_out_of_scope ? '#dc2626' : '#059669'};">
              ${l.is_out_of_scope ? 'Refused' : 'Grounded'}
            </span>
          </div>
          <div class="history-card-meta">
            <span>${Math.round(l.confidence * 100)}% Confidence</span>
            <span class="meta-dot">&bull;</span>
            <span>${l.duration_ms}ms</span>
            <span class="meta-dot">&bull;</span>
            <span>${formatRelativeTime(l.timestamp)}</span>
          </div>
        </div>
      `).join("");
    } else if (auditEvents.length === 0) {
      html += `<div class="empty-threads-notice">No query audit logs recorded yet.</div>`;
    }

    listEl.innerHTML = html;
  } catch (err) {
    listEl.innerHTML = `<div class="empty-threads-notice">Error: ${escapeHtml(err.message)}</div>`;
  }
}

function startInquiryFromHistory(query) {
  closeAuditModal();
  submitQuery(query);
}

async function updateHistoryBadge() {
  try {
    const res = await apiFetch("/api/v1/query/history");
    if (res.ok) {
      const logs = await res.json();
      const badge = document.getElementById("history-count-badge");
      if (badge) badge.textContent = logs.length;
    }
  } catch (e) {
    // silent fallback
  }
}

async function runBenchmarkEvaluation() {
  const container = document.getElementById("benchmark-results-container");
  const runBtn = document.getElementById("btn-run-benchmark");

  if (!container) return;
  container.style.display = "block";
  container.innerHTML = `
    <div class="benchmark-loading">
      <div class="loading-spinner"></div>
      <span>Executing 5 factual golden benchmark queries across corpus...</span>
    </div>
  `;
  if (runBtn) runBtn.disabled = true;

  try {
    const res = await apiFetch("/api/v1/evaluation/benchmark");
    if (!res.ok) throw new Error("Benchmark execution failed");
    const results = await res.json();

    const passedCount = results.filter(r => r.citation_found && r.grounded).length;
    const totalCount = results.length;
    const accuracyPct = totalCount > 0 ? Math.round((passedCount / totalCount) * 100) : 0;
    const avgLatency = Math.round(results.reduce((sum, r) => sum + r.latency_ms, 0) / (totalCount || 1));

    container.innerHTML = `
      <div class="benchmark-summary-bar">
        <span>Accuracy: <strong>${passedCount} / ${totalCount} Passed (${accuracyPct}%)</strong></span>
        <span>Mean Latency: <strong>${avgLatency}ms</strong></span>
      </div>
      ${results.map((r, i) => `
        <div class="benchmark-row">
          <span>${i + 1}. ${escapeHtml(r.query)}</span>
          <span class="${r.grounded ? 'benchmark-pass' : 'benchmark-fail'}">${r.grounded ? 'PASSED' : 'FAILED'} (${r.latency_ms}ms)</span>
        </div>
      `).join("")}
    `;
  } catch (err) {
    container.innerHTML = `<div class="synthesis-paragraph" style="color:#dc2626;">Benchmark Error: ${escapeHtml(err.message)}</div>`;
  } finally {
    if (runBtn) runBtn.disabled = false;
  }
}

async function openInspectorModal(docId) {
  if (!docId) return;
  const modal = document.getElementById("doc-inspector-modal");
  const titleEl = document.getElementById("inspector-doc-title");
  const metaRow = document.getElementById("inspector-meta-row");
  const chunksList = document.getElementById("inspector-chunks-list");

  if (!modal) return;
  modal.style.display = "flex";

  if (titleEl) titleEl.textContent = "Loading Document Vectors...";
  if (metaRow) metaRow.innerHTML = "";
  if (chunksList) {
    chunksList.innerHTML = `
      <div class="benchmark-loading">
        <div class="loading-spinner"></div>
        <span>Retrieving partition chunks and metadata...</span>
      </div>
    `;
  }

  try {
    const res = await apiFetch(`/api/v1/documents/${docId}`);
    if (!res.ok) throw new Error("Could not retrieve document");
    const data = await res.json();
    const doc = data.document;
    const chunks = data.chunks || [];

    if (titleEl) titleEl.textContent = doc.title;
    if (metaRow) {
      metaRow.innerHTML = `
        <span>Format: ${escapeHtml(doc.file_type)}</span>
        <span class="meta-dot">&bull;</span>
        <span>Version: v${doc.version || 1}</span>
        <span class="meta-dot">&bull;</span>
        <span>${chunks.length} Partitions</span>
        <span class="meta-dot">&bull;</span>
        <span>Checksum: ${(doc.sha256_checksum || '').substring(0, 8)}</span>
      `;
    }

    if (chunksList) {
      if (chunks.length === 0) {
        chunksList.innerHTML = `<div class="empty-threads-notice">No chunks found for this document.</div>`;
        return;
      }
      chunksList.innerHTML = chunks.map(c => `
        <div class="inspector-chunk-card">
          <div class="inspector-chunk-header">
            <span>Partition #${c.chunk_index + 1} &bull; ${escapeHtml(c.section_title || 'General')}</span>
            <span>${c.word_count} words (chars ${c.char_start}-${c.char_end})</span>
          </div>
          <div class="inspector-chunk-body">${escapeHtml(c.text)}</div>
        </div>
      `).join("");
    }
  } catch (err) {
    if (chunksList) {
      chunksList.innerHTML = `<div class="synthesis-paragraph" style="color:#dc2626;">Error inspecting document: ${escapeHtml(err.message)}</div>`;
    }
  }
}

function closeInspectorModal() {
  const modal = document.getElementById("doc-inspector-modal");
  if (modal) modal.style.display = "none";
}

function closeInspectorModalOnBackdrop(e) {
  if (e.target.id === "doc-inspector-modal") closeInspectorModal();
}

// Utilities

function scrollCanvasToBottom() {
  const scrollEl = document.getElementById("thread-messages-scroll");
  if (scrollEl) {
    setTimeout(() => {
      scrollEl.scrollTo({ top: scrollEl.scrollHeight, behavior: "smooth" });
    }, 50);
  }
}

function formatRelativeTime(isoString) {
  try {
    const date = new Date(isoString);
    const now = new Date();
    const diffSec = Math.floor((now - date) / 1000);
    if (diffSec < 60) return "Just now";
    if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
    if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`;
    return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  } catch (e) {
    return "";
  }
}

function formatTimeOnly(isoString) {
  try {
    return new Date(isoString).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch (e) {
    return "";
  }
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
