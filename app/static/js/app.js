/** PrivateRAG Web UI — vanilla JS, no build step.
 *  Copyright (c) 2026 PrivateRAG. All rights reserved.
 */

const STORAGE_KEY = "privaterag.conversation_id";
const SETTINGS_KEY = "privaterag.settings";
const SCOPE_KEY = "privaterag.document_scope";
const TOKEN_KEY = "privaterag.token";

const $ = (sel) => document.querySelector(sel);

const messagesEl = $("#messages");
const citationsList = $("#citations-list");
const citationsEmpty = $("#citations-empty");
const metaPanel = $("#meta-panel");
const metaProvider = $("#meta-provider");
const metaGrounded = $("#meta-grounded");
const metaNotes = $("#meta-notes");
const healthPill = $("#health-pill");
const docList = $("#doc-list");
const chatForm = $("#chat-form");
const queryInput = $("#query-input");
const sendBtn = $("#send-btn");
const fileInput = $("#file-input");
const uploadZone = $("#upload-zone");
const uploadProgress = $("#upload-progress");
const uploadProgressLabel = $("#upload-progress-label");
const uploadProgressPct = $("#upload-progress-pct");
const uploadProgressBar = $("#upload-progress-bar");
const loginScreen = $("#login-screen");
const appShell = $("#app-shell");
const openAdminBtn = $("#open-admin-btn");
const adminModal = $("#admin-modal");
const userBar = $("#user-bar");
const userLabel = $("#user-label");

let conversationId = localStorage.getItem(STORAGE_KEY) || null;
let isBusy = false;
let engineBeforeStream = null;
let allDocumentIds = [];
let selectedDocIds = new Set();
let currentUser = { username: "local", role: "admin", auth_enabled: false };
let authToken = localStorage.getItem(TOKEN_KEY) || "";

const STAGE_LABELS = {
  saving: "Saving file",
  extracting: "Extracting text",
  chunking: "Chunking",
  chunked: "Chunks ready",
  embedding: "Embedding",
  indexing: "Indexing",
  done: "Done",
};

function authHeaders(extra = {}) {
  const headers = { ...extra };
  if (authToken) headers.Authorization = `Bearer ${authToken}`;
  return headers;
}

async function apiFetch(url, options = {}) {
  const opts = { ...options };
  opts.headers = authHeaders(opts.headers || {});
  const res = await fetch(url, opts);
  if (res.status === 401 && currentUser.auth_enabled) {
    clearSession();
    showLogin("Session expired. Please sign in again.");
    throw new Error("Authentication required");
  }
  return res;
}

function loadSettings() {
  try {
    return JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}");
  } catch {
    return {};
  }
}

function saveSettings() {
  const settings = {
    mode: $("#mode-select").value,
    engine: $("#engine-select").value,
    stream: $("#stream-toggle").checked,
    top_k: Number($("#top-k-input").value) || 3,
  };
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
}

function applySettings() {
  const s = loadSettings();
  if (s.mode) $("#mode-select").value = s.mode;
  if (s.engine) $("#engine-select").value = s.engine;
  if (s.stream !== undefined) $("#stream-toggle").checked = s.stream;
  if (s.top_k) $("#top-k-input").value = s.top_k;
}

function loadScopeIds() {
  try {
    const raw = JSON.parse(localStorage.getItem(SCOPE_KEY) || "null");
    return Array.isArray(raw) ? raw : null;
  } catch {
    return null;
  }
}

function saveScopeIds() {
  localStorage.setItem(SCOPE_KEY, JSON.stringify([...selectedDocIds]));
}

function chatPayload(query) {
  const body = {
    query,
    mode: $("#mode-select").value,
    engine: $("#engine-select").value,
    top_k: Number($("#top-k-input").value) || 3,
  };
  if (conversationId) body.conversation_id = conversationId;
  if (
    selectedDocIds.size > 0 &&
    allDocumentIds.length > 0 &&
    selectedDocIds.size < allDocumentIds.length
  ) {
    body.document_ids = [...selectedDocIds];
  }
  return body;
}

function scrollMessages() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function providerLabel(provider) {
  const labels = {
    openai: "ChatGPT",
    gemini: "Gemini",
    ollama: "Ollama",
    extractive: "Extractive",
    none: "No LLM",
  };
  return labels[provider] || provider;
}

function providerAvatar(provider) {
  const icons = {
    openai: "✦",
    gemini: "✧",
    ollama: "◎",
    extractive: "◇",
    none: "◇",
  };
  return icons[provider] || "◇";
}

function formatAssistantHtml(text) {
  const lines = escapeHtml(text || "").split("\n");
  const html = [];
  for (const line of lines) {
    let l = line;
    if (/^\s*---+\s*$/.test(l)) {
      html.push('<hr class="md-hr" />');
      continue;
    }
    if (!l.trim()) {
      html.push('<div class="md-spacer"></div>');
      continue;
    }
    l = l.replace(
      /\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]/g,
      '<sup class="cite-ref">[$1]</sup>'
    );
    l = l.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    l = l.replace(/`([^`]+)`/g, "<code>$1</code>");
    if (/^###\s+/.test(l)) {
      html.push(`<h4 class="md-h">${l.replace(/^###\s+/, "")}</h4>`);
      continue;
    }
    if (/^##\s+/.test(l)) {
      html.push(`<h3 class="md-h">${l.replace(/^##\s+/, "")}</h3>`);
      continue;
    }
    if (/^#\s+/.test(l)) {
      html.push(`<h2 class="md-h">${l.replace(/^#\s+/, "")}</h2>`);
      continue;
    }
    if (/^[-*]\s+/.test(l)) {
      html.push(
        `<div class="md-li"><span class="md-bullet">•</span><span>${l.replace(/^[-*]\s+/, "")}</span></div>`
      );
      continue;
    }
    if (/^\d+\.\s+/.test(l)) {
      html.push(`<div class="md-li num">${l}</div>`);
      continue;
    }
    if (/^>\s+/.test(l)) {
      html.push(`<blockquote class="md-quote">${l.replace(/^>\s+/, "")}</blockquote>`);
      continue;
    }
    html.push(`<p>${l}</p>`);
  }
  return html.join("");
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function paintFrame() {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

function setMessageContent(contentEl, text, markdown) {
  if (markdown) {
    contentEl.innerHTML = formatAssistantHtml(text);
  } else {
    contentEl.textContent = text;
  }
}

function setAssistantMeta(row, provider, grounded) {
  const meta = row.querySelector(".message-meta");
  if (!meta) return;
  meta.innerHTML = "";
  if (provider) {
    const badge = document.createElement("span");
    badge.className = `provider-badge ${provider}`;
    badge.textContent = providerLabel(provider);
    meta.appendChild(badge);
  }
  if (grounded !== null && grounded !== undefined) {
    const badge = document.createElement("span");
    badge.className = `grounded-badge ${grounded ? "yes" : "no"}`;
    badge.textContent = grounded ? "Grounded" : "Unverified";
    meta.appendChild(badge);
  }
}

function applyProviderChrome(row, provider) {
  if (!provider) return;
  const known = ["openai", "gemini", "ollama", "extractive", "none"];
  for (const p of known) row.classList.remove(`provider-${p}`);
  row.classList.add(`provider-${provider}`);

  const avatar = row.querySelector(".message-avatar.assistant");
  if (avatar) {
    avatar.textContent = providerAvatar(provider);
    avatar.className = `message-avatar assistant provider-${provider}`;
  }
  setAssistantMeta(
    row,
    provider,
    row.dataset.grounded === "true"
      ? true
      : row.dataset.grounded === "false"
        ? false
        : undefined
  );
}

function updateAssistantMessage(row, text, { provider, grounded, streaming } = {}) {
  const content = row.querySelector(".message-content");
  const bubble = row.querySelector(".message-bubble");
  if (content) {
    // While streaming, keep plain text for speed; format when complete.
    setMessageContent(content, text, streaming !== true);
  }
  if (provider) {
    if (grounded !== null && grounded !== undefined) {
      row.dataset.grounded = grounded ? "true" : "false";
    }
    applyProviderChrome(row, provider);
    setAssistantMeta(row, provider, grounded);
  }
  if (streaming === false && bubble) {
    bubble.classList.remove("streaming");
    if (content) setMessageContent(content, text, true);
  }
  scrollMessages();
}

function appendMessage(role, text, options = {}) {
  const extraClass =
    typeof options === "string" ? options : options.extraClass || "";
  const opts = typeof options === "string" ? { extraClass } : options;
  const { provider = null, grounded = null, markdown = role === "assistant" } = opts;

  const row = document.createElement("div");
  row.className = `message-row ${role}`;
  if (provider) {
    row.classList.add(`provider-${provider}`);
  }
  if (grounded !== null && grounded !== undefined) {
    row.dataset.grounded = grounded ? "true" : "false";
  }

  if (role === "system") {
    const el = document.createElement("div");
    el.className = "message-system";
    el.textContent = text;
    row.appendChild(el);
  } else if (role === "user") {
    row.innerHTML = `
      <div class="message-avatar user" aria-hidden="true">You</div>
      <div class="message-bubble user">
        <div class="message-content"></div>
      </div>`;
    row.querySelector(".message-content").textContent = text;
  } else {
    const icon = providerAvatar(provider || "none");
    row.innerHTML = `
      <div class="message-avatar assistant${provider ? ` provider-${provider}` : ""}" aria-hidden="true">${icon}</div>
      <div class="message-bubble assistant ${extraClass}">
        <div class="message-meta"></div>
        <div class="message-content"></div>
      </div>`;
    setAssistantMeta(row, provider, grounded);
    setMessageContent(
      row.querySelector(".message-content"),
      text,
      markdown && text.length > 0
    );
  }

  messagesEl.appendChild(row);
  scrollMessages();
  return row;
}

function createAssistantMessage() {
  return appendMessage("assistant", "", { extraClass: "streaming", markdown: false });
}

function setBusy(busy) {
  isBusy = busy;
  sendBtn.disabled = busy;
  queryInput.disabled = busy;
}

function formatBytes(n) {
  if (n >= 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  if (n >= 1024) return `${Math.round(n / 1024)} KB`;
  return `${n} B`;
}

function formatDate(iso) {
  if (!iso) return "";
  try {
    const d = new Date(iso);
    return d.toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

function setUploadProgress(visible, percent = 0, label = "") {
  uploadProgress.classList.toggle("hidden", !visible);
  uploadProgressBar.style.width = `${Math.max(0, Math.min(100, percent))}%`;
  uploadProgressPct.textContent = `${Math.round(percent)}%`;
  uploadProgressLabel.textContent = label || "Working…";
}

function renderCitations(citations) {
  citationsList.innerHTML = "";
  if (!citations || citations.length === 0) {
    citationsEmpty.classList.remove("hidden");
    return;
  }
  citationsEmpty.classList.add("hidden");
  for (const c of citations) {
    const fullText = c.chunk_text || c.snippet || "";
    const pageLabel = c.page != null ? ` · p.${c.page}` : "";
    const card = document.createElement("article");
    card.className = "citation-card";
    card.tabIndex = 0;
    card.setAttribute("role", "button");
    card.setAttribute("aria-expanded", "false");
    card.innerHTML = `
      <header>
        <span>[${c.marker}] ${escapeHtml(c.filename)}${pageLabel}</span>
        <span>${Number(c.score).toFixed(2)}</span>
      </header>
      <div class="snippet">${escapeHtml(c.snippet)}</div>
      <div class="citation-full">${escapeHtml(fullText)}</div>
      <div class="citation-hint">Click to expand full passage</div>
    `;
    const toggle = (e) => {
      e.stopPropagation();
      const willOpen = !card.classList.contains("expanded");
      citationsList.querySelectorAll(".citation-card.expanded").forEach((el) => {
        if (el !== card) {
          el.classList.remove("expanded");
          el.setAttribute("aria-expanded", "false");
        }
      });
      card.classList.toggle("expanded", willOpen);
      card.setAttribute("aria-expanded", willOpen ? "true" : "false");
    };
    card.addEventListener("click", toggle);
    card.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        toggle(e);
      }
    });
    citationsList.appendChild(card);
  }
}

function renderMeta(provider, grounded, notes) {
  metaPanel.classList.remove("hidden");
  metaProvider.textContent = provider;
  metaGrounded.textContent = grounded ? "yes" : "no";
  metaGrounded.style.color = grounded ? "var(--ok)" : "var(--warn)";
  metaNotes.innerHTML = "";
  for (const note of notes || []) {
    const li = document.createElement("li");
    li.textContent = note;
    metaNotes.appendChild(li);
  }
}

async function fetchHealth() {
  try {
    const res = await apiFetch("/health");
    const data = await res.json();
    healthPill.textContent = `${data.documents} docs · ${data.chunks} chunks · ${data.embedding_provider}`;
    healthPill.className = "health-pill ok";
  } catch {
    healthPill.textContent = "API offline";
    healthPill.className = "health-pill err";
  }
}

function canDeleteDocs() {
  return currentUser.role === "admin";
}

async function loadDocuments() {
  try {
    const res = await apiFetch("/documents");
    if (!res.ok) throw new Error("Failed to load documents");
    const data = await res.json();
    docList.innerHTML = "";
    allDocumentIds = data.documents.map((doc) => doc.id);
    if (!data.documents.length) {
      selectedDocIds = new Set();
      docList.innerHTML =
        '<li class="muted doc-empty">No documents yet — upload a PDF or DOCX to start.</li>';
      return;
    }

    const savedScope = loadScopeIds();
    if (savedScope) {
      selectedDocIds = new Set(savedScope.filter((id) => allDocumentIds.includes(id)));
    } else {
      selectedDocIds = new Set(allDocumentIds);
    }
    if (selectedDocIds.size === 0) {
      selectedDocIds = new Set(allDocumentIds);
    }

    for (const doc of data.documents) {
      const li = document.createElement("li");
      li.className = "doc-item";
      const checked = selectedDocIds.has(doc.id);
      const meta = `${doc.num_chunks} chunks · ${formatBytes(doc.num_chars)} · ${formatDate(doc.uploaded_at)}`;
      li.innerHTML = `
        <label class="doc-scope">
          <input type="checkbox" data-id="${doc.id}" ${checked ? "checked" : ""} />
          <span class="doc-meta-wrap">
            <span class="doc-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</span>
            <span class="doc-meta">${escapeHtml(meta)}</span>
          </span>
        </label>
        ${
          canDeleteDocs()
            ? `<button type="button" data-id="${doc.id}" class="doc-delete" aria-label="Delete">✕</button>`
            : ""
        }
      `;
      const checkbox = li.querySelector('input[type="checkbox"]');
      checkbox.addEventListener("change", () => {
        if (checkbox.checked) selectedDocIds.add(doc.id);
        else selectedDocIds.delete(doc.id);
        if (selectedDocIds.size === 0) {
          selectedDocIds = new Set(allDocumentIds);
          docList.querySelectorAll('input[type="checkbox"]').forEach((el) => {
            el.checked = true;
          });
        }
        saveScopeIds();
      });
      const delBtn = li.querySelector(".doc-delete");
      if (delBtn) delBtn.addEventListener("click", () => deleteDocument(doc.id));
      docList.appendChild(li);
    }
    saveScopeIds();
  } catch (err) {
    console.error(err);
  }
}

async function uploadFile(file) {
  const form = new FormData();
  form.append("file", file);
  setUploadProgress(true, 2, `Uploading ${file.name}…`);

  const res = await apiFetch("/documents/upload/stream", {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const detail = err.detail;
    const message =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
          : "Upload failed";
    setUploadProgress(false);
    throw new Error(message);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let doneDoc = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";
    for (const part of parts) {
      const line = part.split("\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      const event = JSON.parse(line.slice(5).trim());
      if (event.type === "progress") {
        const label = `${STAGE_LABELS[event.stage] || event.stage}: ${event.filename || file.name}`;
        setUploadProgress(true, event.percent || 0, label);
        await paintFrame();
      } else if (event.type === "done") {
        doneDoc = event.document;
        setUploadProgress(true, 100, `Indexed ${file.name}`);
      } else if (event.type === "error") {
        setUploadProgress(false);
        throw new Error(event.message || "Upload failed");
      }
    }
  }

  setUploadProgress(false);
  await loadDocuments();
  await fetchHealth();
  const chunks = doneDoc?.num_chunks != null ? ` (${doneDoc.num_chunks} chunks)` : "";
  appendMessage("system", `Uploaded ${file.name}${chunks}`);
}

async function uploadFiles(fileList) {
  const files = [...fileList];
  for (const file of files) {
    await uploadFile(file);
  }
}

async function deleteDocument(id) {
  if (!confirm("Delete this document from the index?")) return;
  const res = await apiFetch(`/documents/${id}`, { method: "DELETE" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Delete failed");
  }
  await loadDocuments();
  await fetchHealth();
}

async function chatNonStream(query) {
  const res = await apiFetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(chatPayload(query)),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

async function chatStream(query, assistantRow) {
  const res = await apiFetch("/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(chatPayload(query)),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }

  metaPanel.classList.remove("hidden");
  metaProvider.textContent = "streaming…";
  metaGrounded.textContent = "—";
  setAssistantMeta(assistantRow, null, null);
  const streamingBadge = document.createElement("span");
  streamingBadge.className = "provider-badge streaming";
  streamingBadge.textContent = "Streaming…";
  assistantRow.querySelector(".message-meta")?.appendChild(streamingBadge);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let fullText = "";
  let tokenCount = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";

    for (const part of parts) {
      const line = part.split("\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      const event = JSON.parse(line.slice(5).trim());

      if (event.type === "citations") {
        renderCitations(event.citations);
      } else if (event.type === "token") {
        fullText += event.text;
        updateAssistantMessage(assistantRow, fullText, { streaming: true });
        tokenCount += 1;
        await paintFrame();
      } else if (event.type === "done") {
        if (event.conversation_id) {
          conversationId = event.conversation_id;
          localStorage.setItem(STORAGE_KEY, conversationId);
        }
        renderMeta(event.provider, event.grounded, event.validation_notes);
        updateAssistantMessage(assistantRow, fullText, {
          provider: event.provider,
          grounded: event.grounded,
          streaming: false,
        });
        if (event.provider === "extractive" && tokenCount <= 1) {
          metaNotes.innerHTML = "";
          const li = document.createElement("li");
          li.textContent =
            "Extractive mode returns the full answer in one chunk (no LLM token stream).";
          metaNotes.appendChild(li);
        }
      } else if (event.type === "error") {
        throw new Error(event.message || "Stream failed");
      }
    }
  }

  updateAssistantMessage(assistantRow, fullText, { streaming: false });
  return fullText;
}

async function handleSubmit(e) {
  e.preventDefault();
  if (isBusy) return;

  const query = queryInput.value.trim();
  if (!query) return;

  saveSettings();
  appendMessage("user", query);
  queryInput.value = "";

  const useStream = $("#stream-toggle").checked;
  setBusy(true);

  try {
    if (useStream) {
      const assistantRow = createAssistantMessage();
      await chatStream(query, assistantRow);
    } else {
      const data = await chatNonStream(query);
      appendMessage("assistant", data.answer, {
        provider: data.provider,
        grounded: data.grounded,
        markdown: true,
      });
      renderCitations(data.citations);
      renderMeta(data.provider, data.grounded, data.validation_notes);
      if (data.conversation_id) {
        conversationId = data.conversation_id;
        localStorage.setItem(STORAGE_KEY, conversationId);
      }
    }
  } catch (err) {
    appendMessage("system", `Error: ${err.message}`);
  } finally {
    setBusy(false);
    queryInput.focus();
  }
}

function newConversation() {
  conversationId = null;
  localStorage.removeItem(STORAGE_KEY);
  messagesEl.innerHTML = "";
  renderCitations([]);
  metaPanel.classList.add("hidden");
  citationsEmpty.classList.remove("hidden");
  appendMessage("system", "Started a new conversation.");
}

function showLogin(errorMsg) {
  loginScreen.classList.remove("hidden");
  appShell.classList.add("hidden");
  const err = $("#login-error");
  if (errorMsg) {
    err.textContent = errorMsg;
    err.classList.remove("hidden");
  } else {
    err.classList.add("hidden");
  }
}

function showApp() {
  loginScreen.classList.add("hidden");
  appShell.classList.remove("hidden");
}

function clearSession() {
  authToken = "";
  localStorage.removeItem(TOKEN_KEY);
}

function applyRoleUi() {
  const isAdmin = currentUser.role === "admin";
  openAdminBtn.classList.toggle("hidden", !isAdmin);
  if (currentUser.auth_enabled) {
    userBar.classList.remove("hidden");
    userLabel.textContent = `${currentUser.username} · ${currentUser.role}`;
  } else {
    userBar.classList.add("hidden");
  }
}

function openAdminModal() {
  adminModal.classList.remove("hidden");
  document.body.classList.add("modal-open");
  $("#config-status").textContent = "";
  loadAdminConfig();
}

function closeAdminModal() {
  adminModal.classList.add("hidden");
  document.body.classList.remove("modal-open");
}

async function loadAdminConfig() {
  if (currentUser.role !== "admin") return;
  const res = await apiFetch("/admin/config");
  if (!res.ok) return;
  const data = await res.json();
  const c = data.config || {};
  const set = (id, val) => {
    const el = $(id);
    if (!el || val == null) return;
    if (el.type === "checkbox") el.checked = !!val;
    else el.value = val;
  };
  set("#cfg-llm_provider", c.llm_provider);
  set("#cfg-openai_chat_model", c.openai_chat_model);
  set("#cfg-gemini_model", c.gemini_model);
  set("#cfg-ollama_base_url", c.ollama_base_url);
  set("#cfg-ollama_model", c.ollama_model);
  set("#cfg-top_k", c.top_k);
  set("#cfg-min_score", c.min_score);
  set("#cfg-chunk_size", c.chunk_size);
  set("#cfg-chunk_overlap", c.chunk_overlap);
  set("#cfg-ocr_enabled", c.ocr_enabled);
  set("#cfg-rerank_enabled", c.rerank_enabled);
  set("#cfg-mmr_enabled", c.mmr_enabled);
  set("#cfg-mmr_lambda", c.mmr_lambda);
  set("#cfg-prompt_injection_enabled", c.prompt_injection_enabled);
  $("#cfg-openai_api_key").placeholder = c.openai_api_key_set
    ? "set — leave blank to keep"
    : "not set — paste OpenAI sk-… key";
  $("#cfg-gemini_api_key").placeholder = c.gemini_api_key_set
    ? "set — leave blank to keep"
    : "not set — paste Gemini API key";
  $("#cfg-openai_api_key").value = "";
  $("#cfg-gemini_api_key").value = "";

  // Keep sidebar chat mode aligned with admin default provider.
  if (c.llm_provider && $("#mode-select")) {
    const mode = c.llm_provider;
    if ([...$("#mode-select").options].some((o) => o.value === mode)) {
      $("#mode-select").value = mode;
      saveSettings();
    }
  }
}

function looksLikeGeminiKey(key) {
  return /^(AQ\.|AIza)/i.test(key);
}

function looksLikeOpenAIKey(key) {
  return /^sk-/i.test(key);
}

async function saveAdminConfig(e) {
  e.preventDefault();
  const status = $("#config-status");
  status.textContent = "Saving…";
  const body = {
    llm_provider: $("#cfg-llm_provider").value,
    openai_chat_model: $("#cfg-openai_chat_model").value,
    gemini_model: $("#cfg-gemini_model").value,
    ollama_base_url: $("#cfg-ollama_base_url").value,
    ollama_model: $("#cfg-ollama_model").value,
    top_k: Number($("#cfg-top_k").value),
    min_score: Number($("#cfg-min_score").value),
    chunk_size: Number($("#cfg-chunk_size").value),
    chunk_overlap: Number($("#cfg-chunk_overlap").value),
    ocr_enabled: $("#cfg-ocr_enabled").checked,
    rerank_enabled: $("#cfg-rerank_enabled").checked,
    mmr_enabled: $("#cfg-mmr_enabled").checked,
    mmr_lambda: Number($("#cfg-mmr_lambda").value),
    prompt_injection_enabled: $("#cfg-prompt_injection_enabled").checked,
  };
  let openaiKey = $("#cfg-openai_api_key").value.trim();
  let geminiKey = $("#cfg-gemini_api_key").value.trim();

  // Prevent saving masked placeholders or putting the wrong key in the wrong box.
  if (openaiKey.includes("•")) openaiKey = "";
  if (geminiKey.includes("•")) geminiKey = "";
  if (openaiKey && looksLikeGeminiKey(openaiKey) && !geminiKey) {
    geminiKey = openaiKey;
    openaiKey = "";
    if (body.llm_provider === "auto" || body.llm_provider === "openai") {
      body.llm_provider = "gemini";
      $("#cfg-llm_provider").value = "gemini";
    }
  }
  if (geminiKey && looksLikeOpenAIKey(geminiKey) && !openaiKey) {
    openaiKey = geminiKey;
    geminiKey = "";
    if (body.llm_provider === "auto" || body.llm_provider === "gemini") {
      body.llm_provider = "openai";
      $("#cfg-llm_provider").value = "openai";
    }
  }
  if (openaiKey) body.openai_api_key = openaiKey;
  if (geminiKey) body.gemini_api_key = geminiKey;

  const res = await apiFetch("/admin/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    status.textContent = err.detail || "Save failed";
    return;
  }
  const data = await res.json();
  status.textContent = data.message || "Saved.";

  // Apply default provider to the chat mode dropdown immediately.
  const provider = body.llm_provider || "auto";
  if ([...$("#mode-select").options].some((o) => o.value === provider)) {
    $("#mode-select").value = provider;
    saveSettings();
  }

  await loadAdminConfig();
  await fetchHealth();
  setTimeout(() => closeAdminModal(), 1200);
}

async function bootstrapSession() {
  const statusRes = await fetch("/auth/status");
  const status = await statusRes.json();

  if (!status.auth_enabled) {
    // Zero-login mode: synthetic admin.
    currentUser = { username: "local", role: "admin", auth_enabled: false };
    showApp();
    applyRoleUi();
    return true;
  }

  if (!authToken) {
    showLogin();
    return false;
  }

  const meRes = await fetch("/auth/me", {
    headers: { Authorization: `Bearer ${authToken}` },
  });
  if (!meRes.ok) {
    clearSession();
    showLogin("Please sign in.");
    return false;
  }
  const me = await meRes.json();
  currentUser = me;
  showApp();
  applyRoleUi();
  return true;
}

async function handleLogin(e) {
  e.preventDefault();
  const err = $("#login-error");
  err.classList.add("hidden");
  const username = $("#login-username").value.trim();
  const password = $("#login-password").value;
  const res = await fetch("/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    err.textContent = data.detail || "Login failed";
    err.classList.remove("hidden");
    return;
  }
  const data = await res.json();
  authToken = data.token;
  localStorage.setItem(TOKEN_KEY, authToken);
  currentUser = {
    username: data.username,
    role: data.role,
    auth_enabled: true,
  };
  showApp();
  applyRoleUi();
  await initAppData();
}

async function handleLogout() {
  await fetch("/auth/logout", { method: "POST" }).catch(() => {});
  clearSession();
  showLogin();
}

async function initAppData() {
  applySettings();
  syncStreamEngineUi();
  await fetchHealth();
  await loadDocuments();
  if (currentUser.role === "admin") await loadAdminConfig();

  if (conversationId) {
    appendMessage("system", "Continuing previous conversation.");
  } else {
    appendMessage("system", "Ask a question about your uploaded documents.");
  }
}

// Upload handlers
uploadZone.addEventListener("click", () => fileInput.click());
uploadZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadZone.classList.add("dragover");
});
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
uploadZone.addEventListener("drop", async (e) => {
  e.preventDefault();
  uploadZone.classList.remove("dragover");
  const files = e.dataTransfer.files;
  if (!files?.length) return;
  try {
    setBusy(true);
    await uploadFiles(files);
  } catch (err) {
    appendMessage("system", `Upload error: ${err.message}`);
    setUploadProgress(false);
  } finally {
    setBusy(false);
  }
});

fileInput.addEventListener("change", async () => {
  const files = fileInput.files;
  if (!files?.length) return;
  fileInput.value = "";
  try {
    setBusy(true);
    await uploadFiles(files);
  } catch (err) {
    appendMessage("system", `Upload error: ${err.message}`);
    setUploadProgress(false);
  } finally {
    setBusy(false);
  }
});

$("#mode-select").addEventListener("change", saveSettings);
$("#engine-select").addEventListener("change", saveSettings);
function syncStreamEngineUi() {
  const streaming = $("#stream-toggle").checked;
  const engineSelect = $("#engine-select");
  const hint = $("#engine-hint");

  if (streaming) {
    if (engineSelect.value !== "langchain") {
      engineBeforeStream = engineSelect.value;
      engineSelect.value = "langchain";
    }
  } else if (engineBeforeStream) {
    engineSelect.value = engineBeforeStream;
    engineBeforeStream = null;
  }

  engineSelect.disabled = streaming;
  hint.classList.toggle("hidden", !streaming);
  engineSelect.title = streaming
    ? "Streaming uses the LangChain path (/chat/stream)"
    : "";
  saveSettings();
}

$("#stream-toggle").addEventListener("change", syncStreamEngineUi);
$("#top-k-input").addEventListener("change", saveSettings);
$("#new-chat-btn").addEventListener("click", newConversation);
chatForm.addEventListener("submit", handleSubmit);
$("#login-form").addEventListener("submit", handleLogin);
$("#logout-btn").addEventListener("click", handleLogout);
$("#config-form").addEventListener("submit", saveAdminConfig);
openAdminBtn.addEventListener("click", openAdminModal);
$("#close-admin-btn").addEventListener("click", closeAdminModal);
$("#cancel-admin-btn").addEventListener("click", closeAdminModal);
$("#admin-modal-backdrop").addEventListener("click", closeAdminModal);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !adminModal.classList.contains("hidden")) {
    closeAdminModal();
  }
});

queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    chatForm.requestSubmit();
  }
});

(async function boot() {
  const ok = await bootstrapSession();
  if (ok) await initAppData();
})();
