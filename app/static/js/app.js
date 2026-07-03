/** PrivateRAG Web UI — vanilla JS, no build step. */

const STORAGE_KEY = "privaterag.conversation_id";
const SETTINGS_KEY = "privaterag.settings";
const SCOPE_KEY = "privaterag.document_scope";

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

let conversationId = localStorage.getItem(STORAGE_KEY) || null;
let isBusy = false;
let engineBeforeStream = null;
let allDocumentIds = [];
let selectedDocIds = new Set();

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

function formatAssistantHtml(text) {
  const lines = escapeHtml(text).split("\n");
  return lines
    .map((line) => {
      let l = line;
      l = l.replace(
        /\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]/g,
        '<sup class="cite-ref">[$1]</sup>'
      );
      l = l.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
      l = l.replace(/`([^`]+)`/g, "<code>$1</code>");
      if (/^[-*]\s+/.test(l)) {
        return `<div class="md-li"><span class="md-bullet">•</span><span>${l.replace(/^[-*]\s+/, "")}</span></div>`;
      }
      if (/^\d+\.\s+/.test(l)) {
        return `<div class="md-li num">${l}</div>`;
      }
      if (!l.trim()) {
        return "";
      }
      return `<p>${l}</p>`;
    })
    .filter(Boolean)
    .join("");
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
  if (provider && provider !== "none") {
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

function updateAssistantMessage(row, text, { provider, grounded, streaming } = {}) {
  const content = row.querySelector(".message-content");
  const bubble = row.querySelector(".message-bubble");
  if (content) {
    setMessageContent(content, text, true);
  }
  if (provider) {
    row.classList.add(`provider-${provider}`);
    setAssistantMeta(row, provider, grounded);
  }
  if (streaming === false && bubble) {
    bubble.classList.remove("streaming");
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
    row.innerHTML = `
      <div class="message-avatar assistant" aria-hidden="true">◇</div>
      <div class="message-bubble assistant ${extraClass}">
        <div class="message-meta"></div>
        <div class="message-content"></div>
      </div>`;
    setAssistantMeta(row, provider, grounded);
    setMessageContent(row.querySelector(".message-content"), text, markdown && text.length > 0);
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

function renderCitations(citations) {
  citationsList.innerHTML = "";
  if (!citations || citations.length === 0) {
    citationsEmpty.classList.remove("hidden");
    return;
  }
  citationsEmpty.classList.add("hidden");
  for (const c of citations) {
    const fullText = c.chunk_text || c.snippet || "";
    const card = document.createElement("article");
    card.className = "citation-card";
    card.tabIndex = 0;
    card.setAttribute("role", "button");
    card.setAttribute("aria-expanded", "false");
    card.innerHTML = `
      <header>
        <span>[${c.marker}] ${escapeHtml(c.filename)}</span>
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
        toggle();
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

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/** Let the browser paint between token updates (otherwise one TCP chunk = instant full text). */
function paintFrame() {
  return new Promise((resolve) => requestAnimationFrame(resolve));
}

async function fetchHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    healthPill.textContent = `${data.documents} docs · ${data.chunks} chunks · ${data.llm_provider}`;
    healthPill.className = "health-pill ok";
  } catch {
    healthPill.textContent = "API offline";
    healthPill.className = "health-pill err";
  }
}

async function loadDocuments() {
  try {
    const res = await fetch("/documents");
    const data = await res.json();
    docList.innerHTML = "";
    allDocumentIds = data.documents.map((doc) => doc.id);
    if (!data.documents.length) {
      selectedDocIds = new Set();
      docList.innerHTML = '<li class="muted" style="list-style:none;font-size:0.8rem">No documents yet</li>';
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
      li.innerHTML = `
        <label class="doc-scope">
          <input type="checkbox" data-id="${doc.id}" ${checked ? "checked" : ""} />
          <span title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</span>
        </label>
        <button type="button" data-id="${doc.id}" aria-label="Delete">✕</button>
      `;
      const checkbox = li.querySelector('input[type="checkbox"]');
      checkbox.addEventListener("change", () => {
        if (checkbox.checked) {
          selectedDocIds.add(doc.id);
        } else {
          selectedDocIds.delete(doc.id);
        }
        if (selectedDocIds.size === 0) {
          selectedDocIds = new Set(allDocumentIds);
          docList.querySelectorAll('input[type="checkbox"]').forEach((el) => {
            el.checked = true;
          });
        }
        saveScopeIds();
      });
      li.querySelector("button").addEventListener("click", () => deleteDocument(doc.id));
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
  const res = await fetch("/documents/upload", { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Upload failed");
  }
  await loadDocuments();
  await fetchHealth();
  appendMessage("system", `Uploaded ${file.name}`);
}

async function deleteDocument(id) {
  if (!confirm("Delete this document from the index?")) return;
  const res = await fetch(`/documents/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error("Delete failed");
  await loadDocuments();
  await fetchHealth();
}

async function chatNonStream(query) {
  const res = await fetch("/chat", {
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
  const res = await fetch("/chat/stream", {
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
        // Without this, many tokens from one network read paint as a single frame.
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
  const file = e.dataTransfer.files[0];
  if (!file) return;
  try {
    setBusy(true);
    await uploadFile(file);
  } catch (err) {
    appendMessage("system", `Upload error: ${err.message}`);
  } finally {
    setBusy(false);
  }
});

fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) return;
  fileInput.value = "";
  try {
    setBusy(true);
    await uploadFile(file);
  } catch (err) {
    appendMessage("system", `Upload error: ${err.message}`);
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

queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    chatForm.requestSubmit();
  }
});

applySettings();
syncStreamEngineUi();
fetchHealth();
loadDocuments();

if (conversationId) {
  appendMessage("system", "Continuing previous conversation.");
} else {
  appendMessage("system", "Ask a question about your uploaded documents.");
}
