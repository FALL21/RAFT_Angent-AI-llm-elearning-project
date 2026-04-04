/**
 * RAG-LLM Multi-Agent — Frontend JavaScript
 */

// ── State ──────────────────────────────────────────────────
const state = {
  currentTab: "chat",
  isProcessing: false,
};

// ── DOM Elements ───────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ── Tab Navigation ─────────────────────────────────────────
$$(".nav-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const tab = btn.dataset.tab;
    $$(".nav-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    $$(".tab-content").forEach((t) => t.classList.remove("active"));
    $(`#tab-${tab}`).classList.add("active");
    state.currentTab = tab;

    // Auto-load data
    if (tab === "upload") loadPDFs();
    if (tab === "dataset") loadDataset();
    if (tab === "benchmark") loadBenchmark();
    if (tab === "settings") loadSettings();
  });
});

// ── Chat (rendu réponse assistant) ─────────────────────────
/**
 * Gras **...** (multiligne autorisé), le reste échappé.
 */
function formatInlineMarkdown(text) {
  let out = "";
  let last = 0;
  const re = /\*\*([\s\S]+?)\*\*/g;
  let m;
  while ((m = re.exec(text)) !== null) {
    out += escapeHtml(text.slice(last, m.index));
    out += `<strong class="message-strong">${escapeHtml(m[1])}</strong>`;
    last = re.lastIndex;
  }
  out += escapeHtml(text.slice(last));
  return out;
}

/**
 * Découpe une section finale Sources / Conclusion / Références (souvent collée au corps).
 */
function splitAnswerSections(raw) {
  const t = raw.replace(/\r\n/g, "\n");
  const re =
    /^([\s\S]+?)(\n+(?:#{1,3}\s*)?(?:\*\*)?\s*(?:Sources?|Conclusion|Références)\b(?:\*\*)?\s*[:\.]?[\s\S]*)$/i;
  const m = t.match(re);
  if (m && m[1].trim().length > 0) {
    return { main: m[1].trim(), footer: m[2].trim() };
  }
  return { main: t.trim(), footer: null };
}

function footerSectionLabel(footerText) {
  const first = footerText.split("\n")[0] || "";
  const cleaned = first
    .replace(/\*\*/g, "")
    .replace(/^[#\s]+/, "")
    .replace(/[:：]\s*$/, "")
    .trim();
  if (cleaned.length > 0 && cleaned.length < 80) return cleaned;
  return "Sources & suite";
}

/**
 * Paragraphes (séparés par ligne vide), listes - / * / • et 1. 2.
 */
function renderBotMarkdownBlock(raw) {
  const lines = raw.replace(/\r\n/g, "\n").split("\n");
  const segments = [];
  let i = 0;

  function flushParagraph(buf) {
    if (!buf.length) return;
    const inner = buf
      .map((line, idx) =>
        idx < buf.length - 1
          ? `${formatInlineMarkdown(line)}<br>`
          : formatInlineMarkdown(line),
      )
      .join("");
    segments.push(`<p class="message-para">${inner}</p>`);
  }

  let paraBuf = [];
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      flushParagraph(paraBuf);
      paraBuf = [];
      i++;
      continue;
    }
    if (/^[\-\*•]\s+/.test(line)) {
      flushParagraph(paraBuf);
      paraBuf = [];
      const items = [];
      while (i < lines.length && /^[\-\*•]\s+/.test(lines[i])) {
        items.push(
          formatInlineMarkdown(lines[i].replace(/^[\-\*•]\s+/, "").trim()),
        );
        i++;
      }
      segments.push(
        `<ul class="message-list message-list--bullets">${items.map((li) => `<li>${li}</li>`).join("")}</ul>`,
      );
      continue;
    }
    if (/^\d+\.\s*/.test(line)) {
      flushParagraph(paraBuf);
      paraBuf = [];
      const items = [];
      while (i < lines.length && /^\d+\.\s*/.test(lines[i])) {
        items.push(
          formatInlineMarkdown(lines[i].replace(/^\d+\.\s*/, "").trim()),
        );
        i++;
      }
      segments.push(
        `<ol class="message-list message-list--numbered">${items.map((li) => `<li>${li}</li>`).join("")}</ol>`,
      );
      continue;
    }
    paraBuf.push(line);
    i++;
  }
  flushParagraph(paraBuf);
  return segments.join("");
}

function renderBotAnswerHtml(text) {
  const { main, footer } = splitAnswerSections(text);
  let html = `<div class="message-body">${renderBotMarkdownBlock(main)}</div>`;
  if (footer) {
    const label = escapeHtml(footerSectionLabel(footer));
    html += `<aside class="message-footer-card" aria-label="${label}">
            <div class="message-footer-card__label">${label}</div>
            <div class="message-footer-card__body">${renderBotMarkdownBlock(footer)}</div>
        </aside>`;
  }
  return html;
}

// ── Chat ───────────────────────────────────────────────────
const chatMessages = $("#chat-messages");
const questionInput = $("#question-input");
const sendBtn = $("#send-btn");
const pipelineSelect = $("#pipeline-select");

sendBtn.addEventListener("click", sendMessage);
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// Auto-resize textarea
questionInput.addEventListener("input", () => {
  questionInput.style.height = "auto";
  questionInput.style.height = Math.min(questionInput.scrollHeight, 120) + "px";
});

async function sendMessage() {
  const question = questionInput.value.trim();
  if (!question || state.isProcessing) return;

  state.isProcessing = true;
  sendBtn.disabled = true;

  // Add user message
  addMessage(question, "user");
  questionInput.value = "";
  questionInput.style.height = "auto";

  // Show loading
  const loadingEl = addLoading();

  try {
    const response = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: question,
        pipeline: pipelineSelect.value,
      }),
    });

    const data = await response.json();
    loadingEl.remove();

    if (data.error) {
      addMessage(`Erreur : ${data.error}`, "bot");
    } else {
      addMessage(data.answer, "bot", data.sources);
    }
  } catch (err) {
    loadingEl.remove();
    addMessage(`Erreur de connexion : ${err.message}`, "bot");
  }

  state.isProcessing = false;
  sendBtn.disabled = false;
}

function addMessage(text, type, sources = []) {
  const div = document.createElement("div");
  div.className = `message ${type}`;

  const icon = type === "user" ? "fa-user" : "fa-robot";
  let sourcesHtml = "";

  if (sources && sources.length > 0) {
    sourcesHtml = `<div class="sources-list">
            <span class="sources-list__title"><i class="fas fa-layer-group"></i> Documents cités</span>
            ${sources
              .map(
                (s) =>
                  `<div class="source-item"><i class="fas fa-file-alt"></i> <span class="source-item__name">${escapeHtml(String(s.source || "N/A"))}</span>${s.score != null ? `<span class="source-item__score">${(Number(s.score) * 100).toFixed(0)}%</span>` : ""}</div>`,
              )
              .join("")}
        </div>`;
  }

  const bodyHtml =
    type === "bot"
      ? renderBotAnswerHtml(text)
      : `<div class="message-body message-body--plain"><p class="message-para">${escapeHtml(text)}</p></div>`;

  div.innerHTML = `
        <div class="message-avatar"><i class="fas ${icon}"></i></div>
        <div class="message-content message-content--${type}">
            ${bodyHtml}
            ${sourcesHtml}
        </div>
    `;

  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return div;
}

function addLoading() {
  const div = document.createElement("div");
  div.className = "message bot";
  div.innerHTML = `
        <div class="message-avatar"><i class="fas fa-robot"></i></div>
        <div class="message-content">
            <div class="loading"><span></span><span></span><span></span></div>
        </div>
    `;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return div;
}

// ── Upload ─────────────────────────────────────────────────
const uploadZone = $("#upload-zone");
const fileInput = $("#file-input");

uploadZone.addEventListener("click", () => fileInput.click());
uploadZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadZone.classList.add("dragover");
});
uploadZone.addEventListener("dragleave", () =>
  uploadZone.classList.remove("dragover"),
);
uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("dragover");
  handleFiles(e.dataTransfer.files);
});
fileInput.addEventListener("change", () => handleFiles(fileInput.files));

async function handleFiles(files) {
  for (const file of files) {
    if (!file.name.endsWith(".pdf")) continue;
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setStatus(data.message || data.error);
    } catch (err) {
      setStatus(`Erreur upload : ${err.message}`);
    }
  }
  loadPDFs();
}

async function loadPDFs() {
  try {
    const res = await fetch("/api/pdfs");
    const data = await res.json();
    const list = $("#pdf-items");
    list.innerHTML = data.pdfs
      .map((pdf) => `<li><i class="fas fa-file-pdf"></i> ${pdf}</li>`)
      .join("");
    if (data.pdfs.length === 0) {
      list.innerHTML =
        "<li><i class='fas fa-info-circle'></i> Aucun PDF. Uploadez des documents.</li>";
    }
  } catch (err) {
    console.error(err);
  }
}

// Index button
$("#index-btn").addEventListener("click", async () => {
  setStatus("Indexation en cours...");
  try {
    const res = await fetch("/api/index", { method: "POST" });
    const data = await res.json();
    if (data.error) {
      setStatus(`Erreur : ${data.error}`);
    } else {
      setStatus(`Indexation OK : ${data.stats.total_chunks} chunks`);
    }
  } catch (err) {
    setStatus(`Erreur : ${err.message}`);
  }
});

$("#refresh-pdfs").addEventListener("click", loadPDFs);

// ── Dataset ────────────────────────────────────────────────
async function loadDataset() {
  try {
    const res = await fetch("/api/dataset");
    const data = await res.json();

    const stats = $("#dataset-stats");
    const tbody = $("#dataset-table tbody");

    if (data.error) {
      stats.innerHTML = `<p class="placeholder">${escapeHtml(data.error)}</p>`;
      tbody.innerHTML = "";
      return;
    }

    const rows = Array.isArray(data.data) ? data.data : [];

    if (rows.length === 0) {
      const hint =
        data.metadata && data.metadata.hint
          ? data.metadata.hint
          : "Aucune donnée.";
      stats.innerHTML = `<p class="placeholder">${escapeHtml(hint)}</p>`;
      tbody.innerHTML = "";
      return;
    }

    const domains = {};
    rows.forEach((d) => {
      domains[d.domain] = (domains[d.domain] || 0) + 1;
    });

    const gen =
      data.metadata && data.metadata.generation
        ? `<div class="stat-card"><div class="value">${escapeHtml(data.metadata.generation)}</div><div class="label">Source</div></div>`
        : "";

    stats.innerHTML = `
                <div class="stat-card"><div class="value">${rows.length}</div><div class="label">Questions</div></div>
                ${gen}
                ${Object.entries(domains)
                  .map(
                    ([d, c]) =>
                      `<div class="stat-card"><div class="value">${c}</div><div class="label">${escapeHtml(d)}</div></div>`,
                  )
                  .join("")}
            `;

    tbody.innerHTML = rows
      .slice(0, 50)
      .map(
        (d, i) =>
          `<tr><td>${i + 1}</td><td>${escapeHtml(d.domain || "")}</td><td>${escapeHtml(d.question || "").substring(0, 80)}...</td><td>${escapeHtml(d.answer || "").substring(0, 100)}...</td></tr>`,
      )
      .join("");
  } catch (err) {
    console.error(err);
  }
}

$("#generate-dataset").addEventListener("click", async () => {
  const input = $("#dataset-max-pairs");
  let maxPairs = input ? parseInt(input.value, 10) : 60;
  if (Number.isNaN(maxPairs) || maxPairs < 1) maxPairs = 60;
  if (maxPairs > 300) maxPairs = 300;

  setStatus("Génération depuis les PDFs (patientez)...");
  try {
    const res = await fetch("/api/dataset/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ max_pairs: maxPairs }),
    });
    const out = await res.json();
    if (!res.ok) {
      setStatus(`Erreur : ${escapeHtml(out.error || res.statusText)}`);
      return;
    }
    setStatus(`Dataset enregistré : ${out.count} question(s).`);
    await loadDataset();
  } catch (err) {
    setStatus(`Erreur : ${err.message}`);
  }
});

// ── Benchmark ──────────────────────────────────────────────
async function loadBenchmark() {
  const container = $("#benchmark-cards");
  if (!container) return;

  try {
    const res = await fetch("/api/benchmark");
    const data = await res.json();

    if (data.error) {
      container.innerHTML = `<p class="placeholder">${escapeHtml(data.error)}</p>`;
      return;
    }

    if (!data.results || Object.keys(data.results).length === 0) {
      const msg = data.message || "Pas encore de résultats.";
      container.innerHTML = `<p class="placeholder">${escapeHtml(msg)}</p>`;
      return;
    }

    const best = data.best_model?.name || "";
    const bestNote = data.best_model?.note
      ? `<p class="placeholder benchmark-note">${escapeHtml(data.best_model.note)}</p>`
      : "";

    container.innerHTML =
      bestNote +
      Object.entries(data.results)
        .map(
          ([model, results]) => `
                <div class="benchmark-card ${model === best ? "winner" : ""}">
                    <h3>${model === best ? "🏆 " : ""}${escapeHtml(model)}</h3>
                    ${results
                      .map(
                        (r) => `
                        <div class="metric-row"><span>Domaine</span><span class="metric-value">${escapeHtml(String(r.domain))}</span></div>
                        <div class="metric-row"><span>Accuracy</span><span class="metric-value">${((r.accuracy || 0) * 100).toFixed(1)}%</span></div>
                        <div class="metric-row"><span>F1</span><span class="metric-value">${(r.f1_score || 0).toFixed(3)}</span></div>
                        <div class="metric-row"><span>ROUGE-L</span><span class="metric-value">${(r.rouge_l || 0).toFixed(3)}</span></div>
                        <div class="metric-row"><span>Latence</span><span class="metric-value">${(r.avg_latency_ms || 0).toFixed(0)} ms</span></div>
                    `,
                      )
                      .join("")}
                </div>
            `,
        )
        .join("");
  } catch (err) {
    console.error(err);
    container.innerHTML = `<p class="placeholder">Erreur de chargement : ${escapeHtml(err.message)}</p>`;
  }
}

const runBenchmarkBtn = $("#run-benchmark");
if (runBenchmarkBtn) {
  runBenchmarkBtn.addEventListener("click", async () => {
    const input = $("#benchmark-max-q");
    let n = input ? parseInt(input.value, 10) : 15;
    if (Number.isNaN(n) || n < 1) n = 15;
    if (n > 100) n = 100;

    setStatus("Benchmark en cours (patientez)...");
    try {
      const res = await fetch("/api/benchmark/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ max_questions: n }),
      });
      const out = await res.json();
      if (!res.ok) {
        setStatus(`Erreur : ${escapeHtml(out.error || res.statusText)}`);
        return;
      }
      const bn = out.best_model?.name;
      const note = out.best_model?.note;
      const ca = out.chat_alignment;
      let msg = bn
        ? `Benchmark terminé — meilleur modèle : ${bn}`
        : `Benchmark terminé${note ? " — " + note : " (aucun gagnant : voir l’onglet)"}`;
      if (ca?.applied && ca.hf_chat_model_id) {
        msg += ` — Chat : ${ca.hf_chat_model_id}`;
        if (!ca.partial) msg += " (LLM_PROVIDER → huggingface)";
      }
      setStatus(msg);
      await loadBenchmark();
      if (ca?.applied) await loadSettings();
    } catch (err) {
      setStatus(`Erreur : ${err.message}`);
    }
  });
}

// ── Paramètres ─────────────────────────────────────────────
async function loadSettings() {
  const statusEl = $("#settings-status");
  const hintOa = $("#openai-key-hint");
  const hintHf = $("#hf-token-hint");
  try {
    const res = await fetch("/api/settings");
    const d = await res.json();
    $("#llm-provider").value = d.llm_provider || "auto";
    $("#openai-chat-model").value = d.openai_chat_model || "";
    $("#hf-chat-model").value = d.hf_chat_model_id || "";
    const routerEl = $("#hf-router-url");
    if (routerEl) routerEl.textContent = d.hf_router_base_url || "";
    $("#top-k").value = String(d.rag_top_k ?? 5);
    $("#chunk-size").value = String(d.chunk_size ?? 512);
    $("#openai-key").value = "";
    $("#hf-token").value = "";
    if (hintOa) {
      hintOa.textContent = d.openai_key_set
        ? `Clé enregistrée (${d.openai_key_hint})`
        : "Aucune clé enregistrée";
    }
    if (hintHf) {
      hintHf.textContent = d.hf_token_set
        ? `Token enregistré (${d.hf_token_hint})`
        : "Aucun token enregistré";
    }
    const pre = $("#benchmark-models-pre");
    if (pre && d.benchmark_models) {
      pre.textContent = Object.entries(d.benchmark_models)
        .map(([k, v]) => `${k}: ${v}`)
        .join("\n");
    }
    if (statusEl) {
      statusEl.textContent = "";
      statusEl.classList.remove("settings-status--err");
    }
  } catch (e) {
    if (statusEl) {
      statusEl.textContent = "Impossible de charger les paramètres.";
      statusEl.classList.add("settings-status--err");
    }
  }
}

const saveSettingsBtn = $("#settings-save");
if (saveSettingsBtn) {
  saveSettingsBtn.addEventListener("click", async () => {
    const statusEl = $("#settings-status");
    const body = {
      llm_provider: $("#llm-provider").value,
      openai_chat_model: $("#openai-chat-model").value.trim(),
      hf_chat_model_id: $("#hf-chat-model").value.trim(),
      rag_top_k: parseInt($("#top-k").value, 10),
      chunk_size: parseInt($("#chunk-size").value, 10),
    };
    const oa = $("#openai-key").value;
    const hf = $("#hf-token").value;
    if (oa.trim()) body.openai_api_key = oa.trim();
    if (hf.trim()) body.huggingface_token = hf.trim();
    try {
      const res = await fetch("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const out = await res.json();
      if (!res.ok) {
        statusEl.textContent = out.error || "Erreur";
        statusEl.classList.add("settings-status--err");
        return;
      }
      statusEl.classList.remove("settings-status--err");
      statusEl.textContent = out.note || "Enregistré.";
      $("#openai-key").value = "";
      $("#hf-token").value = "";
      await loadSettings();
    } catch (err) {
      statusEl.textContent = err.message;
      statusEl.classList.add("settings-status--err");
    }
  });
}

// ── Utilities ──────────────────────────────────────────────
function setStatus(text) {
  $("#status-text").textContent = text;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// ── Init ───────────────────────────────────────────────────
fetch("/api/health")
  .then((r) => r.json())
  .then((d) => setStatus(`Prêt — ${d.pdfs_available} PDFs`))
  .catch(() => setStatus("Serveur non connecté"));
