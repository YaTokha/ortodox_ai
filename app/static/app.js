const STORAGE_HISTORY_KEY = "orthodox_sermon_history_v1";
const STORAGE_FEEDBACK_KEY = "orthodox_sermon_feedback_v1";

let selectedTemplate = "short";
let currentResult = null;
let readingMode = false;
let isGenerating = false;
let appInitialized = false;
let lastRequestId = "";

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function byId(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function formatSermonHtml(text) {
  let html = escapeHtml(text || "");
  html = html
    .replace(/(^|\n)(Проповедь на тему:[^\n]*)/gi, "$1<span class=\"sermon-title\">$2</span>")
    .replace(/(^|\n)(Вступление\.)/gi, "$1<strong>$2</strong>")
    .replace(/(^|\n)(Основная часть\.)/gi, "$1<strong>$2</strong>")
    .replace(/(^|\n)(Заключение\.)/gi, "$1<strong>$2</strong>");
  return html;
}

function formatDateTime(ts) {
  const date = new Date(ts);
  return date.toLocaleString("ru-RU", { hour12: false });
}

function metricLabelByValue(value) {
  const val = Number(value || 0);
  if (val >= 82) {
    return { text: "Высокое", cls: "good" };
  }
  if (val >= 62) {
    return { text: "Среднее", cls: "medium" };
  }
  return { text: "Низкое", cls: "low" };
}

function qualityRows(metrics) {
  return [
    { key: "overall_score", title: "Общий балл" },
    { key: "topic_relevance", title: "Релевантность теме" },
    { key: "structure_score", title: "Структура" },
    { key: "substance_score", title: "Содержательность" },
    { key: "diversity_score", title: "Разнообразие" },
    { key: "repetition_control_score", title: "Контроль повторов" },
  ].map((row) => ({ ...row, value: Number(metrics?.[row.key] ?? 0) }));
}

function renderQualitySingle(metrics) {
  const out = byId("qualityOut");
  if (!out) {
    return;
  }
  if (!metrics) {
    out.innerHTML = "<p class=\"note\">Метрики пока недоступны.</p>";
    return;
  }
  const overall = Number(metrics.overall_score || 0);
  const badge = metricLabelByValue(overall);
  const rows = qualityRows(metrics)
    .map(
      (r) => `
      <div class="quality-row">
        <div class="quality-head"><span>${escapeHtml(r.title)}</span><strong>${r.value.toFixed(2)}%</strong></div>
        <div class="quality-bar"><div class="quality-fill" style="width:${Math.max(0, Math.min(100, r.value))}%"></div></div>
      </div>
    `
    )
    .join("");

  const notes = Array.isArray(metrics?.notes) && metrics.notes.length
    ? `<ul class="quality-notes">${metrics.notes.map((n) => `<li>${escapeHtml(n)}</li>`).join("")}</ul>`
    : "";

  out.innerHTML = `
    <p><span class="quality-pill ${badge.cls}">${badge.text}</span></p>
    ${rows}
    ${notes}
  `;
}

function renderQualityCompare(metricsA, metricsB) {
  const out = byId("qualityOut");
  if (!out) {
    return;
  }
  const variantHtml = (title, m) => {
    if (!m) {
      return `<div class="quality-variant"><h4>${escapeHtml(title)}</h4><p class="note">Метрики недоступны.</p></div>`;
    }
    const badge = metricLabelByValue(Number(m.overall_score || 0));
    const rows = qualityRows(m)
      .map(
        (r) => `
          <div class="quality-row">
            <div class="quality-head"><span>${escapeHtml(r.title)}</span><strong>${r.value.toFixed(2)}%</strong></div>
            <div class="quality-bar"><div class="quality-fill" style="width:${Math.max(0, Math.min(100, r.value))}%"></div></div>
          </div>
        `
      )
      .join("");
    return `
      <div class="quality-variant">
        <h4>${escapeHtml(title)} <span class="quality-pill ${badge.cls}">${badge.text}</span></h4>
        ${rows}
      </div>
    `;
  };

  out.innerHTML = `
    <div class="quality-compare">
      ${variantHtml("Вариант A", metricsA)}
      ${variantHtml("Вариант B", metricsB)}
    </div>
  `;
}

function normalizeSentenceKey(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/[^а-яёa-z0-9 ]+/gi, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function splitIntoSentences(text) {
  return String(text || "")
    .split(/(?<=[.!?])\s+/)
    .map((x) => x.trim())
    .filter((x) => x.length > 20);
}

function renderCompareDiff(textA, textB) {
  const diffBlock = byId("diffBlock");
  const diffOut = byId("diffOut");
  if (!diffBlock || !diffOut) {
    return;
  }
  const aSent = splitIntoSentences(textA);
  const bSent = splitIntoSentences(textB);
  const aMap = new Map(aSent.map((s) => [normalizeSentenceKey(s), s]));
  const bMap = new Map(bSent.map((s) => [normalizeSentenceKey(s), s]));

  const onlyA = [];
  for (const [k, s] of aMap.entries()) {
    if (!bMap.has(k)) {
      onlyA.push(s);
    }
  }
  const onlyB = [];
  for (const [k, s] of bMap.entries()) {
    if (!aMap.has(k)) {
      onlyB.push(s);
    }
  }

  const overlap = Math.max(0, aMap.size + bMap.size - onlyA.length - onlyB.length);
  const union = new Set([...aMap.keys(), ...bMap.keys()]).size || 1;
  const similarity = (overlap / union) * 100;

  diffOut.innerHTML = `
    <div class="diff-stat"><strong>Похожесть вариантов:</strong> ${similarity.toFixed(2)}%</div>
    <div class="diff-cols">
      <div>
        <strong>Уникально в A</strong>
        <ul class="diff-list">${onlyA.slice(0, 5).map((s) => `<li>${escapeHtml(s)}</li>`).join("") || "<li>Существенных отличий нет.</li>"}</ul>
      </div>
      <div>
        <strong>Уникально в B</strong>
        <ul class="diff-list">${onlyB.slice(0, 5).map((s) => `<li>${escapeHtml(s)}</li>`).join("") || "<li>Существенных отличий нет.</li>"}</ul>
      </div>
    </div>
  `;
  diffBlock.classList.remove("hidden");
}

function hideDiffBlock() {
  const diffBlock = byId("diffBlock");
  if (!diffBlock) {
    return;
  }
  diffBlock.classList.add("hidden");
}

function topicMarkers(topic, prompt) {
  const source = `${topic || ""} ${prompt || ""}`
    .toLowerCase()
    .replace(/[^а-яёa-z0-9 ]+/gi, " ");
  const stop = new Set(["проповедь", "подготовь", "сгенерируй", "тему", "тема", "о", "про", "на", "и", "для", "по"]);
  return source
    .split(/\s+/)
    .map((w) => w.trim())
    .filter((w) => w.length >= 4 && !stop.has(w))
    .slice(0, 8);
}

function hasAny(text, markers) {
  const low = String(text || "").toLowerCase();
  return markers.some((m) => low.includes(m));
}

function buildSermonChecklist(sermon, payload = {}) {
  const low = String(sermon || "").toLowerCase();
  const markers = topicMarkers(payload.topic, payload.prompt);
  const topicCovered = markers.length === 0 ? true : markers.some((m) => low.includes(m.slice(0, Math.max(4, m.length - 1))));

  return [
    {
      label: "Есть четкая структура: Вступление / Основная часть / Заключение",
      ok: low.includes("вступление.") && low.includes("основная часть.") && low.includes("заключение."),
    },
    {
      label: "Тема раскрыта в тексте (не только в заголовке)",
      ok: topicCovered,
    },
    {
      label: "Добавлено изречение из Ветхого Завета",
      ok: low.includes("ветхий завет"),
    },
    {
      label: "Добавлено изречение из Посланий апостолов",
      ok: low.includes("послание святых апостолов"),
    },
    {
      label: "Есть святоотеческое наставление",
      ok: hasAny(low, ["свт.", "свят", "прп.", "наставляет"]),
    },
    {
      label: "Есть проповедническая ссылка на современную традицию",
      ok: hasAny(low, ["проповеднической традиции звучит слово", "митрополит", "протоиерей", "священник"]),
    },
  ];
}

function checklistItemHtml(item) {
  const icon = item.ok ? "✓" : "•";
  const cls = item.ok ? "ok" : "todo";
  return `<li class="check-item ${cls}"><span class="check-icon">${icon}</span><span>${escapeHtml(item.label)}</span></li>`;
}

function renderChecklistSingle(sermon, payload) {
  const out = byId("checklistOut");
  if (!out) {
    return;
  }
  const list = buildSermonChecklist(sermon, payload);
  out.innerHTML = `<ul class="checklist-list">${list.map(checklistItemHtml).join("")}</ul>`;
}

function renderChecklistCompare(sermonA, sermonB, payload) {
  const out = byId("checklistOut");
  if (!out) {
    return;
  }
  const listA = buildSermonChecklist(sermonA, payload);
  const listB = buildSermonChecklist(sermonB, payload);
  out.innerHTML = `
    <div class="checklist-compare">
      <div>
        <h4>Вариант A</h4>
        <ul class="checklist-list">${listA.map(checklistItemHtml).join("")}</ul>
      </div>
      <div>
        <h4>Вариант B</h4>
        <ul class="checklist-list">${listB.map(checklistItemHtml).join("")}</ul>
      </div>
    </div>
  `;
}

function getTemplateConfig(templateId) {
  const configs = {
    short: {
      style: "краткая",
      max_new_tokens: 420,
      extraPrompt: "Сделай проповедь краткой и ясной, без потери богословского смысла.",
    },
    normal: {
      style: "пастырский",
      max_new_tokens: 620,
      extraPrompt: "Сделай проповедь цельной и средней длины.",
    },
    festive: {
      style: "торжественный",
      max_new_tokens: 760,
      extraPrompt: "Сделай проповедь торжественной, с праздничным настроением и духовной глубиной.",
    },
    strict_sin: {
      style: "строгий",
      max_new_tokens: 780,
      extraPrompt: "Если тема связана с грехом, сделай строгую обличительную проповедь с ясным призывом к покаянию.",
    },
  };
  return configs[templateId] || configs.normal;
}

function showValidation(message) {
  const msg = byId("validationMsg");
  if (!msg) {
    return;
  }
  msg.textContent = message || "";
  msg.classList.toggle("visible", Boolean(message));
}

function setGenerationStatus(message, tone = "muted") {
  const el = byId("generationStatus");
  if (!el) {
    return;
  }
  el.textContent = message || "";
  el.classList.remove("good", "warn", "error");
  if (tone === "good") {
    el.classList.add("good");
  } else if (tone === "warn") {
    el.classList.add("warn");
  } else if (tone === "error") {
    el.classList.add("error");
  }
}

function validateInput(payload) {
  const prompt = (payload.prompt || "").trim();
  const topic = (payload.topic || "").trim();

  if (!prompt && !topic) {
    return "Укажите промт или тему проповеди.";
  }
  if (prompt && prompt.length < 12) {
    return "Промт слишком короткий. Добавьте конкретику: тему, аудиторию или библейский фрагмент.";
  }
  if (topic && topic.length < 3) {
    return "Тема слишком короткая.";
  }
  return "";
}

function buildPayload() {
  const promptEl = byId("prompt");
  const topicEl = byId("topic");
  const occasionEl = byId("occasion");
  const audienceEl = byId("audience");
  const bibleTextEl = byId("bibleText");

  const cfg = getTemplateConfig(selectedTemplate);

  const rawPrompt = (promptEl?.value || "").trim();
  const finalPrompt = rawPrompt;

  return {
    prompt: finalPrompt || null,
    topic: (topicEl?.value || "").trim() || null,
    occasion: (occasionEl?.value || "").trim() || null,
    audience: (audienceEl?.value || "").trim() || "приход",
    bible_text: (bibleTextEl?.value || "").trim() || null,
    style: cfg.style,
    variant_tag: null,
    max_new_tokens: cfg.max_new_tokens,
  };
}

async function callApi(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const requestId = res.headers.get("x-request-id") || "";

  if (!res.ok) {
    let message = `Ошибка запроса (HTTP ${res.status}).`;
    try {
      const data = await res.json();
      if (typeof data?.error === "string") {
        message = data.error;
      }
      if (typeof data?.request_id === "string" && data.request_id.trim()) {
        message += `\nID запроса: ${data.request_id}`;
      } else if (requestId) {
        message += `\nID запроса: ${requestId}`;
      }
      if (Array.isArray(data?.details) && data.details.length > 0) {
        message += `\n- ${data.details.join("\n- ")}`;
      } else if (Array.isArray(data?.detail) && data.detail.length > 0) {
        message += "\n- Проверьте корректность заполнения полей формы.";
      } else if (typeof data?.detail === "string" && data.detail.trim()) {
        message += `\n${data.detail}`;
      }
    } catch {
      const text = await res.text();
      if (text && text.trim()) {
        message = `${message}\n${text}`;
      }
    }
    throw new Error(message);
  }
  return { data: await res.json(), requestId };
}

function getHistory() {
  try {
    const raw = localStorage.getItem(STORAGE_HISTORY_KEY);
    if (!raw) {
      return [];
    }
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function setHistory(items) {
  localStorage.setItem(STORAGE_HISTORY_KEY, JSON.stringify(items));
}

function addHistoryItem(item) {
  const history = getHistory();
  history.unshift(item);
  const trimmed = history.slice(0, 10);
  setHistory(trimmed);
  renderHistory();
}

function renderHistory() {
  const list = byId("historyList");
  if (!list) {
    return;
  }
  const history = getHistory();
  if (history.length === 0) {
    list.innerHTML = "<li class=\"history-empty\">История пока пуста.</li>";
    return;
  }
  list.innerHTML = history
    .map(
      (item, idx) =>
        `<li>
          <button type="button" class="history-item" data-history-index="${idx}">
            <span class="history-title">${escapeHtml(item.title)}</span>
            <span class="history-meta">${escapeHtml(formatDateTime(item.ts))}</span>
          </button>
        </li>`
    )
    .join("");

  list.querySelectorAll(".history-item").forEach((btn) => {
    btn.addEventListener("click", () => {
      const index = Number(btn.getAttribute("data-history-index") || "-1");
      const selected = history[index];
      if (!selected) {
        return;
      }
      const out = byId("generateOut");
      const outA = byId("compareOutA");
      const outB = byId("compareOutB");
      const compareWrap = byId("compareWrap");
      const singleWrap = byId("singleOutWrap");
      if (selected.mode === "compare") {
        if (singleWrap) {
          singleWrap.classList.add("hidden");
        }
        if (compareWrap) {
          compareWrap.classList.remove("hidden");
        }
        if (outA) {
          outA.innerHTML = formatSermonHtml(selected.sermonA || "");
        }
        if (outB) {
          outB.innerHTML = formatSermonHtml(selected.sermonB || "");
        }
        currentResult = {
          type: "compare",
          a: { sermon: selected.sermonA || "", quality: selected.qualityA || null },
          b: { sermon: selected.sermonB || "", quality: selected.qualityB || null },
        };
        renderQualityCompare(selected.qualityA || null, selected.qualityB || null);
        renderCompareDiff(selected.sermonA || "", selected.sermonB || "");
        renderChecklistCompare(selected.sermonA || "", selected.sermonB || "", {
          topic: selected.title || "",
          prompt: selected.title || "",
        });
      } else {
        if (out) {
          out.innerHTML = formatSermonHtml(selected.sermon || "");
        }
        if (singleWrap) {
          singleWrap.classList.remove("hidden");
        }
        if (compareWrap) {
          compareWrap.classList.add("hidden");
        }
        currentResult = {
          type: "single",
          data: selected,
        };
        renderQualitySingle(selected.quality || null);
        hideDiffBlock();
        renderChecklistSingle(selected.sermon || "", {
          topic: selected.title || "",
          prompt: selected.title || "",
        });
      }
      renderSources(selected.citations || []);
      applyReadability();
    });
  });
}

function renderSources(citations) {
  const list = byId("sourcesOut");
  if (!list) {
    return;
  }
  if (!Array.isArray(citations) || citations.length === 0) {
    list.innerHTML = "<li>Источники не найдены в текущем корпусе.</li>";
    return;
  }
  const short = citations.slice(0, 5);
  list.innerHTML = short
    .map((c) => {
      const kind = c?.source_type || "source";
      const author = c?.author || c?.title || "без автора";
      const ref = c?.reference || c?.title || c?.id || "";
      return `<li><strong>${escapeHtml(kind)}</strong>: ${escapeHtml(author)}${ref ? ` — ${escapeHtml(ref)}` : ""}</li>`;
    })
    .join("");
}

function setOutputLoading(message) {
  const out = byId("generateOut");
  const outA = byId("compareOutA");
  const outB = byId("compareOutB");
  const html = `
    <div class="loading-box">
      <span class="loading-spinner" aria-hidden="true"></span>
      <span>${escapeHtml(message || "Генерация...")}</span>
    </div>
  `;
  if (out) {
    out.innerHTML = html;
  }
  if (outA) {
    outA.innerHTML = html;
  }
  if (outB) {
    outB.innerHTML = html;
  }
}

function setGenerateBusy(busy) {
  const btn = byId("generateBtn");
  if (btn) {
    btn.disabled = busy;
    btn.textContent = busy ? "Генерация..." : "Сгенерировать проповедь";
  }
  const ids = [
    "compareMode",
    "prompt",
    "topic",
    "occasion",
    "audience",
    "bibleText",
    "calendarUseBtn",
    "topicPreset",
    "occasionPreset",
    "audiencePreset",
    "fullscreenBtn",
  ];
  ids.forEach((id) => {
    const el = byId(id);
    if (el) {
      el.disabled = busy;
    }
  });
  document.querySelectorAll(".quick-topic, .template-chip").forEach((el) => {
    el.disabled = busy;
  });
}

function scrollToResult() {
  const target = byId("singleOutWrap") || byId("compareWrap");
  if (target) {
    target.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function getActiveSermonContainer() {
  const compareWrap = byId("compareWrap");
  const singleWrap = byId("singleOutWrap");
  if (compareWrap && !compareWrap.classList.contains("hidden")) {
    return compareWrap;
  }
  return singleWrap;
}

function updateFullscreenButton() {
  const btn = byId("fullscreenBtn");
  if (!btn) {
    return;
  }
  btn.textContent = document.fullscreenElement ? "Выйти из полного экрана" : "Во весь экран";
}

async function toggleFullscreenView() {
  const target = getActiveSermonContainer();
  if (!target) {
    showValidation("Сначала сгенерируйте проповедь.");
    return;
  }
  if (!document.fullscreenEnabled) {
    showValidation("Полноэкранный режим не поддерживается в этом браузере.");
    return;
  }
  try {
    if (document.fullscreenElement) {
      await document.exitFullscreen();
    } else {
      await target.requestFullscreen();
    }
  } catch (err) {
    showValidation(`Не удалось открыть полноэкранный режим: ${err.message}`);
  } finally {
    updateFullscreenButton();
  }
}

async function runGenerateSingle(payload) {
  const singleWrap = byId("singleOutWrap");
  const compareWrap = byId("compareWrap");
  const out = byId("generateOut");
  if (singleWrap) {
    singleWrap.classList.remove("hidden");
  }
  if (compareWrap) {
    compareWrap.classList.add("hidden");
  }

  setOutputLoading("Формируем проповедь...");
  const { data, requestId } = await callApi("/api/generate", payload);
  lastRequestId = requestId || "";
  if (out) {
    out.innerHTML = formatSermonHtml(data.sermon);
  }
  renderSources(data.citations || []);
  renderQualitySingle(data.quality || null);
  renderChecklistSingle(data.sermon || "", payload);
  hideDiffBlock();
  currentResult = { type: "single", data };
  addHistoryItem({
    ts: Date.now(),
    title: payload.prompt || payload.topic || "Проповедь",
    sermon: data.sermon,
    quality: data.quality || null,
    citations: data.citations || [],
  });
  applyReadability();
  scrollToResult();
  setGenerationStatus(
    `Готово. Сформирована 1 проповедь.${lastRequestId ? ` ID запроса: ${lastRequestId}` : ""}`,
    "good"
  );
}

async function runGenerateCompare(payloadBase) {
  const singleWrap = byId("singleOutWrap");
  const compareWrap = byId("compareWrap");
  const outA = byId("compareOutA");
  const outB = byId("compareOutB");

  if (singleWrap) {
    singleWrap.classList.add("hidden");
  }
  if (compareWrap) {
    compareWrap.classList.remove("hidden");
  }
  setOutputLoading("Генерация двух вариантов...");

  const baseTemp = Number(payloadBase.temperature ?? 0.78);
  const baseTopP = Number(payloadBase.top_p ?? 0.92);
  const baseRep = Number(payloadBase.repetition_penalty ?? 1.12);

  const payloadA = {
    ...payloadBase,
    variant_tag: "A",
    temperature: clamp(baseTemp - 0.07, 0.58, 1.2),
    top_p: clamp(baseTopP - 0.04, 0.8, 0.98),
    repetition_penalty: clamp(baseRep + 0.02, 1.0, 2.0),
  };
  const payloadB = {
    ...payloadBase,
    variant_tag: "B",
    temperature: clamp(baseTemp + 0.05, 0.58, 1.25),
    top_p: clamp(baseTopP + 0.02, 0.82, 0.99),
    repetition_penalty: clamp(baseRep + 0.06, 1.0, 2.0),
  };

  const [{ data: a, requestId: requestIdA }, { data: b, requestId: requestIdB }] = await Promise.all([
    callApi("/api/generate", payloadA),
    callApi("/api/generate", payloadB),
  ]);
  lastRequestId = requestIdB || requestIdA || "";

  if (outA) {
    outA.innerHTML = formatSermonHtml(a.sermon);
  }
  if (outB) {
    outB.innerHTML = formatSermonHtml(b.sermon);
  }
  renderSources(a.citations || b.citations || []);
  renderQualityCompare(a.quality || null, b.quality || null);
  renderCompareDiff(a.sermon || "", b.sermon || "");
  renderChecklistCompare(a.sermon || "", b.sermon || "", payloadBase);
  currentResult = { type: "compare", a, b };
  addHistoryItem({
    ts: Date.now(),
    title: `${payloadBase.prompt || payloadBase.topic || "Проповедь"} (сравнение)`,
    mode: "compare",
    sermon: `Вариант A:\n\n${a.sermon}\n\n\nВариант B:\n\n${b.sermon}`,
    sermonA: a.sermon,
    sermonB: b.sermon,
    qualityA: a.quality || null,
    qualityB: b.quality || null,
    citations: a.citations || [],
  });
  applyReadability();
  scrollToResult();
  setGenerationStatus(
    `Готово. Сформированы 2 варианта проповеди.${lastRequestId ? ` ID запроса: ${lastRequestId}` : ""}`,
    "good"
  );
}

async function runGenerate() {
  if (isGenerating) {
    showValidation("Генерация уже выполняется. Дождитесь завершения текущего запроса.");
    setGenerationStatus("Запрос уже выполняется. Подождите завершения текущей генерации.", "warn");
    return;
  }

  showValidation("");
  setGenerationStatus("");
  const compareMode = byId("compareMode")?.checked;
  const payload = buildPayload();
  const validation = validateInput(payload);
  if (validation) {
    showValidation(validation);
    setGenerationStatus("Проверьте входные поля перед генерацией.", "warn");
    return;
  }

  isGenerating = true;
  setGenerateBusy(true);
  setGenerationStatus(
    compareMode
      ? "Запрос принят. Генерируются два варианта проповеди..."
      : "Запрос принят. Генерируется проповедь...",
    "warn"
  );
  try {
    if (compareMode) {
      await runGenerateCompare(payload);
    } else {
      await runGenerateSingle(payload);
    }
  } catch (err) {
    const out = byId("generateOut");
    if (out) {
      out.textContent = `Ошибка: ${err.message}`;
    }
    const message = String(err?.message || "");
    const match = message.match(/ID запроса:\s*([a-zA-Z0-9-]+)/);
    if (match && match[1]) {
      lastRequestId = match[1];
    }
    showValidation("Не удалось сгенерировать проповедь. Проверьте интернет и данные формы.");
    setGenerationStatus(
      `Ошибка генерации.${lastRequestId ? ` ID запроса: ${lastRequestId}` : ""}`,
      "error"
    );
  } finally {
    isGenerating = false;
    setGenerateBusy(false);
  }
}

async function runHealth() {
  const out = byId("healthOut");
  if (!out) {
    return;
  }
  out.innerHTML = "<p class=\"note\">Проверка состояния сервиса...</p>";
  try {
    const res = await fetch("/api/health/human");
    const data = await res.json();
    const serviceStatus = String(data?.service_status || "Нет данных");
    const generationStatus = String(data?.generation_status || "Нет данных");
    const modelName = String(data?.model_name || "не указана");
    const modelLoaded = Boolean(data?.model_loaded);
    const adapterLoaded = Boolean(data?.adapter_loaded);
    const uptime = String(data?.uptime_human || "—");
    const rateLimitNote = String(data?.rate_limit_note || "");

    out.innerHTML = `
      <div class="health-item">
        <span class="health-badge ${modelLoaded ? "good" : "warn"}">${modelLoaded ? "Готов" : "Ограниченно"}</span>
        <p><strong>${escapeHtml(serviceStatus)}</strong></p>
      </div>
      <p>${escapeHtml(generationStatus)}</p>
      <p><strong>Модель:</strong> ${escapeHtml(modelName)}</p>
      <p><strong>Адаптер:</strong> ${adapterLoaded ? "подключен" : "не подключен"}</p>
      <p><strong>Время работы:</strong> ${escapeHtml(uptime)}</p>
      <p class="note">${escapeHtml(rateLimitNote)}</p>
    `;
  } catch (err) {
    out.innerHTML = `<p class="health-error">Не удалось получить состояние сервиса: ${escapeHtml(err.message)}</p>`;
  }
}

function setTemplate(templateId) {
  selectedTemplate = templateId;
  document.querySelectorAll(".template-chip").forEach((el) => {
    el.classList.toggle("active", el.getAttribute("data-template") === templateId);
  });
}

function getCurrentSermonText() {
  if (!currentResult) {
    return "";
  }
  if (currentResult.type === "single") {
    return currentResult.data?.sermon || "";
  }
  if (currentResult.type === "compare") {
    return `Вариант A:\n\n${currentResult.a?.sermon || ""}\n\n\nВариант B:\n\n${currentResult.b?.sermon || ""}`;
  }
  return "";
}

function downloadBlob(filename, blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function bindQuickSelects() {
  const topicPreset = byId("topicPreset");
  const topicInput = byId("topic");
  const occasionPreset = byId("occasionPreset");
  const occasionInput = byId("occasion");
  const audiencePreset = byId("audiencePreset");
  const audienceInput = byId("audience");

  if (topicPreset && topicInput) {
    topicPreset.addEventListener("change", () => {
      const value = String(topicPreset.value || "").trim();
      if (value) {
        topicInput.value = value;
      }
    });
  }

  if (occasionPreset && occasionInput) {
    occasionPreset.addEventListener("change", () => {
      const value = String(occasionPreset.value || "").trim();
      if (value) {
        occasionInput.value = value;
      }
    });
  }

  if (audiencePreset && audienceInput) {
    audiencePreset.addEventListener("change", () => {
      const value = String(audiencePreset.value || "").trim();
      if (value) {
        audienceInput.value = value;
      }
    });
  }
}

async function exportDocx(text) {
  if (!window.JSZip) {
    throw new Error("Библиотека экспорта DOCX не загружена.");
  }
  const zip = new window.JSZip();
  const esc = (s) =>
    String(s)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");

  const paragraphs = text
    .split(/\n+/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => `<w:p><w:r><w:t xml:space="preserve">${esc(line)}</w:t></w:r></w:p>`)
    .join("");

  zip.file(
    "[Content_Types].xml",
    `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>`
  );
  zip.folder("_rels").file(
    ".rels",
    `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>`
  );
  zip.folder("word").file(
    "document.xml",
    `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas"
 xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"
 xmlns:o="urn:schemas-microsoft-com:office:office"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"
 xmlns:v="urn:schemas-microsoft-com:vml"
 xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing"
 xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
 xmlns:w10="urn:schemas-microsoft-com:office:word"
 xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
 xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"
 xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup"
 xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk"
 xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml"
 xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"
 mc:Ignorable="w14 wp14">
  <w:body>
    ${paragraphs}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>
    </w:sectPr>
  </w:body>
</w:document>`
  );

  const blob = await zip.generateAsync({
    type: "blob",
    mimeType: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  });
  downloadBlob("propoved.docx", blob);
}

function printCurrent() {
  const text = getCurrentSermonText();
  if (!text) {
    showValidation("Сначала сгенерируйте проповедь.");
    return;
  }
  const win = window.open("", "_blank");
  if (!win) {
    return;
  }
  win.document.write(
    `<html><head><title>Печать проповеди</title><style>
      body { font-family: 'Times New Roman', serif; margin: 2cm; line-height: 1.5; white-space: pre-wrap; }
    </style></head><body>${escapeHtml(text)}</body></html>`
  );
  win.document.close();
  win.focus();
  win.print();
}

function applyReadability() {
  const fontSizeVal = Number(byId("fontSizeRange")?.value || 18);
  const lineHeightRaw = Number(byId("lineHeightRange")?.value || 17);
  const lineHeightVal = (lineHeightRaw / 10).toFixed(2);
  document.documentElement.style.setProperty("--sermon-font-size", `${fontSizeVal}px`);
  document.documentElement.style.setProperty("--sermon-line-height", `${lineHeightVal}`);
}

function calendarFallback(today) {
  const mmdd = `${String(today.getMonth() + 1).padStart(2, "0")}-${String(today.getDate()).padStart(2, "0")}`;
  const fixedFeasts = {
    "01-07": "Рождество Христово",
    "01-19": "Крещение Господне (Богоявление)",
    "02-15": "Сретение Господне",
    "04-07": "Благовещение Пресвятой Богородицы",
    "08-19": "Преображение Господне",
    "08-28": "Успение Пресвятой Богородицы",
    "09-21": "Рождество Пресвятой Богородицы",
    "09-27": "Воздвижение Креста Господня",
    "12-04": "Введение во храм Пресвятой Богородицы",
  };
  const feast = fixedFeasts[mmdd] || (today.getDay() === 0 ? "Воскресный день" : "");
  return {
    topic_of_day: feast || "Память святых дня",
    main_feast: feast || null,
    feasts: feast ? [feast] : [],
    saints: ["Память святых дня"],
    fasting: null,
    source: "local-fallback",
  };
}

function buildCalendarHtml(payload, dateRu) {
  const feasts = Array.isArray(payload?.feasts) ? payload.feasts.filter(Boolean).slice(0, 4) : [];
  const saints = Array.isArray(payload?.saints) ? payload.saints.filter(Boolean).slice(0, 12) : [];
  const mainFeast = (payload?.main_feast || "").trim();
  const topic = (payload?.topic_of_day || "").trim() || mainFeast || "Память святых дня";
  const fasting = (payload?.fasting || "").trim();
  const source = (payload?.source || "").trim();

  const feastHtml = feasts.length
    ? `<ul class="calendar-list">${feasts.map((x) => `<li>${escapeHtml(x)}</li>`).join("")}</ul>`
    : "<p>Сегодня великий двунадесятый праздник не отмечается.</p>";
  const saintsHtml = saints.length
    ? `<ul class="calendar-list">${saints.map((x) => `<li>${escapeHtml(x)}</li>`).join("")}</ul>`
    : "<p>Данные о памяти святых временно недоступны.</p>";
  const fastingHtml = fasting ? `<p><strong>Пост / седмица:</strong> ${escapeHtml(fasting)}</p>` : "";
  const sourceHtml = source ? `<p class="calendar-source">Источник: ${escapeHtml(source)}</p>` : "";
  const sourceWarnHtml = source === "local-fallback"
    ? "<p class=\"calendar-source\">Не удалось получить данные из интернета, показан локальный календарь.</p>"
    : "";

  return {
    topic,
    html: `
      <p><strong>Сегодня:</strong> ${escapeHtml(dateRu)}</p>
      <p><strong>Главная тема дня:</strong> ${escapeHtml(topic)}</p>
      <p><strong>Праздник:</strong></p>
      ${feastHtml}
      <p><strong>Сегодня вспоминаются святые:</strong></p>
      ${saintsHtml}
      ${fastingHtml}
      ${sourceHtml}
      ${sourceWarnHtml}
    `,
  };
}

async function renderCalendar(forceRefresh = false) {
  const box = byId("calendarInfo");
  if (!box) {
    return;
  }
  const today = new Date();
  const isoDate = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, "0")}-${String(today.getDate()).padStart(2, "0")}`;
  const dateRu = today.toLocaleDateString("ru-RU", {
    day: "2-digit",
    month: "long",
    year: "numeric",
    weekday: "long",
  });
  box.innerHTML = forceRefresh
    ? "<p>Обновляем православный календарь дня...</p>"
    : "<p>Загружаем православный календарь дня...</p>";

  let payload = calendarFallback(today);
  try {
    const params = new URLSearchParams();
    params.set("day", isoDate);
    if (forceRefresh) {
      params.set("force_refresh", "true");
    }
    const res = await fetch(`/api/calendar/day?${params.toString()}`);
    if (res.ok) {
      const data = await res.json();
      if (data && typeof data === "object") {
        payload = { ...payload, ...data };
      }
    }
  } catch {
    payload = calendarFallback(today);
  }

  const built = buildCalendarHtml(payload, payload.date_ru || dateRu);
  const dayTopic = built.topic;
  box.innerHTML = built.html;

  const useBtn = byId("calendarUseBtn");
  if (useBtn) {
    useBtn.onclick = () => {
      const promptEl = byId("prompt");
      if (promptEl) {
        promptEl.value = `Подготовь проповедь на тему: ${dayTopic}. Опирайся на евангельское чтение и святоотеческую традицию дня, сделай пастырское слово для приходской аудитории с практическими выводами.`;
      }
      const topicEl = byId("topic");
      if (topicEl) {
        topicEl.value = dayTopic;
      }
      showValidation("");
    };
  }

  const refreshBtn = byId("calendarRefreshBtn");
  if (refreshBtn) {
    refreshBtn.onclick = async () => {
      refreshBtn.disabled = true;
      try {
        await renderCalendar(true);
      } finally {
        refreshBtn.disabled = false;
      }
    };
  }
}

function bindFeedback() {
  const msg = byId("feedbackMsg");
  document.querySelectorAll(".feedback-chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      const key = btn.getAttribute("data-feedback") || "other";
      let store = {};
      try {
        store = JSON.parse(localStorage.getItem(STORAGE_FEEDBACK_KEY) || "{}");
      } catch {
        store = {};
      }
      store[key] = Number(store[key] || 0) + 1;
      localStorage.setItem(STORAGE_FEEDBACK_KEY, JSON.stringify(store));
      if (msg) {
        msg.textContent = "Спасибо, отметка сохранена.";
      }
    });
  });
}

function bindButtons() {
  const generateBtn = byId("generateBtn");
  const healthBtn = byId("healthBtn");
  const copyBtn = byId("copyBtn");
  const txtBtn = byId("txtBtn");
  const docxBtn = byId("docxBtn");
  const printBtn = byId("printBtn");
  const readingModeBtn = byId("readingModeBtn");
  const fullscreenBtn = byId("fullscreenBtn");
  const clearHistoryBtn = byId("clearHistoryBtn");

  if (generateBtn) {
    generateBtn.addEventListener("click", runGenerate);
  }
  if (healthBtn) {
    healthBtn.addEventListener("click", runHealth);
  }

  if (copyBtn) {
    copyBtn.addEventListener("click", async () => {
      const text = getCurrentSermonText();
      if (!text) {
        showValidation("Сначала сгенерируйте проповедь.");
        return;
      }
      await navigator.clipboard.writeText(text);
      showValidation("Текст проповеди скопирован.");
    });
  }

  if (txtBtn) {
    txtBtn.addEventListener("click", () => {
      const text = getCurrentSermonText();
      if (!text) {
        showValidation("Сначала сгенерируйте проповедь.");
        return;
      }
      downloadBlob("propoved.txt", new Blob([text], { type: "text/plain;charset=utf-8" }));
    });
  }

  if (docxBtn) {
    docxBtn.addEventListener("click", async () => {
      const text = getCurrentSermonText();
      if (!text) {
        showValidation("Сначала сгенерируйте проповедь.");
        return;
      }
      try {
        await exportDocx(text);
      } catch (err) {
        showValidation(`Не удалось экспортировать DOCX: ${err.message}`);
      }
    });
  }

  if (printBtn) {
    printBtn.addEventListener("click", printCurrent);
  }

  if (readingModeBtn) {
    readingModeBtn.addEventListener("click", () => {
      readingMode = !readingMode;
      document.body.classList.toggle("reading-mode", readingMode);
    });
  }
  if (fullscreenBtn) {
    fullscreenBtn.addEventListener("click", toggleFullscreenView);
  }

  if (clearHistoryBtn) {
    clearHistoryBtn.addEventListener("click", () => {
      localStorage.removeItem(STORAGE_HISTORY_KEY);
      renderHistory();
    });
  }

  document.querySelectorAll(".quick-topic").forEach((btn) => {
    btn.addEventListener("click", () => {
      const promptEl = byId("prompt");
      const text = btn.getAttribute("data-prompt") || "";
      if (promptEl) {
        promptEl.value = text;
      }
      showValidation("");
    });
  });

  document.querySelectorAll(".template-chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      setTemplate(btn.getAttribute("data-template") || "normal");
    });
  });

  const fontSize = byId("fontSizeRange");
  const lineHeight = byId("lineHeightRange");
  if (fontSize) {
    fontSize.addEventListener("input", applyReadability);
  }
  if (lineHeight) {
    lineHeight.addEventListener("input", applyReadability);
  }

  bindFeedback();
  bindQuickSelects();
}

function initApp() {
  if (appInitialized) {
    return;
  }
  appInitialized = true;
  setTemplate(selectedTemplate);
  bindButtons();
  renderHistory();
  renderCalendar();
  applyReadability();
  renderQualitySingle(null);
  const checklistOut = byId("checklistOut");
  if (checklistOut) {
    checklistOut.innerHTML = "<p class=\"note\">Чек-лист заполнится после генерации проповеди.</p>";
  }
  setGenerationStatus("Готов к работе. Укажите тему или промт и нажмите «Сгенерировать проповедь».");
  runHealth();
  hideDiffBlock();
  updateFullscreenButton();
  document.addEventListener("fullscreenchange", updateFullscreenButton);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initApp);
} else {
  initApp();
}

window.runGenerate = runGenerate;
window.runHealth = runHealth;
