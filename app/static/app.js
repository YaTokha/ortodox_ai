function byId(id) {
  return document.getElementById(id);
}

async function callApi(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    let message = `Ошибка запроса (HTTP ${res.status}).`;
    try {
      const data = await res.json();
      if (typeof data?.error === "string") {
        message = data.error;
      }
      if (Array.isArray(data?.details) && data.details.length > 0) {
        message += `\n- ${data.details.join("\n- ")}`;
      } else if (Array.isArray(data?.detail) && data.detail.length > 0) {
        message += `\n- Проверьте корректность заполнения полей формы.`;
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
  return await res.json();
}

async function runGenerate() {
  const out = byId("generateOut");
  const promptEl = byId("prompt");
  const topicEl = byId("topic");
  const occasionEl = byId("occasion");
  const audienceEl = byId("audience");
  const bibleTextEl = byId("bibleText");
  const temperatureEl = byId("temperature");
  const topPEl = byId("topP");
  const maxTokensEl = byId("maxTokens");
  if (!out || !promptEl || !topicEl || !occasionEl || !audienceEl || !bibleTextEl || !temperatureEl || !topPEl || !maxTokensEl) {
    return;
  }

  out.textContent = "Генерация...";

  const payload = {
    prompt: promptEl.value || null,
    topic: topicEl.value || null,
    occasion: occasionEl.value || null,
    audience: audienceEl.value || "приход",
    bible_text: bibleTextEl.value || null,
    temperature: Number(temperatureEl.value || 0.95),
    top_p: Number(topPEl.value || 0.97),
    max_new_tokens: Number(maxTokensEl.value || 520),
  };

  try {
    const data = await callApi("/api/generate", payload);
    out.textContent = data.sermon;
  } catch (err) {
    out.textContent = `Ошибка: ${err.message}`;
  }
}

async function runHealth() {
  const out = byId("healthOut");
  if (!out) {
    return;
  }
  out.textContent = "Проверка...";
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    out.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    out.textContent = `Ошибка: ${err.message}`;
  }
}

function bindButtons() {
  const generateBtn = byId("generateBtn");
  if (generateBtn && !generateBtn.getAttribute("onclick")) {
    generateBtn.addEventListener("click", runGenerate);
  }

  const healthBtn = byId("healthBtn");
  if (healthBtn && !healthBtn.getAttribute("onclick")) {
    healthBtn.addEventListener("click", runHealth);
  }
}

document.addEventListener("DOMContentLoaded", bindButtons);
bindButtons();

window.runGenerate = runGenerate;
window.runHealth = runHealth;
