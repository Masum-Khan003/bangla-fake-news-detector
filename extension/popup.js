const API_BASE = "https://redis-production-b1ef.up.railway.app";
const API_KEY  = "eadc0835de4dee07947c3e697441b63b0ba9ca21e265c4b3be2f4086e746f467"; // ← same key as background.js

const COLORS = {
  Fake:       { text: "#b83030", bar: "#b83030", emoji: "🔴" },
  Credible:   { text: "#1a6b40", bar: "#1a6b40", emoji: "🟢" },
  Unverified: { text: "#a05c0a", bar: "#e0a030", emoji: "🟡" },
};

// DOM refs
const resultSection   = document.getElementById("result-section");
const manualSection   = document.getElementById("manual-section");
const verdictEmoji    = document.getElementById("verdict-emoji");
const verdictLabel    = document.getElementById("verdict-label");
const verdictConf     = document.getElementById("verdict-conf");
const confBarFill     = document.getElementById("confidence-bar-fill");
const metaInfo        = document.getElementById("meta-info");
const feedbackRow     = document.getElementById("feedback-row");
const fbThanks        = document.getElementById("fb-thanks");
const fbYes           = document.getElementById("fb-yes");
const fbNo            = document.getElementById("fb-no");
const manualText      = document.getElementById("manual-text");
const analyzeBtn      = document.getElementById("analyze-btn");
const loadingEl       = document.getElementById("loading");
const errorEl         = document.getElementById("error-msg");
const statusDot       = document.getElementById("status-dot");

function showResult(result) {
  const c   = COLORS[result.label] || COLORS.Unverified;
  const pct = Math.round(result.confidence * 100);

  verdictEmoji.textContent  = c.emoji;
  verdictLabel.textContent  = result.label;
  verdictLabel.style.color  = c.text;
  verdictConf.textContent   = `${pct}% confidence`;
  confBarFill.style.width   = `${pct}%`;
  confBarFill.style.background = c.bar;

  metaInfo.textContent =
    `Language: ${result.language_detected} · ` +
    `${Math.round(result.inference_ms)}ms` +
    (result.cached ? " · cached" : "") +
    (result.was_chunked ? ` · ${result.n_chunks} chunks` : "");

  resultSection.classList.remove("hidden");

  // Feedback
  let currentRequestId = result.request_id;
  fbYes.onclick = () => sendFeedback(currentRequestId, "Credible");
  fbNo.onclick  = () => sendFeedback(currentRequestId, "Fake");
}

function sendFeedback(requestId, label) {
  chrome.runtime.sendMessage({ type: "FEEDBACK", requestId, label });
  feedbackRow.classList.add("hidden");
  fbThanks.classList.remove("hidden");
}

async function runPredict(text) {
  loadingEl.classList.remove("hidden");
  errorEl.classList.add("hidden");
  analyzeBtn.disabled = true;

  try {
    const response = await fetch(`${API_BASE}/v1/predict`, {
      method:  "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key":    API_KEY,
      },
      body: JSON.stringify({ text: text.slice(0, 10000) }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `API error ${response.status}`);
    }

    const result = await response.json();
    showResult(result);
    manualSection.querySelector(".hint").textContent = "Analyze another article:";
  } catch (err) {
    errorEl.textContent = `Error: ${err.message}`;
    errorEl.classList.remove("hidden");
  } finally {
    loadingEl.classList.add("hidden");
    analyzeBtn.disabled = false;
  }
}

// Check API health and update status dot
async function checkApiStatus() {
  try {
    const r = await fetch(`${API_BASE}/health/ready`, { method: "GET" });
    statusDot.className = r.ok ? "online" : "offline";
    statusDot.title     = r.ok ? "API Online" : "API Offline";
  } catch {
    statusDot.className = "offline";
    statusDot.title     = "API Offline";
  }
}

// Analyze button
analyzeBtn.addEventListener("click", () => {
  const text = manualText.value.trim();
  if (text.length < 200) {
    errorEl.textContent = "Please paste a longer article (min 200 characters)";
    errorEl.classList.remove("hidden");
    return;
  }
  errorEl.classList.add("hidden");
  runPredict(text);
});

// Check if current tab already has a result from content script
chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
  const tab = tabs[0];
  if (!tab?.url) return;

  const key  = `result_tab_${tab.url}`;
  const data = await chrome.storage.local.get(key);
  if (data[key]) {
    showResult(data[key]);
  }
});

// Check API status on open
checkApiStatus();
