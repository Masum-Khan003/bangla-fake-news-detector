/**
 * Content script — runs on BD news sites.
 * Uses MutationObserver to handle JS-rendered content.
 */

const SITE_SELECTORS = {
  "prothomalo.com":   ".story-element-text p, .storyCard p",
  "thedailystar.net": ".text-formatted p",
  "bdnews24.com":     ".details-brief p, .dNewsDesc p",
  "tbsnews.net":      ".section-content p",
  "dhakatribune.com": ".article-body p",
  "samakal.com":      ".dNewsDesc p",
  "kalerkantho.com":  ".details-module__Vu_sVW__detailsBody p, .news-details-content p",
  "jugantor.com":     ".news-details p",
  "ittefaq.com.bd":   ".news-details-content p",
  "risingbd.com":     ".full-details p",
};

const BADGE_ID       = "bfnd-verdict-badge";
const MIN_TEXT_LEN   = 200;
const MAX_RETRIES    = 5;
const RETRY_DELAY_MS = 1500;

let analysisStarted = false;

function getDomain() {
  return window.location.hostname.replace("www.", "");
}

function extractArticleText() {
  const domain   = getDomain();
  const selector = SITE_SELECTORS[domain] ||
    "[class*='text-formatted'] p, [class*='dNews'] p, " +
    "[class*='article'] p, [class*='details'] p, " +
    "[class*='story'] p, article p, main p";

  const paragraphs = Array.from(document.querySelectorAll(selector))
    .map(p => p.innerText.trim())
    .filter(t => t.length > 50);

  const headline = document.querySelector(
    "h1, .article-title, .story-title, .news-title, " +
    "[class*='headline'], [class*='title']"
  )?.innerText?.trim() || "";

  const body = paragraphs.join(" ");
  if (!body || body.length < MIN_TEXT_LEN) return null;

  return `${headline} ${body}`.slice(0, 10_000);
}

function isArticlePage() {
  const url  = window.location.href;
  const path = window.location.pathname;

  // Skip home pages, category pages, search pages
  const skipPatterns = [
    /^\/$/,
    /^\/category\//,
    /^\/tag\//,
    /^\/search/,
    /^\/author\//,
    /\?s=/,
  ];
  if (skipPatterns.some(p => p.test(path))) return false;

  // Must have meaningful path depth (not just homepage)
  if (path.split("/").filter(Boolean).length < 1) return false;

  return true;
}

function createBadge() {
  if (document.getElementById(BADGE_ID)) return;

  const badge = document.createElement("div");
  badge.id = BADGE_ID;
  badge.innerHTML = `
    <div id="bfnd-inner">
      <div id="bfnd-header">
        <span id="bfnd-logo">🔍</span>
        <span id="bfnd-title">Fake News Detector</span>
        <button id="bfnd-close" title="Close">×</button>
      </div>
      <div id="bfnd-body">
        <div id="bfnd-status">Analyzing article...</div>
      </div>
    </div>
  `;

  badge.style.cssText = `
    position: fixed; bottom: 24px; right: 24px; z-index: 999999;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 14px; width: 300px; background: #fff;
    border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.18);
    border: 1px solid #e5e7eb; overflow: hidden;
  `;

  document.body.appendChild(badge);

  document.getElementById("bfnd-header").style.cssText = `
    display:flex; align-items:center; gap:8px;
    padding:10px 14px; background:#0d0f14; color:#fff;
  `;
  document.getElementById("bfnd-title").style.cssText = `
    flex:1; font-weight:600; font-size:13px;
  `;
  document.getElementById("bfnd-close").style.cssText = `
    background:none; border:none; color:#fff; font-size:20px;
    cursor:pointer; line-height:1; opacity:0.7; padding:0;
  `;
  document.getElementById("bfnd-status").style.cssText = `
    padding:14px; color:#666; font-size:13px;
  `;

  document.getElementById("bfnd-close")
    .addEventListener("click", () => badge.remove());
}

function updateBadge(result) {
  const statusEl = document.getElementById("bfnd-status");
  if (!statusEl) return;

  const colors = {
    Fake:       { bg:"#fef2f2", border:"#b83030", text:"#b83030", emoji:"🔴" },
    Credible:   { bg:"#f0faf4", border:"#1a6b40", text:"#1a6b40", emoji:"🟢" },
    Unverified: { bg:"#fef8ec", border:"#a05c0a", text:"#a05c0a", emoji:"🟡" },
  };

  const c   = colors[result.label] || colors.Unverified;
  const pct = Math.round(result.confidence * 100);

  document.getElementById("bfnd-inner").style.background = c.bg;

  statusEl.innerHTML = `
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
      <span style="font-size:24px;">${c.emoji}</span>
      <div>
        <div style="font-weight:700;color:${c.text};font-size:15px;">${result.label}</div>
        <div style="color:#666;font-size:11px;">${pct}% confidence</div>
      </div>
    </div>
    <div style="background:#e5e7eb;border-radius:4px;height:4px;margin-bottom:10px;">
      <div style="background:${c.border};width:${pct}%;height:4px;border-radius:4px;"></div>
    </div>
    <div style="font-size:11px;color:#888;margin-bottom:10px;">
      ${result.language_detected} · ${Math.round(result.inference_ms)}ms
      ${result.was_chunked ? ` · ${result.n_chunks} chunks` : ""}
      ${result.cached ? " · cached" : ""}
    </div>
    <div style="display:flex;gap:8px;align-items:center;margin-bottom:8px;">
      <span style="font-size:11px;color:#999;">Correct?</span>
      <button data-label="Credible" class="bfnd-fb" style="
        padding:3px 10px;border-radius:4px;border:1px solid #1a6b40;
        background:#f0faf4;color:#1a6b40;cursor:pointer;font-size:11px;">
        👍 Yes
      </button>
      <button data-label="Fake" class="bfnd-fb" style="
        padding:3px 10px;border-radius:4px;border:1px solid #b83030;
        background:#fef2f2;color:#b83030;cursor:pointer;font-size:11px;">
        👎 No
      </button>
    </div>
    <div style="font-size:10px;color:#bbb;">
      AI-assisted · Not a definitive fact-check
    </div>
  `;

  document.querySelectorAll(".bfnd-fb").forEach(btn => {
    btn.addEventListener("click", () => {
      chrome.runtime.sendMessage({
        type:      "FEEDBACK",
        requestId: result.request_id,
        label:     btn.dataset.label,
      });
      btn.parentElement.innerHTML =
        `<span style="font-size:11px;color:#1a6b40;">✓ Feedback sent</span>`;
    });
  });
}

function showError(msg) {
  const statusEl = document.getElementById("bfnd-status");
  if (!statusEl) return;
  statusEl.innerHTML = `
    <div style="color:#b83030;font-size:12px;">⚠ ${msg}</div>
    <div style="font-size:11px;color:#999;margin-top:6px;">
      Use the extension popup to analyze manually.
    </div>
  `;
}

async function analyzeArticle() {
  if (analysisStarted) return;
  if (!isArticlePage()) return;

  const text = extractArticleText();
  if (!text) return;

  analysisStarted = true;
  createBadge();

  try {
    const response = await chrome.runtime.sendMessage({
      type: "PREDICT",
      text: text,
      url:  window.location.href,
    });

    if (response?.success) {
      updateBadge(response.data);
      chrome.storage.local.set({
        [`result_tab_${window.location.href}`]: response.data,
      });
    } else {
      showError(response?.error || "Analysis failed");
      analysisStarted = false;
    }
  } catch (err) {
    showError("Could not reach extension background");
    analysisStarted = false;
  }
}

// ── Retry strategy ────────────────────────────────────────
// Attempt 1: immediately (fast pages)
// Attempt 2+: with MutationObserver (JS-rendered pages)

function startWithRetry() {
  // Immediate attempt
  analyzeArticle();

  // If not started after 1.5s, watch DOM for content to appear
  setTimeout(() => {
    if (analysisStarted) return;

    let retries = 0;
    const observer = new MutationObserver(() => {
      if (analysisStarted || retries >= MAX_RETRIES) {
        observer.disconnect();
        return;
      }
      const text = extractArticleText();
      if (text) {
        observer.disconnect();
        analyzeArticle();
      }
      retries++;
    });

    observer.observe(document.body, {
      childList: true,
      subtree:   true,
    });

    // Hard stop after 10s regardless
    setTimeout(() => observer.disconnect(), 10_000);
  }, RETRY_DELAY_MS);
}

startWithRetry();
