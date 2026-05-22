/**
 * Content script — runs on BD news sites.
 * Extracts article text using site-specific selectors,
 * injects verdict badge into the page.
 */

// Site-specific CSS selectors for article body text
const SITE_SELECTORS = {
  "prothomalo.com":   ".story-element-text p, .storyCard p",
  "thedailystar.net": ".text-formatted p",
  "bdnews24.com":     ".details-brief p, .dNewsDesc p",
  "tbsnews.net":      ".article-body p, .article-content p",
  "dhakatribune.com": ".article-body p",
  "samakal.com":      ".dNewsDesc p",
  "kalerkantho.com":  ".details-module__Vu_sVW__detailsBody p, .news-details-content p",
  "jugantor.com":     ".news-details p",
  "ittefaq.com.bd":   ".news-details-content p",
  "risingbd.com":     ".full-details p",
};

const BADGE_ID = "bfnd-verdict-badge";
const MIN_TEXT_LENGTH = 200; // chars — don't analyze snippets


function getDomain() {
  return window.location.hostname.replace("www.", "");
}

function extractArticleText() {
  const domain   = getDomain();
  const selector = SITE_SELECTORS[domain] || "[class*='text-formatted'] p, [class*='article'] p, article p, main p";

  const paragraphs = Array.from(document.querySelectorAll(selector))
    .map(p => p.innerText.trim())
    .filter(t => t.length > 50);

  const headline = document.querySelector(
    "h1, .article-title, .story-title, .news-title"
  )?.innerText?.trim() || "";

  const body = paragraphs.join(" ");

  if (!body || body.length < MIN_TEXT_LENGTH) return null;

  // Headline + body, capped at 10,000 chars
  return `${headline} ${body}`.slice(0, 10_000);
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
        <button id="bfnd-close">×</button>
      </div>
      <div id="bfnd-body">
        <div id="bfnd-status">Analyzing article...</div>
      </div>
    </div>
  `;

  // Styles (inline to avoid CSP issues)
  badge.style.cssText = `
    position: fixed;
    bottom: 24px;
    right: 24px;
    z-index: 999999;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 14px;
    width: 300px;
    background: #fff;
    border-radius: 10px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    border: 1px solid #e5e7eb;
    overflow: hidden;
  `;

  document.body.appendChild(badge);

  // Style sub-elements
  const inner = document.getElementById("bfnd-inner");
  const header = document.getElementById("bfnd-header");
  const closeBtn = document.getElementById("bfnd-close");

  header.style.cssText = `
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 14px;
    background: #0d0f14;
    color: #fff;
  `;

  document.getElementById("bfnd-title").style.cssText = `
    flex: 1;
    font-weight: 600;
    font-size: 13px;
  `;

  closeBtn.style.cssText = `
    background: none;
    border: none;
    color: #fff;
    font-size: 18px;
    cursor: pointer;
    line-height: 1;
    opacity: 0.7;
  `;

  document.getElementById("bfnd-status").style.cssText = `
    padding: 14px;
    color: #666;
    font-size: 13px;
  `;

  closeBtn.addEventListener("click", () => badge.remove());
}

function updateBadge(result) {
  const statusEl = document.getElementById("bfnd-status");
  if (!statusEl) return;

  const colors = {
    "Fake":       { bg: "#fef2f2", border: "#b83030", text: "#b83030", emoji: "🔴" },
    "Credible":   { bg: "#f0faf4", border: "#1a6b40", text: "#1a6b40", emoji: "🟢" },
    "Unverified": { bg: "#fef8ec", border: "#a05c0a", text: "#a05c0a", emoji: "🟡" },
  };

  const c = colors[result.label] || colors["Unverified"];
  const pct = Math.round(result.confidence * 100);

  // Update badge body background
  document.getElementById("bfnd-inner").style.background = c.bg;

  statusEl.innerHTML = `
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
      <span style="font-size:22px;">${c.emoji}</span>
      <div>
        <div style="font-weight:700;color:${c.text};font-size:15px;">${result.label}</div>
        <div style="color:#666;font-size:11px;">${pct}% confidence</div>
      </div>
    </div>
    <div style="background:#e5e7eb;border-radius:4px;height:4px;margin-bottom:10px;">
      <div style="background:${c.border};width:${pct}%;height:4px;border-radius:4px;"></div>
    </div>
    <div style="font-size:11px;color:#888;margin-bottom:10px;">
      Language: ${result.language_detected} · ${result.inference_ms}ms
      ${result.was_chunked ? ` · ${result.n_chunks} chunks` : ""}
    </div>
    <div style="display:flex;gap:8px;align-items:center;">
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
    <div style="font-size:10px;color:#bbb;margin-top:8px;">
      AI-assisted · Not a definitive fact-check
    </div>
  `;

  // Feedback buttons
  document.querySelectorAll(".bfnd-fb").forEach(btn => {
    btn.addEventListener("click", () => {
      const correctLabel = btn.dataset.label;
      chrome.runtime.sendMessage({
        type:      "FEEDBACK",
        requestId: result.request_id,
        label:     correctLabel,
      });
      btn.parentElement.innerHTML =
        `<span style="font-size:11px;color:#1a6b40;">✓ Feedback sent</span>`;
    });
  });
}

function showError(message) {
  const statusEl = document.getElementById("bfnd-status");
  if (!statusEl) return;
  statusEl.innerHTML = `
    <div style="color:#b83030;font-size:12px;">
      ⚠ ${message}
    </div>
    <div style="font-size:11px;color:#999;margin-top:6px;">
      Try opening the extension popup to analyze manually.
    </div>
  `;
}

async function analyzeArticle() {
  const text = extractArticleText();
  if (!text) return; // Not an article page — skip silently

  createBadge();

  try {
    const response = await chrome.runtime.sendMessage({
      type: "PREDICT",
      text: text,
      url:  window.location.href,
    });

    if (response?.success) {
      updateBadge(response.data);

      // Store result for popup to read
      await chrome.storage.local.set({
        [`result_tab_${window.location.href}`]: response.data,
      });
    } else {
      showError(response?.error || "Analysis failed");
    }
  } catch (err) {
    showError("Could not connect to extension");
  }
}

// Run on page load — with retry for slow-rendering sites
// First attempt immediately, retry after 2s if no text found
analyzeArticle();
setTimeout(() => {
  if (!document.getElementById("bfnd-verdict-badge")) {
    analyzeArticle();
  }
}, 2000);

