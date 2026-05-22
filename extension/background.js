/**
 * Background service worker — MV3
 *
 * CRITICAL MV3 CONSTRAINT: This service worker is killed after ~5 minutes
 * of inactivity. NEVER store state in memory variables here.
 * ALL state goes through chrome.storage.local.
 */

const API_BASE = "https://redis-production-b1ef.up.railway.app";
// API key stored in chrome.storage.local, set on first install
// For portfolio: hardcode a read-only public key
const API_KEY  = "eadc0835de4dee07947c3e697441b63b0ba9ca21e265c4b3be2f4086e746f467"; // ← replace with actual key

/**
 * Listen for messages from content scripts and popup.
 * This is the only way to make API calls from content scripts
 * (content scripts can't access cross-origin APIs directly in MV3).
 */
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "PREDICT") {
    handlePredict(message.text, message.url)
      .then(result => sendResponse({ success: true, data: result }))
      .catch(err  => sendResponse({ success: false, error: err.message }));
    return true; // Keep message channel open for async response
  }

  if (message.type === "FEEDBACK") {
    handleFeedback(message.requestId, message.label)
      .then(() => sendResponse({ success: true }))
      .catch(err => sendResponse({ success: false, error: err.message }));
    return true;
  }

  if (message.type === "GET_RESULT") {
    // Content script asking for cached result for current tab
    const tabId = sender.tab?.id;
    if (!tabId) { sendResponse(null); return; }
    chrome.storage.local.get(`result_${tabId}`, (data) => {
      sendResponse(data[`result_${tabId}`] || null);
    });
    return true;
  }
});

async function handlePredict(text, url) {
  // Check storage cache first (tab-level, not Redis)
  const cacheKey = `url_${btoa(url).slice(0, 40)}`;
  const cached   = await chrome.storage.local.get(cacheKey);
  if (cached[cacheKey]) {
    return { ...cached[cacheKey], cached: true };
  }

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

  // Cache result for 1 hour
  const toStore = { ...result, cached: false, timestamp: Date.now() };
  await chrome.storage.local.set({
    [cacheKey]: toStore,
    [`result_${Date.now()}`]: toStore, // backup by time
  });

  return toStore;
}

async function handleFeedback(requestId, correctLabel) {
  await fetch(`${API_BASE}/v1/feedback`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({
      request_id:    requestId,
      correct_label: correctLabel,
    }),
  });
}

// Clean up old cached results (>1 hour) on startup
chrome.runtime.onStartup.addListener(async () => {
  const all  = await chrome.storage.local.get(null);
  const now  = Date.now();
  const keys = Object.keys(all).filter(k =>
    k.startsWith("url_") && all[k].timestamp &&
    now - all[k].timestamp > 3600000
  );
  if (keys.length > 0) {
    await chrome.storage.local.remove(keys);
  }
});
