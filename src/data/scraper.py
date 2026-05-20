"""
Article scraper — waterfall strategy:
  1. trafilatura   (fast, handles most sites)
  2. newspaper4k   (fallback for trafilatura failures)
  3. Playwright    (last resort for JS-rendered pages)

Respects robots.txt. Rate limit: 1 req / 3s per domain.
Results cached to data/external/ as Parquet.
"""

import time
import logging
import hashlib
import json
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse

import httpx
import trafilatura
import pandas as pd

log = logging.getLogger(__name__)

# ── Rate limiting: last request time per domain ───────────
_domain_last_req: dict[str, float] = {}
MIN_DELAY_SEC = 3.0   # minimum seconds between requests to the same domain

# ── User agents ───────────────────────────────────────────
USER_AGENTS = [
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

# ── Site-specific CSS selectors for BD news portals ───────
SITE_SELECTORS = {
    "prothomalo.com":      ".story-element p, .story-element-text p",
    "thedailystar.net":    ".pb-article-body p, .article-body p",
    "bdnews24.com":        ".article-content p, .pf-content p",
    "tbsnews.net":         ".article-body p",
    "dhakatribune.com":    ".article-body p",
    "samakal.com":         ".content-details p",
    "kalerkantho.com":     ".news-details p",
    "ittefaq.com.bd":      ".news-details-content p",
    "jugantor.com":        ".news-details p",
}


@dataclass
class ScrapedArticle:
    url:          str
    domain:       str
    headline:     str
    content:      str
    pub_date:     str
    language:     str
    label:        int         # 1=Credible (from trusted source), 0=Fake
    scrape_method: str        # trafilatura | newspaper4k | playwright | failed
    scraped_at:   str
    text_hash:    str
    char_len:     int


# ── Robots.txt cache ──────────────────────────────────────
_robots_cache: dict[str, RobotFileParser] = {}

def _can_fetch(url: str) -> bool:
    """Check robots.txt before scraping. Cache per domain."""
    parsed = urlparse(url)
    domain = parsed.netloc
    if domain not in _robots_cache:
        rp = RobotFileParser()
        robots_url = f"{parsed.scheme}://{domain}/robots.txt"
        try:
            rp.set_url(robots_url)
            rp.read()
            _robots_cache[domain] = rp
            log.info(f"robots.txt loaded for {domain}")
        except Exception:
            log.warning(f"Could not fetch robots.txt for {domain} — allowing")
            return True
    return _robots_cache[domain].can_fetch("*", url)


def _rate_limit(domain: str) -> None:
    """Enforce minimum delay between requests to same domain."""
    now  = time.time()
    last = _domain_last_req.get(domain, 0)
    wait = MIN_DELAY_SEC - (now - last)
    if wait > 0:
        jitter = random.uniform(0, 1.0)   # avoid pattern detection
        time.sleep(wait + jitter)
    _domain_last_req[domain] = time.time()


def _get_html(url: str) -> str | None:
    """Fetch raw HTML with httpx."""
    domain = urlparse(url).netloc
    _rate_limit(domain)
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    try:
        resp = httpx.get(url, headers=headers, timeout=15, follow_redirects=True)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        log.warning(f"httpx fetch failed for {url}: {e}")
        return None


def _scrape_trafilatura(url: str, html: str | None = None) -> dict | None:
    """Primary scraper — trafilatura."""
    try:
        if html is None:
            html = trafilatura.fetch_url(url)
        if not html:
            return None
        result = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
            output_format="json",
        )
        if result:
            data = json.loads(result)
            if data.get("text") and len(data["text"]) > 100:
                return {
                    "headline": data.get("title", ""),
                    "content":  data["text"],
                    "pub_date": data.get("date", ""),
                    "language": data.get("language", ""),
                    "method":   "trafilatura",
                }
    except Exception as e:
        log.debug(f"trafilatura failed for {url}: {e}")
    return None


def _scrape_newspaper4k(url: str) -> dict | None:
    """Fallback scraper — newspaper4k."""
    try:
        from newspaper import Article
        article = Article(url)
        article.download()
        article.parse()
        if article.text and len(article.text) > 100:
            return {
                "headline": article.title or "",
                "content":  article.text,
                "pub_date": str(article.publish_date or ""),
                "language": article.meta_lang or "",
                "method":   "newspaper4k",
            }
    except Exception as e:
        log.debug(f"newspaper4k failed for {url}: {e}")
    return None


def _scrape_playwright(url: str,
                        selector: str = "article p") -> dict | None:
    """Last-resort scraper — Playwright for JS-rendered pages."""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page    = browser.new_page(
                user_agent=random.choice(USER_AGENTS)
            )
            page.goto(url, timeout=20_000, wait_until="domcontentloaded")
            page.wait_for_timeout(2000)   # let JS render

            # Try site-specific selector first
            domain = urlparse(url).netloc.replace("www.", "")
            sel    = SITE_SELECTORS.get(domain, selector)

            paragraphs = page.query_selector_all(sel)
            text = " ".join(
                p_el.inner_text() for p_el in paragraphs
                if len(p_el.inner_text()) > 50
            )
            title = page.title()
            browser.close()

            if text and len(text) > 100:
                return {
                    "headline": title,
                    "content":  text,
                    "pub_date": "",
                    "language": "",
                    "method":   "playwright",
                }
    except Exception as e:
        log.debug(f"playwright failed for {url}: {e}")
    return None


def scrape_article(url: str, label: int) -> ScrapedArticle | None:
    """
    Scrape a single article URL using the waterfall strategy.
    label: 1 = Credible source, 0 = Known fake/low-quality source
    """
    domain = urlparse(url).netloc.replace("www.", "")

    # robots.txt check
    if not _can_fetch(url):
        log.info(f"robots.txt disallows scraping: {url}")
        return None

    log.info(f"Scraping [{label}] {url}")

    # Waterfall: try each method in order
    html   = _get_html(url)
    result = (
        _scrape_trafilatura(url, html) or
        _scrape_newspaper4k(url)       or
        _scrape_playwright(url)
    )

    if not result:
        log.warning(f"All scraping methods failed: {url}")
        return None

    content   = result["content"].strip()
    text_hash = hashlib.sha256(content.lower().encode()).hexdigest()

    return ScrapedArticle(
        url           = url,
        domain        = domain,
        headline      = result["headline"],
        content       = content,
        pub_date      = result["pub_date"],
        language      = result["language"],
        label         = label,
        scrape_method = result["method"],
        scraped_at    = datetime.utcnow().isoformat(),
        text_hash     = text_hash,
        char_len      = len(content),
    )


def scrape_batch(url_label_pairs: list[tuple[str, int]],
                 cache_path: Path,
                 resume: bool = True) -> pd.DataFrame:
    """
    Scrape a batch of URLs.
    Saves progress after each article (crash-safe).

    Args:
        url_label_pairs : list of (url, label) tuples
        cache_path      : parquet file to save results
        resume          : skip already-scraped URLs if cache exists
    """
    # Load existing cache
    done_hashes: set[str] = set()
    results: list[dict]   = []

    if resume and cache_path.exists():
        existing = pd.read_parquet(cache_path)
        results  = existing.to_dict("records")
        done_hashes = set(existing["url"].tolist())
        log.info(f"Resuming — {len(done_hashes)} already scraped")

    for url, label in url_label_pairs:
        if url in done_hashes:
            continue
        article = scrape_article(url, label)
        if article:
            results.append(asdict(article))
            # Save after every article (crash-safe)
            pd.DataFrame(results).to_parquet(cache_path, index=False)
        done_hashes.add(url)

    df = pd.DataFrame(results) if results else pd.DataFrame()
    log.info(f"Scraping complete: {len(df)} articles saved to {cache_path}")
    return df


# ── Seed URLs for BD news portals ────────────────────────
# label=1 : established, reputable Bangla news outlets
# label=0 : known low-credibility portals (use with caution)
CREDIBLE_SEED_URLS = [
    # Verified: robots.txt ✓ allowed + HTTP 200 (checked 2025-05)
    "https://www.thedailystar.net/news/bangladesh",   # English BD news
    "https://samakal.com/bangladesh",                  # Bangla news
    # All others disallowed by robots.txt — do not add without re-checking
]

# NOTE: Scraping deferred — both allowed sites are credible-only.
# Adding only credible samples worsens 39:1 imbalance further.
# Resume scraping when fake news sources with permissive robots.txt
# are identified, or when Rumor Scanner data sharing is confirmed.



if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # Quick smoke test on 3 URLs
    TEST_URLS = [
        ("https://www.prothomalo.com/bangladesh", 1),
        ("https://www.thedailystar.net/bangla",   1),
        ("https://bdnews24.com/bangla",            1),
    ]

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    cache_path   = PROJECT_ROOT / "data" / "external" / "scraped_articles.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    print("Running smoke test on 3 URLs (respecting rate limits)...")
    df = scrape_batch(TEST_URLS, cache_path)

    if len(df) > 0:
        print(f"\n✓ Scraped {len(df)} articles")
        print(df[["domain","headline","char_len","scrape_method","label"]].to_string())
    else:
        print("✗ No articles scraped — check network or site availability")
