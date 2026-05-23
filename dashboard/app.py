"""
Streamlit demo dashboard — Bangla Fake News Detector
"""

import streamlit as st
import requests
import plotly.graph_objects as go

# ── Config ────────────────────────────────────────────────
API_BASE = "https://redis-production-b1ef.up.railway.app"
try:
    API_KEY = st.secrets["API_SECRET_KEY"]
except Exception:
    API_KEY = ""

MODEL_STATS = {
    "macro_f1":        0.9361,
    "fake_precision":  0.9066,
    "fake_recall":     0.8462,
    "fake_f1":         0.8753,
    "credible_f1":     0.9969,
}

# Full-length samples — verified to classify correctly
SAMPLE_FAKE = """নির্বাচনের দিন যানবাহন চলাচল বন্ধ থাকার পরেও যেভাবে ভোট দিতে যাবেন । আসন্ন একাদশ জাতীয় সংসদ নির্বাচন ঘিরে সব ধরনের যান চলাচলে বিধিনিষেধ জারি করেছে সড়ক পরিবহন ও সেতু মন্ত্রণালয়। এ বিষয়ে রোববার জারি করা এক প্রজ্ঞাপনে জানানো হয়, ২৯ ডিসেম্বর দিনগত মধ্যরাত ১২টা থেকে ৩০ ডিসেম্বর মধ্যরাত ১২টা পর্যন্ত সড়কপথে সব ধরনের যান চলাচল বন্ধ ঘোষণা করেছে সড়ক পরিবহন মন্ত্রণালয়। এর আওতায় রয়েছে বেবি ট্যাক্সি, অটো রিকশা, ইজিবাইক, ট্যাক্সি ক্যাব, মাইক্রোবাস, জিপ, পিকআপ, কার, বাস, ট্রাক। কিন্তু নির্বাচনের দিন ভোট দিতে হলে ভোটকেন্দ্রে তো যেতেই হবে! যদি কোনো যানবাহন না চলে তাহলে কীভাবে যাবেন? ১# বেলুনে চেপে: উষ্ণ বায়ু দিয়ে ফুলানো বেলুনে চেপে শহরের সান্ধ্যকালীন দৃশ্য উপভোগ করা পাশ্চাত্যে বেশ জনপ্রিয়। বেলুনে চেপে মন্থর বেগে কেন্দ্রের দিকে ভেসে যাবেন এবং সুস্থিরভাবে ভোট দিয়ে শ'খানেক ফুট উপর থেকে কেন্দ্রের মনোরম দৃশ্য উপভোগ করতে করতে বাড়ি ফিরবেন। ২# গরুর গাড়িতে করে: আবহমান বাংলার ঐতিহ্যবাহী গরুর গাড়িতে চেপে ভোটকেন্দ্রে যাওয়া হতে পারে এক অনন্য অভিজ্ঞতা। পথে পথে দেশের মানুষের সাথে কুশল বিনিময় করতে করতে ভোট দিতে যাবেন।"""

SAMPLE_CREDIBLE = """প্রধানমন্ত্রী শেখ হাসিনা আজ জাতীয় সংসদে চলতি অর্থবছরের বার্ষিক বাজেট পেশ করেছেন। এবারের বাজেটের আকার ধরা হয়েছে সাত লাখ ৯৭ হাজার কোটি টাকা, যা গত বছরের তুলনায় ১৫ দশমিক চার শতাংশ বেশি। অর্থমন্ত্রী আ হ ম মুস্তফা কামাল বাজেট বক্তৃতায় জানান, শিক্ষা খাতে মোট বাজেটের ১২ শতাংশ এবং স্বাস্থ্য খাতে পাঁচ শতাংশ বরাদ্দ রাখা হয়েছে। দেশের অবকাঠামো উন্নয়নে বিশেষ গুরুত্ব দেওয়া হয়েছে এই বাজেটে। পদ্মা সেতু রেল সংযোগ প্রকল্প এবং মেট্রোরেল সম্প্রসারণে বরাদ্দ রাখা হয়েছে উল্লেখযোগ্য পরিমাণ অর্থ। কৃষি খাতে ভর্তুকি বাড়ানো হয়েছে এবং ক্ষুদ্র ও মাঝারি উদ্যোক্তাদের জন্য বিশেষ প্রণোদনা প্যাকেজ ঘোষণা করা হয়েছে। সামাজিক সুরক্ষা কার্যক্রমের আওতায় বয়স্ক ভাতা, বিধবা ভাতা এবং প্রতিবন্ধী ভাতা বৃদ্ধি করা হয়েছে। বাজেটে রাজস্ব আয়ের লক্ষ্যমাত্রা নির্ধারণ করা হয়েছে পাঁচ লাখ কোটি টাকা। জাতীয় রাজস্ব বোর্ড এই লক্ষ্য অর্জনে নতুন কৌশল গ্রহণ করবে বলে জানানো হয়েছে।"""

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title = "Bangla Fake News Detector",
    page_icon  = "🔍",
    layout     = "wide",
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
  /* Override primary button to blue */
  .stButton > button[kind="primary"] {
    background-color: #1a4080 !important;
    border-color: #1a4080 !important;
    color: white !important;
  }
  .stButton > button[kind="primary"]:hover {
    background-color: #15336a !important;
    border-color: #15336a !important;
  }
  .verdict-fake     { color:#b83030; font-weight:800; font-size:2rem; margin:0; }
  .verdict-credible { color:#1a6b40; font-weight:800; font-size:2rem; margin:0; }
  .meta-text { font-size:0.78rem; color:#9097b0; margin-top:0.25rem; }
  .disclaimer {
    font-size:0.72rem; color:#9097b0; margin-top:1rem;
    padding-top:0.5rem; border-top:1px solid #e5e7eb;
  }
  .section-header { font-size:0.8rem; font-weight:600;
    text-transform:uppercase; letter-spacing:0.08em;
    color:#5a5f72; margin-bottom:0.5rem; }
  div[data-testid="stMetric"] {
    background:#121212; border:1px solid rgba(0,0,0,0.08);
    border-radius:8px; padding:1rem;
  }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────
if "article_text" not in st.session_state:
    st.session_state.article_text = ""
if "result" not in st.session_state:
    st.session_state.result = None
if "error" not in st.session_state:
    st.session_state.error = None


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Bangla Fake News\nDetector")
    st.caption("XLM-RoBERTa · BanFakeNews-2.0")
    st.divider()

    st.markdown("**📊 Model Performance**")
    st.metric("Macro F1",       f"{MODEL_STATS['macro_f1']:.4f}")
    st.metric("Fake Precision", f"{MODEL_STATS['fake_precision']:.4f}")
    st.metric("Fake Recall",    f"{MODEL_STATS['fake_recall']:.4f}")
    st.metric("Fake F1",        f"{MODEL_STATS['fake_f1']:.4f}")

    st.divider()
    st.markdown("**🔗 Links**")
    st.markdown(
        "[![HuggingFace](https://img.shields.io/badge/🤗-Model-yellow)]"
        "(https://huggingface.co/maksays-003/bangla-fake-news-xlmr)"
    )
    st.markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-Repo-black)]"
        "(https://github.com/Masum-Khan003/bangla-fake-news-detector)"
    )
    st.markdown(
        "[![API](https://img.shields.io/badge/API-Docs-blue)]"
        "(https://redis-production-b1ef.up.railway.app/docs)"
    )

    st.divider()
    st.markdown("**🟢 API Status**")
    try:
        r = requests.get(f"{API_BASE}/health/ready", timeout=5)
        if r.ok:
            d = r.json()
            st.success(f"Online · {d.get('uptime_seconds', 0):.0f}s uptime")
        else:
            st.error("API Offline")
    except Exception:
        st.warning("Cannot reach API")


# ── Header ────────────────────────────────────────────────
st.markdown("# 🔍 Bangla Fake News Detector")
st.markdown(
    "AI-powered misinformation detection for **Bangla** and **English** news articles · "
    "Fine-tuned XLM-RoBERTa · Macro-F1: **0.9361**"
)
st.divider()

tab1, tab2, tab3 = st.tabs(["🔎 Analyze Article", "📊 Model Metrics", "ℹ️ About"])


# ════════════════════════════════════════════════════════
# TAB 1 — Analyze
# ════════════════════════════════════════════════════════
with tab1:

    left, right = st.columns([1.1, 0.9], gap="large")

    # ── LEFT: Input ───────────────────────────────────────
    with left:
        st.markdown('<p class="section-header">Input</p>', unsafe_allow_html=True)

        # ── Sample buttons ────────────────────────────────
        # KEY FIX: set session_state key THEN st.rerun()
        # The text_area uses key= (not value=) so session_state controls it
        sc1, sc2 = st.columns(2)
        if sc1.button("🔴 Sample Fake Article", use_container_width=True):
            st.session_state.article_text = SAMPLE_FAKE
            st.session_state.result       = None
            st.session_state.error        = None
            st.rerun()

        if sc2.button("🟢 Sample Credible Article", use_container_width=True):
            st.session_state.article_text = SAMPLE_CREDIBLE
            st.session_state.result       = None
            st.session_state.error        = None
            st.rerun()

        st.caption("↑ Click to load a verified sample, then click Analyze")
        st.markdown("")

        # ── Text area — uses key= only (no value=) ────────
        # Streamlit manages value through st.session_state["article_text"]
        st.text_area(
            label       = "Or paste article text directly:",
            key         = "article_text",   # ← session_state key, not value=
            height      = 220,
            placeholder = "Paste Bangla or English news article text here "
                          "(minimum 200 characters for reliable results)...",
        )

        char_count = len(st.session_state.article_text.strip())
        if char_count == 0:
            st.caption("0 characters")
        elif char_count < 200:
            st.warning(f"⚠ {char_count} chars — too short. Min 200 for reliable results.")
        else:
            if char_count < 500:
            st.warning(
                f"⚠️ {char_count} chars — results may be unreliable. "
                "For best accuracy, provide 500+ characters of full article text."
            )
        else:
            st.caption(f"✓ {char_count:,} characters — ready to analyze")

        # ── Analyze button ────────────────────────────────
        analyze_btn = st.button(
            "🔍 Analyze Article",
            type             = "primary",
            use_container_width = True,
            disabled         = (char_count < 200),
        )

        st.markdown("---")

        # ── URL input ─────────────────────────────────────
        st.markdown('<p class="section-header">Or analyze by URL</p>',
                    unsafe_allow_html=True)
        url_input = st.text_input(
            label       = "Article URL",
            placeholder = "https://www.thedailystar.net/...",
            label_visibility = "collapsed",
        )
        url_btn = st.button(
            "🔗 Scrape & Analyze URL",
            use_container_width = True,
            disabled = (not url_input or not url_input.startswith("http")),
        )

    # ── RIGHT: Result ─────────────────────────────────────
    with right:
        st.markdown('<p class="section-header">Result</p>', unsafe_allow_html=True)
        result_placeholder = st.empty()

        if st.session_state.result is None and st.session_state.error is None:
            # Show warning if token count suggests short input
        if res.get("token_count", 999) < 200 or res.get("warning"):
            st.warning(
                "⚠️ **Short text warning:** This result may be unreliable. "
                "The model needs full article text (500+ characters) to classify accurately. "
                "Headlines and short snippets almost always return 'Credible' regardless of content."
            )
        with result_placeholder.container():
                st.info(
                    "👈 Load a sample article or paste text, "
                    "then click **Analyze Article**."
                )

    # ── API call helper ───────────────────────────────────
    def call_api(endpoint: str, payload: dict) -> None:
        try:
            resp = requests.post(
                f"{API_BASE}{endpoint}",
                headers = {
                    "X-API-Key":    API_KEY,
                    "Content-Type": "application/json",
                },
                json    = payload,
                timeout = 30,
            )
            if resp.status_code == 200:
                st.session_state.result = resp.json()
                st.session_state.error  = None
            else:
                err = resp.json().get("detail", resp.text)
                st.session_state.result = None
                st.session_state.error  = f"API error {resp.status_code}: {err}"
        except requests.Timeout:
            st.session_state.result = None
            st.session_state.error  = (
                "Request timed out. The API may be warming up — "
                "please try again in 10 seconds."
            )
        except Exception as e:
            st.session_state.result = None
            st.session_state.error  = str(e)

    # ── Trigger analysis ──────────────────────────────────
    if analyze_btn and char_count >= 200:
        with st.spinner("Analyzing article..."):
            call_api("/v1/predict",
                     {"text": st.session_state.article_text.strip()})

    if url_btn and url_input:
        with st.spinner("Scraping and analyzing..."):
            call_api("/v1/predict-url", {"url": url_input.strip()})

    # ── Render result ─────────────────────────────────────
    if st.session_state.error:
        with result_placeholder.container():
            st.error(st.session_state.error)

    if st.session_state.result:
        res   = st.session_state.result
        label = res["label"]
        pct   = int(res["confidence"] * 100)
        emoji = "🔴" if label=="Fake" else "🟢" if label=="Credible" else "🟡"
        color = "#b83030" if label=="Fake" else \
                "#1a6b40" if label=="Credible" else "#a05c0a"
        css   = "verdict-fake" if label=="Fake" else "verdict-credible"

        with result_placeholder.container():

            # Verdict
            st.markdown(
                f'<p class="{css}">{emoji} {label}</p>',
                unsafe_allow_html=True,
            )

            # Gauge
            fig = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = pct,
                title = {"text": "Confidence", "font": {"size": 13}},
                gauge = {
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar":  {"color": color, "thickness": 0.25},
                    "bgcolor": "#f5f4f0",
                    "borderwidth": 0,
                    "steps": [{"range": [0, 100], "color": "#f0f0eb"}],
                },
                number = {"suffix": "%", "font": {"size": 32, "color": color}},
            ))
            fig.update_layout(
                height=190,
                margin=dict(t=50, b=10, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Probability bars
            st.markdown("**Calibrated class probabilities:**")
            st.progress(
                min(float(res["fake_prob"]), 1.0),
                text=f"🔴 Fake: {res['fake_prob']:.1%}",
            )
            st.progress(
                min(float(res["credible_prob"]), 1.0),
                text=f"🟢 Credible: {res['credible_prob']:.1%}",
            )

            # Metadata
            chunk_info = (
                f' · {res["n_chunks"]} chunks'
                if res["was_chunked"]
                else ""
            )

            st.markdown(
                f'<p class="meta-text">'
                f'Language: <code>{res["language_detected"]}</code> · '
                f'Model: <code>{res["model_branch"]}</code> · '
                f'{res["inference_ms"]:.0f}ms'
                f'{chunk_info}'
                f'</p>',
                unsafe_allow_html=True,
            )

            st.divider()

            # Feedback
            st.markdown("**Was this prediction correct?**")
            fb1, fb2 = st.columns(2)
            if fb1.button("👍 Yes, correct", use_container_width=True):
                requests.post(
                    f"{API_BASE}/v1/feedback",
                    json    = {"request_id":    res["request_id"],
                               "correct_label": res["label"]},
                    timeout = 5,
                )
                st.success("✓ Thank you for the feedback!")

            if fb2.button("👎 No, incorrect", use_container_width=True):
                wrong = "Credible" if label == "Fake" else "Fake"
                requests.post(
                    f"{API_BASE}/v1/feedback",
                    json    = {"request_id":    res["request_id"],
                               "correct_label": wrong},
                    timeout = 5,
                )
                st.success("✓ Thank you for the feedback!")

            st.markdown(
                '<p class="disclaimer">'
                'AI-assisted tool · Not a replacement for professional fact-checking · '
                'False positives and false negatives will occur · '
                'Do not use as the sole basis for editorial decisions.'
                '</p>',
                unsafe_allow_html=True,
            )


# ════════════════════════════════════════════════════════
# TAB 2 — Model Metrics
# ════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### Test Set Performance")
    st.caption(
        "Stratified test set · 7,784 total · 195 fake · 7,589 credible · 2024+ articles"
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, val, delta in [
        (c1, "Macro F1",        MODEL_STATS["macro_f1"],       "+8.6% vs target"),
        (c2, "Fake Precision",  MODEL_STATS["fake_precision"],  "+8.7% vs target"),
        (c3, "Fake Recall",     MODEL_STATS["fake_recall"],     "-3.4% vs target"),
        (c4, "Fake F1",         MODEL_STATS["fake_f1"],         "Binary class"),
        (c5, "Credible F1",     MODEL_STATS["credible_f1"],     "Near-perfect"),
    ]:
        col.metric(label=label, value=f"{val:.4f}", delta=delta)

    st.divider()

    # Bar chart
    fig = go.Figure()
    for cls, prec, rec, f1, color in [
        ("Fake",     MODEL_STATS["fake_precision"],
                     MODEL_STATS["fake_recall"],
                     MODEL_STATS["fake_f1"],     "#b83030"),
        ("Credible", 0.9961, 0.9978,
                     MODEL_STATS["credible_f1"], "#1a6b40"),
    ]:
        fig.add_trace(go.Bar(
            name         = cls,
            x            = ["Precision", "Recall", "F1-Score"],
            y            = [prec, rec, f1],
            marker_color = color,
            text         = [f"{v:.4f}" for v in [prec, rec, f1]],
            textposition = "outside",
        ))
    fig.update_layout(
        barmode       = "group",
        yaxis         = dict(range=[0, 1.12], title="Score", gridcolor="#e5e7eb"),
        title         = "Per-Class Precision / Recall / F1",
        height        = 380,
        plot_bgcolor  = "#faf9f5",
        paper_bgcolor = "white",
        legend        = dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Training Configuration**")
        st.table({
            "Parameter":  ["Base model", "Train samples", "Epochs",
                           "Batch size", "Learning rate", "GPU", "Time"],
            "Value":      ["xlm-roberta-base", "36,322", "5 (early stop at 5)",
                           "16", "2e-5", "Kaggle T4", "~2.7 hours"],
        })
    with c2:
        st.markdown("**Calibration Configuration**")
        st.table({
            "Parameter": ["Class weight Fake", "Class weight Credible",
                          "Imbalance ratio", "Temperature T",
                          "Threshold Fake", "W&B run"],
            "Value":     ["20.04", "0.51", "39:1",
                          "0.5308", "0.05", "g9qn9beq"],
        })

    st.info(
        "⚠ **Known Limitation:** ~15% of fake articles are missed at threshold=0.05 "
        "(false negatives). Short texts under 200 characters produce unreliable results — "
        "the model requires full article context to classify accurately."
    )


# ════════════════════════════════════════════════════════
# TAB 3 — About
# ════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
#### Why This Exists

Bangladesh has 2,000+ unregulated online news portals. During elections, health crises,
and political upheavals, fabricated Bangla news spreads virally on Facebook and WhatsApp —
causing documented real-world harm. Every existing automated detection tool is English-only.
This project builds the first open, deployable fake news detector for the Bangla-language internet.

#### System Architecture

Chrome Extension (13 BD news portals — auto-detect)
↓
FastAPI Backend (Railway)
/v1/predict · /v1/predict-url · /v1/predict/batch
↓
XLM-RoBERTa-base (278M parameters)
Fine-tuned on BanFakeNews-2.0 (36K samples)
↓
Temperature Scaling (T=0.5308) → Calibrated probabilities
Decision Threshold (0.05) → Fake / Credible

#### Training Data Splits

| Split | Total | Fake | Credible | Fake % |
|---|---|---|---|---|
| Train | 36,322 | 906 | 35,416 | 2.5% |
| Validation | 7,783 | 194 | 7,589 | 2.5% |
| Test | 7,784 | 195 | 7,589 | 2.5% |

#### Known Limitations
- **Binary only** — Fake vs Credible (no Unverified/Satire class yet)
- **Short texts** — articles under 200 characters are unreliable
- **Temporal drift** — trained on data up to 2024; misinformation patterns evolve
- **False negatives** — ~15% of fake articles missed at current threshold
- **Banglish** — code-switching handled by XLM-R but with lower confidence

#### Citation
```bibtex
@software{bangla_fake_news_2026,
  author  = {Masum Khan},
  title   = {Bangla Fake News Detector},
  year    = {2026},
  url     = {https://github.com/Masum-Khan003/bangla-fake-news-detector}
}
```
    """)

    st.divider()
    st.caption(
        "Model hosted on HuggingFace · API on Railway · "
        "Extension on Chrome Web Store · Source on GitHub"
    )