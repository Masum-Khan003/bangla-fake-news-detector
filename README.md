# 🔍 Bangla Fake News Detector

> AI-powered misinformation detection for Bangla and English news articles · End-to-end ML system · Chrome Extension · Live API

![Macro-F1](https://img.shields.io/badge/Macro--F1-0.9361-1a4080?style=flat-square)
![Fake Recall](https://img.shields.io/badge/Fake%20Recall-0.8462-b83030?style=flat-square)
![API](https://img.shields.io/badge/API-Live-1a6b40?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-gray?style=flat-square)

---

## Demo

**[🎯 Try the Live Dashboard](https://huggingface.co/spaces/maksays-003/bangla-fake-news-detector)**
· **[🌐 API Docs](https://redis-production-b1ef.up.railway.app/docs)**
· **[🤗 Model on HuggingFace](https://huggingface.co/maksays-003/bangla-fake-news-xlmr)**

---

## The Problem

Bangladesh has 2,000+ unregulated online news portals. During elections, health
crises, and political upheavals, fabricated Bangla-language news spreads virally
on Facebook and WhatsApp causing documented real-world harm. Every existing
automated detection tool is **English-only**.

This project builds the first open, deployable fake news detector for the
Bangla-language internet.

---

## Model Performance

| Metric | Score |
|---|---|
| **Macro F1** | **0.9361** |
| Fake Precision | 0.9066 |
| Fake Recall | 0.8462 |
| Fake F1 | 0.8753 |
| Credible F1 | 0.9969 |

Evaluated on a stratified temporal test set (7,784 samples · 195 fake · 7,589 credible).
Training used class-weighted loss (Fake weight: 20.04) to handle the 39:1 imbalance.

---

## Quick Start

**Level 1 — Try it now (no setup):**
```bash
curl -X POST https://redis-production-b1ef.up.railway.app/v1/predict \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "paste your Bangla or English article text here..."}'
```

**Level 2 — Python:**
```python
import requests

response = requests.post(
    "https://redis-production-b1ef.up.railway.app/v1/predict",
    headers={"X-API-Key": "YOUR_API_KEY"},
    json={"text": "your article text here"}
)
result = response.json()
print(f"{result['label']} — {result['confidence']:.1%} confidence")
```

**Level 3 — Run locally:**
```bash
git clone https://github.com/Masum-Khan003/bangla-fake-news-detector
cd bangla-fake-news-detector
python3.10 -m venv venv && source venv/bin/activate
pip install -r requirements-prod.txt
cp .env.example .env  # fill in your keys
uvicorn api.main:app --port 8000
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/v1/predict` | POST | Classify a single text |
| `/v1/predict-url` | POST | Scrape URL and classify |
| `/v1/predict/batch` | POST | Classify up to 32 texts |
| `/v1/explain` | POST | LIME token attribution |
| `/v1/feedback` | POST | Submit correction |
| `/health/ready` | GET | Health check |

All prediction endpoints require `X-API-Key` header.

---

## Chrome Extension

Auto-detects and analyzes articles on **13 Bangladeshi news portals**:

Prothom Alo · The Daily Star · bdnews24 · TBS News · Dhaka Tribune ·
Samakal · Kaler Kantho · Jugantor · Ittefaq · Rising BD ·
Dhaka Post · Mzamin · Daily Amar Desh

---

## System Architecture

Chrome Extension (13 BD news portals)
↓
FastAPI Backend (Railway)
/v1/predict · /v1/predict-url · /v1/predict/batch
↓
XLM-RoBERTa-base (278M parameters)
Fine-tuned on BanFakeNews-2.0 (36K samples)
↓
Temperature Scaling (T=0.5308) → Calibrated probabilities
Decision Threshold (0.05) → Fake / Credible verdict
↓
Redis Cache (1hr TTL) + Feedback DB (JSONL)

---

## Training Details

Dataset     : BanFakeNews-2.0 (60K total · 36K train after dedup)
Split       : Stratified 70/15/15 (temporal-aware)
Imbalance   : 39:1 authentic-to-fake → class weights [20.04, 0.51]
GPU         : Kaggle T4 (~2.7 hours)
Optimizer   : AdamW · LR=2e-5 · warmup=10%
Early stop  : patience=2 on val Macro-F1
Calibration : Temperature scaling (T=0.5308) + threshold tuning
W&B run     : g9qn9beq

---

## Repository Structure

bangla-fake-news-detector/
├── src/
│   ├── data/          # Preprocessing, scraping, temporal split
│   ├── model/         # Training, inference, calibration, chunking
│   └── utils/         # Label mapping, text utilities
├── api/               # FastAPI backend (all endpoints)
├── extension/         # Chrome Extension (MV3)
├── dashboard/         # Streamlit demo (HuggingFace Spaces)
├── notebooks/         # EDA, training, calibration notebooks
├── calibration/       # Temperature + threshold configs
├── Dockerfile         # Multi-stage, CPU-only torch (0.39GB)
└── data/splits/       # Train/val/test split metadata

---

## Known Limitations

- **Binary only** - Fake vs Credible (no Unverified/Satire class yet)
- **Short texts** - articles under 200 characters are unreliable
- **Temporal drift** - trained on data up to 2024; misinformation patterns evolve
- **False negatives** - ~15% of fake articles missed at threshold=0.05
- **Banglish** - code-switching handled but with lower confidence
- **BanFakeNews license** - CC BY-NC-SA 4.0 (non-commercial use only)

---

## Future Work

- [ ] Third class: Unverified/Satire with more labeled data
- [ ] ONNX quantization → reduce inference from 1.4s to ~200ms
- [ ] BanglaBERT ensemble for pure Bangla articles
- [ ] Model drift monitoring dashboard (Grafana/Prometheus)
- [ ] Mobile companion app (Android/iOS)
- [ ] Support for more South Asian languages (Hindi, Urdu)

---

## License & Citation

Code: MIT License
Dataset: BanFakeNews-2.0 · CC BY-NC-SA 4.0 (non-commercial)

```bibtex
@software{bangla_fake_news_2026,
  author  = {Md. Masum Khan},
  title   = {Bangla Fake News Detector},
  year    = {2026},
  url     = {https://github.com/Masum-Khan003/bangla-fake-news-detector}
}
```
