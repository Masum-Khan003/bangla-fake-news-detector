# ── Stage 1: Builder ──────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc curl git \
    && rm -rf /var/lib/apt/lists/*

# Copy ONLY the production requirements
COPY requirements-cpu.txt requirements-prod.txt ./

# Step 1: Install CPU-only PyTorch first
# This prevents pip from ever touching the CUDA wheels
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements-cpu.txt

# Step 2: Install production dependencies
# torch is already installed above — pip will skip it here
RUN pip install --no-cache-dir -r requirements-prod.txt

# Verify torch is CPU-only — fail build if CUDA crept in
RUN python -c "import torch; assert '+cpu' in torch.__version__ or not torch.cuda.is_available(), f'CUDA torch detected: {torch.__version__} — build aborted'; print(f'✓ torch {torch.__version__} — CPU only confirmed')"

# ── Stage 2: Runtime ──────────────────────────────────────
FROM python:3.10-slim AS runtime

WORKDIR /app

# Only runtime system libs — no build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy ONLY installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages \
                    /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/         ./src/
COPY api/         ./api/
COPY calibration/ ./calibration/
COPY scripts/     ./scripts/

# Runtime directories
RUN mkdir -p data/external data/feedback models

# Non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health/ready || exit 1

EXPOSE 8000

CMD ["sh", "-c", \
     "python scripts/download_model.py && \
      uvicorn api.main:app \
      --host 0.0.0.0 \
      --port ${PORT:-8000} \
      --workers 1 \
      --log-level info"]