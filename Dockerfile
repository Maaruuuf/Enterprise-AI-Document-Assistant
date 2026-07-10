FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only torch FIRST and explicitly — the default PyPI torch wheel
# bundles full CUDA/GPU support (~700MB+), which is unnecessary here and
# was causing out-of-memory crashes on Render's 512MB free tier. This
# CPU-only build is a fraction of the size and satisfies sentence-transformers'
# torch requirement without pip pulling in the GPU version.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY scripts/ ./scripts/

ENV PORT=8080
EXPOSE 8080

CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT