# Use a Python version with mature wheel support for all our dependencies
# (pydantic-core, torch, etc.) — avoids the 3.14 build-from-source issues
# seen on Streamlit Cloud.
FROM python:3.11-slim

WORKDIR /app

# System dependencies:
# - build-essential: needed for any package that falls back to source builds
# - libgomp1: required by torch/sentence-transformers for CPU inference
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (separate layer) so Docker can cache
# this step and skip re-downloading ~1GB of ML packages on every code change.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY documents/ ./documents/
COPY scripts/ ./scripts/

# Render (and most PaaS) inject the port to bind via $PORT — default to 8080
# for local `docker run` testing.
ENV PORT=8080
EXPOSE 8080

# Shell form so $PORT expands correctly at container start
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT