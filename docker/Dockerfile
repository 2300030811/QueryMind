# ═══════════════════════════════════════════════════════════
# QueryMind Agent — Dockerfile
# Python 3.11 + RL dependencies
# ═══════════════════════════════════════════════════════════
FROM python:3.11-slim-bookworm

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[dev]"

# Copy application code
COPY querymind/ ./querymind/
COPY pyproject.toml ./

# Install the package
RUN pip install --no-cache-dir -e .

# Default: run training
CMD ["python", "-m", "querymind.agent.train"]
