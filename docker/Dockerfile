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
RUN pip install --no-cache-dir -e ".[dev]" 2>/dev/null || \
    pip install --no-cache-dir \
    gymnasium==0.29.1 \
    "stable-baselines3==2.3.2" \
    "sqlalchemy==2.0.36" \
    "psycopg2-binary==2.9.10" \
    "sqlglot==25.34.1" \
    "numpy>=1.26,<2.0" \
    "torch>=2.1" \
    "wandb>=0.18" \
    "fastapi==0.115.6" \
    "uvicorn[standard]==0.34.0" \
    "pydantic>=2.0" \
    "rich>=13.0" \
    "typer>=0.12" \
    "scikit-learn>=1.5"

# Copy application code
COPY querymind/ ./querymind/
COPY pyproject.toml ./

# Install the package
RUN pip install --no-cache-dir -e .

# Default: run training
CMD ["python", "-m", "querymind.agent.train"]
