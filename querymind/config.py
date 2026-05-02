"""
QueryMind configuration — Shared constants and settings.

Environment variables:
    QUERYMIND_DB_URL:       PostgreSQL connection string
    QUERYMIND_MODEL_PATH:   Path to trained PPO checkpoint
    QUERYMIND_LOG_LEVEL:    Logging level (default: INFO)
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Global configuration loaded from environment variables."""

    db_url: str = os.getenv(
        "QUERYMIND_DB_URL",
        "postgresql://querymind:querymind@localhost:5434/tpch",
    )
    model_path: str = os.getenv(
        "QUERYMIND_MODEL_PATH",
        "checkpoints/querymind_final",
    )
    log_level: str = os.getenv("QUERYMIND_LOG_LEVEL", "INFO")

    # Training defaults
    total_timesteps: int = 500_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    seed: int = 42

    # Environment defaults
    obs_dim: int = 128
    query_timeout_ms: int = 30_000
    max_tables: int = 8

    # Benchmark defaults
    n_benchmark_runs: int = 5
    warmup_runs: int = 2


# Singleton
config = Config()
