"""
QueryMind FastAPI — Serves the trained PPO agent as an HTTP API.

Endpoints:
    POST /optimize  — Given a SQL query, returns the optimal planner config
    GET  /health    — Health check
    GET  /stats     — Agent and database statistics

The API loads a trained PPO model and runs inference only (no DB execution
at serve time). Returns the recommended planner hint string that can be
prepended to any query or applied via SET statements.

Usage:
    uvicorn querymind.api.main:app --host 0.0.0.0 --port 8000
    # or
    querymind-serve
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from querymind.env.hint_builder import HintBuilder
from querymind.featurizer.encoder import QueryFeatureEncoder
from querymind.featurizer.query_parser import QueryParser

logger = logging.getLogger(__name__)

# ── Global state (loaded at startup) ───────────────────────────────────────
_model: Any = None
_encoder: QueryFeatureEncoder | None = None
_hint_builder: HintBuilder = HintBuilder()
_parser: QueryParser = QueryParser()


# ── Request/Response schemas ────────────────────────────────────────────────

class OptimizeRequest(BaseModel):
    """Request to optimize a SQL query."""

    sql: str = Field(..., description="SQL query to optimize", min_length=10)
    return_explanation: bool = Field(
        default=False,
        description="Include explanation of the chosen configuration",
    )


class OptimizeResponse(BaseModel):
    """Response with optimal planner configuration."""

    hint: str = Field(..., description="pg_hint_plan compatible hint string")
    set_statements: list[str] = Field(
        ..., description="PostgreSQL SET statements to apply the config"
    )
    optimized_sql: str = Field(
        ..., description="SQL query with hint prepended (if applicable)"
    )
    predicted_speedup: float = Field(
        ..., description="Predicted speedup ratio vs default planner"
    )
    action: int = Field(..., description="Internal action index chosen by agent")
    config: dict[str, bool] = Field(
        ..., description="Planner knob configuration"
    )
    query_info: dict[str, Any] = Field(
        default_factory=dict,
        description="Parsed query metadata",
    )
    inference_time_ms: float = Field(
        ..., description="Agent inference time in milliseconds"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    version: str


# ── Lifespan (startup/shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    """Load model and encoder at startup."""
    global _model, _encoder

    model_path = os.getenv("QUERYMIND_MODEL_PATH", "checkpoints/querymind_final")
    db_url = os.getenv(
        "QUERYMIND_DB_URL",
        "postgresql://querymind:querymind@localhost:5434/tpch",
    )

    # Load model
    try:
        from stable_baselines3 import PPO

        _model = PPO.load(model_path)
        logger.info(f"Model loaded from {model_path}")
    except Exception as e:
        logger.warning(f"Failed to load model: {e} — API will run without agent")

    # Initialize encoder
    try:
        _encoder = QueryFeatureEncoder(db_url=db_url)
        logger.info("Feature encoder initialized")
    except Exception as e:
        logger.warning(f"Failed to init encoder: {e}")

    yield  # App running

    # Cleanup
    logger.info("Shutting down QueryMind API")


# ── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="QueryMind API",
    description=(
        "RL-powered SQL query optimizer. Predicts optimal PostgreSQL "
        "planner configurations to minimize query latency."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.post("/optimize", response_model=OptimizeResponse)
async def optimize_query(request: OptimizeRequest) -> OptimizeResponse:
    """Optimize a SQL query by selecting the best planner configuration.

    The agent analyzes the query structure and predicts which PostgreSQL
    planner knobs should be enabled/disabled for optimal execution.
    No database execution occurs at serve time — only inference.
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Set QUERYMIND_MODEL_PATH env var.",
        )
    if _encoder is None:
        raise HTTPException(
            status_code=503,
            detail="Encoder not initialized. Check database connection.",
        )

    # Encode query features
    start = time.perf_counter()
    obs = _encoder.encode(request.sql)

    # Agent inference
    action, _ = _model.predict(obs, deterministic=True)
    action_int = int(action)
    inference_ms = (time.perf_counter() - start) * 1000.0

    # Decode action to config
    config = _hint_builder.decode_action(action_int)
    set_stmts = config.to_set_statements()

    # Build hint comment
    knob_str = ", ".join(
        f"{k}={'ON' if v else 'OFF'}"
        for k, v in config.to_dict().items()
    )
    hint = f"/* QueryMind: {knob_str} */"

    # Parse query for metadata
    parsed = _parser.parse(request.sql)
    query_info: dict[str, Any] = {}
    if request.return_explanation:
        query_info = {
            "num_tables": parsed.num_tables,
            "num_joins": parsed.num_joins,
            "tables": parsed.tables,
            "has_aggregation": parsed.has_aggregation,
            "has_subquery": parsed.has_subquery,
        }

    return OptimizeResponse(
        hint=hint,
        set_statements=set_stmts,
        optimized_sql=f"{hint}\n{request.sql}",
        predicted_speedup=1.0,  # TODO: predict from critic value
        action=action_int,
        config=config.to_dict(),
        query_info=query_info,
        inference_time_ms=inference_ms,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=_model is not None,
        version="0.1.0",
    )


@app.get("/stats")
async def get_stats() -> dict[str, Any]:
    """Return agent and system statistics."""
    stats: dict[str, Any] = {
        "model_loaded": _model is not None,
        "encoder_ready": _encoder is not None,
        "action_space_size": 64,
        "valid_actions": _hint_builder.num_valid_actions,
    }
    if _model is not None:
        stats["policy_class"] = type(_model.policy).__name__
    return stats


# ── CLI Entry Point ─────────────────────────────────────────────────────────

def start() -> None:
    """Start the QueryMind API server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    uvicorn.run(
        "querymind.api.main:app",
        host=os.getenv("QUERYMIND_HOST", "0.0.0.0"),
        port=int(os.getenv("QUERYMIND_PORT", "8000")),
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    start()
