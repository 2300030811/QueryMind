"""
QueryFeatureEncoder — Converts parsed SQL queries into fixed-size numpy vectors.

The observation vector is structured as:
    [query_meta (8) | table_stats (MAX_TABLES * 5) | join_graph (MAX_TABLES^2) | plan_cost (1)]

All features are normalized using running statistics (online StandardScaler).
Padded to a fixed OBS_DIM for consistent Gymnasium observation space.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from querymind.featurizer.query_parser import QueryParser, ParsedQuery
from querymind.featurizer.stats_extractor import StatsExtractor, TableStats

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────
MAX_TABLES = 8
FEATURES_PER_TABLE = 5  # row_count, page_count, avg_width, avg_n_distinct, avg_correlation
QUERY_META_DIM = 8      # num_tables, num_joins, has_agg, has_subquery, has_order, has_group, num_predicates, padding
JOIN_GRAPH_DIM = MAX_TABLES * MAX_TABLES  # flattened adjacency matrix
PLAN_COST_DIM = 1       # default plan cost from EXPLAIN


class RunningNormalizer:
    """Online mean/variance normalization (Welford's algorithm).

    Maintains running statistics for feature normalization without
    requiring a pre-fitted scaler. Updates incrementally as new
    observations arrive during training.
    """

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        self._dim = dim
        self._eps = eps
        self._count = 0
        self._mean = np.zeros(dim, dtype=np.float64)
        self._var = np.ones(dim, dtype=np.float64)
        self._m2 = np.zeros(dim, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        """Update running statistics with a new observation."""
        self._count += 1
        delta = x - self._mean
        self._mean += delta / self._count
        delta2 = x - self._mean
        self._m2 += delta * delta2
        if self._count > 1:
            self._var = self._m2 / (self._count - 1)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize an observation using running statistics."""
        std = np.sqrt(self._var + self._eps)
        return ((x - self._mean) / std).astype(np.float32)


class QueryFeatureEncoder:
    """Encodes SQL queries into fixed-size numpy feature vectors.

    Combines three information sources:
    1. Query structure (from sqlglot AST parsing)
    2. Table statistics (from PostgreSQL system catalogs)
    3. Join graph topology (adjacency matrix)

    The resulting vector is normalized and padded to obs_dim.

    Args:
        db_url: PostgreSQL connection string for stats extraction.
        obs_dim: Fixed output dimension (zero-padded if shorter).
    """

    def __init__(self, db_url: str, obs_dim: int = 128) -> None:
        self._parser = QueryParser()
        self._stats_extractor = StatsExtractor(db_url)
        self._obs_dim = obs_dim

        raw_dim = QUERY_META_DIM + (MAX_TABLES * FEATURES_PER_TABLE) + JOIN_GRAPH_DIM + PLAN_COST_DIM
        self._normalizer = RunningNormalizer(dim=raw_dim)

    def encode(self, sql: str, plan_cost: float = 0.0) -> np.ndarray:
        """Encode a SQL query into a fixed-size normalized feature vector.

        Args:
            sql: SQL query string.
            plan_cost: Optional estimated cost from EXPLAIN (default 0.0).

        Returns:
            Numpy array of shape (obs_dim,) with float32 values.
        """
        # Parse query structure
        parsed = self._parser.parse(sql)

        # Extract table statistics
        table_stats = self._stats_extractor.get_multi_table_stats(parsed.tables)

        # Build raw feature vector
        raw = self._build_raw_features(parsed, table_stats, plan_cost)

        # Update running normalizer and normalize
        self._normalizer.update(raw)
        normalized = self._normalizer.normalize(raw)

        # Pad or truncate to obs_dim
        return self._pad_to_dim(normalized)

    def encode_batch(self, queries: list[str]) -> np.ndarray:
        """Encode a batch of queries into a feature matrix.

        Args:
            queries: List of SQL strings.

        Returns:
            Numpy array of shape (len(queries), obs_dim).
        """
        return np.stack([self.encode(q) for q in queries])

    def _build_raw_features(
        self,
        parsed: ParsedQuery,
        table_stats: dict[str, TableStats],
        plan_cost: float,
    ) -> np.ndarray:
        """Build the raw (unnormalized) feature vector."""
        features: list[float] = []

        # ── Query metadata (8 dims) ────────────────────────────────────────
        features.extend([
            float(parsed.num_tables),
            float(parsed.num_joins),
            float(parsed.has_aggregation),
            float(parsed.has_subquery),
            float(parsed.has_order_by),
            float(parsed.has_group_by),
            float(len(parsed.predicates)),
            0.0,  # padding
        ])

        # ── Per-table statistics (MAX_TABLES * 5 dims) ─────────────────────
        for i in range(MAX_TABLES):
            if i < len(parsed.tables):
                table_name = parsed.tables[i]
                stats = table_stats.get(table_name)
                if stats:
                    features.extend([
                        np.log1p(stats.row_count),       # log-scaled row count
                        np.log1p(stats.page_count),      # log-scaled page count
                        stats.avg_row_width,
                        stats.avg_n_distinct,
                        stats.avg_correlation,
                    ])
                else:
                    features.extend([0.0] * FEATURES_PER_TABLE)
            else:
                features.extend([0.0] * FEATURES_PER_TABLE)

        # ── Join graph (MAX_TABLES^2 dims) ─────────────────────────────────
        adj = self._parser.build_join_graph(parsed)
        for i in range(MAX_TABLES):
            for j in range(MAX_TABLES):
                if i < len(adj) and j < len(adj):
                    features.append(adj[i][j])
                else:
                    features.append(0.0)

        # ── Plan cost (1 dim) ──────────────────────────────────────────────
        features.append(np.log1p(plan_cost))

        return np.array(features, dtype=np.float64)

    def _pad_to_dim(self, vec: np.ndarray) -> np.ndarray:
        """Pad or truncate a vector to the target observation dimension."""
        if len(vec) >= self._obs_dim:
            return vec[: self._obs_dim].astype(np.float32)
        padded = np.zeros(self._obs_dim, dtype=np.float32)
        padded[: len(vec)] = vec
        return padded
