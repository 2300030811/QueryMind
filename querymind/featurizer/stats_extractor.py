"""
StatsExtractor — Queries PostgreSQL system catalogs for table statistics.

Extracts per-table and per-column statistics from:
    - pg_class: row count, page count
    - pg_stats: n_distinct, null_frac, avg_width, correlation
    - pg_statistic: raw statistics for advanced features

These statistics form part of the RL agent's observation vector,
giving it the same information PostgreSQL's planner uses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


@dataclass
class ColumnStats:
    """Statistics for a single column."""

    column_name: str
    n_distinct: float = 0.0
    null_frac: float = 0.0
    avg_width: int = 0
    correlation: float = 0.0


@dataclass
class TableStats:
    """Statistics for a single table."""

    table_name: str
    row_count: int = 0
    page_count: int = 0
    avg_row_width: float = 0.0
    columns: list[ColumnStats] = field(default_factory=list)

    @property
    def n_columns(self) -> int:
        """Number of columns with available statistics."""
        return len(self.columns)

    @property
    def avg_n_distinct(self) -> float:
        """Average number of distinct values across columns."""
        if not self.columns:
            return 0.0
        vals = [c.n_distinct for c in self.columns]
        return sum(vals) / len(vals)

    @property
    def avg_null_frac(self) -> float:
        """Average null fraction across columns."""
        if not self.columns:
            return 0.0
        vals = [c.null_frac for c in self.columns]
        return sum(vals) / len(vals)

    @property
    def avg_correlation(self) -> float:
        """Average physical/logical correlation across columns."""
        if not self.columns:
            return 0.0
        vals = [abs(c.correlation) for c in self.columns]
        return sum(vals) / len(vals)


class StatsExtractor:
    """Extracts table and column statistics from PostgreSQL system catalogs.

    These stats mirror what PostgreSQL's own cost-based optimizer uses,
    giving the RL agent equivalent information for decision-making.

    Args:
        db_url: SQLAlchemy connection string to the PostgreSQL database.
        schema: Database schema to query (default: 'public').

    Example:
        >>> extractor = StatsExtractor("postgresql://user:pass@localhost/tpch")
        >>> stats = extractor.get_table_stats("orders")
        >>> stats.row_count
        1500000
    """

    # Query to get table-level stats from pg_class
    _TABLE_STATS_SQL = text("""
        SELECT
            c.relname AS table_name,
            c.reltuples::bigint AS row_count,
            c.relpages AS page_count
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = :schema
          AND c.relname = :table_name
          AND c.relkind = 'r'
    """)

    # Query to get column-level stats from pg_stats
    _COLUMN_STATS_SQL = text("""
        SELECT
            attname AS column_name,
            n_distinct,
            null_frac,
            avg_width,
            correlation
        FROM pg_stats
        WHERE schemaname = :schema
          AND tablename = :table_name
        ORDER BY attname
    """)

    # Query to get all table names in the schema
    _LIST_TABLES_SQL = text("""
        SELECT c.relname
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = :schema
          AND c.relkind = 'r'
        ORDER BY c.relname
    """)

    def __init__(self, db_url: str, schema: str = "public") -> None:
        self._engine: Engine = create_engine(db_url, pool_size=1, max_overflow=0)
        self._schema = schema

    def get_table_stats(self, table_name: str) -> TableStats:
        """Get statistics for a single table.

        Args:
            table_name: Name of the PostgreSQL table.

        Returns:
            TableStats with row count, page count, and per-column stats.
        """
        stats = TableStats(table_name=table_name)

        try:
            with self._engine.connect() as conn:
                # Table-level stats
                result = conn.execute(
                    self._TABLE_STATS_SQL,
                    {"schema": self._schema, "table_name": table_name},
                )
                row = result.fetchone()
                if row:
                    stats.row_count = row.row_count or 0
                    stats.page_count = row.page_count or 0

                # Column-level stats
                result = conn.execute(
                    self._COLUMN_STATS_SQL,
                    {"schema": self._schema, "table_name": table_name},
                )
                for row in result:
                    stats.columns.append(
                        ColumnStats(
                            column_name=row.column_name,
                            n_distinct=float(row.n_distinct or 0),
                            null_frac=float(row.null_frac or 0),
                            avg_width=int(row.avg_width or 0),
                            correlation=float(row.correlation or 0),
                        )
                    )

                # Compute average row width from column widths
                if stats.columns:
                    stats.avg_row_width = sum(c.avg_width for c in stats.columns) / len(
                        stats.columns
                    )

        except Exception as e:
            logger.error(f"Failed to get stats for table '{table_name}': {e}")

        return stats

    def get_multi_table_stats(self, table_names: list[str]) -> dict[str, TableStats]:
        """Get statistics for multiple tables.

        Args:
            table_names: List of table names.

        Returns:
            Dictionary mapping table name → TableStats.
        """
        return {name: self.get_table_stats(name) for name in table_names}

    def list_tables(self) -> list[str]:
        """List all user tables in the schema.

        Returns:
            Sorted list of table names.
        """
        try:
            with self._engine.connect() as conn:
                result = conn.execute(self._LIST_TABLES_SQL, {"schema": self._schema})
                return [row[0] for row in result]
        except Exception as e:
            logger.error(f"Failed to list tables: {e}")
            return []

    def close(self) -> None:
        """Dispose the database engine."""
        self._engine.dispose()
