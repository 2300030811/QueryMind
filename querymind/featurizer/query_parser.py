"""
QueryParser — Extracts structural features from SQL queries using sqlglot.

Replaces pglast with sqlglot for better TPC-H compatibility and maintenance.
Extracts: table names, aliases, join conditions, predicates, aggregations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

logger = logging.getLogger(__name__)


@dataclass
class ParsedQuery:
    """Structured representation of a parsed SQL query."""

    tables: list[str] = field(default_factory=list)
    aliases: dict[str, str] = field(default_factory=dict)  # alias → table
    join_conditions: list[tuple[str, str, str]] = field(default_factory=list)
    # (left_col, op, right_col) e.g. ('a.id', '=', 'b.id')
    predicates: list[str] = field(default_factory=list)
    has_aggregation: bool = False
    has_subquery: bool = False
    has_order_by: bool = False
    has_group_by: bool = False
    num_joins: int = 0

    @property
    def num_tables(self) -> int:
        """Number of tables referenced in the query."""
        return len(self.tables)


class QueryParser:
    """Parse SQL queries and extract structural features using sqlglot.

    Uses sqlglot's AST to identify tables, joins, predicates, and
    other structural elements that inform the RL agent's observations.

    Example:
        >>> parser = QueryParser()
        >>> parsed = parser.parse("SELECT * FROM orders o JOIN customer c ON o.custkey = c.custkey")
        >>> parsed.tables
        ['orders', 'customer']
        >>> parsed.num_joins
        1
    """

    def __init__(self, dialect: str = "postgres") -> None:
        self._dialect = dialect

    def parse(self, sql: str) -> ParsedQuery:
        """Parse a SQL query string into a ParsedQuery.

        Args:
            sql: SQL query string.

        Returns:
            ParsedQuery with extracted structural features.
        """
        result = ParsedQuery()

        try:
            parsed = sqlglot.parse_one(sql, dialect=self._dialect)
        except sqlglot.errors.ParseError as e:
            logger.warning(f"Failed to parse SQL: {e}")
            return result

        # ── Extract tables and aliases ──────────────────────────────────────
        for table in parsed.find_all(exp.Table):
            table_name = table.name
            if table_name and table_name not in result.tables:
                result.tables.append(table_name)

            alias_node = table.find(exp.TableAlias)
            if alias_node and alias_node.name:
                result.aliases[alias_node.name] = table_name

        # ── Extract join conditions ─────────────────────────────────────────
        for join in parsed.find_all(exp.Join):
            result.num_joins += 1
            on_clause = join.find(exp.EQ)
            if on_clause:
                left = on_clause.left
                right = on_clause.right
                result.join_conditions.append(
                    (str(left), "=", str(right))
                )

        # ── Extract WHERE predicates ────────────────────────────────────────
        where = parsed.find(exp.Where)
        if where:
            for cond in where.find_all(exp.Predicate):
                result.predicates.append(str(cond))

        # ── Detect structural features ──────────────────────────────────────
        result.has_aggregation = bool(
            parsed.find(exp.AggFunc)
            or parsed.find(exp.Count)
            or parsed.find(exp.Sum)
            or parsed.find(exp.Avg)
        )
        result.has_subquery = bool(parsed.find(exp.Subquery))
        result.has_order_by = bool(parsed.find(exp.Order))
        result.has_group_by = bool(parsed.find(exp.Group))

        return result

    def extract_table_names(self, sql: str) -> list[str]:
        """Quick extraction of table names from a query.

        Args:
            sql: SQL query string.

        Returns:
            List of table names.
        """
        return self.parse(sql).tables

    def build_join_graph(self, parsed: ParsedQuery) -> list[list[float]]:
        """Build an adjacency matrix representing the join graph.

        Args:
            parsed: A ParsedQuery with tables and join conditions.

        Returns:
            NxN adjacency matrix (list of lists) where N = num_tables.
            Entry [i][j] = 1.0 if tables i and j are joined, else 0.0.
        """
        n = parsed.num_tables
        if n == 0:
            return []

        table_idx = {t: i for i, t in enumerate(parsed.tables)}
        adj = [[0.0] * n for _ in range(n)]

        for left_col, _op, right_col in parsed.join_conditions:
            # Try to resolve which tables are involved
            left_table = self._resolve_table(left_col, parsed)
            right_table = self._resolve_table(right_col, parsed)

            if left_table and right_table and left_table != right_table:
                i = table_idx.get(left_table)
                j = table_idx.get(right_table)
                if i is not None and j is not None:
                    adj[i][j] = 1.0
                    adj[j][i] = 1.0

        return adj

    @staticmethod
    def _resolve_table(column_ref: str, parsed: ParsedQuery) -> str | None:
        """Resolve a column reference like 'a.id' to a table name."""
        parts = column_ref.split(".")
        if len(parts) >= 2:
            prefix = parts[0]
            # Check if it's an alias
            if prefix in parsed.aliases:
                return parsed.aliases[prefix]
            # Check if it's a direct table name
            if prefix in parsed.tables:
                return prefix
        return None
