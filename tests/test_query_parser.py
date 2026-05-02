"""
Tests for the QueryParser — SQL parsing and feature extraction.

Uses sqlglot to parse TPC-H queries. No database connection required.
"""

import pytest

from querymind.featurizer.query_parser import QueryParser, ParsedQuery


@pytest.fixture
def parser() -> QueryParser:
    return QueryParser()


class TestQueryParser:
    """Tests for SQL query parsing with sqlglot."""

    def test_simple_select(self, parser: QueryParser) -> None:
        sql = "SELECT * FROM orders WHERE o_orderdate > '1995-01-01'"
        parsed = parser.parse(sql)
        assert "orders" in parsed.tables
        assert parsed.num_tables == 1
        assert parsed.num_joins == 0

    def test_two_table_join(self, parser: QueryParser) -> None:
        sql = """
            SELECT o.o_orderkey, c.c_name
            FROM orders o
            JOIN customer c ON o.o_custkey = c.c_custkey
        """
        parsed = parser.parse(sql)
        assert parsed.num_tables == 2
        assert parsed.num_joins == 1
        assert "orders" in parsed.tables
        assert "customer" in parsed.tables

    def test_three_table_join(self, parser: QueryParser) -> None:
        """TPC-H Q3 style: customer, orders, lineitem."""
        sql = """
            SELECT l_orderkey, o_orderdate
            FROM customer, orders, lineitem
            WHERE c_mktsegment = 'BUILDING'
              AND c_custkey = o_custkey
              AND l_orderkey = o_orderkey
        """
        parsed = parser.parse(sql)
        assert parsed.num_tables == 3
        assert "customer" in parsed.tables
        assert "orders" in parsed.tables
        assert "lineitem" in parsed.tables

    def test_aggregation_detection(self, parser: QueryParser) -> None:
        sql = "SELECT COUNT(*), SUM(l_quantity) FROM lineitem"
        parsed = parser.parse(sql)
        assert parsed.has_aggregation is True

    def test_subquery_detection(self, parser: QueryParser) -> None:
        sql = """
            SELECT * FROM orders
            WHERE o_custkey IN (SELECT c_custkey FROM customer WHERE c_acctbal > 0)
        """
        parsed = parser.parse(sql)
        assert parsed.has_subquery is True

    def test_order_by_detection(self, parser: QueryParser) -> None:
        sql = "SELECT * FROM orders ORDER BY o_orderdate"
        parsed = parser.parse(sql)
        assert parsed.has_order_by is True

    def test_group_by_detection(self, parser: QueryParser) -> None:
        sql = "SELECT o_orderpriority, COUNT(*) FROM orders GROUP BY o_orderpriority"
        parsed = parser.parse(sql)
        assert parsed.has_group_by is True

    def test_extract_table_names(self, parser: QueryParser) -> None:
        sql = "SELECT * FROM nation n JOIN region r ON n.n_regionkey = r.r_regionkey"
        tables = parser.extract_table_names(sql)
        assert "nation" in tables
        assert "region" in tables

    def test_alias_resolution(self, parser: QueryParser) -> None:
        sql = """
            SELECT * FROM orders o
            JOIN customer c ON o.o_custkey = c.c_custkey
        """
        parsed = parser.parse(sql)
        assert "o" in parsed.aliases
        assert "c" in parsed.aliases
        assert parsed.aliases["o"] == "orders"
        assert parsed.aliases["c"] == "customer"

    def test_empty_query(self, parser: QueryParser) -> None:
        parsed = parser.parse("")
        assert parsed.num_tables == 0
        assert parsed.num_joins == 0

    def test_invalid_sql(self, parser: QueryParser) -> None:
        """Should return empty ParsedQuery on parse error, not crash."""
        parsed = parser.parse("NOT VALID SQL AT ALL ;; $$")
        assert isinstance(parsed, ParsedQuery)


class TestJoinGraph:
    """Tests for join graph adjacency matrix construction."""

    def test_no_joins(self, parser: QueryParser) -> None:
        sql = "SELECT * FROM orders"
        parsed = parser.parse(sql)
        adj = parser.build_join_graph(parsed)
        assert len(adj) == 1
        assert adj[0][0] == 0.0

    def test_two_table_graph(self, parser: QueryParser) -> None:
        sql = """
            SELECT * FROM orders o
            JOIN customer c ON o.o_custkey = c.c_custkey
        """
        parsed = parser.parse(sql)
        adj = parser.build_join_graph(parsed)
        assert len(adj) == 2
        # Should have edge between orders and customer
        # (order depends on alias resolution working)

    def test_empty_graph(self, parser: QueryParser) -> None:
        parsed = ParsedQuery()
        adj = parser.build_join_graph(parsed)
        assert adj == []


class TestTPCHQueries:
    """Smoke tests: parse actual TPC-H queries without errors."""

    def test_parse_tpch_q3(self, parser: QueryParser) -> None:
        sql = """
            SELECT l_orderkey, SUM(l_extendedprice * (1 - l_discount)) AS revenue,
                   o_orderdate, o_shippriority
            FROM customer, orders, lineitem
            WHERE c_mktsegment = 'BUILDING'
              AND c_custkey = o_custkey
              AND l_orderkey = o_orderkey
              AND o_orderdate < DATE '1995-03-15'
              AND l_shipdate > DATE '1995-03-15'
            GROUP BY l_orderkey, o_orderdate, o_shippriority
            ORDER BY revenue DESC, o_orderdate
            LIMIT 10
        """
        parsed = parser.parse(sql)
        assert parsed.num_tables == 3
        assert parsed.has_aggregation is True
        assert parsed.has_group_by is True
        assert parsed.has_order_by is True

    def test_parse_tpch_q5(self, parser: QueryParser) -> None:
        """Q5 has 6 tables — the most complex join pattern."""
        sql = """
            SELECT n_name, SUM(l_extendedprice * (1 - l_discount)) AS revenue
            FROM customer, orders, lineitem, supplier, nation, region
            WHERE c_custkey = o_custkey
              AND l_orderkey = o_orderkey
              AND l_suppkey = s_suppkey
              AND c_nationkey = s_nationkey
              AND s_nationkey = n_nationkey
              AND n_regionkey = r_regionkey
              AND r_name = 'ASIA'
            GROUP BY n_name
            ORDER BY revenue DESC
        """
        parsed = parser.parse(sql)
        assert parsed.num_tables == 6
        assert parsed.has_aggregation is True
