"""
Tests for the TPC-H query loader.
"""

import pytest

from querymind.benchmark.tpch_loader import (
    JOIN_HEAVY_IDS,
    TEST_QUERY_IDS,
    TRAIN_QUERY_IDS,
    TPCH_QUERIES,
    TPCHLoader,
)


@pytest.fixture
def loader() -> TPCHLoader:
    return TPCHLoader()


class TestTPCHLoader:
    """Tests for TPC-H query loading and splitting."""

    def test_all_22_queries_present(self) -> None:
        """TPC-H has 22 standard queries."""
        assert len(TPCH_QUERIES) == 22

    def test_query_ids_format(self) -> None:
        """All query IDs should be Q1-Q22."""
        for i in range(1, 23):
            assert f"Q{i}" in TPCH_QUERIES

    def test_train_test_split_sizes(self) -> None:
        assert len(TRAIN_QUERY_IDS) == 16
        assert len(TEST_QUERY_IDS) == 6

    def test_train_test_no_overlap(self) -> None:
        overlap = set(TRAIN_QUERY_IDS) & set(TEST_QUERY_IDS)
        assert len(overlap) == 0, f"Train/test overlap: {overlap}"

    def test_train_test_covers_all(self) -> None:
        all_ids = set(TRAIN_QUERY_IDS) | set(TEST_QUERY_IDS)
        assert all_ids == set(TPCH_QUERIES.keys())

    def test_join_heavy_subset(self) -> None:
        """Join-heavy queries should be a subset of train queries."""
        assert set(JOIN_HEAVY_IDS).issubset(set(TRAIN_QUERY_IDS))

    def test_loader_get_train(self, loader: TPCHLoader) -> None:
        queries, ids = loader.get_train_queries()
        assert len(queries) == 16
        assert len(ids) == 16
        assert ids == TRAIN_QUERY_IDS

    def test_loader_get_test(self, loader: TPCHLoader) -> None:
        queries, ids = loader.get_test_queries()
        assert len(queries) == 6
        assert len(ids) == 6

    def test_loader_get_join_heavy(self, loader: TPCHLoader) -> None:
        queries, ids = loader.get_join_heavy_queries()
        assert len(queries) == 6
        assert "Q3" in ids
        assert "Q5" in ids

    def test_queries_are_valid_sql(self, loader: TPCHLoader) -> None:
        """All queries should contain SELECT and FROM."""
        for qid in loader.all_query_ids:
            sql = loader.get_query(qid)
            assert "SELECT" in sql.upper(), f"{qid} missing SELECT"
            assert "FROM" in sql.upper(), f"{qid} missing FROM"

    def test_num_queries(self, loader: TPCHLoader) -> None:
        assert loader.num_queries == 22
