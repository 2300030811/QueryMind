"""
Tests for the HintBuilder — action encoding/decoding and validation.

These tests verify the core action-space logic without requiring
a PostgreSQL connection.
"""

import pytest

from querymind.env.hint_builder import (
    ACTION_SPACE_SIZE,
    NUM_KNOBS,
    PLANNER_KNOBS,
    HintBuilder,
    PlannerConfig,
)


@pytest.fixture
def builder() -> HintBuilder:
    return HintBuilder()


class TestPlannerConfig:
    """Tests for the PlannerConfig dataclass."""

    def test_to_set_statements(self) -> None:
        config = PlannerConfig(
            enable_hashjoin=True,
            enable_mergejoin=False,
            enable_nestloop=True,
            enable_seqscan=True,
            enable_indexscan=False,
            enable_sort=True,
        )
        stmts = config.to_set_statements()
        assert len(stmts) == NUM_KNOBS
        assert "SET enable_hashjoin = ON;" in stmts
        assert "SET enable_mergejoin = OFF;" in stmts
        assert "SET enable_indexscan = OFF;" in stmts

    def test_to_dict(self) -> None:
        config = PlannerConfig(
            enable_hashjoin=True,
            enable_mergejoin=True,
            enable_nestloop=True,
            enable_seqscan=True,
            enable_indexscan=True,
            enable_sort=True,
        )
        d = config.to_dict()
        assert len(d) == NUM_KNOBS
        assert all(v is True for v in d.values())


class TestHintBuilder:
    """Tests for the HintBuilder action encoding/decoding."""

    def test_action_space_size(self, builder: HintBuilder) -> None:
        assert ACTION_SPACE_SIZE == 64

    def test_decode_all_on(self, builder: HintBuilder) -> None:
        """Action 63 (0b111111) should have all knobs ON."""
        config = builder.decode_action(63)
        assert config.enable_hashjoin is True
        assert config.enable_mergejoin is True
        assert config.enable_nestloop is True
        assert config.enable_seqscan is True
        assert config.enable_indexscan is True
        assert config.enable_sort is True

    def test_decode_all_off(self, builder: HintBuilder) -> None:
        """Action 0 (0b000000) should have all knobs OFF."""
        config = builder.decode_action(0)
        assert config.enable_hashjoin is False
        assert config.enable_mergejoin is False
        assert config.enable_nestloop is False
        assert config.enable_seqscan is False
        assert config.enable_indexscan is False
        assert config.enable_sort is False

    def test_decode_out_of_range(self, builder: HintBuilder) -> None:
        with pytest.raises(ValueError):
            builder.decode_action(64)
        with pytest.raises(ValueError):
            builder.decode_action(-1)

    def test_valid_action_all_on(self, builder: HintBuilder) -> None:
        """All-ON (default) should always be valid."""
        assert builder.is_valid_action(63)

    def test_invalid_action_all_off(self, builder: HintBuilder) -> None:
        """All-OFF should be invalid (no join method, no scan type)."""
        assert not builder.is_valid_action(0)

    def test_valid_actions_not_empty(self, builder: HintBuilder) -> None:
        """There should be valid actions available."""
        assert builder.num_valid_actions > 0
        assert builder.num_valid_actions < ACTION_SPACE_SIZE  # not all valid

    def test_valid_actions_have_join_and_scan(self, builder: HintBuilder) -> None:
        """Every valid action must have at least one join and one scan method."""
        for action in builder.valid_actions:
            config = builder.decode_action(action)
            has_join = (
                config.enable_hashjoin
                or config.enable_mergejoin
                or config.enable_nestloop
            )
            has_scan = config.enable_seqscan or config.enable_indexscan
            assert has_join, f"Action {action} has no join method"
            assert has_scan, f"Action {action} has no scan type"

    def test_default_action(self, builder: HintBuilder) -> None:
        assert builder.get_default_action() == 63

    def test_reset_statements(self, builder: HintBuilder) -> None:
        stmts = builder.get_reset_statements()
        assert len(stmts) == NUM_KNOBS
        assert all("ON" in s for s in stmts)

    def test_get_set_statements(self, builder: HintBuilder) -> None:
        stmts = builder.get_set_statements(63)
        assert len(stmts) == NUM_KNOBS

    def test_decode_specific_action(self, builder: HintBuilder) -> None:
        """Test a known bit pattern: action 36 = 0b100100."""
        # 0b100100: bit5=1 (hashjoin), bit2=1 (seqscan), rest OFF
        config = builder.decode_action(36)
        assert config.enable_hashjoin is True
        assert config.enable_mergejoin is False
        assert config.enable_nestloop is False
        assert config.enable_seqscan is True
        assert config.enable_indexscan is False
        assert config.enable_sort is False
