from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from melee_policy.integration.natural_game_end import has_decisive_zero_stock


def _gamestate(player_1_stocks: int, player_2_stocks: int) -> SimpleNamespace:
    return SimpleNamespace(
        players={
            1: SimpleNamespace(stock=player_1_stocks),
            2: SimpleNamespace(stock=player_2_stocks),
        }
    )


@pytest.mark.parametrize(
    ("stocks", "expected"),
    [
        ((4, 4), False),
        ((1, 0), True),
        ((0, 2), True),
        ((0, 0), False),
    ],
)
def test_decisive_zero_stock_latches_only_an_unambiguous_stock_result(
    stocks: tuple[int, int],
    expected: bool,
) -> None:
    assert has_decisive_zero_stock(_gamestate(*stocks), (1, 2)) is expected


def test_decisive_zero_stock_requires_two_distinct_ports() -> None:
    with pytest.raises(ValueError, match="two distinct ports"):
        has_decisive_zero_stock(_gamestate(1, 0), (1, 1))


def test_every_live_benchmark_loop_latches_a_decisive_stock_out() -> None:
    project_root = Path(__file__).resolve().parents[2]
    expected_counts = {
        project_root / "src/melee_policy/integration/frisson_cpu_match.py": 1,
        project_root / "src/melee_policy/integration/frisson_match.py": 1,
        project_root / "src/melee_policy/integration/frisson_slippi_match.py": 1,
        project_root / "src/melee_policy/integration/match_runtime.py": 1,
        project_root / "src/melee_policy/integration/slippi_match.py": 2,
    }

    for path, expected_count in expected_counts.items():
        source = path.read_text(encoding="utf-8")
        assert source.count("has_decisive_zero_stock(gamestate,") == expected_count, path
