from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from melee_policy.integration.frame_watchdog import (
    DEFAULT_IN_GAME_NO_FRAME_TIMEOUT_SECONDS,
    InGameNoFrameTimeout,
    InGameNoFrameWatchdog,
)


def _clock(values: list[float]) -> Callable[[], float]:
    iterator = iter(values)

    def read() -> float:
        return next(iterator)

    return read


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("inf"), float("nan")])
def test_watchdog_rejects_invalid_timeout(timeout: float) -> None:
    with pytest.raises(ValueError, match="finite and greater than zero"):
        InGameNoFrameWatchdog(timeout)


def test_watchdog_does_not_sample_clock_before_gameplay_or_on_received_states() -> None:
    def forbidden_clock() -> float:
        raise AssertionError("clock must only be sampled for an in-game None result")

    watchdog = InGameNoFrameWatchdog(clock=forbidden_clock)

    watchdog.observe_step_result(None, gameplay_started=False)
    watchdog.observe_step_result(object(), gameplay_started=False)
    watchdog.observe_step_result(object(), gameplay_started=True)

    assert watchdog.consecutive_none_results == 0
    assert watchdog.timing_empty_results is False


def test_watchdog_times_only_one_consecutive_in_game_none_run_and_resets() -> None:
    watchdog = InGameNoFrameWatchdog(
        timeout_seconds=30.0,
        clock=_clock([10.0, 39.9, 50.0, 79.9]),
    )

    watchdog.observe_step_result(None, gameplay_started=True)
    watchdog.observe_step_result(None, gameplay_started=True)
    assert watchdog.consecutive_none_results == 2
    assert watchdog.timing_empty_results is True

    watchdog.observe_step_result(object(), gameplay_started=True)
    assert watchdog.consecutive_none_results == 0
    assert watchdog.timing_empty_results is False

    watchdog.observe_step_result(None, gameplay_started=True)
    watchdog.observe_step_result(None, gameplay_started=True)
    assert watchdog.consecutive_none_results == 2


def test_watchdog_raises_at_timeout_boundary_with_diagnostics() -> None:
    watchdog = InGameNoFrameWatchdog(
        timeout_seconds=30.0,
        clock=_clock([100.0, 129.0, 130.0]),
    )

    watchdog.observe_step_result(None, gameplay_started=True)
    watchdog.observe_step_result(None, gameplay_started=True)
    with pytest.raises(InGameNoFrameTimeout) as caught:
        watchdog.observe_step_result(None, gameplay_started=True)

    error = caught.value
    assert error.timeout_seconds == 30.0
    assert error.elapsed_seconds == 30.0
    assert error.consecutive_none_results == 3
    assert "Slippi stopped providing states during gameplay" in str(error)


def test_default_timeout_is_conservative() -> None:
    assert DEFAULT_IN_GAME_NO_FRAME_TIMEOUT_SECONDS == 30.0


def test_every_benchmark_transport_loop_observes_the_watchdog() -> None:
    project_root = Path(__file__).resolve().parents[2]
    expected_counts = {
        project_root / "src/melee_policy/integration/frisson_cpu_match.py": 1,
        project_root / "src/melee_policy/integration/frisson_match.py": 1,
        project_root / "src/melee_policy/integration/frisson_slippi_match.py": 1,
        project_root / "src/melee_policy/integration/match_runtime.py": 1,
        project_root / "src/melee_policy/integration/slippi_match.py": 2,
    }
    step_statement = "gamestate = transport.step()"
    observed_statement = "no_frame_watchdog.observe_step_result(gamestate, gameplay_started=in_game)"

    for path, expected_count in expected_counts.items():
        source = path.read_text(encoding="utf-8")
        assert source.count(observed_statement) == expected_count, path
