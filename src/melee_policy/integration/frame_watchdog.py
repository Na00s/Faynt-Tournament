"""Fail closed when an in-progress Slippi game stops yielding states."""

from __future__ import annotations

import math
import time
from collections.abc import Callable

DEFAULT_IN_GAME_NO_FRAME_TIMEOUT_SECONDS = 30.0


class InGameNoFrameTimeout(TimeoutError):
    """Raised after a consecutive in-game run of empty Slippi polls."""

    def __init__(
        self,
        *,
        timeout_seconds: float,
        elapsed_seconds: float,
        consecutive_none_results: int,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds
        self.consecutive_none_results = consecutive_none_results
        super().__init__(
            "Slippi stopped providing states during gameplay for "
            f"{elapsed_seconds:.3f}s (timeout={timeout_seconds:.3f}s, "
            f"consecutive_none_results={consecutive_none_results})"
        )


class InGameNoFrameWatchdog:
    """Time only consecutive ``None`` step results after gameplay begins.

    A received state clears the timer. Menu-time empty polls also clear it. The
    clock is therefore never sampled on a normal state-processing path, so the
    watchdog does not enter policy inference or controller-boundary timing.
    """

    def __init__(
        self,
        timeout_seconds: float = DEFAULT_IN_GAME_NO_FRAME_TIMEOUT_SECONDS,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        timeout_seconds = float(timeout_seconds)
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise ValueError("in-game no-frame timeout must be finite and greater than zero")
        self.timeout_seconds = timeout_seconds
        self._clock = clock
        self._none_since: float | None = None
        self._consecutive_none_results = 0

    @property
    def consecutive_none_results(self) -> int:
        return self._consecutive_none_results

    @property
    def timing_empty_results(self) -> bool:
        return self._none_since is not None

    def observe_step_result(self, step_result: object | None, *, gameplay_started: bool) -> None:
        """Observe one completed transport step and raise on an in-game stall."""
        if step_result is not None or not gameplay_started:
            self._none_since = None
            self._consecutive_none_results = 0
            return

        now = self._clock()
        self._consecutive_none_results += 1
        if self._none_since is None:
            self._none_since = now
            return

        elapsed = now - self._none_since
        if elapsed >= self.timeout_seconds:
            raise InGameNoFrameTimeout(
                timeout_seconds=self.timeout_seconds,
                elapsed_seconds=elapsed,
                consecutive_none_results=self._consecutive_none_results,
            )
