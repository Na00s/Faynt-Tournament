"""Shared natural-end detection for live stock-match launch loops."""

from __future__ import annotations

from typing import Any


def has_decisive_zero_stock(gamestate: Any, ports: tuple[int, int]) -> bool:
    """Return whether one player, and only one player, has reached zero stocks.

    Call this after both policies have processed and dispatched the current
    frame.  A double-zero frame must continue so Melee can resolve sudden
    death through its normal state transition.
    """

    if len(set(ports)) != 2:
        raise ValueError("natural-end detection requires two distinct ports")
    stocks = tuple(int(gamestate.players[port].stock) for port in ports)
    return sum(stock == 0 for stock in stocks) == 1
