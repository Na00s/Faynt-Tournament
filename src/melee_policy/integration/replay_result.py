"""Fail-closed rule and outcome audit for tournament Slippi replays.

The public entry point imports ``peppi_py`` lazily so importing the integration
package does not require the E010 environment.  ``audit_parsed_game`` and
``derive_outcome`` are pure helpers and intentionally accept structural objects,
which keeps terminal-state policy testable without constructing Arrow columns.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

from melee_policy.e000.enums import (
    CHARACTER_FOLDER_TO_INTERNAL,
    EXTERNAL_CHARACTER_TO_INTERNAL,
    SLIPPI_STAGE_TO_LIBMELEE,
)

SCHEMA_VERSION = "integration.replay_result.v1"
MINIMUM_SLIPPI_VERSION = (3, 18, 0)
EXPECTED_PEPPI_VERSION = "0.9.2"

_TIEBREAKER_SHARED_RULE_CHECKS = (
    "slippi_version_at_least_3_18",
    "ntsc",
    "singles",
    "exact_expected_ports",
    "decreasing_timer_enabled",
    "stock_game_mode",
    "no_player_gameplay_modifiers",
    "versus_mode_scene",
    "items_off_frequency",
    "items_off_bitfield",
    "no_raining_bombs",
    "damage_ratio_one",
    "standard_self_destruct_score",
    "human_slots",
    "standard_handicap",
    "standard_offense_ratio",
    "standard_defense_ratio",
    "standard_model_scale",
    "ucf_dashback",
    "ucf_shielddrop",
    "frozen_pokemon_stadium",
    "exact_expected_stage",
    "exact_expected_characters",
    "exact_expected_costumes",
)

_STAGE_NAME_TO_LIBMELEE = {
    "BATTLEFIELD": 24,
    "DREAMLAND": 26,
    "FINAL_DESTINATION": 25,
    "FOUNTAIN_OF_DREAMS": 8,
    "POKEMON_STADIUM": 18,
    "YOSHIS_STORY": 6,
}
_LIBMELEE_STAGE_TO_NAME = {value: key for key, value in _STAGE_NAME_TO_LIBMELEE.items()}
_INTERNAL_CHARACTER_TO_NAME = {
    internal: name
    for name, internal_values in CHARACTER_FOLDER_TO_INTERNAL.items()
    for internal in internal_values
}


def _scalar(value: Any) -> Any:
    if hasattr(value, "as_py"):
        return value.as_py()
    if isinstance(value, Enum):
        return value.value
    return value


def _integer(value: Any) -> int | None:
    value = _scalar(value)
    if value is None:
        return None
    try:
        return int(value)
    except TypeError:
        return None
    except ValueError:
        return None
    except OverflowError:
        return None


def _finite(value: Any) -> float | None:
    value = _scalar(value)
    if value is None:
        return None
    try:
        result = float(value)
    except TypeError:
        return None
    except ValueError:
        return None
    except OverflowError:
        return None
    return result if math.isfinite(result) else None


def _name(value: Any) -> str | None:
    if value is None:
        return None
    enum_name = getattr(value, "name", None)
    if enum_name is not None:
        return str(enum_name).upper()
    if isinstance(value, str):
        return value.upper()
    return str(value).upper()


def _column(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "to_pylist"):
        return list(value.to_pylist())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    try:
        return list(value)
    except TypeError:
        return []


def _port(value: Any) -> int | None:
    """Convert a zero-indexed raw Slippi port to a one-indexed public port."""
    raw = _integer(value)
    if raw is None or raw not in range(4):
        return None
    return raw + 1


def _expected_stage(value: Any) -> tuple[int, str | None]:
    if isinstance(value, str):
        normalized = value.strip().upper()
        try:
            return _STAGE_NAME_TO_LIBMELEE[normalized], normalized
        except KeyError as exc:
            raise ValueError(f"unsupported expected stage: {value!r}") from exc
    stage_id = _integer(getattr(value, "value", value))
    if stage_id not in _LIBMELEE_STAGE_TO_NAME:
        raise ValueError(f"unsupported expected libmelee stage ID: {stage_id!r}")
    return stage_id, _LIBMELEE_STAGE_TO_NAME[stage_id]


def _expected_character(value: Any) -> tuple[tuple[int, ...], str | None]:
    if isinstance(value, str):
        normalized = value.strip().upper()
        try:
            return tuple(CHARACTER_FOLDER_TO_INTERNAL[normalized]), normalized
        except KeyError as exc:
            raise ValueError(f"unsupported expected character: {value!r}") from exc
    character_id = _integer(getattr(value, "value", value))
    if character_id is None or character_id not in _INTERNAL_CHARACTER_TO_NAME:
        raise ValueError(f"unsupported expected internal character ID: {character_id!r}")
    return (character_id,), _INTERNAL_CHARACTER_TO_NAME[character_id]


def _expectations(
    expected_stage: Any, expected_characters: Mapping[int, Any]
) -> tuple[int, str | None, dict[int, tuple[tuple[int, ...], str | None]]]:
    stage_id, stage_name = _expected_stage(expected_stage)
    characters: dict[int, tuple[tuple[int, ...], str | None]] = {}
    for raw_port, character in expected_characters.items():
        port = _integer(raw_port)
        if port is None or port not in range(1, 5):
            raise ValueError(f"expected character port must be in 1..4, received {raw_port!r}")
        if port in characters:
            raise ValueError(f"duplicate expected character port: {port}")
        characters[port] = _expected_character(character)
    if len(characters) != 2:
        raise ValueError("a singles replay audit requires exactly two expected character ports")
    return stage_id, stage_name, characters


def _costume_expectations(
    expected_costumes: Mapping[int, int] | None,
    expected_ports: Mapping[int, Any],
) -> dict[int, int] | None:
    if expected_costumes is None:
        return None
    costumes: dict[int, int] = {}
    for raw_port, raw_costume in expected_costumes.items():
        port = _integer(raw_port)
        costume = _integer(raw_costume)
        if port is None or port not in range(1, 5):
            raise ValueError(f"expected costume port must be in 1..4, received {raw_port!r}")
        if isinstance(raw_costume, bool) or costume is None or costume not in range(6):
            raise ValueError(f"expected costume for port {port} must be an integer in 0..5")
        costumes[port] = costume
    if set(costumes) != set(expected_ports):
        raise ValueError("expected costume ports must exactly match expected character ports")
    return costumes


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    return str(value)


def _equal_float(value: Any, expected: float) -> bool:
    actual = _finite(value)
    return actual is not None and math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-6)


def _version(start: Any) -> tuple[int, ...] | None:
    slippi = getattr(start, "slippi", None)
    raw = getattr(slippi, "version", None)
    values = _column(raw)
    if len(values) < 3:
        return None
    version = tuple(_integer(item) for item in values[:3])
    if any(item is None for item in version):
        return None
    return tuple(int(item) for item in version if item is not None)


def _player_metadata(player: Any) -> dict[str, Any]:
    port = _port(getattr(player, "port", None))
    external_character = _integer(getattr(player, "character", None))
    internal_character = (
        EXTERNAL_CHARACTER_TO_INTERNAL.get(external_character) if external_character is not None else None
    )
    ucf = getattr(player, "ucf", None)
    return {
        "port": port,
        "external_character_id": external_character,
        "internal_character_id": internal_character,
        "character": (
            _INTERNAL_CHARACTER_TO_NAME.get(internal_character) if internal_character is not None else None
        ),
        "costume": _integer(getattr(player, "costume", None)),
        "bitfield": _integer(getattr(player, "bitfield", None)),
        "player_type": _name(getattr(player, "type", None)),
        "stocks": _integer(getattr(player, "stocks", None)),
        "handicap": _integer(getattr(player, "handicap", None)),
        "offense_ratio": _finite(getattr(player, "offense_ratio", None)),
        "defense_ratio": _finite(getattr(player, "defense_ratio", None)),
        "model_scale": _finite(getattr(player, "model_scale", None)),
        "ucf": {
            "dashback": _name(getattr(ucf, "dash_back", None)),
            "shielddrop": _name(getattr(ucf, "shield_drop", None)),
        },
    }


def _initial_slots(
    game: Any, player_entries: list[tuple[int, int | None]]
) -> tuple[int | None, dict[int, dict[str, Any]]]:
    frames = getattr(game, "frames", None)
    frame_ids = _column(getattr(frames, "id", None))
    frame_ports = tuple(getattr(frames, "ports", ()) or ())
    columns: dict[int, tuple[list[Any], list[Any], list[Any]]] = {}
    for frame_port_index, public_port in player_entries:
        if public_port is None or frame_port_index >= len(frame_ports):
            continue
        leader = getattr(frame_ports[frame_port_index], "leader", None)
        post = getattr(leader, "post", None)
        columns[public_port] = (
            _column(getattr(post, "stocks", getattr(post, "stock", None))),
            _column(getattr(post, "percent", None)),
            _column(getattr(post, "character", None)),
        )
    common_length = min(
        [
            len(frame_ids),
            *(len(stocks) for stocks, _, _ in columns.values()),
            *(len(percents) for _, percents, _ in columns.values()),
        ],
        default=0,
    )
    for ordinal in range(common_length):
        if not all(
            _integer(stocks[ordinal]) is not None and _finite(percents[ordinal]) is not None
            for stocks, percents, _ in columns.values()
        ):
            continue
        frame_id = _integer(frame_ids[ordinal])
        return frame_id, {
            port: {
                "stocks": _integer(stocks[ordinal]),
                "percent": _finite(percents[ordinal]),
                "internal_character_id": _integer(characters[ordinal]),
            }
            for port, (stocks, percents, characters) in columns.items()
        }
    return None, {}


def _terminal_slots(
    game: Any, player_entries: list[tuple[int, int | None]]
) -> tuple[int | None, dict[int, dict[str, Any]]]:
    frames = getattr(game, "frames", None)
    frame_ids = _column(getattr(frames, "id", None))
    frame_ports = tuple(getattr(frames, "ports", ()) or ())
    columns: dict[int, tuple[list[Any], list[Any], list[Any]]] = {}
    for frame_port_index, public_port in player_entries:
        if public_port is None or frame_port_index >= len(frame_ports):
            continue
        leader = getattr(frame_ports[frame_port_index], "leader", None)
        post = getattr(leader, "post", None)
        stocks = _column(getattr(post, "stocks", getattr(post, "stock", None)))
        percents = _column(getattr(post, "percent", None))
        characters = _column(getattr(post, "character", None))
        columns[public_port] = (stocks, percents, characters)

    common_length = min(
        [len(frame_ids), *(len(stock_values) for stock_values, _, _ in columns.values())],
        default=0,
    )
    terminal_ordinal: int | None = None
    for ordinal in range(common_length - 1, -1, -1):
        if all(_integer(stock_values[ordinal]) is not None for stock_values, _, _ in columns.values()):
            terminal_ordinal = ordinal
            break
    if terminal_ordinal is None or len(columns) != len(player_entries):
        return None, {}

    frame_id = _integer(frame_ids[terminal_ordinal])
    result: dict[int, dict[str, Any]] = {}
    for port, (stocks, percents, characters) in columns.items():
        percent = percents[terminal_ordinal] if terminal_ordinal < len(percents) else None
        character = characters[terminal_ordinal] if terminal_ordinal < len(characters) else None
        internal_character = _integer(character)
        result[port] = {
            "port": port,
            "stocks": _integer(stocks[terminal_ordinal]),
            "percent": _finite(percent),
            "internal_character_id": internal_character,
            "character": (
                _INTERNAL_CHARACTER_TO_NAME.get(internal_character)
                if internal_character is not None
                else None
            ),
        }
    return frame_id, result


def _end_metadata(end: Any) -> dict[str, Any]:
    if end is None:
        return {
            "present": False,
            "method": None,
            "lras_initiator_port": None,
            "placements": [],
        }
    placements: list[dict[str, int | None]] = []
    for player in getattr(end, "players", ()) or ():
        placements.append(
            {
                "port": _port(getattr(player, "port", None)),
                "placement": _integer(getattr(player, "placement", None)),
            }
        )
    placements.sort(key=lambda item: (item["port"] is None, item["port"] or 0))
    return {
        "present": True,
        "method": _name(getattr(end, "method", None)),
        "lras_initiator_port": _port(getattr(end, "lras_initiator", None)),
        "placements": placements,
    }


def _outcome(
    status: str,
    *,
    game_complete: bool,
    winner_port: int | None = None,
    loser_port: int | None = None,
    draw: bool = False,
    requires_tiebreak: bool = False,
    reason: str,
    evidence: list[str],
) -> dict[str, Any]:
    return {
        "status": status,
        "game_complete": game_complete,
        "conclusive": status in {"win", "draw"},
        "winner_port": winner_port,
        "loser_port": loser_port,
        "draw": draw,
        "requires_tiebreak": requires_tiebreak,
        "reason": reason,
        "evidence": evidence,
    }


def derive_outcome(
    end: Any,
    terminal_slots: Mapping[int, Mapping[str, Any]],
    *,
    sudden_death: bool = False,
) -> dict[str, Any]:
    """Derive a two-player result without inventing evidence.

    Timeout placement bytes are not used because old Slippi writers could emit
    an incorrect placement.  Stocks and percent are the actual timeout rules.
    For a normal stock game, terminal stocks and End placements must agree when
    both are decisive.  A simultaneous final-stock loss is a draw that requires
    a tiebreak game even if an End placement claims a winner.
    """
    end_data = _end_metadata(end)
    method = end_data["method"]
    ports = sorted(int(port) for port in terminal_slots)
    placements = end_data["placements"]
    placement_winners = [
        int(item["port"]) for item in placements if item["port"] is not None and item["placement"] == 0
    ]
    placement_winner = placement_winners[0] if len(placement_winners) == 1 else None

    if not end_data["present"]:
        return _outcome(
            "incomplete",
            game_complete=False,
            reason="missing_game_end",
            evidence=[
                "No Game End event is present; terminal frames alone do not prove recording completion."
            ],
        )
    if len(ports) != 2:
        return _outcome(
            "inconclusive",
            game_complete=False,
            reason="terminal_slots_unavailable",
            evidence=[f"Expected two complete terminal slots, found {ports}."],
        )

    first, second = ports
    first_stocks = _integer(terminal_slots[first].get("stocks"))
    second_stocks = _integer(terminal_slots[second].get("stocks"))
    if first_stocks is None or second_stocks is None:
        return _outcome(
            "inconclusive",
            game_complete=False,
            reason="terminal_stocks_unavailable",
            evidence=["Both terminal stock counts are required."],
        )

    prefix = "sudden_death_" if sudden_death else ""
    if method in {"NO_CONTEST", "UNRESOLVED"}:
        initiator = end_data["lras_initiator_port"]
        if initiator not in ports:
            return _outcome(
                "inconclusive",
                game_complete=False,
                reason=f"{prefix}no_contest_without_lras_initiator",
                evidence=[f"Game End method is {method}, but no active LRAS initiator is recorded."],
            )
        winner = second if initiator == first else first
        if placement_winner is not None and placement_winner != winner:
            return _outcome(
                "inconclusive",
                game_complete=True,
                reason=f"{prefix}lras_placement_conflict",
                evidence=[
                    f"LRAS identifies port {winner} as winner, but placement identifies port "
                    f"{placement_winner}."
                ],
            )
        return _outcome(
            "win",
            game_complete=True,
            winner_port=winner,
            loser_port=initiator,
            reason=f"{prefix}lras_forfeit",
            evidence=[f"Port {initiator} initiated LRAS; port {winner} wins the two-player game."],
        )

    if method == "TIME":
        first_percent = _finite(terminal_slots[first].get("percent"))
        second_percent = _finite(terminal_slots[second].get("percent"))
        if first_stocks != second_stocks:
            winner = first if first_stocks > second_stocks else second
            loser = second if winner == first else first
            return _outcome(
                "win",
                game_complete=True,
                winner_port=winner,
                loser_port=loser,
                reason=f"{prefix}timeout_stocks",
                evidence=[
                    f"Timeout terminal stocks are port {first}: {first_stocks}, "
                    f"port {second}: {second_stocks}."
                ],
            )
        if first_percent is None or second_percent is None:
            return _outcome(
                "inconclusive",
                game_complete=True,
                reason=f"{prefix}timeout_percent_unavailable",
                evidence=["Tied terminal stocks require both terminal percents."],
            )
        if not math.isclose(first_percent, second_percent, rel_tol=0.0, abs_tol=1e-6):
            winner = first if first_percent < second_percent else second
            loser = second if winner == first else first
            return _outcome(
                "win",
                game_complete=True,
                winner_port=winner,
                loser_port=loser,
                reason=f"{prefix}timeout_percent",
                evidence=[
                    f"Timeout terminal percent is port {first}: {first_percent}, "
                    f"port {second}: {second_percent}; lower percent wins tied stocks."
                ],
            )
        return _outcome(
            "draw",
            game_complete=True,
            draw=True,
            requires_tiebreak=True,
            reason=f"{prefix}timeout_exact_tie",
            evidence=["Timeout terminal stocks and percent are tied; another tiebreak game is required."],
        )

    if method not in {"GAME", "RESOLVED"}:
        return _outcome(
            "inconclusive",
            game_complete=False,
            reason="unsupported_game_end_method",
            evidence=[f"Unsupported or missing Game End method: {method!r}."],
        )

    if first_stocks == 0 and second_stocks == 0:
        return _outcome(
            "draw",
            game_complete=True,
            draw=True,
            requires_tiebreak=True,
            reason=f"{prefix}simultaneous_final_stock_loss",
            evidence=["Both players have zero stocks on the terminal frame."],
        )
    if (first_stocks == 0) != (second_stocks == 0):
        winner = second if first_stocks == 0 else first
        loser = first if first_stocks == 0 else second
        if placement_winner is not None and placement_winner != winner:
            return _outcome(
                "inconclusive",
                game_complete=True,
                reason=f"{prefix}stock_out_placement_conflict",
                evidence=[
                    f"Terminal stocks identify port {winner} as winner, but placement identifies "
                    f"port {placement_winner}."
                ],
            )
        return _outcome(
            "win",
            game_complete=True,
            winner_port=winner,
            loser_port=loser,
            reason=f"{prefix}stock_out",
            evidence=[f"Port {loser} has zero terminal stocks; port {winner} has stocks remaining."],
        )
    if method == "RESOLVED" and placement_winner in ports:
        loser = second if placement_winner == first else first
        return _outcome(
            "win",
            game_complete=True,
            winner_port=placement_winner,
            loser_port=loser,
            reason=f"{prefix}resolved_end_placement",
            evidence=[f"Resolved Game End records port {placement_winner} in first place."],
        )
    return _outcome(
        "inconclusive",
        game_complete=True,
        reason=f"{prefix}normal_end_without_decisive_terminal_state",
        evidence=[
            f"Game End method is {method}, but neither player is at zero stocks and no supported "
            "resolution proves the winner."
        ],
    )


def audit_parsed_game(
    game: Any,
    *,
    expected_stage: Any,
    expected_characters: Mapping[int, Any],
    expected_costumes: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    """Audit an already parsed peppi-compatible game object."""
    expected_stage_id, expected_stage_name, expected_slots = _expectations(
        expected_stage, expected_characters
    )
    expected_costume_slots = _costume_expectations(expected_costumes, expected_slots)
    start = getattr(game, "start", None)
    if start is None:
        raise ValueError("parsed game has no Game Start event")

    version = _version(start)
    raw_stage_id = _integer(getattr(start, "stage", None))
    actual_stage_id = SLIPPI_STAGE_TO_LIBMELEE.get(raw_stage_id) if raw_stage_id is not None else None
    raw_players = list(getattr(start, "players", ()) or ())
    players = [(index, player) for index, player in enumerate(raw_players) if player is not None]
    slots = [_player_metadata(player) for _, player in players]
    actual_ports = [slot["port"] for slot in slots if slot["port"] is not None]
    exact_ports = sorted(actual_ports) == sorted(expected_slots)
    game_bitfield = tuple(_integer(item) for item in _column(getattr(start, "bitfield", None)))
    timer_mode = game_bitfield[0] & 0x03 if len(game_bitfield) == 4 and game_bitfield[0] is not None else None
    game_mode = game_bitfield[0] & 0xE0 if len(game_bitfield) == 4 and game_bitfield[0] is not None else None
    scene = getattr(start, "scene", None)
    scene_major = _integer(getattr(scene, "major", None))
    scene_minor = _integer(getattr(scene, "minor", None))
    match = getattr(start, "match", None)
    tiebreaker_number = _integer(getattr(match, "tiebreaker", None))
    sudden_death = tiebreaker_number is not None and tiebreaker_number > 0

    player_entries = [(index, slot["port"]) for (index, _), slot in zip(players, slots, strict=True)]
    initial_frame_id, initial_slots = _initial_slots(game, player_entries)
    frame_id, terminal_slots = _terminal_slots(
        game,
        player_entries,
    )
    end_data = _end_metadata(getattr(game, "end", None))
    outcome = derive_outcome(getattr(game, "end", None), terminal_slots, sudden_death=sudden_death)

    checks: dict[str, bool] = {}
    failures: list[dict[str, Any]] = []

    def check(name: str, passed: bool, expected: Any, actual: Any) -> None:
        checks[name] = bool(passed)
        if not passed:
            failures.append(
                {
                    "check": name,
                    "expected": _json_safe(expected),
                    "actual": _json_safe(actual),
                }
            )

    check(
        "no_lras_technical_termination",
        not str(outcome.get("reason", "")).endswith("lras_forfeit"),
        "natural stock, timeout, draw, or supported resolved Game End",
        outcome.get("reason"),
    )
    check("base_game_not_tiebreaker", tiebreaker_number == 0, 0, tiebreaker_number)

    check(
        "slippi_version_at_least_3_18",
        version is not None and version >= MINIMUM_SLIPPI_VERSION,
        list(MINIMUM_SLIPPI_VERSION),
        list(version) if version is not None else None,
    )
    check("ntsc", getattr(start, "is_pal", None) is False, False, getattr(start, "is_pal", None))
    check(
        "singles",
        getattr(start, "is_teams", None) is False and len(slots) == 2,
        {"is_teams": False, "player_count": 2},
        {"is_teams": getattr(start, "is_teams", None), "player_count": len(slots)},
    )
    check("exact_expected_ports", exact_ports, sorted(expected_slots), sorted(actual_ports))
    check(
        "four_stocks_each",
        len(slots) == 2 and all(slot["stocks"] == 4 for slot in slots),
        {str(port): 4 for port in sorted(expected_slots)},
        {str(slot["port"]): slot["stocks"] for slot in slots},
    )
    check(
        "timer_480_seconds",
        _integer(getattr(start, "timer", None)) == 480,
        480,
        getattr(start, "timer", None),
    )
    check(
        "decreasing_timer_enabled",
        timer_mode == 2,
        {"game_bitfield_length": 4, "timer_mode": 2, "meaning": "DECREASING"},
        {"game_bitfield": game_bitfield, "timer_mode": timer_mode},
    )
    check(
        "stock_game_mode",
        game_mode == 0x20,
        {"game_mode_mask": "0xe0", "game_mode": "0x20", "meaning": "STOCK"},
        {"game_bitfield": game_bitfield, "game_mode": game_mode},
    )
    check(
        "no_player_gameplay_modifiers",
        len(slots) == 2
        and all(isinstance(slot["bitfield"], int) and slot["bitfield"] & 0x3F == 0 for slot in slots),
        "low six gameplay-modifier bits clear for both players",
        {str(slot["port"]): slot["bitfield"] for slot in slots},
    )
    check(
        "initial_post_frame_four_stocks_zero_percent",
        exact_ports
        and initial_frame_id is not None
        and all(
            initial_slots.get(port, {}).get("stocks") == 4
            and _equal_float(initial_slots.get(port, {}).get("percent"), 0.0)
            for port in expected_slots
        ),
        {str(port): {"stocks": 4, "percent": 0.0} for port in sorted(expected_slots)},
        {"frame_id": initial_frame_id, "slots": initial_slots},
    )
    check(
        "versus_mode_scene",
        (scene_major, scene_minor) == (2, 2),
        {"major": 2, "minor": 2},
        {"major": scene_major, "minor": scene_minor},
    )
    check(
        "items_off_frequency",
        _integer(getattr(start, "item_spawn_frequency", None)) == -1,
        -1,
        getattr(start, "item_spawn_frequency", None),
    )
    item_bitfield = tuple(_integer(item) for item in _column(getattr(start, "item_spawn_bitfield", None)))
    check(
        "items_off_bitfield",
        item_bitfield == (255, 255, 255, 255, 255),
        [255, 255, 255, 255, 255],
        item_bitfield,
    )
    check(
        "no_raining_bombs",
        getattr(start, "is_raining_bombs", None) is False,
        False,
        getattr(start, "is_raining_bombs", None),
    )
    check(
        "damage_ratio_one",
        _equal_float(getattr(start, "damage_ratio", None), 1.0),
        1.0,
        getattr(start, "damage_ratio", None),
    )
    check(
        "standard_self_destruct_score",
        _integer(getattr(start, "self_destruct_score", None)) == -1,
        -1,
        getattr(start, "self_destruct_score", None),
    )
    check(
        "human_slots",
        len(slots) == 2 and all(slot["player_type"] == "HUMAN" for slot in slots),
        "HUMAN for both slots",
        {str(slot["port"]): slot["player_type"] for slot in slots},
    )
    check(
        "standard_handicap",
        len(slots) == 2 and all(slot["handicap"] == 9 for slot in slots),
        9,
        {str(slot["port"]): slot["handicap"] for slot in slots},
    )
    for field in ("offense_ratio", "defense_ratio", "model_scale"):
        check(
            f"standard_{field}",
            len(slots) == 2 and all(_equal_float(slot[field], 1.0) for slot in slots),
            1.0,
            {str(slot["port"]): slot[field] for slot in slots},
        )
    check(
        "ucf_dashback",
        len(slots) == 2 and all(slot["ucf"]["dashback"] == "UCF" for slot in slots),
        "UCF for both slots",
        {str(slot["port"]): slot["ucf"]["dashback"] for slot in slots},
    )
    check(
        "ucf_shielddrop",
        len(slots) == 2 and all(slot["ucf"]["shielddrop"] == "UCF" for slot in slots),
        "UCF for both slots",
        {str(slot["port"]): slot["ucf"]["shielddrop"] for slot in slots},
    )
    check(
        "frozen_pokemon_stadium",
        getattr(start, "is_frozen_ps", None) is True,
        True,
        getattr(start, "is_frozen_ps", None),
    )
    check(
        "exact_expected_stage",
        actual_stage_id == expected_stage_id,
        {"name": expected_stage_name, "libmelee_id": expected_stage_id},
        {
            "name": (_LIBMELEE_STAGE_TO_NAME.get(actual_stage_id) if actual_stage_id is not None else None),
            "libmelee_id": actual_stage_id,
            "raw_slippi_id": raw_stage_id,
        },
    )
    observed_characters = {
        int(slot["port"]): slot["internal_character_id"] for slot in slots if slot["port"] is not None
    }
    character_match = exact_ports and all(
        observed_characters.get(port) in accepted_ids for port, (accepted_ids, _) in expected_slots.items()
    )
    check(
        "exact_expected_characters",
        character_match,
        {
            str(port): {"name": name, "internal_ids": list(ids)}
            for port, (ids, name) in sorted(expected_slots.items())
        },
        {
            str(slot["port"]): {
                "name": slot["character"],
                "internal_id": slot["internal_character_id"],
                "external_id": slot["external_character_id"],
            }
            for slot in slots
        },
    )
    if expected_costume_slots is not None:
        observed_costumes = {int(slot["port"]): slot["costume"] for slot in slots if slot["port"] is not None}
        check(
            "exact_expected_costumes",
            exact_ports and observed_costumes == expected_costume_slots,
            {str(port): costume for port, costume in sorted(expected_costume_slots.items())},
            {str(port): costume for port, costume in sorted(observed_costumes.items())},
        )

    rules_passed = bool(checks) and all(checks.values())
    audit_passed = rules_passed and outcome["game_complete"] and outcome["conclusive"]
    tournament_result_ready = audit_passed and outcome["status"] in {"win", "draw"}
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "settings": {
            "slippi_version": list(version) if version is not None else None,
            "region": (
                "PAL"
                if getattr(start, "is_pal", None) is True
                else "NTSC"
                if getattr(start, "is_pal", None) is False
                else "unknown"
            ),
            "is_teams": getattr(start, "is_teams", None),
            "stage": {
                "raw_slippi_id": raw_stage_id,
                "libmelee_id": actual_stage_id,
                "name": (
                    _LIBMELEE_STAGE_TO_NAME.get(actual_stage_id) if actual_stage_id is not None else None
                ),
            },
            "timer_seconds": _integer(getattr(start, "timer", None)),
            "game_bitfield": list(game_bitfield),
            "timer_mode": timer_mode,
            "timer_mode_name": "DECREASING" if timer_mode == 2 else None,
            "game_mode": game_mode,
            "game_mode_name": "STOCK" if game_mode == 0x20 else None,
            "pause_mode_enabled": (
                bool(game_bitfield[2] & 0x80)
                if len(game_bitfield) == 4 and game_bitfield[2] is not None
                else None
            ),
            "scene": {"major": scene_major, "minor": scene_minor},
            "item_spawn_frequency": _integer(getattr(start, "item_spawn_frequency", None)),
            "item_spawn_bitfield": list(item_bitfield),
            "is_raining_bombs": getattr(start, "is_raining_bombs", None),
            "damage_ratio": _finite(getattr(start, "damage_ratio", None)),
            "self_destruct_score": _integer(getattr(start, "self_destruct_score", None)),
            "game_random_seed": _integer(getattr(start, "random_seed", None)),
            "is_frozen_pokemon_stadium": getattr(start, "is_frozen_ps", None),
            "tiebreaker_number": tiebreaker_number,
            "sudden_death": sudden_death,
            "slots": slots,
        },
        "expected": {
            "stage": {"name": expected_stage_name, "libmelee_id": expected_stage_id},
            "characters": {
                str(port): {"name": name, "internal_ids": list(ids)}
                for port, (ids, name) in sorted(expected_slots.items())
            },
            "costumes": (
                None
                if expected_costume_slots is None
                else {str(port): costume for port, costume in sorted(expected_costume_slots.items())}
            ),
        },
        "terminal": {
            "frame_id": frame_id,
            "slots": {str(port): slot for port, slot in sorted(terminal_slots.items())},
        },
        "initial": {
            "frame_id": initial_frame_id,
            "slots": {str(port): slot for port, slot in sorted(initial_slots.items())},
        },
        "end": end_data,
        "checks": checks,
        "failures": failures,
        "rules_passed": rules_passed,
        "outcome": outcome,
        "audit_passed": audit_passed,
        "tournament_result_ready": tournament_result_ready,
    }
    safe_result = _json_safe(result)
    if not isinstance(safe_result, dict):
        raise TypeError("internal replay audit result was not a JSON object")
    return safe_result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _peppi_version(module: Any) -> str | None:
    for distribution in ("peppi-py", "peppi-py-vladfi"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    raw = getattr(module, "__version__", None)
    return str(raw) if raw is not None else None


def audit_replay(
    replay_path: str | Path,
    *,
    expected_stage: Any,
    expected_characters: Mapping[int, Any],
    expected_costumes: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    """Parse and audit one saved ``.slp`` replay.

    ``expected_stage`` accepts a canonical libmelee stage name, integer value,
    or enum.  ``expected_characters`` must map exactly two one-indexed ports to
    canonical libmelee character names, integer values, or enums. Optional
    ``expected_costumes`` maps the same ports to exact Slippi costume indices.
    """
    path = Path(replay_path).expanduser().resolve()
    # Validate caller expectations before loading optional parser dependencies.
    _, _, expected_slots = _expectations(expected_stage, expected_characters)
    _costume_expectations(expected_costumes, expected_slots)
    replay: dict[str, Any] = {
        "path": str(path),
        "sha256": None,
        "raw_byte_length": None,
        "parser": "peppi-py",
        "parser_version": None,
        "parse_status": "error",
        "parse_error": None,
    }
    try:
        replay["raw_byte_length"] = path.stat().st_size
        replay["sha256"] = _sha256(path)
    except OSError as exc:
        replay["parse_error"] = f"{type(exc).__name__}: {exc}"
        return {
            "schema_version": SCHEMA_VERSION,
            "replay": replay,
            "checks": {"replay_parsed": False},
            "failures": [
                {"check": "replay_parsed", "expected": "readable .slp", "actual": replay["parse_error"]}
            ],
            "rules_passed": False,
            "outcome": _outcome(
                "incomplete",
                game_complete=False,
                reason="replay_unreadable",
                evidence=[str(replay["parse_error"])],
            ),
            "audit_passed": False,
            "tournament_result_ready": False,
        }

    try:
        import peppi_py

        replay["parser_version"] = _peppi_version(peppi_py)
        game = peppi_py.read_slippi(str(path))
    except Exception as exc:
        replay["parse_error"] = f"{type(exc).__name__}: {exc}"
        return {
            "schema_version": SCHEMA_VERSION,
            "replay": replay,
            "checks": {"replay_parsed": False},
            "failures": [
                {"check": "replay_parsed", "expected": "valid .slp", "actual": replay["parse_error"]}
            ],
            "rules_passed": False,
            "outcome": _outcome(
                "incomplete",
                game_complete=False,
                reason="replay_parse_failure",
                evidence=[str(replay["parse_error"])],
            ),
            "audit_passed": False,
            "tournament_result_ready": False,
        }

    result = audit_parsed_game(
        game,
        expected_stage=expected_stage,
        expected_characters=expected_characters,
        expected_costumes=expected_costumes,
    )
    replay["parse_status"] = "ok"
    result["replay"] = replay
    parser_exact = replay["parser_version"] == EXPECTED_PEPPI_VERSION
    result["checks"] = {
        "replay_parsed": True,
        "peppi_parser_version_exact": parser_exact,
        **result["checks"],
    }
    if not parser_exact:
        result["failures"].append(
            {
                "check": "peppi_parser_version_exact",
                "expected": EXPECTED_PEPPI_VERSION,
                "actual": replay["parser_version"],
            }
        )
    result["rules_passed"] = bool(result["rules_passed"] and parser_exact)
    result["audit_passed"] = bool(
        result["rules_passed"] and result["outcome"]["game_complete"] and result["outcome"]["conclusive"]
    )
    result["tournament_result_ready"] = bool(
        result["audit_passed"] and result["outcome"]["status"] in {"win", "draw"}
    )
    return result


def _tiebreaker_audit_from_base_audit(base_audit: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a normal replay audit into a non-scoring tiebreaker-settings audit."""

    raw_checks = base_audit.get("checks")
    base_checks = raw_checks if isinstance(raw_checks, Mapping) else {}
    raw_settings = base_audit.get("settings")
    settings = raw_settings if isinstance(raw_settings, Mapping) else {}
    tiebreaker_number = _integer(settings.get("tiebreaker_number"))
    checks = {
        "replay_parsed": base_checks.get("replay_parsed") is True,
        "peppi_parser_version_exact": base_checks.get("peppi_parser_version_exact") is True,
        "positive_tiebreaker_number": tiebreaker_number is not None and tiebreaker_number > 0,
        **{f"shared_rule_{name}": base_checks.get(name) is True for name in _TIEBREAKER_SHARED_RULE_CHECKS},
    }
    failures = [name for name, passed in checks.items() if not passed]
    passed = not failures
    return {
        "schema_version": "integration.replay_tiebreaker.v1",
        "classification": (
            "verified non-scoring sudden-death transition replay"
            if passed
            else "unverified or incomplete non-scoring sudden-death transition replay"
        ),
        "decision": "pass" if passed else "fail",
        "checks": checks,
        "failures": failures,
        "settings": dict(settings),
        "replay": base_audit.get("replay"),
    }


def audit_tiebreaker_replay(
    replay_path: str | Path,
    *,
    expected_stage: Any,
    expected_characters: Mapping[int, Any],
    expected_costumes: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    """Verify an auxiliary replay as a same-rules, positive-number tiebreaker.

    Tiebreaker-specific stock, timer, initial-state, and outcome fields are not
    scored. All shared tournament settings, participants, parser identity, and
    the positive Slippi tiebreaker number remain fail-closed.
    """

    return _tiebreaker_audit_from_base_audit(
        audit_replay(
            replay_path,
            expected_stage=expected_stage,
            expected_characters=expected_characters,
            expected_costumes=expected_costumes,
        )
    )
