from __future__ import annotations

import json
from dataclasses import dataclass, replace
from enum import IntEnum
from pathlib import Path
from typing import Any

import pytest

from melee_policy.integration.replay_result import (
    _tiebreaker_audit_from_base_audit,
    audit_parsed_game,
    audit_replay,
    derive_outcome,
)


class Port(IntEnum):
    P1 = 0
    P2 = 1
    P3 = 2
    P4 = 3


class PlayerType(IntEnum):
    HUMAN = 0
    CPU = 1


class UcfValue(IntEnum):
    UCF = 1
    ARDUINO = 2


class EndMethod(IntEnum):
    UNRESOLVED = 0
    TIME = 1
    GAME = 2
    RESOLVED = 3
    NO_CONTEST = 7


@dataclass(frozen=True)
class Slippi:
    version: tuple[int, int, int]


@dataclass(frozen=True)
class Ucf:
    dash_back: UcfValue | None = UcfValue.UCF
    shield_drop: UcfValue | None = UcfValue.UCF


@dataclass(frozen=True)
class Player:
    port: Port
    character: int = 2
    costume: int = 0
    bitfield: int = 192
    type: PlayerType = PlayerType.HUMAN
    stocks: int = 4
    handicap: int = 9
    offense_ratio: float = 1.0
    defense_ratio: float = 1.0
    model_scale: float = 1.0
    ucf: Ucf | None = Ucf()


@dataclass(frozen=True)
class Match:
    tiebreaker: int = 0


@dataclass(frozen=True)
class Scene:
    major: int = 2
    minor: int = 2


@dataclass(frozen=True)
class Start:
    slippi: Slippi = Slippi((3, 19, 1))
    is_pal: bool | None = False
    is_teams: bool = False
    stage: int = 31
    timer: int = 480
    bitfield: tuple[int, ...] = (50, 1, 134, 76)
    scene: Scene = Scene()
    item_spawn_frequency: int = -1
    item_spawn_bitfield: tuple[int, ...] = (255, 255, 255, 255, 255)
    is_raining_bombs: bool = False
    damage_ratio: float = 1.0
    self_destruct_score: int = -1
    random_seed: int = 123456789
    is_frozen_ps: bool | None = True
    players: tuple[Player, ...] = (Player(Port.P1, costume=0), Player(Port.P2, costume=1))
    match: Match | None = Match()


@dataclass(frozen=True)
class PlayerEnd:
    port: Port
    placement: int


@dataclass(frozen=True)
class End:
    method: EndMethod
    lras_initiator: Port | None = None
    players: tuple[PlayerEnd, ...] | None = (
        PlayerEnd(Port.P1, 0),
        PlayerEnd(Port.P2, 1),
    )


@dataclass(frozen=True)
class Post:
    stocks: list[int | None]
    percent: list[float | None]
    character: list[int | None]


@dataclass(frozen=True)
class Leader:
    post: Post


@dataclass(frozen=True)
class FramePort:
    leader: Leader


@dataclass(frozen=True)
class Frames:
    id: list[int]
    ports: tuple[FramePort, ...]


@dataclass(frozen=True)
class Game:
    start: Start
    end: End | None
    frames: Frames


_NORMAL_END = End(EndMethod.GAME)


def _game(
    *,
    start: Start | None = None,
    end: End | None = _NORMAL_END,
    p1_stocks: int = 2,
    p2_stocks: int = 0,
    p1_percent: float = 40.0,
    p2_percent: float = 120.0,
) -> Game:
    return Game(
        start=start or Start(),
        end=end,
        frames=Frames(
            id=[-123, 100],
            ports=(
                FramePort(Leader(Post([4, p1_stocks], [0.0, p1_percent], [1, 1]))),
                FramePort(Leader(Post([4, p2_stocks], [0.0, p2_percent], [1, 1]))),
            ),
        ),
    )


def _terminal(
    p1_stocks: int,
    p2_stocks: int,
    p1_percent: float = 0.0,
    p2_percent: float = 0.0,
) -> dict[int, dict[str, Any]]:
    return {
        1: {"stocks": p1_stocks, "percent": p1_percent},
        2: {"stocks": p2_stocks, "percent": p2_percent},
    }


def test_valid_rules_and_stock_out_are_json_safe() -> None:
    result = audit_parsed_game(
        _game(),
        expected_stage="BATTLEFIELD",
        expected_characters={1: "FOX", 2: "FOX"},
        expected_costumes={1: 0, 2: 1},
    )

    assert result["rules_passed"] is True
    assert all(result["checks"].values())
    assert result["checks"]["exact_expected_costumes"] is True
    assert result["settings"]["game_random_seed"] == 123456789
    assert result["outcome"] == {
        "status": "win",
        "game_complete": True,
        "conclusive": True,
        "winner_port": 1,
        "loser_port": 2,
        "draw": False,
        "requires_tiebreak": False,
        "reason": "stock_out",
        "evidence": ["Port 2 has zero terminal stocks; port 1 has stocks remaining."],
    }
    assert result["audit_passed"] is True
    assert result["tournament_result_ready"] is True
    json.dumps(result, allow_nan=False)


def test_valid_completed_draw_is_a_tournament_result_without_resampling() -> None:
    result = audit_parsed_game(
        _game(p1_stocks=0, p2_stocks=0),
        expected_stage="BATTLEFIELD",
        expected_characters={1: "FOX", 2: "FOX"},
        expected_costumes={1: 0, 2: 1},
    )

    assert result["rules_passed"] is True
    assert result["audit_passed"] is True
    assert result["outcome"]["status"] == "draw"
    assert result["outcome"]["requires_tiebreak"] is True
    assert result["tournament_result_ready"] is True


def test_rule_checks_fail_closed_on_missing_or_nonstandard_values() -> None:
    invalid_players = (
        replace(Player(Port.P1), type=PlayerType.CPU, stocks=3, handicap=8, offense_ratio=0.9),
        replace(
            Player(Port.P2),
            defense_ratio=0.9,
            model_scale=0.9,
            ucf=Ucf(UcfValue.ARDUINO, None),
        ),
    )
    start = replace(
        Start(),
        slippi=Slippi((3, 17, 9)),
        is_pal=None,
        is_teams=True,
        stage=32,
        timer=479,
        bitfield=(48, 1, 134, 76),
        scene=Scene(major=1, minor=2),
        item_spawn_frequency=0,
        item_spawn_bitfield=(0, 0, 0, 0, 0),
        is_raining_bombs=True,
        damage_ratio=1.2,
        self_destruct_score=0,
        is_frozen_ps=None,
        players=invalid_players,
    )
    result = audit_parsed_game(
        _game(start=start),
        expected_stage="BATTLEFIELD",
        expected_characters={1: "FOX", 2: "MARTH"},
        expected_costumes={1: 1, 2: 0},
    )

    expected_failures = {
        "slippi_version_at_least_3_18",
        "ntsc",
        "singles",
        "four_stocks_each",
        "timer_480_seconds",
        "decreasing_timer_enabled",
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
    }
    assert expected_failures <= {name for name, passed in result["checks"].items() if not passed}
    assert result["rules_passed"] is False
    assert result["audit_passed"] is False
    assert result["tournament_result_ready"] is False


@pytest.mark.parametrize(
    "start",
    [
        replace(Start(), bitfield=(18, 1, 134, 76)),
        replace(
            Start(),
            players=(
                replace(Player(Port.P1), bitfield=193),
                Player(Port.P2),
            ),
        ),
    ],
)
def test_stock_mode_and_player_gameplay_modifiers_fail_closed(start: Start) -> None:
    result = audit_parsed_game(
        _game(start=start),
        expected_stage="BATTLEFIELD",
        expected_characters={1: "FOX", 2: "FOX"},
    )
    assert result["rules_passed"] is False
    assert not (result["checks"]["stock_game_mode"] and result["checks"]["no_player_gameplay_modifiers"])


def test_nonzero_starting_damage_fails_closed() -> None:
    game = _game()
    damaged_frames = replace(
        game.frames,
        ports=(
            FramePort(Leader(Post([4, 2], [12.0, 40.0], [1, 1]))),
            game.frames.ports[1],
        ),
    )
    result = audit_parsed_game(
        replace(game, frames=damaged_frames),
        expected_stage="BATTLEFIELD",
        expected_characters={1: "FOX", 2: "FOX"},
    )
    assert result["checks"]["initial_post_frame_four_stocks_zero_percent"] is False
    assert result["tournament_result_ready"] is False


def test_tiebreak_replay_cannot_be_selected_as_primary_result() -> None:
    result = audit_parsed_game(
        _game(start=replace(Start(), match=Match(tiebreaker=1))),
        expected_stage="BATTLEFIELD",
        expected_characters={1: "FOX", 2: "FOX"},
    )
    assert result["settings"]["sudden_death"] is True
    assert result["checks"]["base_game_not_tiebreaker"] is False
    assert result["tournament_result_ready"] is False


def test_exact_expected_ports_are_checked_independently() -> None:
    players = (Player(Port.P1), Player(Port.P3))
    result = audit_parsed_game(
        _game(start=replace(Start(), players=players)),
        expected_stage="BATTLEFIELD",
        expected_characters={1: "FOX", 2: "FOX"},
    )
    assert result["checks"]["exact_expected_ports"] is False
    assert result["checks"]["exact_expected_characters"] is False


def test_timeout_uses_terminal_stocks_then_percent_not_end_placement() -> None:
    wrong_placement = End(
        EndMethod.TIME,
        players=(PlayerEnd(Port.P1, 1), PlayerEnd(Port.P2, 0)),
    )
    by_stocks = derive_outcome(wrong_placement, _terminal(2, 1, 150.0, 0.0))
    by_percent = derive_outcome(wrong_placement, _terminal(2, 2, 73.08, 136.88))

    assert by_stocks["winner_port"] == 1
    assert by_stocks["reason"] == "timeout_stocks"
    assert by_percent["winner_port"] == 1
    assert by_percent["reason"] == "timeout_percent"


@pytest.mark.parametrize(
    ("end", "terminal", "expected_status", "expected_reason"),
    [
        (
            End(EndMethod.TIME),
            _terminal(2, 2, 50.0, 50.0),
            "draw",
            "timeout_exact_tie",
        ),
        (
            End(EndMethod.GAME),
            _terminal(0, 0, 80.0, 120.0),
            "draw",
            "simultaneous_final_stock_loss",
        ),
        (
            None,
            _terminal(2, 0),
            "incomplete",
            "missing_game_end",
        ),
        (
            End(
                EndMethod.GAME,
                players=(PlayerEnd(Port.P1, 1), PlayerEnd(Port.P2, 0)),
            ),
            _terminal(2, 0),
            "inconclusive",
            "stock_out_placement_conflict",
        ),
    ],
)
def test_draw_incomplete_and_conflicting_results_never_guess(
    end: End | None,
    terminal: dict[int, dict[str, Any]],
    expected_status: str,
    expected_reason: str,
) -> None:
    result = derive_outcome(end, terminal)
    assert result["status"] == expected_status
    assert result["reason"] == expected_reason
    assert result["winner_port"] is None
    if expected_status == "draw":
        assert result["requires_tiebreak"] is True


def test_lras_and_sudden_death_are_explicit() -> None:
    lras = derive_outcome(
        End(
            EndMethod.NO_CONTEST,
            lras_initiator=Port.P2,
            players=(PlayerEnd(Port.P1, 0), PlayerEnd(Port.P2, 1)),
        ),
        _terminal(1, 1),
    )
    sudden_death = derive_outcome(
        End(EndMethod.GAME),
        _terminal(1, 0),
        sudden_death=True,
    )

    assert (lras["winner_port"], lras["loser_port"], lras["reason"]) == (1, 2, "lras_forfeit")
    assert sudden_death["winner_port"] == 1
    assert sudden_death["reason"] == "sudden_death_stock_out"

    audited_lras = audit_parsed_game(
        _game(
            end=End(
                EndMethod.NO_CONTEST,
                lras_initiator=Port.P2,
                players=(PlayerEnd(Port.P1, 0), PlayerEnd(Port.P2, 1)),
            ),
            p1_stocks=1,
            p2_stocks=1,
        ),
        expected_stage="BATTLEFIELD",
        expected_characters={1: "FOX", 2: "FOX"},
    )
    assert audited_lras["outcome"]["reason"] == "lras_forfeit"
    assert audited_lras["checks"]["no_lras_technical_termination"] is False
    assert audited_lras["tournament_result_ready"] is False


def test_tiebreaker_audit_accepts_shared_rules_but_never_scores_auxiliary() -> None:
    base_audit = audit_parsed_game(
        _game(start=Start(match=Match(tiebreaker=1))),
        expected_stage="BATTLEFIELD",
        expected_characters={1: "FOX", 2: "FOX"},
        expected_costumes={1: 0, 2: 1},
    )
    base_audit["checks"] = {
        "replay_parsed": True,
        "peppi_parser_version_exact": True,
        **base_audit["checks"],
    }

    result = _tiebreaker_audit_from_base_audit(base_audit)

    assert result["decision"] == "pass"
    assert result["settings"]["tiebreaker_number"] == 1
    assert result["classification"] == "verified non-scoring sudden-death transition replay"

    base_audit["settings"]["tiebreaker_number"] = 0
    rejected = _tiebreaker_audit_from_base_audit(base_audit)
    assert rejected["decision"] == "fail"
    assert "positive_tiebreaker_number" in rejected["failures"]


def test_missing_replay_returns_structured_failure(tmp_path: Path) -> None:
    result = audit_replay(
        tmp_path / "missing.slp",
        expected_stage="BATTLEFIELD",
        expected_characters={1: "FOX", 2: "FOX"},
    )
    assert result["checks"] == {"replay_parsed": False}
    assert result["rules_passed"] is False
    assert result["outcome"]["status"] == "incomplete"
    assert result["tournament_result_ready"] is False
    json.dumps(result, allow_nan=False)


@pytest.mark.local_replays
def test_local_natural_end_double_ko_is_a_draw_not_a_guessed_winner() -> None:
    pytest.importorskip("peppi_py")
    path = Path(".e003-cache/slippi-js-source/slp/placementsTest/same-frame-death.slp")
    if not path.is_file():
        pytest.skip("ignored slippi-js natural-end replay fixture is not available")

    result = audit_replay(
        path,
        expected_stage="FOUNTAIN_OF_DREAMS",
        expected_characters={1: "FALCO", 2: "LUIGI"},
    )

    assert result["replay"]["parse_status"] == "ok"
    assert result["settings"]["slippi_version"] == [3, 19, 0]
    assert result["end"]["method"] == "GAME"
    assert result["outcome"]["status"] == "draw"
    assert result["outcome"]["reason"] == "simultaneous_final_stock_loss"
    assert result["outcome"]["winner_port"] is None
    assert result["outcome"]["requires_tiebreak"] is True
    assert result["checks"]["frozen_pokemon_stadium"] is False
    assert result["rules_passed"] is False
    assert result["tournament_result_ready"] is False
