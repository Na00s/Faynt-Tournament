from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import melee
import pytest

from melee_policy.e000.enums import CHARACTER_FOLDER_TO_INTERNAL, EXTERNAL_CHARACTER_TO_INTERNAL
from melee_policy.integration import frisson_cpu_match as module
from melee_policy.integration.frisson_cpu_match import (
    CPU_LEVEL,
    FINAL_CPU_CHECKPOINT_IDENTITIES,
    FrissonCpuMatchRequest,
    _cpu_menu_character_for_state,
    _cpu_trace_row,
    _CpuFrameResult,
    _drain_cpu_formal_game_end,
    _run_cpu_frame,
    _validate_cpu_first_context,
    _validate_cpu_replay_start,
    _validate_final_cpu_checkpoint,
)
from melee_policy.integration.frisson_match import (
    FRISSON_SUPPORTED_CHARACTERS,
    _replay_character_expectation,
)
from melee_policy.integration.slippi_ai_policy import CanonicalControllerCommand
from melee_policy.integration.slippi_match import _game_start_transport_proof


def _command() -> CanonicalControllerCommand:
    return CanonicalControllerCommand(
        main_stick=(0.25, 0.75),
        c_stick=(0.5, 0.5),
        analog_l=0.2,
        analog_r=0.0,
        buttons=("A",),
    )


def _exact_game_start_transport_proof() -> dict[str, Any]:
    return _game_start_transport_proof(
        {
            "schema_version": "integration.controller_pipe_lockstep.v7",
            "frame_sync": {"kind_commits": {"internal": 1}},
            "group_commit_reasons": {"console-internal": 1},
            "ports": {
                "p1": {"later_internal_flush_requests": 1},
                "p2": {"later_internal_flush_requests": 1},
            },
            "pending_internal_ports": [],
            "pending_unscoped_ports": [],
            "benchmark_scope": {"sealed_before_shutdown_drain": True},
        }
    )


def _neutral_cpu_audit_trace(
    *,
    first_policy_command: CanonicalControllerCommand | None = None,
) -> list[dict[str, Any]]:
    command = CanonicalControllerCommand.neutral().as_dict()
    player_state = {"action": 322, "stocks": 4, "percent": 0.0}
    rows = [
        {
            "game_frame": frame,
            "slots": {
                "p1": {
                    "model": "frisson-ai",
                    "command": command,
                    "inference": {"native_dummy_prefix": False},
                    "controller_dispatch": {"called": True},
                    "player_state": player_state,
                },
                "p2": {
                    "model": "cpu",
                    "command": command,
                    "controller_dispatch": {"called": True},
                    "player_state": player_state,
                },
            },
        }
        for frame in range(-123, -117)
    ]
    if first_policy_command is not None:
        rows[0]["slots"]["p1"]["command"] = first_policy_command.as_dict()
    return rows


def _neutral_cpu_audit_replay(
    *,
    first_policy_replay_frame: int | None = None,
) -> dict[int, dict[int, dict[str, Any]]]:
    p1 = {
        "buttons_physical": (),
        "buttons_processed": (),
        "raw_main_stick": (0, 0),
        "c_stick": (0.0, 0.0),
        "physical_analog_l": 0.0,
        "physical_analog_r": 0.0,
    }
    replay = {frame: {1: dict(p1), 2: {}} for frame in range(-123, -116)}
    if first_policy_replay_frame is not None:
        replay[first_policy_replay_frame][1] = {
            "buttons_physical": ("A",),
            "buttons_processed": ("A",),
            "raw_main_stick": (-40, 40),
            "c_stick": (0.0, 0.0),
            "physical_analog_l": 0.2,
            "physical_analog_r": 0.0,
        }
    return replay


def _install_cpu_audit_records(
    monkeypatch: pytest.MonkeyPatch,
    trace: list[dict[str, Any]],
    replay: dict[int, dict[int, dict[str, Any]]],
) -> None:
    monkeypatch.setattr(
        "melee_policy.integration.slippi_match._read_trace_rows",
        lambda _path: trace,
    )
    monkeypatch.setattr(
        "melee_policy.integration.slippi_match._read_replay_controller_states",
        lambda _path: replay,
    )
    monkeypatch.setattr(
        module,
        "_file_identity",
        lambda path, _root: {"path": path.name, "sha256": "0" * 64, "byte_length": 1},
    )


class _Session:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def step(self, gamestate: Any) -> CanonicalControllerCommand:
        assert gamestate.frame == -123
        self.events.append("inference-complete")
        return _command()


class _Transport:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def begin_boundary(self, *, reason: str, game_frame: int) -> None:
        assert reason == "frisson-vs-cpu9-gameplay"
        assert game_frame == -123
        self.events.append("boundary-begin")

    def schedule_next_boundary(self, port: int, *, reason: str, game_frame: int) -> None:
        assert game_frame == -123
        assert reason in {
            "frisson-zero-delay-next-frame-command",
            "cpu9-neutral-pipe-frame-sync",
        }
        self.events.append(f"schedule-p{port}")

    def commit_boundary(self) -> None:
        self.events.append("boundary-commit")


class _DrainTransport:
    def __init__(
        self,
        events: list[str],
        steps: list[tuple[Any | None, bool]],
        *,
        transport_gate_passed: bool = True,
    ) -> None:
        self.events = events
        self.steps = list(steps)
        self.transport_gate_passed = transport_gate_passed
        self.console = SimpleNamespace(_events_this_frame=[])

    def seal_benchmark_audit(self) -> tuple[dict[str, Any], dict[str, bool]]:
        self.events.append("seal-game-scope")
        return (
            {"benchmark_scope": {"sealed_before_shutdown_drain": True}},
            {"transport": self.transport_gate_passed},
        )

    def step(self) -> Any | None:
        from melee.slippstream import EventType

        self.events.append("step")
        gamestate, formal_game_end = self.steps.pop(0)
        self.console._events_this_frame = [EventType.GAME_END] if formal_game_end else []
        return gamestate

    def begin_boundary(self, *, reason: str, game_frame: int) -> None:
        assert reason == "cpu9-post-stock-out-formal-game-end-drain"
        self.events.append(f"boundary-begin-{game_frame}")

    def schedule_next_boundary(self, port: int, *, reason: str, game_frame: int) -> None:
        expected_reason = {
            1: "frisson-post-stock-out-neutral-finalization",
            2: "cpu9-post-stock-out-neutral-finalization",
        }
        assert reason == expected_reason[port]
        self.events.append(f"schedule-p{port}-{game_frame}")

    def commit_boundary(self) -> None:
        self.events.append("boundary-commit")


def _player(*, character: str = "FOX", stock: int = 4, cpu_level: int = 0) -> Any:
    return SimpleNamespace(
        character=SimpleNamespace(name=character),
        action=SimpleNamespace(value=14),
        stock=stock,
        percent=12.5,
        position=SimpleNamespace(x=1.0, y=2.0),
        costume=0,
        cpu_level=cpu_level,
    )


def test_request_defaults_to_frisson_fox_vs_native_cpu9_fox() -> None:
    request = FrissonCpuMatchRequest()
    request.validate()
    assert request.player_1_model == "frisson-ai"
    assert request.player_2_model == "cpu"
    assert request.cpu_level == CPU_LEVEL == 9
    assert request.require_natural_end


@pytest.mark.parametrize("character", FRISSON_SUPPORTED_CHARACTERS)
def test_request_accepts_each_same_character_cpu9_mirror(character: str) -> None:
    request = FrissonCpuMatchRequest(
        player_1_character=character,
        player_2_character=character,
    )
    request.validate()
    assert request.player_1_character == request.player_2_character == character


@pytest.mark.parametrize(
    "override",
    [
        {"player_1_model": "mimic"},
        {"player_2_model": "mimic"},
        {"stage": "BATTLEFIELD"},
        {"cpu_level": 8},
        {"require_natural_end": False},
    ],
)
def test_request_rejects_changes_to_the_cpu9_benchmark_contract(
    override: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="fixed match contract mismatch"):
        FrissonCpuMatchRequest(**override).validate()


def test_request_rejects_a_nonmirror_character_assignment() -> None:
    with pytest.raises(ValueError, match="same-character mirror"):
        FrissonCpuMatchRequest(
            player_1_character="FOX",
            player_2_character="MARTH",
        ).validate()


@pytest.mark.parametrize("character", ["NANA", "ICE_CLIMBERS", "fox", ""])
def test_request_rejects_nonlaunchable_character_aliases(character: str) -> None:
    with pytest.raises(ValueError, match="launchable character"):
        FrissonCpuMatchRequest(
            player_1_character=character,
            player_2_character=character,
        ).validate()


@pytest.mark.parametrize(
    ("sha256", "expected"),
    tuple(FINAL_CPU_CHECKPOINT_IDENTITIES.items()),
)
def test_cpu_runner_accepts_each_frozen_pretraining_and_posttraining_winner(
    tmp_path: Path,
    sha256: str,
    expected: dict[str, Any],
) -> None:
    identity = {
        "sha256": sha256,
        "path": str((tmp_path / str(expected["relative_path"])).resolve()),
        "all_state_tensors_finite": True,
        **{name: value for name, value in expected.items() if name != "relative_path"},
    }
    selected = _validate_final_cpu_checkpoint(identity, tmp_path)
    assert selected["sha256"] == sha256
    assert selected["profile"] in {"10m", "75m"}
    assert selected["format"] == expected["format"]
    assert selected["step"] == expected["step"]


def test_cpu_runner_rejects_every_checkpoint_outside_the_four_frozen_final_winners(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="one exact frozen final-winner checkpoint"):
        _validate_final_cpu_checkpoint(
            {
                "sha256": "a" * 64,
                "path": str((tmp_path / "legacy-20m.pt").resolve()),
            },
            tmp_path,
        )


@pytest.mark.parametrize("character", FRISSON_SUPPORTED_CHARACTERS)
def test_cpu_css_selection_preserves_each_character_with_the_sheik_transform(
    character: str,
) -> None:
    css = SimpleNamespace(menu_state=melee.Menu.CHARACTER_SELECT)
    selected = _cpu_menu_character_for_state(melee, css, character)
    expected = melee.Character.ZELDA if character == "SHEIK" else melee.Character[character]
    assert selected is expected

    stage = SimpleNamespace(menu_state=melee.Menu.STAGE_SELECT)
    assert _cpu_menu_character_for_state(melee, stage, character) is melee.Character[character]


def test_cpu_frame_infers_frisson_then_commits_p1_and_neutral_p2_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    dispatched: list[tuple[object, CanonicalControllerCommand, bool]] = []
    p1 = object()
    p2 = object()

    def fake_send(
        controller: object,
        command: CanonicalControllerCommand,
        *,
        flush: bool,
    ) -> Any:
        dispatched.append((controller, command, flush))
        events.append("dispatch-p1" if controller is p1 else "dispatch-p2")
        return SimpleNamespace(as_dict=lambda: {"canonical": command.as_dict()})

    monkeypatch.setattr(module, "send_canonical_controller", fake_send)
    result = _run_cpu_frame(
        gamestate=SimpleNamespace(frame=-123),
        frisson_session=_Session(events),  # type: ignore[arg-type]
        frisson_controller=p1,
        cpu_pipe_controller=p2,
        transport=_Transport(events),  # type: ignore[arg-type]
    )

    assert events == [
        "inference-complete",
        "boundary-begin",
        "dispatch-p1",
        "schedule-p1",
        "dispatch-p2",
        "schedule-p2",
        "boundary-commit",
    ]
    assert dispatched[0] == (p1, _command(), False)
    assert dispatched[1] == (p2, CanonicalControllerCommand.neutral(), False)
    assert result.cpu_pipe_dispatch["neutral"] is True
    assert result.cpu_pipe_dispatch["gameplay_owner"] == "melee-native-cpu"


def test_formal_game_end_probe_reads_libmelee_event_ledger() -> None:
    from melee.slippstream import EventType

    console = SimpleNamespace(_events_this_frame=[])
    assert module._console_observed_formal_game_end(console) is False
    console._events_this_frame = [EventType.POST_FRAME, EventType.GAME_END]
    assert module._console_observed_formal_game_end(console) is True


def test_post_stock_out_drain_accepts_formal_game_end_on_first_step_without_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    transport = _DrainTransport(events, [(None, True)])
    monkeypatch.setattr(
        module,
        "send_canonical_controller",
        lambda *_args, **_kwargs: pytest.fail("formal Game End required no extra dispatch"),
    )

    record = _drain_cpu_formal_game_end(
        transport=transport,  # type: ignore[arg-type]
        frisson_controller=object(),
        cpu_pipe_controller=object(),
        terminal_policy_frame=907,
        in_game_menu_states=("IN_GAME", "SUDDEN_DEATH"),
        timeout_seconds=5.0,
    )

    assert events == ["seal-game-scope", "step"]
    assert record["decision"] == "pass"
    assert record["formal_game_end_observed"] is True
    assert record["neutral_boundaries_started"] == 0
    assert record["neutral_boundaries_committed"] == 0
    assert record["neutral_dispatches_by_port"] == {"p1": 0, "p2": 0}
    assert record["policy_inference_calls"] == 0
    assert record["controller_trace_rows_written"] == 0
    assert record["game_scope_transport_audit_sealed_before_drain"] is True
    assert record["game_scope_transport_gate_passed_before_drain"] is True


def test_post_stock_out_drain_uses_only_paired_neutral_commands_until_game_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    state = SimpleNamespace(menu_state="IN_GAME", frame=908)
    transport = _DrainTransport(events, [(state, False), (None, True)])
    p1 = object()
    p2 = object()
    dispatches: list[tuple[Any, CanonicalControllerCommand, bool]] = []

    def fake_send(
        controller: Any,
        command: CanonicalControllerCommand,
        *,
        flush: bool,
    ) -> None:
        dispatches.append((controller, command, flush))
        events.append("dispatch-p1" if controller is p1 else "dispatch-p2")

    monkeypatch.setattr(module, "send_canonical_controller", fake_send)
    record = _drain_cpu_formal_game_end(
        transport=transport,  # type: ignore[arg-type]
        frisson_controller=p1,
        cpu_pipe_controller=p2,
        terminal_policy_frame=907,
        in_game_menu_states=("IN_GAME", "SUDDEN_DEATH"),
        timeout_seconds=5.0,
    )

    assert events == [
        "seal-game-scope",
        "step",
        "boundary-begin-908",
        "dispatch-p1",
        "schedule-p1-908",
        "dispatch-p2",
        "schedule-p2-908",
        "boundary-commit",
        "step",
    ]
    assert dispatches == [
        (p1, CanonicalControllerCommand.neutral(), False),
        (p2, CanonicalControllerCommand.neutral(), False),
    ]
    assert record["decision"] == "pass"
    assert record["step_calls"] == 2
    assert record["returned_states"] == 1
    assert record["neutral_boundaries_started"] == 1
    assert record["neutral_boundaries_committed"] == 1
    assert record["neutral_dispatches_by_port"] == {"p1": 1, "p2": 1}
    assert record["last_returned_game_frame"] == 908
    assert record["policy_inference_calls"] == 0
    assert record["controller_trace_rows_written"] == 0


def test_post_stock_out_drain_fails_closed_if_menu_arrives_without_game_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    state = SimpleNamespace(menu_state="CHARACTER_SELECT", frame=908)
    transport = _DrainTransport(events, [(state, False)])
    monkeypatch.setattr(
        module,
        "send_canonical_controller",
        lambda *_args, **_kwargs: pytest.fail("invalid menu transition must not dispatch"),
    )

    record = _drain_cpu_formal_game_end(
        transport=transport,  # type: ignore[arg-type]
        frisson_controller=object(),
        cpu_pipe_controller=object(),
        terminal_policy_frame=907,
        in_game_menu_states=("IN_GAME", "SUDDEN_DEATH"),
        timeout_seconds=5.0,
    )

    assert record["decision"] == "fail"
    assert record["reason"] == "left-gameplay-without-formal-game-end-event"
    assert record["formal_game_end_observed"] is False
    assert record["neutral_boundaries_committed"] == 0


def test_post_stock_out_drain_refuses_to_advance_after_failed_transport_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    transport = _DrainTransport(
        events,
        [(None, True)],
        transport_gate_passed=False,
    )
    monkeypatch.setattr(
        module,
        "send_canonical_controller",
        lambda *_args, **_kwargs: pytest.fail("failed transport gate must not dispatch"),
    )

    record = _drain_cpu_formal_game_end(
        transport=transport,  # type: ignore[arg-type]
        frisson_controller=object(),
        cpu_pipe_controller=object(),
        terminal_policy_frame=907,
        in_game_menu_states=("IN_GAME", "SUDDEN_DEATH"),
        timeout_seconds=5.0,
    )

    assert events == ["seal-game-scope"]
    assert record["decision"] == "fail"
    assert record["reason"] == "game-scope-controller-transport-gate-failed-before-drain"
    assert record["game_scope_transport_gate_passed_before_drain"] is False


def test_post_stock_out_drain_has_a_hard_step_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    transport = _DrainTransport(events, [(None, False), (None, False)])
    monkeypatch.setattr(module, "POST_STOCK_OUT_DRAIN_MAX_STEP_CALLS", 2)

    record = _drain_cpu_formal_game_end(
        transport=transport,  # type: ignore[arg-type]
        frisson_controller=object(),
        cpu_pipe_controller=object(),
        terminal_policy_frame=907,
        in_game_menu_states=("IN_GAME", "SUDDEN_DEATH"),
        timeout_seconds=5.0,
        clock=lambda: 0.0,
    )

    assert record["decision"] == "fail"
    assert record["reason"] == "formal-game-end-drain-step-limit"
    assert record["step_calls"] == 2
    assert record["none_step_results"] == 2


def test_post_stock_out_drain_records_a_step_exception_as_failed_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    transport = _DrainTransport(events, [])

    def fail_step() -> None:
        events.append("step-failed")
        raise RuntimeError("injected step failure")

    monkeypatch.setattr(transport, "step", fail_step)
    record = _drain_cpu_formal_game_end(
        transport=transport,  # type: ignore[arg-type]
        frisson_controller=object(),
        cpu_pipe_controller=object(),
        terminal_policy_frame=907,
        in_game_menu_states=("IN_GAME", "SUDDEN_DEATH"),
        timeout_seconds=5.0,
    )

    assert events == ["seal-game-scope", "step-failed"]
    assert record["attempted"] is True
    assert record["decision"] == "fail"
    assert record["reason"] == "formal-game-end-drain-exception"
    assert record["failure_phase"] == "console-step"
    assert record["error"] == "RuntimeError: injected step failure"
    assert record["step_calls"] == 0
    assert record["formal_game_end_observed"] is False


def test_post_stock_out_drain_records_an_uncommitted_neutral_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    state = SimpleNamespace(menu_state="IN_GAME", frame=908)
    transport = _DrainTransport(events, [(state, False)])

    def fake_send(
        controller: Any,
        command: CanonicalControllerCommand,
        *,
        flush: bool,
    ) -> None:
        assert command == CanonicalControllerCommand.neutral()
        assert flush is False
        events.append("dispatch")

    def fail_commit() -> None:
        events.append("boundary-commit-failed")
        raise RuntimeError("injected commit failure")

    monkeypatch.setattr(module, "send_canonical_controller", fake_send)
    monkeypatch.setattr(transport, "commit_boundary", fail_commit)
    record = _drain_cpu_formal_game_end(
        transport=transport,  # type: ignore[arg-type]
        frisson_controller=object(),
        cpu_pipe_controller=object(),
        terminal_policy_frame=907,
        in_game_menu_states=("IN_GAME", "SUDDEN_DEATH"),
        timeout_seconds=5.0,
    )

    assert record["attempted"] is True
    assert record["decision"] == "fail"
    assert record["failure_phase"] == "neutral-boundary-commit"
    assert record["error"] == "RuntimeError: injected commit failure"
    assert record["neutral_boundaries_started"] == 1
    assert record["neutral_boundaries_committed"] == 0
    assert record["neutral_dispatches_by_port"] == {"p1": 1, "p2": 1}
    assert events[-1] == "boundary-commit-failed"


def test_cpu_controller_audit_uses_exact_piecewise_game_start_alignment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace = _neutral_cpu_audit_trace()
    replay = _neutral_cpu_audit_replay()
    _install_cpu_audit_records(monkeypatch, trace, replay)

    audit = module._audit_frisson_only_controller_boundary(
        tmp_path / "trace.jsonl",
        tmp_path / "game.slp",
        tmp_path,
        game_start_transport_proof=_exact_game_start_transport_proof(),
    )

    assert audit["schema_version"] == "integration.frisson_vs_cpu9.controller_audit.v3"
    assert audit["gate"]["decision"] == "pass"
    assert all(audit["gate"]["checks"].values())
    assert audit["alignment"]["mode"] == "piecewise-game-start"
    assert audit["alignment"]["startup_mapped_pairs"] == [
        [-123, -121],
        [-122, -120],
        [-121, -119],
        [-120, -118],
    ]
    assert audit["alignment"]["game_start_controller_latch_unobservable_trace_frames"] == [-119]
    assert audit["alignment"]["first_aligned_pair"] == [-123, -121]
    assert audit["alignment"]["first_policy_trace_to_replay_lag_frames"] == 2
    assert (
        audit["alignment"]["first_policy_frame_mapping_evidence"]["decision"]
        == "observationally-equivalent-delayed-first"
    )
    assert audit["alignment"]["last_aligned_pair"] == [-118, -117]
    assert audit["slots"]["p1"]["controller_frame_pairs_compared"] == 5
    assert audit["slots"]["p1"]["all_mapped_commands_hard_gated"] is True
    assert set(audit["slots"]) == {"p1"}
    assert audit["excluded_slots"]["p2"]["model"] == "cpu"


@pytest.mark.parametrize(
    ("first_policy_replay_frame", "expected_decision", "expected_lag"),
    [
        (-122, "early-latched", 1),
        (-121, "delayed-first", 2),
    ],
)
def test_cpu_controller_audit_evidence_selects_each_first_policy_latch_variant(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    first_policy_replay_frame: int,
    expected_decision: str,
    expected_lag: int,
) -> None:
    trace = _neutral_cpu_audit_trace(first_policy_command=_command())
    replay = _neutral_cpu_audit_replay(
        first_policy_replay_frame=first_policy_replay_frame,
    )
    _install_cpu_audit_records(monkeypatch, trace, replay)

    audit = module._audit_frisson_only_controller_boundary(
        tmp_path / "trace.jsonl",
        tmp_path / "game.slp",
        tmp_path,
        game_start_transport_proof=_exact_game_start_transport_proof(),
    )

    assert audit["gate"]["decision"] == "pass"
    assert all(audit["gate"]["checks"].values())
    assert audit["alignment"]["first_aligned_pair"] == [
        -123,
        first_policy_replay_frame,
    ]
    assert audit["alignment"]["startup_mapped_pairs"][0] == [
        -123,
        first_policy_replay_frame,
    ]
    assert audit["alignment"]["first_policy_trace_to_replay_lag_frames"] == expected_lag
    evidence = audit["alignment"]["first_policy_frame_mapping_evidence"]
    assert evidence["decision"] == expected_decision
    assert evidence["selected_replay_frame"] == first_policy_replay_frame
    assert evidence["evidence_ports"] == [1]
    assert evidence["candidate_matches"][expected_decision]["p1"] is True
    assert audit["slots"]["p1"]["intended_raw_main_stick"]["mismatch_components"] == 0


def test_cpu_controller_audit_hard_gates_each_mapped_frisson_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace = _neutral_cpu_audit_trace()
    replay = _neutral_cpu_audit_replay()
    replay[-120][1]["buttons_physical"] = ("A",)
    replay[-120][1]["buttons_processed"] = ("A",)
    _install_cpu_audit_records(monkeypatch, trace, replay)

    audit = module._audit_frisson_only_controller_boundary(
        tmp_path / "trace.jsonl",
        tmp_path / "game.slp",
        tmp_path,
        game_start_transport_proof=_exact_game_start_transport_proof(),
    )

    assert audit["gate"]["decision"] == "fail"
    assert audit["gate"]["checks"]["frisson_physical_buttons_exact"] is False
    mismatch = audit["slots"]["p1"]["digital_buttons"]["physical"]["mismatches"]
    assert mismatch == [
        {
            "trace_frame": -122,
            "replay_frame": -120,
            "expected": [],
            "observed": ["A"],
        }
    ]


def test_cpu_controller_audit_fails_closed_without_sealed_transport_proof(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace = _neutral_cpu_audit_trace()
    replay = _neutral_cpu_audit_replay()
    _install_cpu_audit_records(monkeypatch, trace, replay)
    failed_proof = _exact_game_start_transport_proof()
    failed_proof["decision"] = "fail"
    failed_proof["checks"]["sealed_before_shutdown_drain"] = False

    audit = module._selected_frisson_cpu_audit(
        tmp_path / "trace.jsonl",
        [tmp_path / "game.slp"],
        [{"tournament_result_replay": True}],
        tmp_path,
        game_start_transport_proof=failed_proof,
    )

    assert audit["schema_version"] == "integration.frisson_vs_cpu9.controller_audit.v3"
    assert audit["gate"]["decision"] == "fail"
    assert all(value is False for value in audit["gate"]["checks"].values())
    assert "lacks one exact paired internal commit" in audit["error"]
    assert audit["alignment"]["game_start_transport_proof"] == failed_proof


@pytest.mark.parametrize("character", FRISSON_SUPPORTED_CHARACTERS)
def test_first_context_requires_each_mirror_character_and_live_cpu_level_9(
    character: str,
) -> None:
    gamestate = SimpleNamespace(
        frame=-123,
        stage=SimpleNamespace(name="FINAL_DESTINATION"),
        players={
            1: _player(character=character, cpu_level=0),
            2: _player(character=character, cpu_level=9),
        },
    )
    context = _validate_cpu_first_context(gamestate, character=character)
    assert context["cpu_levels"] == {"p1": 0, "p2": 9}
    assert context["characters"] == {"p1": character, "p2": character}
    assert all(context["checks"].values())

    gamestate.players[2].cpu_level = 8
    with pytest.raises(RuntimeError, match="game context mismatch"):
        _validate_cpu_first_context(gamestate, character=character)


@pytest.mark.parametrize("character", FRISSON_SUPPORTED_CHARACTERS)
def test_cpu_trace_identifies_each_mirror_and_neutral_pipe_role(character: str) -> None:
    gamestate = SimpleNamespace(
        frame=-123,
        players={
            1: _player(character=character),
            2: _player(character=character, stock=3, cpu_level=9),
        },
    )
    result = _CpuFrameResult(
        game_frame=-123,
        frisson_command=_command(),
        frisson_dispatch={"called": True},
        frisson_inference_seconds=0.01,
        cpu_pipe_dispatch={
            "called": True,
            "neutral": True,
            "gameplay_owner": "melee-native-cpu",
        },
        barrier_seconds=0.02,
    )
    row = _cpu_trace_row(gamestate, result, character=character)
    assert row["slots"]["p1"]["model"] == "frisson-ai"
    assert row["slots"]["p1"]["requested_character"] == character
    assert row["slots"]["p2"]["model"] == "cpu"
    assert row["slots"]["p2"]["requested_character"] == character
    assert row["slots"]["p2"]["inference"] == {
        "called": False,
        "owner": "melee-native-cpu",
        "cpu_level": 9,
    }
    assert row["slots"]["p2"]["command"] == CanonicalControllerCommand.neutral().as_dict()


@pytest.mark.parametrize("character", FRISSON_SUPPORTED_CHARACTERS)
def test_replay_start_contract_requires_each_mirror_cpu_type_and_level_9(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    character: str,
) -> None:
    class Type:
        def __init__(self, name: str) -> None:
            self.name = name

    class Port:
        def __init__(self, value: int) -> None:
            self.value = value

    replay_character = _replay_character_expectation(character)
    internal_id = CHARACTER_FOLDER_TO_INTERNAL[replay_character][0]
    external_id = next(
        external for external, internal in EXTERNAL_CHARACTER_TO_INTERNAL.items() if internal == internal_id
    )
    players = (
        SimpleNamespace(port=Port(0), type=Type("HUMAN"), cpu_level=None, character=external_id),
        SimpleNamespace(port=Port(1), type=Type("CPU"), cpu_level=9, character=external_id),
    )
    end_players = (
        SimpleNamespace(port=Port(0), placement=0),
        SimpleNamespace(port=Port(1), placement=1),
    )
    monkeypatch.setattr(
        "peppi_py.read_slippi",
        lambda _path: SimpleNamespace(
            start=SimpleNamespace(players=players, stage=32, random_seed=1234),
            end=SimpleNamespace(players=end_players, method=Type("GAME")),
        ),
    )
    observed_expectations: dict[str, Any] = {}

    def audit_parsed_game(_game: Any, **expectations: Any) -> dict[str, Any]:
        observed_expectations.update(expectations)
        return {
            "outcome": {
                "status": "win",
                "game_complete": True,
                "conclusive": True,
                "winner_port": 1,
            },
            "terminal": {
                "frame_id": 123,
                "slots": {
                    "1": {"stocks": 2, "percent": 73.0},
                    "2": {"stocks": 0, "percent": 120.0},
                },
            },
        }

    monkeypatch.setattr(
        "melee_policy.integration.replay_result.audit_parsed_game",
        audit_parsed_game,
    )
    contract = _validate_cpu_replay_start(tmp_path / "game.slp", character=character)
    assert contract["decision"] == "pass"
    assert all(contract["checks"].values())
    assert contract["requested_character"] == character
    assert contract["replay_character_name"] == replay_character
    assert observed_expectations["expected_characters"] == {
        1: replay_character,
        2: replay_character,
    }
    assert contract["players"]["p2"]["type"] == "CPU"
    assert contract["players"]["p2"]["cpu_level"] == 9
    assert contract["stage"] == {
        "raw_slippi_id": 32,
        "libmelee_id": 25,
        "name": "FINAL_DESTINATION",
    }
    assert contract["game_random_seed"] == 1234
    assert contract["end"]["winner_port"] == 1
    assert contract["outcome"]["winner_port"] == 1

    players[1].cpu_level = 8
    contract = _validate_cpu_replay_start(tmp_path / "game.slp", character=character)
    assert contract["decision"] == "fail"
    assert contract["checks"]["p2_cpu_level_9"] is False
