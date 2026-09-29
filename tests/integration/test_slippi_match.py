from __future__ import annotations
import copy
import json
import signal
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import pytest
import torch
from melee_policy.integration import slippi_match as slippi_match_module
from melee_policy.integration.match_runtime import (
    MimicLivePolicy,
    _load_config,
    _require_exact_player_ports,
    _stop_console,
)
from melee_policy.integration.slippi_ai_policy import (
    MEDIUM_V2_SUPPORTED_CHARACTERS,
    POLICY_DELAY_FRAMES,
    UPSTREAM_EVAL_TWO_CONSOLE_DELAY_FRAMES,
    CanonicalControllerCommand,
)
from melee_policy.integration.slippi_match import (
    CONTROLLER_REPLAY_LAG_FRAMES,
    FIRST_GAMEPLAY_CONTROLLER_LATCH_UNOBSERVABLE_FRAMES,
    FIRST_POLICY_FRAME,
    GAME_START_CONTROLLER_LATCH_TRACE_FRAME,
    GAME_START_CONTROLLER_LATCH_UNOBSERVABLE_FRAMES,
    LAUNCHABLE_PRIMARY_CHARACTERS,
    MIXED_EFFECTIVE_POLICY_DELAY_FRAMES,
    MIXED_RUNTIME_CONSOLE_DELAY_FRAMES,
    SCHEMA_VERSION,
    SlippiMatchRequest,
    _activate_windowed_source,
    _audit_controller_boundary_candidate,
    _audit_controller_boundary_records,
    _audit_mixed_controller_boundary_candidates,
    _build_trace_row,
    _dispatch_slippi_command,
    _drain_formal_game_end,
    _game_start_transport_proof,
    _load_windowed_runtime,
    _make_windowed_policy,
    _normalize_trace_command,
    _observe_windowed_policy,
    _observed_match_context,
    _read_trace_rows,
    _selected_replay_result_binding,
    _slippi_policy_config,
    _validate_character_contracts,
    _validate_config_contract,
    _windowed_exact_timing_checks,
)


@pytest.mark.parametrize("complete", [True, False])
def test_replay_collector_binds_coverage_to_observed_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, complete: bool
) -> None:
    replay = tmp_path / "game.slp"
    replay.write_bytes(b"retained replay")
    calls = []
    validation = {"required_trace_coverage": {"complete": complete}}

    def validate(path: Path, **kwargs: Any) -> dict[str, Any]:
        calls.append((path, kwargs))
        return validation

    monkeypatch.setattr(slippi_match_module, "_validate_saved_replay", validate)
    records = slippi_match_module._collect_replay_records(
        [replay], tmp_path, -123, 10926
    )
    assert calls == [
        (replay, {"required_first_frame": -123, "required_last_frame": 10926})
    ]
    assert records[0]["validation"] == validation
    assert records[0]["byte_length"] == len(b"retained replay")
    assert records[0]["sha256"] == slippi_match_module._sha256_file(replay)


@pytest.mark.parametrize("bounds", [(None, None), (-123, None), (None, 100)])
def test_replay_collector_preserves_startup_failure_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bounds: tuple[int | None, int | None],
) -> None:
    replay = tmp_path / "startup.slp"
    replay.write_bytes(b"startup")
    calls = []

    def validate(path: Path, **kwargs: Any) -> dict[str, Any]:
        calls.append((path, kwargs))
        return {"required_trace_coverage": None}

    monkeypatch.setattr(slippi_match_module, "_validate_saved_replay", validate)
    records = slippi_match_module._collect_replay_records([replay], tmp_path, *bounds)
    assert calls == [(replay, {})]
    assert records[0]["validation"]["required_trace_coverage"] is None


def _reproducibility_fixture() -> dict[str, Any]:

    def identity(path: str) -> dict[str, Any]:
        return {"path": path, "sha256": "a" * 64, "byte_length": 1}

    return {
        "implementation_files": [
            identity(path) for path in slippi_match_module.RUNTIME_IMPLEMENTATION_PATHS
        ],
        "configuration": identity("configs/integration.toml"),
        "dependency_lock": identity("requirements-e010.lock"),
    }


def test_reproducibility_gate_matches_the_full_launch_input_manifest() -> None:
    record = _reproducibility_fixture()
    assert len(record["implementation_files"]) == len(
        slippi_match_module.RUNTIME_IMPLEMENTATION_PATHS
    )
    assert slippi_match_module._reproducibility_inputs_hashed(record)
    record["implementation_files"].reverse()
    assert slippi_match_module._reproducibility_inputs_hashed(record)


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "duplicate", "wrong-path", "old-eleven"]
)
def test_reproducibility_gate_rejects_inexact_input_coverage(mutation: str) -> None:
    record = _reproducibility_fixture()
    files = record["implementation_files"]
    if mutation == "missing":
        files.pop()
    elif mutation == "extra":
        files.append(copy.deepcopy(files[0]))
    elif mutation == "duplicate":
        files[-1] = copy.deepcopy(files[0])
    elif mutation == "wrong-path":
        files[-1]["path"] = "unrelated.py"
    else:
        del files[11:]
    assert not slippi_match_module._reproducibility_inputs_hashed(record)


@pytest.mark.parametrize(
    "field,value",
    [
        ("sha256", None),
        ("sha256", "x" * 64),
        ("sha256", "a" * 63),
        ("byte_length", -1),
        ("byte_length", True),
    ],
)
@pytest.mark.parametrize(
    "target", ["implementation", "configuration", "dependency_lock"]
)
def test_reproducibility_gate_rejects_unhashed_inputs(
    field: str, value: object, target: str
) -> None:
    record = _reproducibility_fixture()
    identity = (
        record["implementation_files"][0]
        if target == "implementation"
        else record[target]
    )
    identity[field] = value
    assert not slippi_match_module._reproducibility_inputs_hashed(record)


def test_release_card_p1_specialists_keep_their_exact_fifo(tmp_path: Path) -> None:
    config = _config()
    config["slippi_ai"].update(
        {
            "source_directory": "pinned-source",
            "checkpoint": "default-medium-v2",
            "default_name": "Master Player",
            "sample_temperature": 1.0,
        }
    )
    request = SlippiMatchRequest(
        player_1_model="slippi-ai",
        player_2_model="mimic",
        player_1_character="DK",
        player_2_character="DK",
        player_1_slippi_release="dk_d18_imitation_v2",
    )
    contract = _validate_config_contract(config, request)
    policy = _slippi_policy_config(config, tmp_path, request)
    characters = _validate_character_contracts(
        request, SimpleNamespace(controlled_character="DK")
    )
    assert contract["release_contract"]["key"] == "dk_d18_imitation_v2"
    assert contract["checkpoint_delay_frames"] == 18
    assert contract["effective_policy_delay_frames"] == 18
    assert policy.release == "dk_d18_imitation_v2"
    assert policy.policy_delay_frames == 18
    assert policy.console_delay_frames == 0
    assert (
        policy.checkpoint_path
        == (tmp_path / ".e001-cache/slippi-ai/models/dk_d18_imitation_v2").resolve()
    )
    assert characters["p1"]["requested_character_covered_by_training"] is True
    assert characters["p1"]["forced_ood_character_transfer"] is False


def test_release_card_double_ood_requires_independent_physical_port_flags() -> None:
    runtime = SimpleNamespace(controlled_character="FOX")
    shared = {
        "player_1_model": "slippi-ai",
        "player_2_model": "mimic",
        "player_1_character": "KIRBY",
        "player_2_character": "KIRBY",
        "player_2_checkpoint": Path("fox-master/model.pt"),
        "player_2_assets": Path("fox-master"),
    }
    with pytest.raises(ValueError, match="Slippi-AI does not support"):
        _validate_character_contracts(
            SlippiMatchRequest(**shared, allow_player_2_ood_character=True), runtime
        )
    with pytest.raises(ValueError, match="MIMIC requires FOX"):
        _validate_character_contracts(
            SlippiMatchRequest(**shared, allow_player_1_ood_character=True), runtime
        )
    request = SlippiMatchRequest(
        **shared,
        allow_player_1_ood_character=True,
        allow_player_2_ood_character=True,
        require_formal_game_end=True,
    )
    contracts = _validate_character_contracts(request, runtime)
    assert contracts["p1"]["forced_ood_character_transfer"] is True
    assert contracts["p1"]["ood_provenance"]["physical_port"] == 1
    assert contracts["p2"]["forced_ood_character_transfer"] is True
    assert contracts["p2"]["ood_provenance"]["physical_port"] == 2


def test_mimic_p2_ood_requires_explicit_checkpoint_and_assets(tmp_path: Path) -> None:
    request = SlippiMatchRequest(
        player_1_character="KIRBY",
        player_2_character="KIRBY",
        allow_player_1_ood_character=True,
        allow_player_2_ood_character=True,
    )
    with pytest.raises(ValueError, match="explicit checkpoint and asset-directory"):
        _load_windowed_runtime(_config(), tmp_path, request)


def test_mimic_native_p2_resolves_its_matching_current_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    config["mimic"] = {"character": "FOX"}
    captured: dict[str, Any] = {}

    def fake_load(
        _config_value: dict[str, Any],
        _root: Path,
        checkpoint_override: Path | None = None,
        asset_directory_override: Path | None = None,
    ) -> Any:
        captured["checkpoint"] = checkpoint_override
        captured["assets"] = asset_directory_override
        return SimpleNamespace(controlled_character="DK")

    monkeypatch.setattr(slippi_match_module, "load_mimic_runtime", fake_load)
    request = SlippiMatchRequest(
        player_1_character="DK",
        player_2_character="DK",
        player_1_slippi_release="dk_d18_imitation_v2",
    )
    _load_windowed_runtime(config, tmp_path, request)
    expected_assets = tmp_path / ".e012-cache/mimic-native-0629eb17/dk"
    assert captured["assets"] == expected_assets.resolve()
    assert captured["checkpoint"] == (expected_assets / "model.pt").resolve()


def test_selected_replay_result_binding_uses_replay_winner_and_live_stocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from melee_policy.integration import replay_result

    replay_path = tmp_path / "game.slp"
    replay_path.write_bytes(b"replay")
    record = {
        "path": "game.slp",
        "byte_length": 6,
        "sha256": "bound-sha256",
        "tournament_result_replay": True,
    }
    outcome = {
        "status": "win",
        "game_complete": True,
        "conclusive": True,
        "winner_port": 2,
        "loser_port": 1,
        "draw": False,
        "requires_tiebreak": False,
        "reason": "stock_out",
        "evidence": ["port 1 has zero stocks"],
    }

    def audit_replay(
        path: Path, *, expected_stage: str, expected_characters: dict[int, str]
    ) -> dict[str, Any]:
        assert path == replay_path
        assert expected_stage == "FINAL_DESTINATION"
        assert expected_characters == {1: "ICE_CLIMBERS", 2: "ICE_CLIMBERS"}
        return {
            "replay": {"sha256": "bound-sha256", "raw_byte_length": 6},
            "audit_passed": True,
            "tournament_result_ready": True,
            "outcome": outcome,
        }

    monkeypatch.setattr(replay_result, "audit_replay", audit_replay)
    binding = _selected_replay_result_binding(
        [replay_path],
        [record],
        0,
        tmp_path,
        SlippiMatchRequest(
            player_1_character="POPO",
            player_2_character="POPO",
            stage="FINAL_DESTINATION",
        ),
        {"p1": 0, "p2": 2},
    )
    assert binding["decision"] == "pass"
    assert all(binding["checks"].values())
    assert binding["winner_port"] == 2
    assert binding["winner"] == "p2"
    assert binding["outcome"] == outcome
    assert binding["live_terminal"] == {"stocks": {"p1": 0, "p2": 2}, "winner": "p2"}


def test_selected_replay_result_binding_rejects_live_replay_winner_disagreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from melee_policy.integration import replay_result

    replay_path = tmp_path / "game.slp"
    replay_path.write_bytes(b"replay")
    monkeypatch.setattr(
        replay_result,
        "audit_replay",
        lambda *_args, **_kwargs: {
            "replay": {"sha256": "bound-sha256", "raw_byte_length": 6},
            "audit_passed": True,
            "tournament_result_ready": True,
            "outcome": {
                "status": "win",
                "game_complete": True,
                "conclusive": True,
                "winner_port": 1,
                "draw": False,
            },
        },
    )
    binding = _selected_replay_result_binding(
        [replay_path],
        [
            {
                "byte_length": 6,
                "sha256": "bound-sha256",
                "tournament_result_replay": True,
            }
        ],
        0,
        tmp_path,
        SlippiMatchRequest(stage="FINAL_DESTINATION"),
        {"p1": 0, "p2": 3},
    )
    assert binding["decision"] == "fail"
    assert binding["winner"] == "p1"
    assert binding["live_terminal"]["winner"] == "p2"
    assert binding["checks"]["replay_winner_matches_live_terminal_stocks"] is False


class _FormalDrainTransport:
    def __init__(self, fail_phase: str | None = None) -> None:
        self.console = SimpleNamespace(step_calls=0)
        self.fail_phase = fail_phase

    def seal_benchmark_audit(self) -> tuple[dict[str, Any], dict[str, bool]]:
        return (
            {"benchmark_scope": {"sealed_before_shutdown_drain": True}},
            {"ok": True},
        )

    def step(self) -> Any:
        if self.fail_phase == "step":
            raise RuntimeError("step failed")
        self.console.step_calls += 1
        return SimpleNamespace(frame=101, menu_state="IN_GAME")

    def begin_boundary(self, **_kwargs: Any) -> None:
        pass

    def schedule_next_boundary(self, _port: int, **_kwargs: Any) -> None:
        pass

    def commit_boundary(self) -> None:
        if self.fail_phase == "commit":
            raise RuntimeError("commit failed")


def test_release_card_formal_game_end_drain_has_zero_additional_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sends: list[tuple[Any, bool]] = []
    monkeypatch.setattr(
        slippi_match_module,
        "send_canonical_controller",
        lambda controller, _command, *, flush: sends.append((controller, flush)),
    )
    transport = _FormalDrainTransport()
    record = _drain_formal_game_end(
        transport=transport,
        controllers={1: "p1", 2: "p2"},
        terminal_policy_frame=100,
        in_game_menu_states=("IN_GAME",),
        timeout_seconds=5.0,
        formal_game_end_probe=lambda console: console.step_calls >= 2,
    )
    assert record["decision"] == "pass"
    assert record["policy_inference_calls"] == 0
    assert record["controller_trace_rows_written"] == 0
    assert record["neutral_boundaries_started"] == 1
    assert record["neutral_boundaries_committed"] == 1
    assert record["neutral_dispatches_by_port"] == {"p1": 1, "p2": 1}
    assert sends == [("p1", False), ("p2", False)]


@pytest.mark.parametrize(
    ("fail_phase", "expected_phase"),
    [("step", "console-step"), ("commit", "neutral-boundary-commit")],
)
def test_release_card_formal_game_end_drain_records_injected_failure(
    monkeypatch: pytest.MonkeyPatch, fail_phase: str, expected_phase: str
) -> None:
    monkeypatch.setattr(
        slippi_match_module, "send_canonical_controller", lambda *_args, **_kwargs: None
    )
    record = _drain_formal_game_end(
        transport=_FormalDrainTransport(fail_phase),
        controllers={1: object(), 2: object()},
        terminal_policy_frame=100,
        in_game_menu_states=("IN_GAME",),
        timeout_seconds=5.0,
        formal_game_end_probe=lambda _console: False,
    )
    assert record["attempted"] is True
    assert record["decision"] == "fail"
    assert record["failure_phase"] == expected_phase
    assert "RuntimeError" in record["error"]


def test_auxiliary_boundary_exception_becomes_failed_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text("", encoding="utf-8")

    def raising_audit(
        _trace: Path, _replay: Path, _root: Path, **_kwargs: Any
    ) -> dict[str, Any]:
        raise ValueError("auxiliary replay has no aligned frames")

    monkeypatch.setattr(
        slippi_match_module, "_audit_controller_boundary", raising_audit
    )
    result = _audit_controller_boundary_candidate(trace, tmp_path / "aux.slp", tmp_path)
    assert result["gate"]["decision"] == "fail"
    assert result["error"] == "ValueError: auxiliary replay has no aligned frames"


def test_mixed_replay_candidates_receive_the_sealed_game_start_transport_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = {
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
    observed: list[tuple[Path, dict[str, Any]]] = []

    def audit_candidate(
        _trace: Path, replay: Path, _root: Path, **kwargs: Any
    ) -> dict[str, Any]:
        observed.append((replay, kwargs["game_start_transport_proof"]))
        return {"gate": {"decision": "pass"}}

    monkeypatch.setattr(
        slippi_match_module, "_audit_controller_boundary_candidate", audit_candidate
    )
    replays = [tmp_path / "base.slp", tmp_path / "aux.slp"]
    candidates = _audit_mixed_controller_boundary_candidates(
        tmp_path / "trace.jsonl", replays, tmp_path, transport
    )
    expected_proof = _game_start_transport_proof(transport)
    assert expected_proof["decision"] == "pass"
    assert candidates == [
        (0, {"gate": {"decision": "pass"}}),
        (1, {"gate": {"decision": "pass"}}),
    ]
    assert observed == [(replay, expected_proof) for replay in replays]


def _config() -> dict[str, Any]:
    return {
        "integration": {"inference_mode": "synchronous-concurrent"},
        "slippi_ai": {
            "repository_url": "https://github.com/vladfi1/slippi-ai",
            "source_revision": "577965a7731dc53e3472ea63d9e9853a4e9d65fa",
            "checkpoint_sha256": "48dcfd87c52fde9899fb37b0293ce2a597fb2384b7b40baf7fb49c760152a96c",
            "checkpoint_byte_length": 95559707,
            "checkpoint_delay": 21,
            "console_delay": 0,
            "effective_output_delay": 21,
            "upstream_eval_two_console_delay": 2,
            "compile": True,
            "async_inference": True,
            "allowed_characters": [
                "FOX",
                "FALCO",
                "MARTH",
                "SHEIK",
                "JIGGLYPUFF",
                "CPTFALCON",
                "PEACH",
                "YOSHI",
                "POPO",
                "LUIGI",
                "PIKACHU",
                "SAMUS",
            ],
        },
        "slippi_integration": {
            "schema_version": "integration.slippi_ai.v6",
            "exact_mode_only": True,
        },
    }


def test_mixed_policy_frames_require_exactly_player_ports_one_and_two() -> None:
    gamestate = SimpleNamespace(players={1: object(), 2: object()})
    _require_exact_player_ports(gamestate, FIRST_POLICY_FRAME)
    gamestate.players = {1: object()}
    with pytest.raises(RuntimeError, match="got \\[1\\]"):
        _require_exact_player_ports(gamestate, FIRST_POLICY_FRAME)
    gamestate.players = {1: object(), 2: object(), 3: object()}
    with pytest.raises(RuntimeError, match="got \\[1, 2, 3\\]"):
        _require_exact_player_ports(gamestate, FIRST_POLICY_FRAME)


@pytest.mark.parametrize(("p1", "p2"), [("mimic", "hal"), ("slippi-ai", "unknown")])
def test_request_rejects_unsupported_non_slippi_pairings(p1: str, p2: str) -> None:
    with pytest.raises(ValueError, match="requires two 'slippi-ai'"):
        SlippiMatchRequest(player_1_model=p1, player_2_model=p2).validate()


def test_request_accepts_two_independent_slippi_slots() -> None:
    request = SlippiMatchRequest(
        player_1_model="slippi-ai",
        player_2_model="slippi-ai",
        player_1_checkpoint=Path("p1-medium-v2"),
        player_2_checkpoint=Path("p2-medium-v2"),
        player_1_name="Master Player",
        player_2_name="Master Player",
        player_1_temperature=0.75,
        player_2_temperature=1.25,
    )
    request.validate()
    assert request.checkpoint_for_port(1) == Path("p1-medium-v2")
    assert request.checkpoint_for_port(2) == Path("p2-medium-v2")
    assert request.temperature_for_port(1) == 0.75
    assert request.temperature_for_port(2) == 1.25
    with pytest.raises(ValueError, match="exactly one"):
        request.port_for("slippi-ai")


def test_dual_slippi_configs_honor_both_overrides_and_eval_two_timing(
    tmp_path: Path,
) -> None:
    config = _config()
    config["slippi_ai"].update(
        {
            "source_directory": "pinned-source",
            "checkpoint": "default-medium-v2",
            "default_name": "Master Player",
            "sample_temperature": 1.0,
        }
    )
    request = SlippiMatchRequest(
        player_1_model="slippi-ai",
        player_2_model="slippi-ai",
        player_1_checkpoint=tmp_path / "p1-medium-v2",
        player_2_checkpoint=tmp_path / "p2-medium-v2",
        player_1_name="Master Player",
        player_2_name="Master Player",
        player_1_temperature=0.75,
        player_2_temperature=1.25,
    )
    configs = {
        port: _slippi_policy_config(
            config,
            tmp_path,
            request,
            port_override=port,
            console_delay_override=UPSTREAM_EVAL_TWO_CONSOLE_DELAY_FRAMES,
        )
        for port in (1, 2)
    }
    assert configs[1] is not configs[2]
    assert configs[1].checkpoint_path == (tmp_path / "p1-medium-v2").resolve()
    assert configs[2].checkpoint_path == (tmp_path / "p2-medium-v2").resolve()
    assert configs[1].sample_temperature == 0.75
    assert configs[2].sample_temperature == 1.25
    assert configs[1].port == 1 and configs[1].opponent_port == 2
    assert configs[2].port == 2 and configs[2].opponent_port == 1
    assert configs[1].console_delay_frames == configs[2].console_delay_frames == 2
    assert configs[1].policy_delay_frames - configs[1].console_delay_frames == 19


@pytest.mark.parametrize("mode", ["watch", "asynchronous-latest"])
def test_request_rejects_frame_dropping_modes(mode: str) -> None:
    with pytest.raises(ValueError, match="recurrent Slippi-AI frames cannot drop"):
        SlippiMatchRequest(inference_mode=mode).validate()


def test_config_contract_is_exact_only_and_has_mixed_21_minus_0_delay() -> None:
    contract = _validate_config_contract(_config(), SlippiMatchRequest())
    assert contract["exact_mode_only"]
    assert contract["checkpoint_delay_frames"] == POLICY_DELAY_FRAMES == 21
    assert (
        contract["mixed_runtime_console_delay_frames"]
        == MIXED_RUNTIME_CONSOLE_DELAY_FRAMES
        == 0
    )
    assert (
        contract["effective_policy_delay_frames"]
        == MIXED_EFFECTIVE_POLICY_DELAY_FRAMES
        == 21
    )
    assert contract["upstream_eval_two_console_delay_frames"] == 2
    assert UPSTREAM_EVAL_TWO_CONSOLE_DELAY_FRAMES == 2
    invalid = _config()
    invalid["integration"]["inference_mode"] = "asynchronous-latest"
    with pytest.raises(ValueError, match="can drop recurrent frames"):
        _validate_config_contract(invalid, SlippiMatchRequest())


def test_mixed_runner_builds_mimic_with_executed_previous_control_timing() -> None:
    runtime = SimpleNamespace(state=SimpleNamespace(prev_sent=None))
    policy = _make_windowed_policy(
        "mimic",
        runtime,
        2,
        online_delay_frames=MIXED_RUNTIME_CONSOLE_DELAY_FRAMES,
        evaluation_seed=1729,
    )
    assert isinstance(policy, MimicLivePolicy)
    assert policy.port == 2
    assert policy.online_delay_frames == 0
    assert policy.policy_rng["evaluation_seed"] == 1729
    assert policy.policy_rng["policy_identity"] == "mimic"
    policy.record_decoded_command(10, {"main_x": 0.25})
    policy.record_decoded_command(11, {"main_x": 0.25})
    assert policy.previous_executed_command(12) == {"main_x": 0.25}
    assert policy.previous_executed_frame == 11


def test_mixed_runner_slippi_dispatch_is_deferred_without_explicit_flush() -> None:

    class Controller:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def press_button(self, _button: object) -> None:
            self.calls.append("press_button")

        def release_button(self, _button: object) -> None:
            self.calls.append("release_button")

        def tilt_analog(self, _button: object, _x: float, _y: float) -> None:
            self.calls.append("tilt_analog")

        def press_shoulder(self, _button: object, _value: float) -> None:
            self.calls.append("press_shoulder")

        def flush(self) -> None:
            self.calls.append("flush")

    controller = Controller()
    dispatch = _dispatch_slippi_command(
        controller, CanonicalControllerCommand.neutral()
    )
    assert dispatch["called"] is True
    assert dispatch["flush"]["project_adapter_count"] == 0
    assert "flush" not in controller.calls


def test_mimic_source_activation_hard_gates_repository_revision_and_clean_tree() -> (
    None
):
    config_path = Path("configs/integration.toml")
    config, project_root = _load_config(config_path)
    source = project_root / config["mimic"]["source_directory"]
    if not source.is_dir():
        pytest.skip("pinned MIMIC source cache is unavailable")
    checks = _activate_windowed_source(config, project_root, "mimic")
    assert all(checks["repositories"]["mimic"]["checks"].values())


@pytest.mark.parametrize("character", ["NANA", "GIGA_BOWSER", "UNKNOWN_CHARACTER"])
def test_request_rejects_non_launchable_primary_characters(character: str) -> None:
    with pytest.raises(ValueError, match="not an independently launchable"):
        SlippiMatchRequest(player_2_character=character).validate()


def test_first_frame_match_context_records_requested_and_observed_values() -> None:
    gamestate = SimpleNamespace(
        stage=SimpleNamespace(name="YOSHIS_STORY"),
        players={
            1: SimpleNamespace(character=SimpleNamespace(name="FOX"), costume=1),
            2: SimpleNamespace(character=SimpleNamespace(name="MARTH"), costume=0),
        },
    )
    request = SlippiMatchRequest(
        player_1_character="FOX", player_2_character="MARTH", stage="YOSHIS_STORY"
    )
    observed = _observed_match_context(gamestate, request)
    assert observed["match"]
    assert observed["observed_stage"] == "YOSHIS_STORY"
    assert observed["slots"]["p2"]["observed_character"] == "MARTH"
    assert observed["slots"]["p1"]["observed_costume"] == 1
    assert observed["slots"]["p2"]["observed_costume"] == 0
    gamestate.players[2].character.name = "FOX"
    mismatch = _observed_match_context(gamestate, request)
    assert not mismatch["match"]
    assert not mismatch["checks"]["both_characters_exact"]
    gamestate.players[2].character.name = "MARTH"
    gamestate.players[3] = SimpleNamespace(
        character=SimpleNamespace(name="FOX"), costume=0
    )
    extra_port = _observed_match_context(gamestate, request)
    assert not extra_port["match"]
    assert not extra_port["checks"]["two_standard_ports_exact"]


def _boundary_trace_rows() -> list[dict[str, Any]]:
    request = SlippiMatchRequest()
    return [
        _build_trace_row(
            frame,
            request,
            {
                "p1": {
                    "command": {
                        "main_stick": [0.0, 0.5],
                        "c_stick": [0.0, 0.5],
                        "analog_l": 0.4,
                        "analog_r": 0.0,
                        "buttons": ["A"],
                    },
                    "controller_dispatch": {"called": True},
                },
                "p2": {
                    "command": {
                        "main_x": 0.025,
                        "main_y": 0.5,
                        "c_x": 0.5,
                        "c_y": 0.5,
                        "l_shldr": 0.0,
                        "r_shldr": 0.0,
                        "btn_BUTTON_Z": 1.0,
                    },
                    "controller_dispatch": {"called": True},
                },
            },
        )
        for frame in (10, 11)
    ]


def _boundary_replay_states() -> dict[int, dict[int, dict[str, Any]]]:
    return {
        replay_frame: {
            1: {
                "buttons_physical": ("A",),
                "buttons_processed": ("A",),
                "raw_main_stick": (-80, 0),
                "main_stick": (0.15, 0.5),
                "c_stick": (-1.0, 0.0),
                "analog_l": 0.35,
                "physical_analog_l": 0.4,
                "physical_analog_r": 0.0,
            },
            2: {
                "buttons_physical": ("Z",),
                "buttons_processed": ("A", "Z"),
                "raw_main_stick": (-76, 0),
                "main_stick": (0.025000005, 0.5),
                "c_stick": (0.0, 0.0),
                "analog_l": 0.0,
                "physical_analog_l": 0.0,
                "physical_analog_r": 0.0,
            },
        }
        for replay_frame in (10, 11, 12)
    }


def test_controller_boundary_audit_gates_all_saved_controller_dimensions() -> None:
    audit = _audit_controller_boundary_records(
        _boundary_trace_rows(), _boundary_replay_states()
    )
    assert CONTROLLER_REPLAY_LAG_FRAMES == 1
    assert audit["gate"]["decision"] == "pass"
    assert audit["alignment"]["online_delay_frames"] == 0
    assert audit["alignment"]["first_aligned_pair"] == [10, 11]
    assert audit["alignment"]["last_aligned_pair"] == [11, 12]
    assert audit["alignment"]["actual_overlap_pairs"] == 2
    assert audit["slots"]["p1"]["excluded_startup_trace_frames"] == []
    assert audit["slots"]["p2"]["excluded_startup_trace_frames"] == []
    assert audit["slots"]["p2"]["controller_frame_pairs_compared"] == 2
    for slot in ("p1", "p2"):
        assert (
            audit["slots"][slot]["digital_buttons"]["physical"]["frames_compared"] == 2
        )
        assert (
            audit["slots"][slot]["digital_buttons"]["physical"]["mismatch_frames"] == 0
        )
        assert (
            audit["slots"][slot]["digital_buttons"]["processed_upstream_observation"][
                "mismatch_frames"
            ]
            == 0
        )
        assert (
            audit["slots"][slot]["intended_raw_main_stick"]["mismatch_components"] == 0
        )
        assert audit["slots"][slot]["processed_c_stick"]["mismatch_components"] == 0
        assert (
            audit["slots"][slot]["physical_analog_shoulders"]["mismatch_components"]
            == 0
        )
        assert audit["slots"][slot]["processed_diagnostics"]["hard_gate"] is False
    assert (
        audit["slots"]["p2"]["digital_buttons"][
            "decoded_physical_vs_processed_diagnostic"
        ]["mismatch_frames"]
        == 2
    )
    assert (
        audit["slots"]["p1"]["processed_diagnostics"]["main_stick"][
            "mismatch_components"
        ]
        == 2
    )
    assert audit["gate"]["checks"]["both_slots_processed_c_stick_within_tolerance"]
    assert audit["gate"]["checks"][
        "both_slots_physical_analog_shoulders_within_tolerance"
    ]


def test_controller_boundary_universally_excludes_the_first_gameplay_latch() -> None:
    rows = _boundary_trace_rows()
    rows[0]["game_frame"] = FIRST_POLICY_FRAME
    rows[1]["game_frame"] = FIRST_POLICY_FRAME + 1
    source_replay = _boundary_replay_states()
    replay = {
        FIRST_POLICY_FRAME: copy.deepcopy(source_replay[10]),
        FIRST_POLICY_FRAME + 1: copy.deepcopy(source_replay[11]),
        FIRST_POLICY_FRAME + 2: copy.deepcopy(source_replay[12]),
    }
    for port in (1, 2):
        replay[FIRST_POLICY_FRAME + 1][port]["buttons_physical"] = ()
        replay[FIRST_POLICY_FRAME + 1][port]["buttons_processed"] = ()
        replay[FIRST_POLICY_FRAME + 1][port]["raw_main_stick"] = (0, 0)
    audit = _audit_controller_boundary_records(rows, replay)
    assert FIRST_GAMEPLAY_CONTROLLER_LATCH_UNOBSERVABLE_FRAMES == 1
    assert audit["gate"]["decision"] == "pass"
    assert audit["gate"]["checks"]["both_slots_first_gameplay_latch_rule_exact"]
    assert audit["alignment"]["first_aligned_pair"] == [
        FIRST_POLICY_FRAME,
        FIRST_POLICY_FRAME + 1,
    ]
    assert audit["alignment"]["first_comparable_controller_pair"] == [
        FIRST_POLICY_FRAME + 1,
        FIRST_POLICY_FRAME + 2,
    ]
    assert audit["alignment"][
        "first_gameplay_controller_latch_unobservable_trace_frames"
    ] == [FIRST_POLICY_FRAME]
    for slot in ("p1", "p2"):
        slot_audit = audit["slots"][slot]
        assert slot_audit["controller_frame_pairs_compared"] == 1
        assert slot_audit["excluded_startup_trace_frames"] == [FIRST_POLICY_FRAME]
        assert slot_audit["excluded_first_gameplay_latch_trace_frames"] == [
            FIRST_POLICY_FRAME
        ]
        assert slot_audit["excluded_no_dispatch_trace_frames"] == []
        assert slot_audit["startup_exclusions"] == [
            {
                "trace_frame": FIRST_POLICY_FRAME,
                "dispatch_called": True,
                "classifications": ["first-gameplay controller latch unavailable"],
            }
        ]
        assert slot_audit["intended_raw_main_stick"]["mismatch_components"] == 0


def _game_start_controller_latch_fixture() -> tuple[
    list[dict[str, Any]], dict[int, dict[int, dict[str, Any]]]
]:
    template = _boundary_trace_rows()[0]
    rows: list[dict[str, Any]] = []
    for frame in range(FIRST_POLICY_FRAME, GAME_START_CONTROLLER_LATCH_TRACE_FRAME + 2):
        row = copy.deepcopy(template)
        row["game_frame"] = frame
        row["slots"]["p1"]["command"] = {
            "main_stick": [0.5, 0.5],
            "c_stick": [0.5, 0.5],
            "analog_l": 0.0,
            "analog_r": 0.0,
            "buttons": [],
        }
        row["slots"]["p2"]["command"] = {
            "main_x": 0.0 if frame == GAME_START_CONTROLLER_LATCH_TRACE_FRAME else 0.5,
            "main_y": 0.5,
            "c_x": 0.5,
            "c_y": 0.5,
            "l_shldr": 0.0,
            "r_shldr": 0.0,
        }
        for slot in ("p1", "p2"):
            row["slots"][slot]["player_state"] = {
                "character": "FOX",
                "action": 322,
                "stocks": 4,
                "percent": 0.0,
                "position": [-60.0 if slot == "p1" else 60.0, 10.0],
            }
            row["slots"][slot]["controller_dispatch"] = {"called": True}
        rows.append(row)
    replay_state = {
        "buttons_physical": (),
        "buttons_processed": (),
        "raw_main_stick": (0, 0),
        "main_stick": (0.5, 0.5),
        "c_stick": (0.0, 0.0),
        "analog_l": 0.0,
        "physical_analog_l": 0.0,
        "physical_analog_r": 0.0,
    }
    replay = {
        frame: {1: copy.deepcopy(replay_state), 2: copy.deepcopy(replay_state)}
        for frame in range(
            FIRST_POLICY_FRAME, GAME_START_CONTROLLER_LATCH_TRACE_FRAME + 3
        )
    }
    return (rows, replay)


def _passing_game_start_transport_proof() -> dict[str, Any]:
    return {"decision": "pass", "checks": {"exactly_one_internal_commit": True}}


def test_game_start_transport_proof_requires_the_exact_paired_internal_commit() -> None:
    transport = {
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
    proof = _game_start_transport_proof(transport)
    assert proof["decision"] == "pass"
    assert all(proof["checks"].values())
    transport["ports"]["p2"]["later_internal_flush_requests"] = 0
    failed = _game_start_transport_proof(transport)
    assert failed["decision"] == "fail"
    assert failed["checks"]["p2_exactly_one_later_internal_flush"] is False


def test_controller_boundary_uses_the_transport_proved_game_start_mapping() -> None:
    rows, replay = _game_start_controller_latch_fixture()
    unclassified = _audit_controller_boundary_records(rows, replay)
    assert unclassified["gate"]["decision"] == "fail"
    assert (
        unclassified["slots"]["p2"]["intended_raw_main_stick"]["mismatch_components"]
        == 1
    )
    audit = _audit_controller_boundary_records(
        rows, replay, game_start_transport_proof=_passing_game_start_transport_proof()
    )
    assert GAME_START_CONTROLLER_LATCH_TRACE_FRAME == -119
    assert GAME_START_CONTROLLER_LATCH_UNOBSERVABLE_FRAMES == 1
    assert audit["gate"]["decision"] == "pass"
    assert audit["gate"]["checks"]["both_slots_game_start_controller_latch_rule_exact"]
    assert audit["alignment"][
        "game_start_controller_latch_unobservable_trace_frames"
    ] == [-119]
    assert audit["alignment"]["first_aligned_pair"] == [-123, -121]
    first_frame_evidence = audit["alignment"]["first_policy_frame_mapping_evidence"]
    assert (
        first_frame_evidence["decision"] == "observationally-equivalent-delayed-first"
    )
    assert first_frame_evidence["matching_candidates"] == [
        "early-latched",
        "delayed-first",
    ]
    assert (
        "controller-observationally equivalent"
        in first_frame_evidence["selection_reason"]
    )
    for slot in ("p1", "p2"):
        assert audit["slots"][slot]["excluded_startup_trace_frames"] == [-119]
        assert audit["slots"][slot][
            "excluded_game_start_controller_latch_trace_frames"
        ] == [-119]
        assert (
            audit["slots"][slot]["intended_raw_main_stick"]["mismatch_components"] == 0
        )


def test_game_start_mapping_proof_fails_closed_without_transport_evidence() -> None:
    rows, replay = _game_start_controller_latch_fixture()
    with pytest.raises(ValueError, match="one exact paired internal commit"):
        _audit_controller_boundary_records(
            rows,
            replay,
            game_start_transport_proof={"decision": "fail", "checks": {"exact": False}},
        )


def _nonneutral_piecewise_game_start_fixture() -> tuple[
    list[dict[str, Any]], dict[int, dict[int, dict[str, Any]]]
]:
    rows, replay = _game_start_controller_latch_fixture()
    commands = {
        -123: {"main_y": 0.5, "l_shldr": 1.0, "btn_BUTTON_B": 0},
        -122: {"main_y": 1.0, "l_shldr": 0.0, "btn_BUTTON_B": 0},
        -121: {"main_y": 1.0, "l_shldr": 0.4, "btn_BUTTON_B": 1},
        -120: {"main_y": 1.0, "l_shldr": 0.4, "btn_BUTTON_B": 1},
        -119: {"main_y": 1.0, "l_shldr": 0.0, "btn_BUTTON_B": 1},
        -118: {"main_y": 1.0, "l_shldr": 0.0, "btn_BUTTON_B": 0},
    }
    for row in rows:
        frame = row["game_frame"]
        command = row["slots"]["p2"]["command"]
        command.update(commands[frame])
    mapped_frames = {-123: -121, -122: -120, -121: -119, -120: -118, -118: -117}
    for trace_frame, replay_frame in mapped_frames.items():
        command = commands[trace_frame]
        button_b = command["btn_BUTTON_B"] == 1
        replay[replay_frame][2].update(
            {
                "buttons_physical": ("B",) if button_b else (),
                "buttons_processed": ("B",) if button_b else (),
                "raw_main_stick": (0, round((command["main_y"] - 0.5) * 160)),
                "main_stick": (0.5, command["main_y"]),
                "physical_analog_l": command["l_shldr"],
            }
        )
    return (rows, replay)


def test_controller_boundary_hard_gates_the_full_piecewise_startup_mapping() -> None:
    rows, replay = _nonneutral_piecewise_game_start_fixture()
    audit = _audit_controller_boundary_records(
        rows, replay, game_start_transport_proof=_passing_game_start_transport_proof()
    )
    assert audit["gate"]["decision"] == "pass"
    assert audit["alignment"]["mode"] == "piecewise-game-start"
    assert audit["alignment"]["first_comparable_controller_pair"] == [-123, -121]
    evidence = audit["alignment"]["first_policy_frame_mapping_evidence"]
    assert evidence["decision"] == "delayed-first"
    assert evidence["matching_candidates"] == ["delayed-first"]
    assert evidence["selection_reason"].startswith("only replay -121 matches")
    assert evidence["candidate_matches"] == {
        "early-latched": {"p1": True, "p2": False, "both_slots": False},
        "delayed-first": {"p1": True, "p2": True, "both_slots": True},
    }
    assert audit["alignment"][
        "game_start_controller_latch_unobservable_trace_frames"
    ] == [-119]
    for slot in ("p1", "p2"):
        assert audit["slots"][slot]["excluded_startup_trace_frames"] == [-119]
        assert audit["slots"][slot]["controller_frame_pairs_compared"] == 5


def _early_latched_piecewise_game_start_fixture() -> tuple[
    list[dict[str, Any]], dict[int, dict[int, dict[str, Any]]]
]:
    rows, replay = _nonneutral_piecewise_game_start_fixture()
    first_command = rows[0]["slots"]["p2"]["command"]
    first_command.update(
        {"main_x": 0.15, "main_y": 0.15, "l_shldr": 0.0, "btn_BUTTON_B": 0}
    )
    replay[-122][2].update({"raw_main_stick": (-56, -56), "main_stick": (0.15, 0.15)})
    replay[-121][2]["physical_analog_l"] = 0.0
    return (rows, replay)


def test_controller_boundary_selects_the_early_latched_first_policy_frame() -> None:
    rows, replay = _early_latched_piecewise_game_start_fixture()
    audit = _audit_controller_boundary_records(
        rows, replay, game_start_transport_proof=_passing_game_start_transport_proof()
    )
    assert audit["gate"]["decision"] == "pass"
    assert audit["alignment"]["first_aligned_pair"] == [-123, -122]
    assert audit["alignment"]["first_comparable_controller_pair"] == [-123, -122]
    assert audit["alignment"]["unpaired_replay_boundary_frames"] == [-123, -121]
    evidence = audit["alignment"]["first_policy_frame_mapping_evidence"]
    assert evidence["decision"] == "early-latched"
    assert evidence["selected_replay_frame"] == -122
    assert evidence["matching_candidates"] == ["early-latched"]
    assert evidence["selection_reason"].startswith("only replay -122 matches")
    assert evidence["candidate_matches"] == {
        "early-latched": {"p1": True, "p2": True, "both_slots": True},
        "delayed-first": {"p1": True, "p2": False, "both_slots": False},
    }
    assert audit["slots"]["p2"]["intended_raw_main_stick"]["mismatch_components"] == 0


def _fixed_normal_game_start_fixture() -> tuple[
    list[dict[str, Any]], dict[int, dict[int, dict[str, Any]]]
]:
    rows, replay = _game_start_controller_latch_fixture()
    commands = {
        -123: (0.15, 0.15, False),
        -122: (0.85, 0.85, True),
        -121: (1.0, 0.5, False),
        -120: (0.5, 1.0, True),
        -119: (0.0, 0.5, False),
        -118: (0.5, 0.0, True),
    }
    for row in rows:
        trace_frame = row["game_frame"]
        main_x, main_y, button_b = commands[trace_frame]
        row["slots"]["p2"]["command"].update(
            {"main_x": main_x, "main_y": main_y, "btn_BUTTON_B": int(button_b)}
        )
        replay[trace_frame + 1][2].update(
            {
                "buttons_physical": ("B",) if button_b else (),
                "buttons_processed": ("B",) if button_b else (),
                "raw_main_stick": (
                    round((main_x - 0.5) * 160),
                    round((main_y - 0.5) * 160),
                ),
                "main_stick": (main_x, main_y),
            }
        )
    return (rows, replay)


def test_controller_boundary_keeps_exact_normal_alignment_through_game_start() -> None:
    rows, replay = _fixed_normal_game_start_fixture()
    audit = _audit_controller_boundary_records(
        rows, replay, game_start_transport_proof=_passing_game_start_transport_proof()
    )
    assert audit["gate"]["decision"] == "pass"
    assert all(audit["gate"]["checks"].values())
    assert audit["alignment"]["mode"] == "fixed-lag-through-game-start"
    assert (
        audit["alignment"]["equation"]
        == "replay_frame = trace_frame + 1 across the complete startup and game"
    )
    assert audit["alignment"]["first_aligned_pair"] == [-123, -122]
    assert audit["alignment"]["last_aligned_pair"] == [-118, -117]
    assert audit["alignment"]["unpaired_trace_boundary_frames"] == []
    assert audit["alignment"]["unpaired_replay_boundary_frames"] == [-123]
    evidence = audit["alignment"]["first_policy_frame_mapping_evidence"]
    assert evidence["decision"] == "fixed-normal-boundary"
    assert evidence["selected_replay_frame"] == -122
    assert evidence["fixed_normal_boundary_evidence"]["decision"] == "pass"
    assert (
        evidence["fixed_normal_boundary_evidence"]["includes_first_policy_frame"]
        is True
    )
    assert evidence["fixed_normal_boundary_evidence"]["pair_count"] == 6
    for slot in ("p1", "p2"):
        assert audit["slots"][slot]["excluded_startup_trace_frames"] == []
        assert audit["slots"][slot]["controller_frame_pairs_compared"] == 6


def test_normal_game_start_alignment_requires_every_two_port_command() -> None:
    rows, replay = _fixed_normal_game_start_fixture()
    replay[-119][2]["raw_main_stick"] = (1, 80)
    audit = _audit_controller_boundary_records(
        rows, replay, game_start_transport_proof=_passing_game_start_transport_proof()
    )
    evidence = audit["alignment"]["first_policy_frame_mapping_evidence"]
    assert evidence["decision"] == "early-latched"
    assert evidence["fixed_normal_boundary_evidence"]["decision"] == "fail"
    assert audit["gate"]["decision"] == "fail"


def test_fixed_normal_alignment_allows_one_proved_paired_neutral_overwrite() -> None:
    rows, replay = _fixed_normal_game_start_fixture()
    overwritten_trace_frame = -120
    replay[overwritten_trace_frame + 1][2].update(
        {
            "buttons_physical": (),
            "buttons_processed": (),
            "raw_main_stick": (0, 0),
            "main_stick": (0.5, 0.5),
        }
    )
    audit = _audit_controller_boundary_records(
        rows, replay, game_start_transport_proof=_passing_game_start_transport_proof()
    )
    assert audit["gate"]["decision"] == "pass"
    assert all(audit["gate"]["checks"].values())
    assert audit["alignment"]["mode"] == "fixed-lag-with-game-start-neutral-overwrite"
    assert audit["alignment"][
        "game_start_controller_latch_unobservable_trace_frames"
    ] == [overwritten_trace_frame]
    evidence = audit["alignment"]["first_policy_frame_mapping_evidence"]
    assert (
        evidence["decision"]
        == "fixed-normal-boundary-with-neutral-game-start-overwrite"
    )
    overwrite = evidence["neutral_game_start_overwrite_evidence"]
    assert overwrite["decision"] == "pass"
    assert overwrite["trace_frame"] == overwritten_trace_frame
    assert overwrite["replay_frame"] == overwritten_trace_frame + 1
    assert overwrite["mismatched_ports"] == [2]
    assert all(overwrite["checks"].values())
    for slot in ("p1", "p2"):
        assert audit["slots"][slot]["excluded_startup_trace_frames"] == [
            overwritten_trace_frame
        ]
        assert audit["slots"][slot]["controller_frame_pairs_compared"] == 5


def test_neutral_overwrite_rule_rejects_a_nonneutral_replay_boundary() -> None:
    rows, replay = _fixed_normal_game_start_fixture()
    replay[-119][2]["raw_main_stick"] = (1, 0)
    replay[-119][2]["main_stick"] = (0.50625, 0.5)
    audit = _audit_controller_boundary_records(
        rows, replay, game_start_transport_proof=_passing_game_start_transport_proof()
    )
    evidence = audit["alignment"]["first_policy_frame_mapping_evidence"]
    overwrite = evidence["neutral_game_start_overwrite_evidence"]
    assert overwrite["decision"] == "fail"
    assert (
        overwrite["checks"]["overwritten_replay_boundary_exactly_neutral_on_both_ports"]
        is False
    )
    assert audit["gate"]["decision"] == "fail"


def _unobservable_first_policy_game_start_fixture() -> tuple[
    list[dict[str, Any]], dict[int, dict[int, dict[str, Any]]]
]:
    rows, replay = _game_start_controller_latch_fixture()
    terminal_row = copy.deepcopy(rows[-1])
    terminal_row["game_frame"] = -117
    rows.append(terminal_row)
    commands = {
        -123: {"main_x": 0.15, "main_y": 0.15, "btn_BUTTON_B": 1},
        -122: {"main_x": 0.5, "main_y": 0.5, "btn_BUTTON_B": 0},
        -121: {"main_x": 0.5, "main_y": 1.0, "btn_BUTTON_B": 1},
        -120: {"main_x": 0.5, "main_y": 1.0, "l_shldr": 0.4, "btn_BUTTON_B": 0},
        -119: {"main_x": 0.0, "main_y": 0.5, "l_shldr": 0.0, "btn_BUTTON_B": 0},
        -118: {"main_x": 0.5, "main_y": 0.5, "btn_BUTTON_B": 1},
        -117: {"main_x": 1.0, "main_y": 0.5, "btn_BUTTON_B": 0},
    }
    for row in rows:
        frame = row["game_frame"]
        command = row["slots"]["p2"]["command"]
        command.update(commands[frame])
    for trace_frame in range(-122, -117):
        command = commands[trace_frame]
        replay[trace_frame + 1][2].update(
            {
                "buttons_physical": ("B",) if command["btn_BUTTON_B"] else (),
                "buttons_processed": ("B",) if command["btn_BUTTON_B"] else (),
                "raw_main_stick": (
                    round((command["main_x"] - 0.5) * 160),
                    round((command["main_y"] - 0.5) * 160),
                ),
                "main_stick": (command["main_x"], command["main_y"]),
                "physical_analog_l": command.get("l_shldr", 0.0),
            }
        )
    return (rows, replay)


def test_controller_boundary_classifies_the_unobservable_first_policy_frame() -> None:
    rows, replay = _unobservable_first_policy_game_start_fixture()
    audit = _audit_controller_boundary_records(
        rows, replay, game_start_transport_proof=_passing_game_start_transport_proof()
    )
    assert audit["schema_version"] == "integration.controller_boundary.v12"
    assert audit["gate"]["decision"] == "pass"
    assert all(audit["gate"]["checks"].values())
    assert (
        audit["alignment"]["mode"]
        == "normal-boundary-after-unobservable-first-gameplay-latch"
    )
    assert audit["alignment"]["first_policy_trace_to_replay_lag_frames"] is None
    assert audit["alignment"]["startup_trace_to_replay_lag_frames"] == 1
    assert audit["alignment"]["first_aligned_pair"] == [-122, -121]
    assert audit["alignment"]["first_comparable_controller_pair"] == [-122, -121]
    assert audit["alignment"]["last_aligned_pair"] == [-118, -117]
    assert audit["alignment"]["permitted_terminal_unobservable_trace_frames"] == [-117]
    assert audit["alignment"][
        "first_gameplay_controller_latch_unobservable_trace_frames"
    ] == [-123]
    assert (
        audit["alignment"]["game_start_controller_latch_unobservable_trace_frames"]
        == []
    )
    evidence = audit["alignment"]["first_policy_frame_mapping_evidence"]
    assert evidence["decision"] == "first-gameplay-latch-unobservable"
    assert evidence["selected_replay_frame"] is None
    assert evidence["matching_candidates"] == []
    assert evidence["candidate_neutrality"] == {
        "early-latched": {"p1": True, "p2": True, "both_slots": True},
        "delayed-first": {"p1": True, "p2": True, "both_slots": True},
    }
    assert evidence["candidate_exact_two_ports"] == {
        "early-latched": True,
        "delayed-first": True,
    }
    normal = evidence["normal_next_boundary_evidence"]
    assert normal["decision"] == "pass"
    assert normal["evidence_ports"] == [1, 2]
    assert normal["first_pair"] == [-122, -121]
    assert normal["last_pair"] == [-118, -117]
    assert normal["pair_count"] == 5
    assert normal["exact_pairs_by_port"] == {"p1": 5, "p2": 5}
    assert normal["terminal_unobservable_trace_frames"] == [-117]
    for slot in ("p1", "p2"):
        assert audit["slots"][slot]["excluded_startup_trace_frames"] == [-123]
        assert audit["slots"][slot]["excluded_first_gameplay_latch_trace_frames"] == [
            -123
        ]
        assert (
            audit["slots"][slot]["excluded_game_start_controller_latch_trace_frames"]
            == []
        )
        assert audit["slots"][slot]["controller_frame_pairs_compared"] == 5


def test_unobservable_first_policy_mapping_wins_when_first_command_is_also_neutral() -> (
    None
):
    rows, replay = _unobservable_first_policy_game_start_fixture()
    rows[0]["slots"]["p2"]["command"].update(
        {"main_x": 0.5, "main_y": 0.5, "l_shldr": 0.0, "btn_BUTTON_B": 0}
    )
    audit = _audit_controller_boundary_records(
        rows, replay, game_start_transport_proof=_passing_game_start_transport_proof()
    )
    evidence = audit["alignment"]["first_policy_frame_mapping_evidence"]
    assert audit["gate"]["decision"] == "pass"
    assert evidence["decision"] == "first-gameplay-latch-unobservable"
    assert evidence["matching_candidates"] == ["early-latched", "delayed-first"]
    assert evidence["normal_next_boundary_evidence"]["decision"] == "pass"
    assert audit["alignment"]["first_aligned_pair"] == [-122, -121]
    assert (
        audit["alignment"]["game_start_controller_latch_unobservable_trace_frames"]
        == []
    )


def _repeated_initial_command_game_start_fixture() -> tuple[
    list[dict[str, Any]], dict[int, dict[int, dict[str, Any]]]
]:
    rows, replay = _unobservable_first_policy_game_start_fixture()
    for row in rows:
        frame = row["game_frame"]
        command = row["slots"]["p2"]["command"]
        command.update(
            {
                "main_x": 0.15,
                "main_y": 0.15,
                "btn_BUTTON_B": int(frame >= -120),
                "l_shldr": 0.0 if frame < -120 else 0.4 if frame == -120 else 1.0,
            }
        )
        if -122 <= frame <= -118:
            replay[frame + 1][2].update(
                {
                    "buttons_physical": ("B",) if command["btn_BUTTON_B"] else (),
                    "buttons_processed": ("B",) if command["btn_BUTTON_B"] else (),
                    "raw_main_stick": (-56, -56),
                    "main_stick": (0.15, 0.15),
                    "physical_analog_l": command["l_shldr"],
                }
            )
    return (rows, replay)


def test_repeated_initial_command_uses_complete_normal_boundary_evidence() -> None:
    rows, replay = _repeated_initial_command_game_start_fixture()
    audit = _audit_controller_boundary_records(
        rows, replay, game_start_transport_proof=_passing_game_start_transport_proof()
    )
    assert audit["gate"]["decision"] == "pass"
    assert all(audit["gate"]["checks"].values())
    evidence = audit["alignment"]["first_policy_frame_mapping_evidence"]
    assert evidence["matching_candidates"] == ["delayed-first"]
    assert evidence["decision"] == "first-gameplay-latch-unobservable"
    assert evidence["repeated_initial_command_normal_boundary_proved"] is True
    assert evidence["normal_next_boundary_evidence"]["exact_pairs_by_port"] == {
        "p1": 5,
        "p2": 5,
    }
    assert audit["alignment"]["first_aligned_pair"] == [-122, -121]
    assert (
        audit["alignment"]["game_start_controller_latch_unobservable_trace_frames"]
        == []
    )
    for slot in ("p1", "p2"):
        assert audit["slots"][slot]["excluded_startup_trace_frames"] == [-123]
        assert audit["slots"][slot]["controller_frame_pairs_compared"] == 5


@pytest.mark.parametrize(
    ("frame", "port", "field", "value"),
    [
        (-122, 1, "physical_analog_r", 0.01),
        (-119, 2, "buttons_physical", ("X",)),
        (-118, 2, "physical_analog_l", 0.4),
        (-117, 1, "c_stick", (0.01, 0.0)),
    ],
)
def test_repeated_initial_command_rejects_inconsistent_controller_evidence(
    frame: int, port: int, field: str, value: Any
) -> None:
    rows, replay = _repeated_initial_command_game_start_fixture()
    replay[frame][port][field] = value
    audit = _audit_controller_boundary_records(
        rows, replay, game_start_transport_proof=_passing_game_start_transport_proof()
    )
    assert audit["gate"]["decision"] == "fail"
    evidence = audit["alignment"]["first_policy_frame_mapping_evidence"]
    assert evidence["repeated_initial_command_normal_boundary_proved"] is False
    assert evidence["decision"] == "delayed-first"


@pytest.mark.parametrize("missing_transport", [False, True])
def test_repeated_initial_command_requires_transport_and_entry_startup(
    missing_transport: bool,
) -> None:
    rows, replay = _repeated_initial_command_game_start_fixture()
    proof = _passing_game_start_transport_proof()
    if missing_transport:
        proof["checks"]["exactly_one_internal_commit"] = False
    else:
        rows[2]["slots"]["p2"]["player_state"]["action"] = 14
    with pytest.raises(ValueError, match="piecewise GAME_START proof"):
        _audit_controller_boundary_records(
            rows, replay, game_start_transport_proof=proof
        )


def test_repeated_initial_command_preserves_a_proved_delayed_startup() -> None:
    rows, replay = _nonneutral_piecewise_game_start_fixture()
    rows[1]["slots"]["p2"]["command"] = copy.deepcopy(rows[0]["slots"]["p2"]["command"])
    replay[-120][2] = copy.deepcopy(replay[-121][2])
    audit = _audit_controller_boundary_records(
        rows, replay, game_start_transport_proof=_passing_game_start_transport_proof()
    )
    assert audit["gate"]["decision"] == "pass"
    evidence = audit["alignment"]["first_policy_frame_mapping_evidence"]
    assert evidence["decision"] == "delayed-first"
    assert evidence["repeated_initial_command_normal_boundary_proved"] is False
    assert evidence["normal_next_boundary_evidence"]["decision"] == "fail"
    assert audit["alignment"][
        "game_start_controller_latch_unobservable_trace_frames"
    ] == [-119]


def test_unobservable_first_policy_mapping_requires_exact_neutral_boundaries() -> None:
    rows, replay = _unobservable_first_policy_game_start_fixture()
    replay[-122][1]["physical_analog_r"] = 0.01
    with pytest.raises(
        ValueError, match="exact first-gameplay-latch classification failed"
    ):
        _audit_controller_boundary_records(
            rows,
            replay,
            game_start_transport_proof=_passing_game_start_transport_proof(),
        )


def test_unobservable_first_policy_mapping_requires_exactly_two_boundary_ports() -> (
    None
):
    rows, replay = _unobservable_first_policy_game_start_fixture()
    del replay[-122][2]
    with pytest.raises(
        ValueError, match="exact first-gameplay-latch classification failed"
    ):
        _audit_controller_boundary_records(
            rows,
            replay,
            game_start_transport_proof=_passing_game_start_transport_proof(),
        )


def test_unobservable_first_policy_mapping_requires_every_subsequent_dimension() -> (
    None
):
    rows, replay = _unobservable_first_policy_game_start_fixture()
    replay[-119][2]["c_stick"] = (0.01, 0.0)
    with pytest.raises(
        ValueError, match="exact first-gameplay-latch classification failed"
    ):
        _audit_controller_boundary_records(
            rows,
            replay,
            game_start_transport_proof=_passing_game_start_transport_proof(),
        )


def test_unobservable_first_policy_mapping_rejects_split_boundary_evidence() -> None:
    rows, replay = _unobservable_first_policy_game_start_fixture()
    rows[0]["slots"]["p1"]["command"] = {
        "main_stick": [1.0, 0.5],
        "c_stick": [0.5, 0.5],
        "analog_l": 0.0,
        "analog_r": 0.0,
        "buttons": [],
    }
    replay[-122][1]["raw_main_stick"] = (80, 0)
    replay[-122][1]["main_stick"] = (1.0, 0.5)
    replay[-121][2].update(
        {
            "buttons_physical": ("B",),
            "buttons_processed": ("B",),
            "raw_main_stick": (-56, -56),
            "main_stick": (0.15, 0.15),
        }
    )
    with pytest.raises(
        ValueError, match="exact first-gameplay-latch classification failed"
    ):
        _audit_controller_boundary_records(
            rows,
            replay,
            game_start_transport_proof=_passing_game_start_transport_proof(),
        )


def test_piecewise_game_start_first_policy_mapping_fails_closed_without_a_match() -> (
    None
):
    rows, replay = _nonneutral_piecewise_game_start_fixture()
    replay[-121][2]["physical_analog_l"] = 0.5
    with pytest.raises(
        ValueError, match="exact first-gameplay-latch classification failed"
    ):
        _audit_controller_boundary_records(
            rows,
            replay,
            game_start_transport_proof=_passing_game_start_transport_proof(),
        )


def test_piecewise_game_start_mapping_fails_closed_on_controller_mismatch() -> None:
    rows, replay = _nonneutral_piecewise_game_start_fixture()
    replay[-119][2]["buttons_physical"] = ("B", "R")
    audit = _audit_controller_boundary_records(
        rows, replay, game_start_transport_proof=_passing_game_start_transport_proof()
    )
    assert audit["gate"]["decision"] == "fail"
    assert audit["slots"]["p2"]["digital_buttons"]["physical"]["mismatch_frames"] == 1


def test_controller_boundary_rejects_internal_no_dispatch_frame() -> None:
    rows = _boundary_trace_rows()
    rows[1]["slots"]["p2"]["controller_dispatch"] = {
        "called": False,
        "reason": "simulated internal dropped dispatch",
    }
    audit = _audit_controller_boundary_records(rows, _boundary_replay_states())
    assert audit["gate"]["decision"] == "fail"
    assert not audit["gate"]["checks"]["both_slots_no_dispatch_rule_exact"]
    assert audit["slots"]["p2"]["excluded_no_dispatch_trace_frames"] == [11]
    assert audit["slots"]["p2"]["expected_no_dispatch_trace_frames"] == []


def test_controller_boundary_requires_explicit_dispatch_evidence() -> None:
    rows = _boundary_trace_rows()
    del rows[0]["slots"]["p2"]["controller_dispatch"]
    with pytest.raises(ValueError, match="lacks explicit controller-dispatch evidence"):
        _audit_controller_boundary_records(rows, _boundary_replay_states())


def test_controller_boundary_audit_rejects_button_or_raw_main_mismatch() -> None:
    replay = _boundary_replay_states()
    replay[11][1]["buttons_physical"] = ("B",)
    replay[11][1]["buttons_processed"] = ("B",)
    replay[12][2]["raw_main_stick"] = (-75, 0)
    audit = _audit_controller_boundary_records(_boundary_trace_rows(), replay)
    assert audit["gate"]["decision"] == "fail"
    assert not audit["gate"]["checks"]["both_slots_physical_buttons_exact"]
    assert not audit["gate"]["checks"]["both_slots_processed_upstream_buttons_exact"]
    assert not audit["gate"]["checks"]["both_slots_intended_raw_main_stick_exact"]
    assert audit["slots"]["p1"]["digital_buttons"]["physical"]["mismatch_frames"] == 1
    assert audit["slots"]["p2"]["intended_raw_main_stick"]["mismatch_components"] == 1


def test_controller_boundary_rejects_c_stick_or_left_trigger_mismatch() -> None:
    replay = _boundary_replay_states()
    replay[11][1]["c_stick"] = (-0.98, 0.0)
    replay[12][1]["physical_analog_l"] = 0.2
    audit = _audit_controller_boundary_records(_boundary_trace_rows(), replay)
    assert audit["gate"]["decision"] == "fail"
    assert not audit["gate"]["checks"]["both_slots_processed_c_stick_within_tolerance"]
    assert not audit["gate"]["checks"][
        "both_slots_physical_analog_shoulders_within_tolerance"
    ]
    assert audit["slots"]["p1"]["processed_c_stick"]["mismatches"] == [
        {
            "trace_frame": 10,
            "replay_frame": 11,
            "component": "x",
            "expected": -1.0,
            "observed": -0.98,
            "absolute_error": pytest.approx(0.02),
        }
    ]
    assert audit["slots"]["p1"]["physical_analog_shoulders"]["mismatches"] == [
        {
            "trace_frame": 11,
            "replay_frame": 12,
            "component": "analog_l",
            "expected": 0.4,
            "observed": 0.2,
            "absolute_error": pytest.approx(0.2),
        }
    ]


def test_controller_boundary_models_melee_radial_c_stick_clamp() -> None:
    rows = _boundary_trace_rows()
    replay = _boundary_replay_states()
    for row in rows:
        row["slots"]["p1"]["command"]["c_stick"] = [0.09375, 0.8125]
    for replay_frame in (11, 12):
        replay[replay_frame][1]["c_stick"] = (-0.7875, 0.6)
    audit = _audit_controller_boundary_records(rows, replay)
    assert audit["gate"]["decision"] == "pass"
    assert audit["slots"]["p1"]["processed_c_stick"]["mismatch_components"] == 0


def test_expected_processed_c_stick_applies_per_axis_deadzone_after_clamp() -> None:
    assert slippi_match_module._expected_processed_c_stick(
        "slippi-ai", {}, (1.0, 0.375)
    ) == (0.9625, 0.0)
    assert slippi_match_module._expected_processed_c_stick(
        "slippi-ai", {}, (0.875, 0.375)
    ) == (0.75, 0.0)


def test_expected_processed_c_stick_preserves_deadzone_boundary() -> None:
    assert slippi_match_module._expected_processed_c_stick(
        "slippi-ai", {}, (0.5 + 23 / 160, 0.5)
    ) == (23 / 80, 0.0)


def test_controller_boundary_rejects_wrong_value_near_radially_clamped_c_stick() -> (
    None
):
    rows = _boundary_trace_rows()
    replay = _boundary_replay_states()
    for row in rows:
        row["slots"]["p1"]["command"]["c_stick"] = [0.09375, 0.8125]
    for replay_frame in (11, 12):
        replay[replay_frame][1]["c_stick"] = (-0.7875, 0.6)
    replay[11][1]["c_stick"] = (-0.7625, 0.6)
    audit = _audit_controller_boundary_records(rows, replay)
    assert audit["gate"]["decision"] == "fail"
    assert not audit["gate"]["checks"]["both_slots_processed_c_stick_within_tolerance"]
    assert audit["slots"]["p1"]["processed_c_stick"]["mismatches"] == [
        {
            "trace_frame": 10,
            "replay_frame": 11,
            "component": "x",
            "expected": -0.7875,
            "observed": -0.7625,
            "absolute_error": pytest.approx(0.025),
        }
    ]


def test_controller_boundary_rejects_adjacent_raw_c_stick_bin() -> None:
    rows = _boundary_trace_rows()
    replay = _boundary_replay_states()
    for row in rows:
        row["slots"]["p1"]["command"]["c_stick"] = [0.09375, 0.8125]
    for replay_frame in (11, 12):
        replay[replay_frame][1]["c_stick"] = (-0.7875, 0.6)
    replay[11][1]["c_stick"] = (-0.775, 0.6)
    audit = _audit_controller_boundary_records(rows, replay)
    assert audit["gate"]["decision"] == "fail"
    assert not audit["gate"]["checks"]["both_slots_processed_c_stick_within_tolerance"]
    mismatch = audit["slots"]["p1"]["processed_c_stick"]["mismatches"]
    assert mismatch == [
        {
            "trace_frame": 10,
            "replay_frame": 11,
            "component": "x",
            "expected": -0.7875,
            "observed": -0.775,
            "absolute_error": pytest.approx(0.0125),
        }
    ]


def test_controller_boundary_rejects_right_trigger_mismatch() -> None:
    rows = _boundary_trace_rows()
    replay = _boundary_replay_states()
    for row in rows:
        row["slots"]["p2"]["command"]["btn_BUTTON_R"] = 1.0
    for replay_frame in (11, 12):
        replay[replay_frame][2]["buttons_physical"] = ("R", "Z")
        replay[replay_frame][2]["buttons_processed"] = ("A", "R", "Z")
        replay[replay_frame][2]["physical_analog_r"] = 1.0
    replay[11][2]["physical_analog_r"] = 0.25
    audit = _audit_controller_boundary_records(rows, replay)
    assert audit["gate"]["decision"] == "fail"
    assert not audit["gate"]["checks"][
        "both_slots_physical_analog_shoulders_within_tolerance"
    ]
    mismatch = audit["slots"]["p2"]["physical_analog_shoulders"]["mismatches"]
    assert mismatch[0]["component"] == "analog_r"
    assert mismatch[0]["expected"] == 1.0
    assert mismatch[0]["observed"] == 0.25


def test_controller_boundary_uses_native_slippi_ai_dummy_c_stick() -> None:
    rows = _boundary_trace_rows()
    replay = _boundary_replay_states()
    for row in rows:
        row["slots"]["p1"]["inference"] = {"native_dummy_prefix": True}
    for replay_frame in (11, 12):
        replay[replay_frame][1]["c_stick"] = (-0.7, -0.7)
    audit = _audit_controller_boundary_records(rows, replay)
    assert audit["gate"]["decision"] == "pass"
    assert audit["slots"]["p1"]["processed_c_stick"]["mismatch_components"] == 0


def test_controller_boundary_allows_only_one_terminal_unobservable_frame() -> None:
    replay = _boundary_replay_states()
    del replay[12]
    audit = _audit_controller_boundary_records(_boundary_trace_rows(), replay)
    assert audit["gate"]["decision"] == "pass"
    assert audit["alignment"]["permitted_terminal_unobservable_trace_frames"] == [11]
    assert audit["alignment"]["expected_overlap_pairs"] == 1
    assert audit["alignment"]["actual_overlap_pairs"] == 1


def test_controller_boundary_rejects_a_severely_truncated_replay() -> None:
    template = _boundary_trace_rows()[0]
    rows = []
    for frame in range(10, 20):
        row = copy.deepcopy(template)
        row["game_frame"] = frame
        rows.append(row)
    replay = {
        frame: value
        for frame, value in _boundary_replay_states().items()
        if frame <= 11
    }
    audit = _audit_controller_boundary_records(rows, replay)
    assert audit["gate"]["decision"] == "fail"
    assert not audit["gate"]["checks"]["replay_covers_every_observed_trace_state"]
    assert not audit["gate"]["checks"]["terminal_unobservable_tail_bounded_by_lag"]
    assert audit["alignment"]["permitted_terminal_unobservable_trace_frames"] == list(
        range(11, 20)
    )


def test_console_shutdown_allows_replay_writer_to_flush() -> None:

    class Controller:
        def __init__(self) -> None:
            self.disconnected = False
            self.flushes = 0

        def flush(self) -> None:
            self.flushes += 1

        def disconnect(self) -> None:
            self.disconnected = True

    class Process:
        def __init__(self) -> None:
            self.signals: list[int] = []
            self.wait_timeouts: list[float] = []

        def poll(self) -> None:
            return None

        def send_signal(self, value: int) -> None:
            self.signals.append(value)

        def wait(self, timeout: float) -> None:
            self.wait_timeouts.append(timeout)

    process = Process()
    controllers = [Controller(), Controller()]
    console = SimpleNamespace(
        _process=process, controllers=controllers, stop=lambda: None
    )
    assert (
        _stop_console(console, replay_finalize_timeout_seconds=30.0)
        == "child-sigint-input-drain"
    )
    assert process.signals == [signal.SIGINT]
    assert len(process.wait_timeouts) == 1
    assert 0 < process.wait_timeouts[0] <= 1.0 / 60.0
    assert all((controller.flushes == 1 for controller in controllers))
    assert all((controller.disconnected for controller in controllers))
    assert console._process is None
    with pytest.raises(ValueError, match="replay finalize timeout must be positive"):
        _stop_console(console, replay_finalize_timeout_seconds=0.0)
