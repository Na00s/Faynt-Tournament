from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from melee_policy.integration import frisson_slippi_match as module
from melee_policy.integration.frisson_policy import FRISSON_FAMILY_CHECKPOINT_FORMAT
from melee_policy.integration.frisson_slippi_match import (
    FINAL_10M_CHECKPOINT_BYTES,
    FINAL_10M_CHECKPOINT_FRAMES,
    FINAL_10M_CHECKPOINT_PARAMETER_COUNT,
    FINAL_10M_CHECKPOINT_SHA256,
    FINAL_10M_CHECKPOINT_STEP,
    FINAL_CHECKPOINT_BYTES,
    FINAL_CHECKPOINT_FRAMES,
    FINAL_CHECKPOINT_PARAMETER_COUNT,
    FINAL_CHECKPOINT_SHA256,
    FINAL_CHECKPOINT_STEP,
    FRISSON_LAUNCHABLE_CHARACTERS,
    SLIPPI_EFFECTIVE_POLICY_DELAY_FRAMES,
    FrissonSlippiMatchRequest,
    _canonicalize_request,
    _css_characters,
    _first_context,
    _run_exact_frame,
    _selected_replay_identity_audit,
    _slippi_config,
    _slippi_request,
    _trace_row,
    _validate_match_contract,
    canonical_frisson_character,
)
from melee_policy.integration.match_runtime import _load_config
from melee_policy.integration.slippi_ai_policy import (
    DEFAULT_PLAYER_NAME,
    MEDIUM_V2_SUPPORTED_CHARACTERS,
    CanonicalControllerCommand,
    slippi_ai_release_contract,
)
from melee_policy.integration.slippi_match import _audit_controller_boundary_records

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/integration.toml"
FINAL_CHECKPOINT = (ROOT / ".e011-cache/final-winners/remote-verification/75m-step-86016.pt").resolve()
FINAL_10M_CHECKPOINT = (ROOT / ".e011-cache/final-winners/remote-verification/10m-step-122064.pt").resolve()


def _command(button: str) -> CanonicalControllerCommand:
    return CanonicalControllerCommand(
        main_stick=(0.25, 0.75),
        c_stick=(0.5, 0.5),
        analog_l=0.2,
        analog_r=0.0,
        buttons=(button,),
    )


def _checkpoint_identity(profile: str = "75m") -> dict[str, Any]:
    if profile == "10m":
        values = {
            "profile": "10m",
            "parameter_count": FINAL_10M_CHECKPOINT_PARAMETER_COUNT,
            "sha256": FINAL_10M_CHECKPOINT_SHA256,
            "byte_length": FINAL_10M_CHECKPOINT_BYTES,
            "path": str(FINAL_10M_CHECKPOINT),
            "step": FINAL_10M_CHECKPOINT_STEP,
            "processed_target_frames": FINAL_10M_CHECKPOINT_FRAMES,
            "trial_id": "10m-muon-low",
            "wandb_run_id": "mpr-00d5f6f00b122510125b",
        }
    else:
        values = {
            "profile": "75m",
            "parameter_count": FINAL_CHECKPOINT_PARAMETER_COUNT,
            "sha256": FINAL_CHECKPOINT_SHA256,
            "byte_length": FINAL_CHECKPOINT_BYTES,
            "path": str(FINAL_CHECKPOINT),
            "step": FINAL_CHECKPOINT_STEP,
            "processed_target_frames": FINAL_CHECKPOINT_FRAMES,
            "trial_id": "75m-muon-low",
            "wandb_run_id": "mpr-7310252a57a246be08f1",
        }
    return {
        "format": FRISSON_FAMILY_CHECKPOINT_FORMAT,
        **{
            name: values[name]
            for name in (
                "profile",
                "parameter_count",
                "sha256",
                "byte_length",
                "path",
                "step",
                "processed_target_frames",
            )
        },
        "slippi_ai_commit": "577965a7731dc53e3472ea63d9e9853a4e9d65fa",
        "codec": {"name": "custom_v1", "vocab_sizes": {"buttons": 728, "main_stick": 85}},
        "training": {
            "trial_id": values["trial_id"],
            "wandb_run_id": values["wandb_run_id"],
        },
        "all_state_tensors_finite": True,
    }


class _Session:
    def __init__(
        self,
        name: str,
        command: CanonicalControllerCommand,
        barrier: threading.Barrier,
        completed: set[str],
    ) -> None:
        self.name = name
        self.command = command
        self.barrier = barrier
        self.completed = completed
        self.observations: list[Any] = []

    def step(self, gamestate: Any) -> CanonicalControllerCommand:
        self.observations.append(gamestate)
        self.barrier.wait(timeout=2)
        self.completed.add(self.name)
        return self.command


class _Transport:
    def __init__(self, completed: set[str]) -> None:
        self.completed = completed
        self.events: list[tuple[Any, ...]] = []

    def begin_boundary(self, *, reason: str, game_frame: int) -> None:
        assert self.completed == {"frisson", "slippi"}
        self.events.append(("begin", reason, game_frame))

    def schedule_next_boundary(self, port: int, *, reason: str, game_frame: int) -> None:
        self.events.append(("schedule", port, reason, game_frame))

    def commit_boundary(self) -> None:
        self.events.append(("commit",))


def _player(character: str) -> Any:
    return SimpleNamespace(
        character=SimpleNamespace(name=character),
        action=SimpleNamespace(value=14),
        stock=4,
        percent=0.0,
        position=SimpleNamespace(x=0.0, y=0.0),
        costume=0,
    )


def test_request_covers_full_frisson_roster_and_all_medium_v2_characters() -> None:
    assert len(FRISSON_LAUNCHABLE_CHARACTERS) == 26
    assert len(MEDIUM_V2_SUPPORTED_CHARACTERS) == 12
    for frisson_character in FRISSON_LAUNCHABLE_CHARACTERS:
        FrissonSlippiMatchRequest(player_1_character=frisson_character).validate()
    for slippi_character in MEDIUM_V2_SUPPORTED_CHARACTERS:
        FrissonSlippiMatchRequest(player_2_character=slippi_character).validate()


def test_player_2_ood_character_transfer_requires_explicit_opt_in() -> None:
    with pytest.raises(ValueError, match="medium-v2 does not support"):
        FrissonSlippiMatchRequest(player_2_character="GANONDORF").validate()

    request = FrissonSlippiMatchRequest(
        player_2_character="GANONDORF",
        allow_player_2_ood_character=True,
    )
    request.validate()
    internal = _slippi_request(request)
    assert internal.allow_player_2_ood_character is True


def test_ood_contract_preserves_checkpoint_roster_and_records_provenance() -> None:
    config, project_root = _load_config(CONFIG_PATH)
    request = _canonicalize_request(
        project_root,
        config,
        FrissonSlippiMatchRequest(
            player_2_character="GANONDORF",
            allow_player_2_ood_character=True,
        ),
    )
    contract = _validate_match_contract(config, project_root, request, _checkpoint_identity())
    slippi = contract["slippi_ai"]
    assert tuple(slippi["checkpoint_declared_characters"]) == MEDIUM_V2_SUPPORTED_CHARACTERS
    assert slippi["requested_character_covered_by_training"] is False
    assert slippi["forced_ood_character_transfer"] is True
    assert slippi["ood_provenance"]["checkpoint_roster_unchanged"] is True

    native = _slippi_config(config, project_root, request)
    assert native.requested_character == "GANONDORF"
    assert native.allow_ood_character is True
    assert native.sample_temperature == 1.0
    assert native.policy_delay_frames == 21


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("POPO", "POPO"),
        ("Ice Climbers", "POPO"),
        ("ice-climbers", "POPO"),
        ("Captain Falcon", "CPTFALCON"),
        ("Dr. Mario", "DOC"),
        ("Mr. Game & Watch", "GAMEANDWATCH"),
        ("Young Link", "YLINK"),
    ],
)
def test_frisson_character_aliases_are_explicit_and_canonical(
    alias: str,
    canonical: str,
) -> None:
    assert canonical_frisson_character(alias) == canonical
    FrissonSlippiMatchRequest(player_1_character=alias).validate()


@pytest.mark.parametrize(
    "override",
    [
        {"player_1_model": "slippi-ai"},
        {"player_2_model": "mimic"},
        {"player_1_character": "MASTER_HAND"},
        {"player_2_character": "GANONDORF"},
        {"stage": "BATTLEFIELD"},
        {"inference_mode": "asynchronous-latest"},
        {"require_natural_end": False},
    ],
)
def test_request_rejects_changes_to_the_frozen_match_contract(
    override: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        FrissonSlippiMatchRequest(**override).validate()


@pytest.mark.parametrize(
    ("checkpoint", "character", "expected_profile", "expected_character"),
    [
        (FINAL_10M_CHECKPOINT, "Ice Climbers", "10m", "POPO"),
        (FINAL_CHECKPOINT, "MARTH", "75m", "MARTH"),
    ],
)
def test_canonical_request_accepts_only_both_exact_final_winners(
    checkpoint: Path,
    character: str,
    expected_profile: str,
    expected_character: str,
) -> None:
    config, project_root = _load_config(CONFIG_PATH)
    request = _canonicalize_request(
        project_root,
        config,
        FrissonSlippiMatchRequest(
            player_1_checkpoint=checkpoint,
            player_1_character=character,
        ),
    )
    assert request.player_1_checkpoint == checkpoint
    assert request.player_1_character == expected_character
    identity = _checkpoint_identity(expected_profile)
    contract = _validate_match_contract(config, project_root, request, identity)
    assert contract["frisson"]["controlled_character"] == expected_character
    assert contract["frisson"]["selected_checkpoint"]["profile"] == expected_profile


def test_canonical_request_rejects_an_arbitrary_checkpoint(tmp_path: Path) -> None:
    config, project_root = _load_config(CONFIG_PATH)
    with pytest.raises(ValueError, match="two final winners"):
        _canonicalize_request(
            project_root,
            config,
            FrissonSlippiMatchRequest(player_1_checkpoint=tmp_path / "other.pt"),
        )


def test_contract_preserves_frisson_zero_delay_and_full_slippi_fifo() -> None:
    config, project_root = _load_config(CONFIG_PATH)
    request = _canonicalize_request(
        project_root,
        config,
        FrissonSlippiMatchRequest(player_2_character="SAMUS", seed=1301),
    )
    contract = _validate_match_contract(
        config,
        project_root,
        request,
        _checkpoint_identity(),
    )
    assert contract["same_current_state"] is True
    assert contract["frisson"]["action_offset_frames"] == 1
    assert contract["frisson"]["delay_frames"] == 0
    assert contract["frisson"]["codec"] == "custom_v1"
    assert contract["frisson"]["controlled_character"] == "FOX"
    assert tuple(contract["frisson"]["allowed_characters"]) == FRISSON_LAUNCHABLE_CHARACTERS
    assert contract["slippi_ai"]["player_name"] == DEFAULT_PLAYER_NAME
    assert contract["slippi_ai"]["sample_temperature"] == 1.0
    assert contract["slippi_ai"]["checkpoint_policy_delay_frames"] == 21
    assert contract["slippi_ai"]["console_delay_frames"] == 0
    assert contract["slippi_ai"]["effective_policy_delay_frames"] == 21
    assert contract["slippi_ai"]["configured_mixed_runtime"]["mixed_runtime_console_delay_frames"] == 0

    native = _slippi_config(config, project_root, request)
    assert native.port == 2 and native.opponent_port == 1
    assert native.name == DEFAULT_PLAYER_NAME
    assert native.sample_temperature == 1.0
    assert native.policy_delay_frames == 21
    assert native.console_delay_frames == 0
    assert native.async_inference is True
    assert native.compile is True


@pytest.mark.parametrize(
    ("release", "character"),
    (("dk_d18_imitation_v2", "DK"), ("doc_d18_imitation_v3", "DOC")),
)
def test_native_specialist_request_resolves_exact_release_and_delay(
    release: str,
    character: str,
) -> None:
    config, project_root = _load_config(CONFIG_PATH)
    request = _canonicalize_request(
        project_root,
        config,
        FrissonSlippiMatchRequest(
            player_1_character=character,
            player_2_character=character,
            player_2_slippi_release=release,
        ),
    )
    selected = slippi_ai_release_contract(release)
    assert request.player_2_checkpoint == selected.checkpoint_path(project_root)
    contract = _validate_match_contract(
        config,
        project_root,
        request,
        _checkpoint_identity(),
    )
    assert contract["slippi_ai"]["release"] == release
    assert contract["slippi_ai"]["checkpoint_sha256"] == selected.checkpoint_sha256
    assert contract["slippi_ai"]["checkpoint_policy_delay_frames"] == 18
    assert contract["slippi_ai"]["effective_policy_delay_frames"] == 18
    assert contract["slippi_ai"]["checkpoint_declared_characters"] == [character]
    assert contract["slippi_ai"]["requested_character_covered_by_training"] is True
    assert contract["slippi_ai"]["forced_ood_character_transfer"] is False

    native = _slippi_config(config, project_root, request)
    assert native.release == release
    assert native.policy_delay_frames == 18
    assert native.console_delay_frames == 0
    assert native.requested_character == character


def test_specialist_request_rejects_a_character_outside_native_release() -> None:
    with pytest.raises(ValueError, match="dk_d18_imitation_v2 does not support 'FOX'"):
        FrissonSlippiMatchRequest(
            player_2_character="FOX",
            player_2_slippi_release="dk_d18_imitation_v2",
        ).validate()


def test_frame_calls_both_policies_concurrently_before_shared_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_barrier = threading.Barrier(2)
    completed: set[str] = set()
    frisson = _Session("frisson", _command("A"), shared_barrier, completed)
    slippi = _Session("slippi", _command("B"), shared_barrier, completed)
    transport = _Transport(completed)
    controllers = {1: object(), 2: object()}
    dispatch_order: list[int] = []

    def fake_send(
        controller: object,
        _command_value: CanonicalControllerCommand,
        *,
        flush: bool,
    ) -> Any:
        assert completed == {"frisson", "slippi"}
        assert flush is False
        port = 1 if controller is controllers[1] else 2
        dispatch_order.append(port)
        return SimpleNamespace(as_dict=lambda: {"port": port})

    monkeypatch.setattr(module, "send_canonical_controller", fake_send)
    gamestate = SimpleNamespace(frame=-102)
    with ThreadPoolExecutor(max_workers=2) as executor:
        result = _run_exact_frame(
            gamestate=gamestate,
            processed_frames=21,
            frisson_session=frisson,  # type: ignore[arg-type]
            slippi_session=slippi,  # type: ignore[arg-type]
            executor=executor,
            controllers=controllers,
            transport=transport,  # type: ignore[arg-type]
        )

    assert frisson.observations == [gamestate]
    assert slippi.observations == [gamestate]
    assert result.both_submitted_before_wait is True
    assert result.slippi_delayed_source_frame == -123
    assert dispatch_order == [1, 2]
    assert transport.events[0] == ("begin", "frisson-vs-slippi-gameplay", -102)
    assert transport.events[-1] == ("commit",)


def test_inference_failure_prevents_controller_transaction() -> None:
    class RaisingSession:
        def step(self, _gamestate: Any) -> CanonicalControllerCommand:
            raise RuntimeError("inference failed")

    class ReturningSession:
        def step(self, _gamestate: Any) -> CanonicalControllerCommand:
            return _command("B")

    transport = _Transport(set())
    with (
        ThreadPoolExecutor(max_workers=2) as executor,
        pytest.raises(RuntimeError, match="inference failed"),
    ):
        _run_exact_frame(
            gamestate=SimpleNamespace(frame=-123),
            processed_frames=0,
            frisson_session=RaisingSession(),  # type: ignore[arg-type]
            slippi_session=ReturningSession(),  # type: ignore[arg-type]
            executor=executor,
            controllers={1: object(), 2: object()},
            transport=transport,  # type: ignore[arg-type]
        )
    assert transport.events == []


def test_trace_records_native_slippi_dummy_prefix_and_delayed_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gamestate = SimpleNamespace(
        frame=-123,
        players={1: _player("MARTH"), 2: _player("SAMUS")},
    )
    barrier = threading.Barrier(2)
    completed: set[str] = set()
    sessions = (
        _Session("frisson", _command("A"), barrier, completed),
        _Session("slippi", _command("B"), barrier, completed),
    )
    transport = _Transport(completed)
    monkeypatch.setattr(
        module,
        "send_canonical_controller",
        lambda _controller, command, *, flush: SimpleNamespace(as_dict=command.as_dict),
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        result = _run_exact_frame(
            gamestate=gamestate,
            processed_frames=0,
            frisson_session=sessions[0],  # type: ignore[arg-type]
            slippi_session=sessions[1],  # type: ignore[arg-type]
            executor=executor,
            controllers={1: object(), 2: object()},
            transport=transport,  # type: ignore[arg-type]
        )
    row = _trace_row(
        gamestate,
        FrissonSlippiMatchRequest(
            player_1_character="MARTH",
            player_2_character="SAMUS",
        ),
        result,
        frisson_display_name="Frisson-AI 10M v2 final step-122064",
    )
    assert row["slots"]["p1"]["requested_character"] == "MARTH"
    assert row["slots"]["p1"]["display_model"] == "Frisson-AI 10M v2 final step-122064"
    assert row["slots"]["p1"]["inference"]["delay_frames"] == 0
    assert row["slots"]["p2"]["inference"]["native_dummy_prefix"] is True
    assert row["slots"]["p2"]["inference"]["source_frame"] is None
    assert row["barrier"]["controller_transaction_opened_after_both"] is True


def test_css_and_first_context_bind_canonical_frisson_character() -> None:
    character_enum = {name: SimpleNamespace(name=name) for name in FRISSON_LAUNCHABLE_CHARACTERS}
    fake_melee = SimpleNamespace(Character=character_enum)
    request = FrissonSlippiMatchRequest(
        player_1_character="Ice Climbers",
        player_2_character="POPO",
    )
    selections = _css_characters(fake_melee, request)
    assert selections[1].name == "POPO"
    assert selections[2].name == "POPO"

    gamestate = SimpleNamespace(
        frame=-123,
        stage=SimpleNamespace(name="FINAL_DESTINATION"),
        players={1: _player("POPO"), 2: _player("POPO")},
    )
    context = _first_context(gamestate, request)
    assert context["requested_characters"] == {"p1": "POPO", "p2": "POPO"}
    assert context["checks"]["frisson_requested_character"] is True


def test_first_context_rejects_wrong_physical_frisson_character() -> None:
    gamestate = SimpleNamespace(
        frame=-123,
        stage=SimpleNamespace(name="FINAL_DESTINATION"),
        players={1: _player("FOX"), 2: _player("SAMUS")},
    )
    with pytest.raises(RuntimeError, match="context mismatch"):
        _first_context(
            gamestate,
            FrissonSlippiMatchRequest(
                player_1_character="MARTH",
                player_2_character="SAMUS",
            ),
        )


def test_public_play_routes_character_and_checkpoint_to_the_dedicated_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from melee_policy.integration import play

    captured: dict[str, Any] = {}

    def fake_run(config: Path, iso_path: Path | None, request: Any) -> dict[str, Any]:
        captured.update(config=config, iso_path=iso_path, request=request)
        return {"result": "captured"}

    monkeypatch.setattr(module, "run_frisson_slippi_match", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "play",
            "--p1",
            "frisson-ai",
            "--p2",
            "slippi-ai",
            "--p1-checkpoint",
            str(FINAL_10M_CHECKPOINT),
            "--p1-character",
            "SAMUS",
            "--p2-character",
            "SAMUS",
            "--p2-name",
            "Master Player",
            "--p2-temperature",
            "1.0",
            "--stage",
            "FINAL_DESTINATION",
            "--seed",
            "1301",
            "--require-natural-end",
        ],
    )
    play.main()
    request = captured["request"]
    assert isinstance(request, FrissonSlippiMatchRequest)
    assert request.player_1_checkpoint == FINAL_10M_CHECKPOINT
    assert request.player_1_character == "SAMUS"
    assert request.player_2_character == "SAMUS"
    assert request.seed == 1301


def test_physical_replay_audit_accepts_both_canonical_policy_ports() -> None:
    commands = {
        "p1": {
            "main_stick": [0.0, 0.5],
            "c_stick": [0.0, 0.5],
            "analog_l": 0.4,
            "analog_r": 0.0,
            "buttons": ["A"],
        },
        "p2": {
            "main_stick": [0.025, 0.5],
            "c_stick": [0.5, 0.5],
            "analog_l": 0.0,
            "analog_r": 0.0,
            "buttons": ["Z"],
        },
    }
    rows = [
        {
            "game_frame": frame,
            "slots": {
                "p1": {
                    "port": 1,
                    "model": "frisson-ai",
                    "command": commands["p1"],
                    "inference": {"native_dummy_prefix": False},
                    "controller_dispatch": {"called": True},
                },
                "p2": {
                    "port": 2,
                    "model": "slippi-ai",
                    "command": commands["p2"],
                    "inference": {"native_dummy_prefix": False},
                    "controller_dispatch": {"called": True},
                },
            },
        }
        for frame in (10, 11)
    ]
    replay = {
        frame: {
            1: {
                "buttons_physical": ("A",),
                "buttons_processed": ("A",),
                "raw_main_stick": (-80, 0),
                "main_stick": (0.0, 0.5),
                "c_stick": (-1.0, 0.0),
                "analog_l": 0.4,
                "physical_analog_l": 0.4,
                "physical_analog_r": 0.0,
            },
            2: {
                "buttons_physical": ("Z",),
                "buttons_processed": ("A", "Z"),
                "raw_main_stick": (-76, 0),
                "main_stick": (0.025, 0.5),
                "c_stick": (0.0, 0.0),
                "analog_l": 0.0,
                "physical_analog_l": 0.0,
                "physical_analog_r": 0.0,
            },
        }
        for frame in (10, 11, 12)
    }
    audit = _audit_controller_boundary_records(rows, replay, lag_frames=1)
    assert audit["gate"]["decision"] == "pass"
    assert audit["slots"]["p1"]["model"] == "frisson-ai"
    assert audit["slots"]["p2"]["model"] == "slippi-ai"
    assert audit["gate"]["checks"]["both_slots_physical_buttons_exact"] is True
    assert audit["gate"]["checks"]["both_slots_intended_raw_main_stick_exact"] is True


def test_selected_replay_identity_audits_both_ports_and_ice_climbers_alias(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from melee_policy.integration import replay_result

    replay_path = tmp_path / "game.slp"
    replay_path.write_bytes(b"physical replay fixture")
    captured: dict[str, Any] = {}

    def fake_audit(
        path: Path,
        *,
        expected_stage: str,
        expected_characters: dict[int, str],
    ) -> dict[str, Any]:
        captured.update(
            path=path,
            expected_stage=expected_stage,
            expected_characters=expected_characters,
        )
        return {
            "checks": {
                "replay_parsed": True,
                "peppi_parser_version_exact": True,
                "exact_expected_ports": True,
                "human_slots": True,
                "exact_expected_stage": True,
                "exact_expected_characters": True,
            }
        }

    monkeypatch.setattr(replay_result, "audit_replay", fake_audit)
    audit = _selected_replay_identity_audit(
        [replay_path],
        [{"tournament_result_replay": True}],
        FrissonSlippiMatchRequest(
            player_1_character="POPO",
            player_2_character="POPO",
        ),
        ROOT,
    )
    assert audit["decision"] == "pass"
    assert captured["expected_stage"] == "FINAL_DESTINATION"
    assert captured["expected_characters"] == {1: "ICE_CLIMBERS", 2: "ICE_CLIMBERS"}
    assert audit["expected"]["characters"] == {"p1": "POPO", "p2": "POPO"}


def test_effective_policy_delay_constant_is_full_native_fifo() -> None:
    assert SLIPPI_EFFECTIVE_POLICY_DELAY_FRAMES == 21
