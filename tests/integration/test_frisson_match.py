from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from melee_policy.integration import frisson_match as frisson_match_module
from melee_policy.integration import replay_result as replay_result_module
from melee_policy.integration.frisson_match import (
    CONTROLLER_REPLAY_LAG_FRAMES,
    EXACT_INFERENCE_MODE,
    FRISSON_CHARACTER,
    FRISSON_SUPPORTED_CHARACTERS,
    MATCH_SEED,
    MATCH_STAGE,
    MIMIC_CHARACTER,
    MIMIC_RELEASED_ASSET_DIRECTORIES,
    MIMIC_SUPPORTED_CHARACTERS,
    FrissonMatchRequest,
    _adapt_trace_rows_for_shared_audit,
    _bind_mimic_runtime_contract,
    _console_options,
    _ExactFrameResult,
    _menu_character_selection,
    _mimic_character_gate_checks,
    _replay_character_expectation,
    _resolve_checkpoint,
    _resolve_mimic_bundle_selection,
    _run_exact_frame,
    _selected_replay_identity_audit,
    _trace_row,
    _validate_config_contract,
    _validate_first_context,
)
from melee_policy.integration.slippi_ai_policy import CanonicalControllerCommand


def _valid_config() -> dict[str, Any]:
    return {
        "integration": {
            "device": "cpu",
            "inference_mode": "synchronous-concurrent",
            "maximum_pending_snapshots_per_model": 1,
        },
        "mimic": {
            "character": "FOX",
            "decode_strategy": "categorical-sampling",
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 0.0,
        },
        "frisson_ai": {
            "source_revision": "268031e7bddebb4e8c7a40026cd0b95f824d9d0d",
            "runtime_source_revision": "2c535c04693dc9a232fc81f244cbd24cb2359771",
            "slippi_ai_commit": "577965a7731dc53e3472ea63d9e9853a4e9d65fa",
            "checkpoint_format": "melee_rl.rl_checkpoint.v1",
            "actor_seed": 0,
            "sample_temperature": 1.0,
            "action_offset_frames": 1,
            "delay_frames": 0,
            "context_mode": "ring",
            "context_length": 256,
            "controller_buttons_vocab": 728,
            "controller_main_stick_vocab": 85,
            "parameter_count": 20_040_877,
            "allowed_characters": list(FRISSON_SUPPORTED_CHARACTERS),
        },
    }


def _inspected_checkpoint_identity(
    tmp_path: Path,
    *,
    checkpoint_format: str,
    profile: str,
    parameter_count: int,
) -> dict[str, Any]:
    return {
        "format": checkpoint_format,
        "profile": profile,
        "path": str((tmp_path / f"{profile}.pt").resolve()),
        "sha256": "a" * 64,
        "byte_length": 123,
        "step": 17,
        "processed_target_frames": 456,
        "parameter_count": parameter_count,
        "slippi_ai_commit": "577965a7731dc53e3472ea63d9e9853a4e9d65fa",
        "codec": {
            "name": "custom_v1",
            "vocab_sizes": {"buttons": 728, "main_stick": 85},
        },
    }


def _command() -> CanonicalControllerCommand:
    return CanonicalControllerCommand(
        main_stick=(0.25, 0.75),
        c_stick=(0.5, 0.5),
        analog_l=0.35,
        analog_r=0.0,
        buttons=("A", "Y"),
    )


def _mimic_command() -> dict[str, Any]:
    return {
        "main_x": 0.5,
        "main_y": 0.5,
        "c_x": 0.5,
        "c_y": 0.5,
        "l_shldr": 0.0,
        "r_shldr": 0.0,
        "btn_BUTTON_A": 1,
        "btn_BUTTON_B": 0,
        "btn_BUTTON_X": 0,
        "btn_BUTTON_Y": 0,
        "btn_BUTTON_Z": 0,
        "btn_BUTTON_L": 0,
        "btn_BUTTON_R": 0,
    }


def _released_mimic_runtime(
    tmp_path: Path,
    *,
    checkpoint_character: str = "FOX",
    hash_bound: bool = True,
) -> Any:
    assets = (tmp_path / "fox-master").resolve()
    checkpoint = assets / "model.pt"
    digest = "d" * 64
    return SimpleNamespace(
        checkpoint_path=checkpoint,
        checkpoint_sha256=digest,
        asset_directory=assets,
        controlled_character=checkpoint_character,
        state=SimpleNamespace(prev_sent=None),
        bundle_identity={
            "name": "fox-master",
            "run_name": "fox-mastfox-20260625",
            "character": checkpoint_character,
            "checkpoint_sha256": digest,
            "checks": {
                "checkpoint_matches_asset_model": hash_bound,
                "metadata_character": True,
                "metadata_run_name": True,
                "config_run_name": True,
                "metadata_config_run_name": True,
                "sequence_length": True,
                "controller_vocabulary": True,
                "architecture": True,
            },
            "state_dictionary": {
                "strict": True,
                "missing_keys": [],
                "unexpected_keys": [],
            },
        },
    )


class _FakeFrissonSession:
    def __init__(self, events: list[str], *, failure: BaseException | None = None) -> None:
        self.events = events
        self.failure = failure

    def step(self, gamestate: Any) -> CanonicalControllerCommand:
        self.events.append("frisson-start")
        if self.failure is not None:
            raise self.failure
        self.events.append("frisson-complete")
        return _command()


class _FakeMimicWorker:
    def __init__(self, events: list[str], *, failure: BaseException | None = None) -> None:
        self.events = events
        self.failure = failure
        self.frame: int | None = None

    def submit(self, frame: int, snapshot: Any) -> None:
        assert snapshot == "snapshot"
        self.frame = frame
        self.events.append("mimic-submit")

    def wait_completed(self, frame: int) -> tuple[int, dict[str, torch.Tensor], float]:
        assert self.frame == frame
        if self.failure is not None:
            raise self.failure
        self.events.append("mimic-complete")
        return frame, {"buttons": torch.zeros(1)}, 0.012


class _FakeController:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def flush(self) -> None:
        self.events.append("mimic-native-flush")


class _FakeMimicPolicy:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.runtime = SimpleNamespace(state=SimpleNamespace(prev_sent=None))
        self.previous_executed_frame: int | None = None

    def observe(self, gamestate: Any) -> bool:
        self.events.append("mimic-observe")
        return True

    def snapshot(self) -> str:
        self.events.append("mimic-snapshot")
        return "snapshot"

    def decode_and_press(
        self,
        controller: _FakeController,
        prediction: dict[str, torch.Tensor],
        previous: dict[str, Any] | None,
        *,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> tuple[dict[str, Any], list[str], list[str]]:
        assert torch.equal(prediction["buttons"], torch.zeros(1))
        assert previous is None
        assert (temperature, top_k, top_p) == (1.0, 0, 0.0)
        self.events.append("mimic-decode")
        controller.flush()
        return _mimic_command(), ["A"], ["A"]

    def record_decoded_command(self, frame: int, sent: dict[str, Any]) -> None:
        assert frame == -123
        assert sent == _mimic_command()
        self.runtime.state.prev_sent = sent
        self.events.append("mimic-record")


class _FakeTransport:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def begin_boundary(self, *, reason: str, game_frame: int) -> None:
        assert reason == "frisson-vs-mimic-gameplay"
        assert game_frame == -123
        self.events.append("boundary-begin")

    def schedule_next_boundary(self, port: int, *, reason: str, game_frame: int) -> None:
        assert port == 1
        assert reason == "frisson-zero-delay-next-frame-command"
        assert game_frame == -123
        self.events.append("frisson-schedule")

    def commit_boundary(self) -> None:
        self.events.append("boundary-commit")


def test_request_defaults_preserve_the_fixed_matchup_and_default_seed() -> None:
    request = FrissonMatchRequest()
    request.validate()
    assert request.player_1_model == "frisson-ai"
    assert request.player_2_model == "mimic"
    assert request.player_1_character == FRISSON_CHARACTER == "FOX"
    assert request.player_2_character == MIMIC_CHARACTER == "FOX"
    assert request.stage == MATCH_STAGE == "FINAL_DESTINATION"
    assert request.seed == MATCH_SEED == 0
    assert request.inference_mode == EXACT_INFERENCE_MODE
    assert request.require_natural_end
    assert request.allow_player_2_ood_character is False


@pytest.mark.parametrize("character", MIMIC_SUPPORTED_CHARACTERS)
def test_request_accepts_each_released_mimic_character_as_a_true_mirror(character: str) -> None:
    request = FrissonMatchRequest(
        player_1_character=character,
        player_2_character=character,
    )
    request.validate()
    assert request.player_1_character == character
    assert request.player_2_character == character


@pytest.mark.parametrize("character", FRISSON_SUPPORTED_CHARACTERS)
def test_request_accepts_every_launchable_frisson_character(character: str) -> None:
    request = FrissonMatchRequest(player_1_character=character)
    request.validate()
    assert request.player_1_character == character


@pytest.mark.parametrize("character", MIMIC_SUPPORTED_CHARACTERS)
def test_menu_selection_binds_each_mirror_to_both_physical_ports(character: str) -> None:
    character_enum = SimpleNamespace(name=character)
    melee_module = SimpleNamespace(Character={character: character_enum})
    selected = _menu_character_selection(
        melee_module,
        FrissonMatchRequest(
            player_1_character=character,
            player_2_character=character,
        ),
    )
    assert selected == {1: character_enum, 2: character_enum}


def test_menu_selection_uses_the_ood_physical_character(tmp_path: Path) -> None:
    fox = SimpleNamespace(name="FOX")
    kirby = SimpleNamespace(name="KIRBY")
    melee_module = SimpleNamespace(Character={"FOX": fox, "KIRBY": kirby})
    assets = tmp_path / "fox-master"
    selected = _menu_character_selection(
        melee_module,
        FrissonMatchRequest(
            player_2_character="KIRBY",
            player_2_checkpoint=assets / "model.pt",
            player_2_assets=assets,
            allow_player_2_ood_character=True,
        ),
    )
    assert selected == {1: fox, 2: kirby}


@pytest.mark.parametrize(
    "override",
    [
        {"player_1_model": "mimic", "player_2_model": "frisson-ai"},
        {"stage": "BATTLEFIELD"},
        {"inference_mode": "asynchronous-latest"},
        {"require_natural_end": False},
    ],
)
def test_request_fails_closed_outside_fixed_scope(override: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="fixed match contract mismatch"):
        FrissonMatchRequest(**override).validate()


def test_request_rejects_a_character_without_a_released_mimic_bundle() -> None:
    with pytest.raises(ValueError, match="released MIMIC character"):
        FrissonMatchRequest(player_2_character="KIRBY").validate()


@pytest.mark.parametrize("value", [1, None, "true"])
def test_request_rejects_nonboolean_mimic_ood_opt_in(value: Any) -> None:
    with pytest.raises(TypeError, match="allow_player_2_ood_character must be a boolean"):
        FrissonMatchRequest(allow_player_2_ood_character=value).validate()


@pytest.mark.parametrize(
    "character",
    tuple(
        character
        for character in FRISSON_SUPPORTED_CHARACTERS
        if character not in MIMIC_SUPPORTED_CHARACTERS
    ),
)
def test_request_accepts_each_launchable_mimic_ood_character_with_explicit_bundle(
    tmp_path: Path,
    character: str,
) -> None:
    assets = tmp_path / "fox-master"
    request = FrissonMatchRequest(
        player_2_character=character,
        player_2_checkpoint=assets / "model.pt",
        player_2_assets=assets,
        allow_player_2_ood_character=True,
    )
    request.validate()
    assert request.player_2_character == character


@pytest.mark.parametrize(
    ("checkpoint", "assets"),
    [
        (None, None),
        (Path("fox-master/model.pt"), None),
        (None, Path("fox-master")),
    ],
)
def test_request_requires_both_explicit_bundle_paths_for_mimic_ood_transfer(
    checkpoint: Path | None,
    assets: Path | None,
) -> None:
    with pytest.raises(ValueError, match="explicit player_2_checkpoint and player_2_assets"):
        FrissonMatchRequest(
            player_2_character="KIRBY",
            player_2_checkpoint=checkpoint,
            player_2_assets=assets,
            allow_player_2_ood_character=True,
        ).validate()


@pytest.mark.parametrize("character", ["NANA", "ICE_CLIMBERS", "fox", ""])
def test_request_rejects_nonlaunchable_mimic_ood_physical_character(character: str) -> None:
    with pytest.raises(ValueError, match="launchable physical player 2 character"):
        FrissonMatchRequest(
            player_2_character=character,
            player_2_checkpoint=Path("fox-master/model.pt"),
            player_2_assets=Path("fox-master"),
            allow_player_2_ood_character=True,
        ).validate()


def test_unused_ood_opt_in_does_not_change_released_character_contract() -> None:
    baseline = _validate_config_contract(_valid_config(), FrissonMatchRequest())
    opted_in = _validate_config_contract(
        _valid_config(),
        FrissonMatchRequest(allow_player_2_ood_character=True),
    )
    assert opted_in == baseline


@pytest.mark.parametrize("character", ["NANA", "ICE_CLIMBERS", "fox", ""])
def test_request_rejects_nonlaunchable_frisson_character_aliases(character: str) -> None:
    with pytest.raises(ValueError, match="launchable Frisson character"):
        FrissonMatchRequest(player_1_character=character).validate()


@pytest.mark.parametrize("seed", [0, 1, 123, 2**32 - 1])
def test_request_accepts_distinct_evaluation_seeds(seed: int) -> None:
    FrissonMatchRequest(seed=seed).validate()


@pytest.mark.parametrize("seed", [-1, 2**32, True])
def test_request_rejects_invalid_evaluation_seeds(seed: int) -> None:
    with pytest.raises(ValueError, match="seed"):
        FrissonMatchRequest(seed=seed).validate()


def test_config_contract_has_no_slippi_ai_delay_or_tensorflow() -> None:
    contract = _validate_config_contract(_valid_config(), FrissonMatchRequest())
    assert contract["frisson_character"] == "FOX"
    assert contract["frisson_supported_characters"] == list(FRISSON_SUPPORTED_CHARACTERS)
    assert contract["frisson_character_selection"] == "actual-live-physical-character-id"
    assert contract["mimic_character"] == "FOX"
    assert contract["mimic_bundle_selection"] == "released-character-bundle"
    assert contract["delay_frames"] == 0
    assert contract["action_offset_frames"] == 1
    assert contract["sample_temperature"] == 1.0
    assert contract["controller_replay_lag_frames"] == CONTROLLER_REPLAY_LAG_FRAMES == 1
    assert contract["actor_trajectory_context_frames"] == 128
    assert contract["kv_cache_capacity_frames"] == 256
    assert contract["observation_filter"] is None
    assert contract["tensorflow_runtime"] is False
    assert contract["slippi_ai_policy_fifo"] is False
    assert contract["native_dummy_prefix"] is False
    assert contract["selected_checkpoint"] == {
        "inspected": False,
        "format": "melee_rl.rl_checkpoint.v1",
        "profile": "20m",
        "parameter_count": 20_040_877,
    }

    invalid = _valid_config()
    invalid["frisson_ai"]["delay_frames"] = 21
    with pytest.raises(ValueError, match=r"frisson\.delay_frames"):
        _validate_config_contract(invalid, FrissonMatchRequest())


@pytest.mark.parametrize(
    ("character", "directory_name"),
    tuple(MIMIC_RELEASED_ASSET_DIRECTORIES.items()),
)
def test_mimic_character_selects_its_complete_released_bundle(
    tmp_path: Path,
    character: str,
    directory_name: str,
) -> None:
    config = {
        "mimic": {
            "checkpoint": ".e001-cache/mimic/fox-master/model.pt",
            "asset_directory": ".e001-cache/mimic/fox-master",
        }
    }
    checkpoint, assets = _resolve_mimic_bundle_selection(
        config,
        tmp_path,
        FrissonMatchRequest(player_2_character=character),
    )
    expected = (
        tmp_path / ".e012-cache" / "mimic-native-0629eb17" / directory_name
    ).resolve()
    assert assets == expected
    assert checkpoint == expected / "model.pt"


def test_current_mimic_roster_uses_master_fox_and_only_true_uncovered_ood() -> None:
    assert len(MIMIC_SUPPORTED_CHARACTERS) == 23
    assert MIMIC_RELEASED_ASSET_DIRECTORIES["FOX"] == "fox-master"
    assert set(FRISSON_SUPPORTED_CHARACTERS) - set(MIMIC_SUPPORTED_CHARACTERS) == {
        "KIRBY",
        "PICHU",
        "ZELDA",
    }


def test_explicit_mimic_assets_imply_their_model_checkpoint(tmp_path: Path) -> None:
    assets = tmp_path / "released-marth"
    checkpoint, selected_assets = _resolve_mimic_bundle_selection(
        {"mimic": {}},
        tmp_path,
        FrissonMatchRequest(player_2_character="MARTH", player_2_assets=assets),
    )
    assert selected_assets == assets
    assert checkpoint == assets / "model.pt"


def test_mimic_ood_bundle_resolution_uses_both_explicit_paths(tmp_path: Path) -> None:
    assets = tmp_path / "fox-master"
    checkpoint = assets / "model.pt"
    selected_checkpoint, selected_assets = _resolve_mimic_bundle_selection(
        {"mimic": {}},
        tmp_path,
        FrissonMatchRequest(
            player_2_character="KIRBY",
            player_2_checkpoint=checkpoint,
            player_2_assets=assets,
            allow_player_2_ood_character=True,
        ),
    )
    assert selected_checkpoint == checkpoint.resolve()
    assert selected_assets == assets.resolve()


def test_mimic_ood_runtime_contract_separates_checkpoint_and_physical_character(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "fox-master"
    request = FrissonMatchRequest(
        player_2_character="KIRBY",
        player_2_checkpoint=assets / "model.pt",
        player_2_assets=assets,
        allow_player_2_ood_character=True,
    )
    baseline = _validate_config_contract(_valid_config(), request)
    runtime = _released_mimic_runtime(tmp_path)
    bound = _bind_mimic_runtime_contract(baseline, request, runtime, tmp_path)

    assert baseline["mimic_bundle_selection"] == "released-character-bundle"
    assert "mimic_ood_character_transfer" not in baseline
    assert bound["mimic_bundle_selection"] == "explicit-released-bundle-ood-character-transfer"
    assert bound["mimic_character"] == "KIRBY"
    assert bound["mimic_physical_character"] == "KIRBY"
    assert bound["mimic_checkpoint_character"] == "FOX"
    assert bound["mimic_requested_character_covered_by_released_checkpoint"] is False
    assert bound["mimic_forced_ood_character_transfer"] is True
    provenance = bound["mimic_ood_character_transfer"]
    assert provenance["physical_character"] == "KIRBY"
    assert provenance["checkpoint_character"] == "FOX"
    assert provenance["assets"]["released_bundle_name"] == "fox-master"
    assert provenance["checkpoint"]["sha256"] == "d" * 64
    assert provenance["checkpoint_roster_unchanged"] is True
    assert provenance["physical_character_covered_by_released_checkpoint"] is False
    assert all(provenance["checks"].values())
    assert provenance["native_inference"] == {
        "live_policy_class": "MimicLivePolicy",
        "observation_builder": "tools.inference_utils.build_frame_p2",
        "model_forward": "MimicRuntime.model",
        "sampler_decoder": "tools.inference_utils.decode_and_press",
        "decode_strategy": "categorical-sampling",
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 0.0,
        "online_delay_frames": 0,
        "inference_mode": "synchronous-concurrent",
        "current_frame_barrier_before_controller_transaction": True,
    }


def test_mimic_ood_runtime_contract_requires_bundle_hash_binding(tmp_path: Path) -> None:
    assets = tmp_path / "fox-master"
    request = FrissonMatchRequest(
        player_2_character="KIRBY",
        player_2_checkpoint=assets / "model.pt",
        player_2_assets=assets,
        allow_player_2_ood_character=True,
    )
    with pytest.raises(RuntimeError, match="checkpoint_sha256_bound_to_released_bundle"):
        _bind_mimic_runtime_contract(
            _validate_config_contract(_valid_config(), request),
            request,
            _released_mimic_runtime(tmp_path, hash_bound=False),
            tmp_path,
        )


@pytest.mark.parametrize("checkpoint_character", ["KIRBY", "NANA", "UNKNOWN"])
def test_mimic_ood_runtime_contract_requires_a_distinct_released_checkpoint_character(
    tmp_path: Path,
    checkpoint_character: str,
) -> None:
    assets = tmp_path / "fox-master"
    request = FrissonMatchRequest(
        player_2_character="KIRBY",
        player_2_checkpoint=assets / "model.pt",
        player_2_assets=assets,
        allow_player_2_ood_character=True,
    )
    with pytest.raises(RuntimeError, match="provenance failed"):
        _bind_mimic_runtime_contract(
            _validate_config_contract(_valid_config(), request),
            request,
            _released_mimic_runtime(tmp_path, checkpoint_character=checkpoint_character),
            tmp_path,
        )


def test_released_mimic_runtime_contract_remains_unchanged(tmp_path: Path) -> None:
    contract = _validate_config_contract(_valid_config(), FrissonMatchRequest())
    bound = _bind_mimic_runtime_contract(
        contract,
        FrissonMatchRequest(),
        _released_mimic_runtime(tmp_path),
        tmp_path,
    )
    assert bound is contract


def test_mimic_ood_gate_binds_native_policy_runtime_and_provenance(tmp_path: Path) -> None:
    assets = tmp_path / "fox-master"
    request = FrissonMatchRequest(
        player_2_character="KIRBY",
        player_2_checkpoint=assets / "model.pt",
        player_2_assets=assets,
        allow_player_2_ood_character=True,
    )
    runtime = _released_mimic_runtime(tmp_path)
    contract = _bind_mimic_runtime_contract(
        _validate_config_contract(_valid_config(), request),
        request,
        runtime,
        tmp_path,
    )
    policy = frisson_match_module.MimicLivePolicy(
        runtime,
        2,
        online_delay_frames=0,
        evaluation_seed=0,
    )
    checks = _mimic_character_gate_checks(request, contract, runtime, policy)
    assert set(checks) == {
        "requested_frisson_p1_vs_explicit_mimic_ood_p2_character_exact",
        "mimic_ood_character_provenance_exact",
        "mimic_ood_native_builder_model_sampler_decoder_preserved",
    }
    assert all(checks.values())


@pytest.mark.parametrize(
    ("checkpoint_format", "profile", "parameter_count"),
    [
        ("melee_rl.rl_checkpoint.v1", "20m", 20_040_877),
        ("melee_policy.bc_checkpoint.v1", "20m", 20_040_877),
        ("melee_policy.final_pretraining_checkpoint.v1", "10m", 10_163_629),
        ("melee_policy.final_pretraining_checkpoint.v1", "75m", 75_305_709),
    ],
)
def test_config_contract_binds_dynamic_inspected_checkpoint_identity(
    tmp_path: Path,
    checkpoint_format: str,
    profile: str,
    parameter_count: int,
) -> None:
    identity = _inspected_checkpoint_identity(
        tmp_path,
        checkpoint_format=checkpoint_format,
        profile=profile,
        parameter_count=parameter_count,
    )
    contract = _validate_config_contract(
        _valid_config(),
        FrissonMatchRequest(),
        checkpoint_identity=identity,
    )
    assert contract["selected_checkpoint"] == {
        "inspected": True,
        "format": checkpoint_format,
        "profile": profile,
        "parameter_count": parameter_count,
        "sha256": "a" * 64,
        "byte_length": 123,
        "path": identity["path"],
        "step": 17,
        "processed_target_frames": 456,
    }


def test_config_contract_rejects_inconsistent_inspected_family_identity(tmp_path: Path) -> None:
    identity = _inspected_checkpoint_identity(
        tmp_path,
        checkpoint_format="melee_policy.final_pretraining_checkpoint.v1",
        profile="10m",
        parameter_count=75_305_709,
    )
    with pytest.raises(ValueError, match="parameter count mismatch"):
        _validate_config_contract(
            _valid_config(),
            FrissonMatchRequest(),
            checkpoint_identity=identity,
        )


def test_family_override_does_not_relax_the_pinned_default_config(tmp_path: Path) -> None:
    invalid = _valid_config()
    invalid["frisson_ai"]["parameter_count"] = 10_163_629
    identity = _inspected_checkpoint_identity(
        tmp_path,
        checkpoint_format="melee_policy.final_pretraining_checkpoint.v1",
        profile="10m",
        parameter_count=10_163_629,
    )
    with pytest.raises(ValueError, match=r"frisson\.parameter_count"):
        _validate_config_contract(
            invalid,
            FrissonMatchRequest(),
            checkpoint_identity=identity,
        )


def test_console_is_blocking_zero_delay_and_saves_unfoldered_replays(tmp_path: Path) -> None:
    options = _console_options(tmp_path / "replays", 51441)
    assert options["blocking_input"] is True
    assert options["online_delay"] == 0
    assert options["save_replays"] is True
    assert options["replay_dir"] == str(tmp_path / "replays")
    assert options["replay_monthly_folders"] is False
    assert options["enable_ffw"] is False


def test_default_and_explicit_checkpoints_must_match_configured_identities(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"frisson-checkpoint")
    configured = {
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_byte_length": checkpoint.stat().st_size,
        "explicit_checkpoint_identities": [],
    }
    assert _resolve_checkpoint(configured, tmp_path, None) == checkpoint.resolve()

    different = tmp_path / "different.pt"
    different.write_bytes(b"different")
    configured["explicit_checkpoint_identities"] = [
        {
            "name": "requested-checkpoint",
            "sha256": hashlib.sha256(different.read_bytes()).hexdigest(),
            "byte_length": different.stat().st_size,
        }
    ]
    assert _resolve_checkpoint(configured, tmp_path, different) == different.resolve()

    unknown = tmp_path / "unknown.pt"
    unknown.write_bytes(b"unknown")
    with pytest.raises(RuntimeError, match="checkpoint identity mismatch"):
        _resolve_checkpoint(configured, tmp_path, unknown)

    configured["checkpoint_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="checkpoint identity mismatch"):
        _resolve_checkpoint(configured, tmp_path, None)


def test_explicit_checkpoint_identity_catalog_fails_closed_when_malformed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    configured = {
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_byte_length": checkpoint.stat().st_size,
        "explicit_checkpoint_identities": [{"name": "broken", "sha256": "not-a-digest", "byte_length": 10}],
    }
    with pytest.raises(ValueError, match="invalid SHA-256"):
        _resolve_checkpoint(configured, tmp_path, checkpoint)


def test_exact_frame_waits_for_both_inferences_before_any_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Dispatch:
        def as_dict(self) -> dict[str, Any]:
            return {"flush": {"called": False}}

    def send(_controller: Any, command: CanonicalControllerCommand, *, flush: bool) -> Dispatch:
        assert command == _command()
        assert flush is False
        events.append("frisson-queue")
        return Dispatch()

    monkeypatch.setattr(frisson_match_module, "send_canonical_controller", send)
    result = _run_exact_frame(
        gamestate=SimpleNamespace(frame=-123),
        frisson_session=_FakeFrissonSession(events),  # type: ignore[arg-type]
        mimic_policy=_FakeMimicPolicy(events),  # type: ignore[arg-type]
        mimic_worker=_FakeMimicWorker(events),  # type: ignore[arg-type]
        frisson_controller=object(),
        mimic_controller=_FakeController(events),
        transport=_FakeTransport(events),  # type: ignore[arg-type]
        mimic_held=_mimic_command(),
        mimic_held_pressed=[],
        mimic_temperature=1.0,
        mimic_top_k=0,
        mimic_top_p=0.0,
    )
    assert result.frisson_command == _command()
    assert result.mimic_source_frame == -123
    begin = events.index("boundary-begin")
    assert events.index("frisson-complete") < begin
    assert events.index("mimic-complete") < begin
    assert events.index("frisson-queue") > begin
    assert events.index("mimic-decode") > begin
    assert events[-1] == "boundary-commit"


@pytest.mark.parametrize("failed_side", ["frisson", "mimic"])
def test_inference_failure_cannot_open_or_commit_a_controller_boundary(
    failed_side: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def forbidden_send(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("controller dispatch must not run after failed inference")

    monkeypatch.setattr(frisson_match_module, "send_canonical_controller", forbidden_send)
    session_failure = RuntimeError("frisson failed") if failed_side == "frisson" else None
    worker_failure = RuntimeError("mimic failed") if failed_side == "mimic" else None
    with pytest.raises(RuntimeError, match=f"{failed_side} failed"):
        _run_exact_frame(
            gamestate=SimpleNamespace(frame=-123),
            frisson_session=_FakeFrissonSession(  # type: ignore[arg-type]
                events,
                failure=session_failure,
            ),
            mimic_policy=_FakeMimicPolicy(events),  # type: ignore[arg-type]
            mimic_worker=_FakeMimicWorker(  # type: ignore[arg-type]
                events,
                failure=worker_failure,
            ),
            frisson_controller=object(),
            mimic_controller=_FakeController(events),
            transport=_FakeTransport(events),  # type: ignore[arg-type]
            mimic_held=_mimic_command(),
            mimic_held_pressed=[],
            mimic_temperature=1.0,
            mimic_top_k=0,
            mimic_top_p=0.0,
        )
    assert "boundary-begin" not in events
    assert "boundary-commit" not in events


@pytest.mark.parametrize("character", MIMIC_SUPPORTED_CHARACTERS)
def test_trace_preserves_each_mirror_character_and_explicit_barrier_evidence(
    character: str,
) -> None:
    player = lambda stock: SimpleNamespace(  # noqa: E731
        character=SimpleNamespace(name=character),
        action=SimpleNamespace(value=14),
        stock=stock,
        percent=12.5,
        position=SimpleNamespace(x=1.0, y=2.0),
    )
    gamestate = SimpleNamespace(frame=-123, players={1: player(4), 2: player(3)})
    result = _ExactFrameResult(
        game_frame=-123,
        frisson_command=_command(),
        frisson_dispatch={"called": True},
        frisson_inference_seconds=0.1,
        mimic_sent=_mimic_command(),
        mimic_pressed=["A"],
        mimic_inference_seconds=0.2,
        barrier_seconds=0.21,
        mimic_source_frame=-123,
    )
    row = _trace_row(
        gamestate,
        result,
        frisson_character=character,
        mimic_character=character,
    )
    assert row["barrier"]["controller_transaction_opened_after_both"] is True
    assert row["slots"]["p1"]["model"] == "frisson-ai"
    assert row["slots"]["p1"]["requested_character"] == character
    assert row["slots"]["p1"]["inference"]["delay_frames"] == 0
    assert row["slots"]["p1"]["inference"]["native_dummy_prefix"] is False
    assert row["slots"]["p2"]["model"] == "mimic"
    assert row["slots"]["p2"]["requested_character"] == character
    assert row["slots"]["p2"]["inference"]["command_age_frames"] == 0


def test_mimic_ood_trace_separates_physical_and_checkpoint_character() -> None:
    def player(character: str) -> Any:
        return SimpleNamespace(
            character=SimpleNamespace(name=character),
            action=SimpleNamespace(value=14),
            stock=4,
            percent=0.0,
            position=SimpleNamespace(x=0.0, y=0.0),
        )

    gamestate = SimpleNamespace(
        frame=-123,
        players={1: player("FOX"), 2: player("KIRBY")},
    )
    result = _ExactFrameResult(
        game_frame=-123,
        frisson_command=_command(),
        frisson_dispatch={"called": True},
        frisson_inference_seconds=0.1,
        mimic_sent=_mimic_command(),
        mimic_pressed=["A"],
        mimic_inference_seconds=0.2,
        barrier_seconds=0.21,
        mimic_source_frame=-123,
    )
    row = _trace_row(
        gamestate,
        result,
        frisson_character="FOX",
        mimic_character="KIRBY",
        mimic_checkpoint_character="FOX",
    )
    assert row["slots"]["p2"]["requested_character"] == "KIRBY"
    assert row["slots"]["p2"]["checkpoint_character"] == "FOX"
    assert row["slots"]["p2"]["player_state"]["character"] == "KIRBY"


def test_shared_replay_audit_adapter_does_not_relabel_saved_trace() -> None:
    original = {
        "game_frame": -123,
        "slots": {
            "p1": {
                "model": "frisson-ai",
                "inference": {"native_dummy_prefix": False},
            },
            "p2": {"model": "mimic"},
        },
    }
    adapted = _adapt_trace_rows_for_shared_audit([original])
    assert original["slots"]["p1"]["model"] == "frisson-ai"
    assert adapted[0]["slots"]["p1"]["model"] == "slippi-ai"
    assert adapted[0]["slots"]["p1"]["inference"]["native_dummy_prefix"] is False


def test_first_context_defaults_require_frame_minus_123_foxes_and_final_destination() -> None:
    def player(costume: int) -> Any:
        return SimpleNamespace(character=SimpleNamespace(name="FOX"), costume=costume)

    gamestate = SimpleNamespace(
        frame=-123,
        stage=SimpleNamespace(name="FINAL_DESTINATION"),
        players={1: player(0), 2: player(1)},
    )
    context = _validate_first_context(gamestate)
    assert all(context["checks"].values())
    assert context["costumes"] == {"p1": 0, "p2": 1}

    gamestate.stage = SimpleNamespace(name="BATTLEFIELD")
    with pytest.raises(RuntimeError, match="game context mismatch"):
        _validate_first_context(gamestate)


@pytest.mark.parametrize("character", MIMIC_SUPPORTED_CHARACTERS)
def test_first_context_accepts_each_requested_mirror_character(character: str) -> None:
    def player(character: str) -> Any:
        return SimpleNamespace(character=SimpleNamespace(name=character), costume=0)

    gamestate = SimpleNamespace(
        frame=-123,
        stage=SimpleNamespace(name="FINAL_DESTINATION"),
        players={1: player(character), 2: player(character)},
    )
    context = _validate_first_context(
        gamestate,
        frisson_character=character,
        mimic_character=character,
    )
    assert all(context["checks"].values())
    assert context["characters"] == {"p1": character, "p2": character}


def test_first_context_checks_the_mimic_ood_physical_character() -> None:
    def player(character: str) -> Any:
        return SimpleNamespace(character=SimpleNamespace(name=character), costume=0)

    gamestate = SimpleNamespace(
        frame=-123,
        stage=SimpleNamespace(name="FINAL_DESTINATION"),
        players={1: player("FOX"), 2: player("KIRBY")},
    )
    context = _validate_first_context(
        gamestate,
        frisson_character="FOX",
        mimic_character="KIRBY",
    )
    assert all(context["checks"].values())
    assert context["characters"] == {"p1": "FOX", "p2": "KIRBY"}


@pytest.mark.parametrize("character", MIMIC_SUPPORTED_CHARACTERS)
def test_selected_replay_identity_audit_proves_each_mirror_character(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    character: str,
) -> None:
    replay_path = tmp_path / f"{character.lower()}.slp"
    replay_path.write_bytes(character.encode())
    digest = hashlib.sha256(replay_path.read_bytes()).hexdigest()
    observed: dict[str, Any] = {}

    def audit_replay(
        path: Path,
        *,
        expected_stage: str,
        expected_characters: dict[int, str],
    ) -> dict[str, Any]:
        observed.update(
            {
                "path": path,
                "stage": expected_stage,
                "characters": expected_characters,
            }
        )
        return {
            "checks": {
                "replay_parsed": True,
                "peppi_parser_version_exact": True,
                "exact_expected_ports": True,
                "human_slots": True,
                "exact_expected_stage": True,
                "exact_expected_characters": True,
            },
            "replay": {
                "sha256": digest,
                "raw_byte_length": replay_path.stat().st_size,
            },
        }

    monkeypatch.setattr(replay_result_module, "audit_replay", audit_replay)
    result = _selected_replay_identity_audit(
        [replay_path],
        [
            {
                "path": replay_path.name,
                "sha256": digest,
                "byte_length": replay_path.stat().st_size,
                "tournament_result_replay": True,
            }
        ],
        tmp_path,
        FrissonMatchRequest(
            player_1_character=character,
            player_2_character=character,
        ),
    )
    assert result["decision"] == "pass"
    assert all(result["checks"].values())
    replay_character = "ICE_CLIMBERS" if character == "POPO" else character
    assert observed == {
        "path": replay_path,
        "stage": "FINAL_DESTINATION",
        "characters": {1: replay_character, 2: replay_character},
    }


def test_replay_identity_uses_the_parser_name_for_popo() -> None:
    assert _replay_character_expectation("POPO") == "ICE_CLIMBERS"
    assert _replay_character_expectation("SHEIK") == "SHEIK"


def test_selected_replay_identity_audit_fails_closed_on_character_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_path = tmp_path / "mismatch.slp"
    replay_path.write_bytes(b"replay")
    digest = hashlib.sha256(replay_path.read_bytes()).hexdigest()

    def audit_replay(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "checks": {
                "replay_parsed": True,
                "peppi_parser_version_exact": True,
                "exact_expected_ports": True,
                "human_slots": True,
                "exact_expected_stage": True,
                "exact_expected_characters": False,
            },
            "replay": {
                "sha256": digest,
                "raw_byte_length": replay_path.stat().st_size,
            },
        }

    monkeypatch.setattr(replay_result_module, "audit_replay", audit_replay)
    result = _selected_replay_identity_audit(
        [replay_path],
        [
            {
                "sha256": digest,
                "byte_length": replay_path.stat().st_size,
                "tournament_result_replay": True,
            }
        ],
        tmp_path,
        FrissonMatchRequest(player_1_character="MARTH", player_2_character="MARTH"),
    )
    assert result["decision"] == "fail"
    assert result["checks"]["exact_expected_characters"] is False
