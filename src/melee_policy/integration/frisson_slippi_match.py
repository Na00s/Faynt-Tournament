"""Run either final Frisson checkpoint against a pinned Slippi-AI release.

Both policy sessions receive the same current libmelee GameState. Their current
recurrent inference work runs concurrently, and the two-port controller
transaction opens only after both calls return. Frisson keeps its native t+1,
zero-delay custom_v1 action contract. Slippi-AI keeps its pinned parser,
observation filter, recurrent state, asynchronous worker, release-native policy
FIFO, player-name code, temperature, and native controller decoder.
"""

from __future__ import annotations

import json
import platform
import random
import signal
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TextIO, cast

import numpy as np
import torch

from melee_policy.integration.frame_watchdog import InGameNoFrameWatchdog
from melee_policy.integration.frisson_match import (
    CONTROLLER_REPLAY_LAG_FRAMES,
    EXACT_INFERENCE_MODE,
    FIRST_POLICY_FRAME,
    FRISSON_CHARACTER,
    FRISSON_PORT,
    FRISSON_SUPPORTED_CHARACTERS,
    MATCH_STAGE,
    FrissonMatchRequest,
    _collect_replays,
    _console_options,
    _frisson_policy_config,
    _player_state,
    _resolve_checkpoint,
    _selected_checkpoint_contract,
)
from melee_policy.integration.frisson_match import (
    _validate_config_contract as _validate_frisson_config_contract,
)
from melee_policy.integration.frisson_policy import (
    FRISSON_FAMILY_CHECKPOINT_FORMAT,
    FRISSON_POSTTRAINING_CHECKPOINT_FORMAT,
    FrissonPolicySession,
    inspect_frisson_checkpoint,
)
from melee_policy.integration.match_runtime import (
    _assert_udp_port_available,
    _attested_emulator_release,
    _ControllerPipeLockstep,
    _create_attested_dolphin_console,
    _display_path,
    _emulator_application_identity,
    _file_identity,
    _game_image_identity,
    _launch_and_connect_attested_dolphin,
    _legacy_replay_gate_checks,
    _load_config,
    _require_exact_player_ports,
    _require_unused_artifact_label,
    _resolve_game_image_path,
    _runtime_reproducibility_record,
    _stop_console,
    _validate_evaluation_seed,
    _write_json,
)
from melee_policy.integration.natural_game_end import has_decisive_zero_stock
from melee_policy.integration.slippi_ai_policy import (
    DEFAULT_PLAYER_NAME,
    SAMPLE_TEMPERATURE,
    SLIPPI_AI_REPOSITORY_URL,
    SLIPPI_AI_SOURCE_REVISION,
    CanonicalControllerCommand,
    SlippiAIPolicyConfig,
    SlippiAIPolicySession,
    send_canonical_controller,
    slippi_ai_release_contract,
)
from melee_policy.integration.slippi_match import (
    SlippiMatchRequest,
    _audit_controller_boundary_candidate,
    _game_start_transport_proof,
    _unavailable_controller_boundary,
)
from melee_policy.integration.slippi_match import (
    _validate_config_contract as _validate_slippi_config_contract,
)

SCHEMA_VERSION = "integration.frisson_vs_slippi_ai.v1"
TRACE_SCHEMA_VERSION = "integration.frisson_vs_slippi_ai.controller_trace.v1"
SLIPPI_DISPLAY_NAME = "vladfi1 Slippi-AI medium-v2 Master Player"
SLIPPI_PORT = 2

FRISSON_LAUNCHABLE_CHARACTERS = FRISSON_SUPPORTED_CHARACTERS
_FRISSON_CHARACTER_ALIASES: dict[str, str] = {
    **{character: character for character in FRISSON_LAUNCHABLE_CHARACTERS},
    "CAPTAIN_FALCON": "CPTFALCON",
    "DONKEY_KONG": "DK",
    "DR_MARIO": "DOC",
    "DRMARIO": "DOC",
    "MR_GAME_AND_WATCH": "GAMEANDWATCH",
    "MR_GAMEANDWATCH": "GAMEANDWATCH",
    "GAME_AND_WATCH": "GAMEANDWATCH",
    "ICE_CLIMBERS": "POPO",
    "ICECLIMBERS": "POPO",
    "YOUNG_LINK": "YLINK",
}


def canonical_frisson_character(value: str) -> str:
    """Map one public fighter name to the physical libmelee leader enum."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Frisson character must be a nonempty string")
    normalized = (
        value.strip()
        .upper()
        .replace(".", "")
        .replace("'", "")
        .replace("\N{RIGHT SINGLE QUOTATION MARK}", "")
        .replace("&", "AND")
        .replace("-", "_")
        .replace(" ", "_")
    )
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    try:
        return _FRISSON_CHARACTER_ALIASES[normalized]
    except KeyError as error:
        raise ValueError(
            f"unsupported Frisson character {value!r}; "
            f"allowed={FRISSON_LAUNCHABLE_CHARACTERS} plus documented aliases"
        ) from error


def _replay_character_name(character: str) -> str:
    """Return the replay-auditor name for a canonical libmelee fighter."""
    canonical = canonical_frisson_character(character)
    return "ICE_CLIMBERS" if canonical == "POPO" else canonical


@dataclass(frozen=True, slots=True)
class _FinalCheckpointSpec:
    profile: str
    checkpoint_format: str
    relative_path: Path
    sha256: str
    byte_length: int
    step: int
    processed_target_frames: int | None
    parameter_count: int
    trial_id: str
    wandb_run_id: str
    validation_nll: float | None

    @property
    def display_name(self) -> str:
        if self.checkpoint_format == "melee_policy.post_rl_ali_checkpoint.v1":
            return f"Frisson-AI {self.profile.upper()} post-RL step-{self.step}"
        qualifier = " v2" if self.profile == "10m" else ""
        return f"Frisson-AI {self.profile.upper()}{qualifier} final step-{self.step}"

    @property
    def artifact_checkpoint_slug(self) -> str:
        return f"{self.profile}-step{self.step}"


FINAL_10M_CHECKPOINT_RELATIVE_PATH = Path(".e011-cache/final-winners/remote-verification/10m-step-122064.pt")
FINAL_10M_CHECKPOINT_SHA256 = "5a54ea4ecfa150198180dff4d06ac4fe6ecec5d9433803cc13b527e41f41d14e"
FINAL_10M_CHECKPOINT_BYTES = 96_451_691
FINAL_10M_CHECKPOINT_STEP = 122_064
FINAL_10M_CHECKPOINT_FRAMES = 7_999_586_304
FINAL_10M_CHECKPOINT_PARAMETER_COUNT = 10_163_629
FINAL_10M_CHECKPOINT_PROFILE = "10m"
FINAL_10M_CHECKPOINT_TRIAL_ID = "10m-muon-low"
FINAL_10M_CHECKPOINT_WANDB_RUN_ID = "mpr-00d5f6f00b122510125b"
FINAL_10M_CHECKPOINT_VALIDATION_NLL = 0.7968172955299399

FINAL_CHECKPOINT_RELATIVE_PATH = Path(".e011-cache/final-winners/remote-verification/75m-step-86016.pt")
FINAL_CHECKPOINT_SHA256 = "8662de0c4c0deaae2879548d6373def792dbb52adec44f2b474adb7fe155c648"
FINAL_CHECKPOINT_BYTES = 644_291_987
FINAL_CHECKPOINT_STEP = 86_016
FINAL_CHECKPOINT_FRAMES = 5_637_144_576
FINAL_CHECKPOINT_PARAMETER_COUNT = 75_305_709
FINAL_CHECKPOINT_PROFILE = "75m"
FINAL_CHECKPOINT_TRIAL_ID = "75m-muon-low"
FINAL_CHECKPOINT_WANDB_RUN_ID = "mpr-7310252a57a246be08f1"
FINAL_CHECKPOINT_VALIDATION_NLL = 0.7648214009464961
FRISSON_DISPLAY_NAME = "Frisson-AI 75M final step-86016"

FINAL_CHECKPOINT_SPECS = (
    _FinalCheckpointSpec(
        profile=FINAL_10M_CHECKPOINT_PROFILE,
        checkpoint_format=FRISSON_FAMILY_CHECKPOINT_FORMAT,
        relative_path=FINAL_10M_CHECKPOINT_RELATIVE_PATH,
        sha256=FINAL_10M_CHECKPOINT_SHA256,
        byte_length=FINAL_10M_CHECKPOINT_BYTES,
        step=FINAL_10M_CHECKPOINT_STEP,
        processed_target_frames=FINAL_10M_CHECKPOINT_FRAMES,
        parameter_count=FINAL_10M_CHECKPOINT_PARAMETER_COUNT,
        trial_id=FINAL_10M_CHECKPOINT_TRIAL_ID,
        wandb_run_id=FINAL_10M_CHECKPOINT_WANDB_RUN_ID,
        validation_nll=FINAL_10M_CHECKPOINT_VALIDATION_NLL,
    ),
    _FinalCheckpointSpec(
        profile=FINAL_CHECKPOINT_PROFILE,
        checkpoint_format=FRISSON_FAMILY_CHECKPOINT_FORMAT,
        relative_path=FINAL_CHECKPOINT_RELATIVE_PATH,
        sha256=FINAL_CHECKPOINT_SHA256,
        byte_length=FINAL_CHECKPOINT_BYTES,
        step=FINAL_CHECKPOINT_STEP,
        processed_target_frames=FINAL_CHECKPOINT_FRAMES,
        parameter_count=FINAL_CHECKPOINT_PARAMETER_COUNT,
        trial_id=FINAL_CHECKPOINT_TRIAL_ID,
        wandb_run_id=FINAL_CHECKPOINT_WANDB_RUN_ID,
        validation_nll=FINAL_CHECKPOINT_VALIDATION_NLL,
    ),
    _FinalCheckpointSpec(
        profile="10m",
        checkpoint_format=FRISSON_POSTTRAINING_CHECKPOINT_FORMAT,
        relative_path=Path(
            ".e013-cache/posttraining-winners/frisson-melee-10m-posttrained-best-val.pt"
        ),
        sha256="63b5ff05ef30476c4f41590c478f4a7218f3b72a8b5ef24a0e336eeb2b7c287b",
        byte_length=96_452_075,
        step=195_248,
        processed_target_frames=12_795_772_928,
        parameter_count=10_163_629,
        trial_id="10m-muon-low",
        wandb_run_id="mp-01fa6731541b0fd42623",
        validation_nll=0.7574608703491693,
    ),
    _FinalCheckpointSpec(
        profile="75m",
        checkpoint_format=FRISSON_POSTTRAINING_CHECKPOINT_FORMAT,
        relative_path=Path(
            ".e013-cache/posttraining-winners/frisson-melee-75m-posttrained-best-val.pt"
        ),
        sha256="8211f1832198646e9f4e3bacde26f181614f93320e68bd326dd5942c4e0f4077",
        byte_length=644_291_987,
        step=127_214,
        processed_target_frames=8_337_096_704,
        parameter_count=75_305_709,
        trial_id="75m-muon-low",
        wandb_run_id="mp-ce9d4e1e59ea7f66ee1a",
        validation_nll=0.7248941140799456,
    ),
)
from melee_policy.integration.post_rl_checkpoints import CHECKPOINTS as POST_RL_CHECKPOINTS

FINAL_CHECKPOINT_SPECS += tuple(
    _FinalCheckpointSpec(
        profile=profile, checkpoint_format=row["format"],
        relative_path=Path(row["relative_path"]), sha256=row["sha256"],
        byte_length=row["byte_length"], step=row["step"], processed_target_frames=None,
        parameter_count=row["parameter_count"], trial_id=f"p21-{profile}-rl",
        wandb_run_id=row["wandb_run_id"], validation_nll=None,
    ) for profile, row in POST_RL_CHECKPOINTS.items()
)
SLIPPI_EFFECTIVE_POLICY_DELAY_FRAMES = 21
SLIPPI_CONSOLE_DELAY_FRAMES = 0


def _checkpoint_spec_for_path(project_root: Path, checkpoint: Path) -> _FinalCheckpointSpec:
    selected = checkpoint.expanduser().resolve()
    for spec in FINAL_CHECKPOINT_SPECS:
        if selected == (project_root / spec.relative_path).resolve():
            return spec
    allowed = [str((project_root / spec.relative_path).resolve()) for spec in FINAL_CHECKPOINT_SPECS]
    raise ValueError(f"Frisson checkpoint must be one of the two final winners: {allowed}")


def character_slug(character: str) -> str:
    """Return the stable artifact-label slug for a launchable fighter."""
    return canonical_frisson_character(character).lower().replace("_", "-")


def _slippi_display_name(request: FrissonSlippiMatchRequest) -> str:
    return slippi_ai_release_contract(request.player_2_slippi_release).display_name


@dataclass(frozen=True, slots=True)
class FrissonSlippiMatchRequest:
    """One exact final Frisson winner versus a pinned Slippi-AI release."""

    player_1_model: str = "frisson-ai"
    player_2_model: str = "slippi-ai"
    player_1_character: str = FRISSON_CHARACTER
    player_2_character: str = "FOX"
    stage: str = MATCH_STAGE
    player_1_checkpoint: Path | None = None
    player_2_checkpoint: Path | None = None
    player_2_slippi_release: str = "medium-v2"
    player_1_assets: Path | None = None
    player_2_assets: Path | None = None
    player_1_name: str | None = None
    player_2_name: str | None = None
    player_1_temperature: float | None = None
    player_2_temperature: float | None = None
    max_game_frames: int | None = None
    inference_mode: str = EXACT_INFERENCE_MODE
    artifact_label: str | None = None
    seed: int = 0
    require_natural_end: bool = True
    save_slp: bool = False
    save_video: bool = False
    allow_player_2_ood_character: bool = False

    def validate(self) -> None:
        release = slippi_ai_release_contract(self.player_2_slippi_release)
        fixed = {
            "player_1_model": (self.player_1_model.replace("_", "-").lower(), "frisson-ai"),
            "player_2_model": (self.player_2_model.replace("_", "-").lower(), "slippi-ai"),
            "stage": (self.stage, MATCH_STAGE),
            "inference_mode": (self.inference_mode, EXACT_INFERENCE_MODE),
            "require_natural_end": (self.require_natural_end, True),
        }
        mismatches = {
            name: {"observed": observed, "required": required}
            for name, (observed, required) in fixed.items()
            if observed != required
        }
        if mismatches:
            raise ValueError(f"Frisson-versus-Slippi fixed match contract mismatch: {mismatches}")
        canonical_frisson_character(self.player_1_character)
        if not isinstance(self.allow_player_2_ood_character, bool):
            raise TypeError("allow_player_2_ood_character must be a boolean")
        if self.player_2_character not in FRISSON_LAUNCHABLE_CHARACTERS:
            raise ValueError(f"player 2 character is not a launchable fighter: {self.player_2_character!r}")
        if (
            self.player_2_character not in release.supported_characters
            and not self.allow_player_2_ood_character
        ):
            raise ValueError(
                f"{release.key} does not support {self.player_2_character!r}; "
                f"allowed={release.supported_characters}"
            )
        if self.player_1_assets is not None or self.player_2_assets is not None:
            raise ValueError("Frisson and Slippi-AI use checkpoint-contained policy assets")
        if self.player_1_name is not None:
            raise ValueError("Frisson does not accept a Slippi-AI player name")
        if self.player_2_name not in (None, DEFAULT_PLAYER_NAME):
            raise ValueError(f"Slippi-AI player name is fixed to {DEFAULT_PLAYER_NAME!r}")
        if self.player_1_temperature not in (None, 1, 1.0):
            raise ValueError("Frisson sample temperature is fixed at 1.0")
        if self.player_2_temperature not in (None, SAMPLE_TEMPERATURE):
            raise ValueError("Slippi-AI sample temperature is fixed at 1.0")
        if self.max_game_frames is not None and self.max_game_frames < 1:
            raise ValueError("max_game_frames must be positive")
        if not isinstance(self.save_slp, bool) or not isinstance(self.save_video, bool):
            raise TypeError("save_slp and save_video must be booleans")
        _validate_evaluation_seed(self.seed)
        if self.artifact_label is not None and (
            not self.artifact_label
            or any(value not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for value in self.artifact_label)
        ):
            raise ValueError(
                "artifact_label must contain only lowercase letters, digits, underscores, or hyphens"
            )


@dataclass(frozen=True, slots=True)
class _PolicyStepResult:
    source_frame: int
    observation_identity: int
    command: CanonicalControllerCommand
    inference_seconds: float


@dataclass(frozen=True, slots=True)
class _ExactFrameResult:
    game_frame: int
    frisson: _PolicyStepResult
    slippi: _PolicyStepResult
    frisson_dispatch: dict[str, Any]
    slippi_dispatch: dict[str, Any]
    slippi_delayed_source_frame: int | None
    barrier_seconds: float
    both_submitted_before_wait: bool


def _resolve_project_path(project_root: Path, value: Path) -> Path:
    expanded = value.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root / expanded).resolve()


def _canonicalize_request(
    project_root: Path,
    config: dict[str, Any],
    request: FrissonSlippiMatchRequest,
) -> FrissonSlippiMatchRequest:
    canonical_player_1_character = canonical_frisson_character(request.player_1_character)
    release = slippi_ai_release_contract(request.player_2_slippi_release)
    final_checkpoint = (
        project_root / FINAL_CHECKPOINT_RELATIVE_PATH
        if request.player_1_checkpoint is None
        else _resolve_project_path(project_root, request.player_1_checkpoint)
    ).resolve()
    _checkpoint_spec_for_path(project_root, final_checkpoint)
    configured_medium_v2 = (project_root / str(config["slippi_ai"]["checkpoint"])).resolve()
    expected_slippi = (
        configured_medium_v2 if release.key == "medium-v2" else release.checkpoint_path(project_root)
    )
    slippi_checkpoint = (
        expected_slippi
        if request.player_2_checkpoint is None
        else _resolve_project_path(project_root, request.player_2_checkpoint)
    )
    canonical = replace(
        request,
        player_1_character=canonical_player_1_character,
        player_1_checkpoint=final_checkpoint,
        player_2_checkpoint=slippi_checkpoint,
        player_2_name=DEFAULT_PLAYER_NAME,
        player_1_temperature=1.0,
        player_2_temperature=SAMPLE_TEMPERATURE,
    )
    canonical.validate()
    if slippi_checkpoint != expected_slippi:
        raise ValueError(f"{release.key} checkpoint path must be {expected_slippi}")
    return canonical


def _frisson_request(request: FrissonSlippiMatchRequest) -> FrissonMatchRequest:
    return FrissonMatchRequest(
        player_1_character=canonical_frisson_character(request.player_1_character),
        player_1_checkpoint=request.player_1_checkpoint,
        player_1_temperature=request.player_1_temperature,
        seed=request.seed,
        max_game_frames=request.max_game_frames,
        inference_mode=request.inference_mode,
        require_natural_end=True,
    )


def _slippi_request(request: FrissonSlippiMatchRequest) -> SlippiMatchRequest:
    return SlippiMatchRequest(
        player_1_model="mimic",
        player_2_model="slippi-ai",
        player_1_character=canonical_frisson_character(request.player_1_character),
        player_2_character=request.player_2_character,
        player_2_checkpoint=request.player_2_checkpoint,
        player_2_name=DEFAULT_PLAYER_NAME,
        player_2_temperature=SAMPLE_TEMPERATURE,
        stage=request.stage,
        max_game_frames=request.max_game_frames,
        inference_mode=request.inference_mode,
        seed=request.seed,
        require_natural_end=True,
        allow_player_2_ood_character=request.allow_player_2_ood_character,
    )


def _final_checkpoint_contract(identity: dict[str, Any], expected_path: Path) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[3]
    spec = _checkpoint_spec_for_path(project_root, expected_path)
    selected = _selected_checkpoint_contract(identity)
    required = {
        "format": spec.checkpoint_format,
        "profile": spec.profile,
        "parameter_count": spec.parameter_count,
        "sha256": spec.sha256,
        "byte_length": spec.byte_length,
        "path": str(expected_path.resolve()),
        "step": spec.step,
        "processed_target_frames": spec.processed_target_frames,
    }
    mismatches = {
        name: {"observed": selected.get(name), "required": expected}
        for name, expected in required.items()
        if selected.get(name) != expected
    }
    training = identity.get("training")
    lineage_required = {
        "training.trial_id": (
            training.get("trial_id") if isinstance(training, dict) else None,
            spec.trial_id,
        ),
        "training.wandb_run_id": (
            training.get("wandb_run_id") if isinstance(training, dict) else None,
            spec.wandb_run_id,
        ),
        "all_state_tensors_finite": (identity.get("all_state_tensors_finite"), True),
    }
    mismatches.update(
        {
            name: {"observed": observed, "required": required_value}
            for name, (observed, required_value) in lineage_required.items()
            if observed != required_value
        }
    )
    if mismatches:
        raise ValueError(f"final {spec.profile.upper()} checkpoint identity mismatch: {mismatches}")
    return {
        **selected,
        "trial_id": spec.trial_id,
        "wandb_run_id": spec.wandb_run_id,
        "validation_nll": spec.validation_nll,
        "display_name": spec.display_name,
        "artifact_checkpoint_slug": spec.artifact_checkpoint_slug,
        "all_state_tensors_finite": True,
    }


def _validate_frisson_character_allowlist(config: dict[str, Any]) -> tuple[str, ...]:
    frisson = config.get("frisson_ai")
    if not isinstance(frisson, dict):
        raise ValueError("config must contain [frisson_ai]")
    raw = frisson.get("allowed_characters")
    if not isinstance(raw, list) or not all(isinstance(value, str) for value in raw):
        raise ValueError("frisson_ai.allowed_characters must be a list of fighter names")
    canonical = tuple(canonical_frisson_character(value) for value in raw)
    if len(canonical) != len(set(canonical)) or set(canonical) != set(FRISSON_LAUNCHABLE_CHARACTERS):
        raise ValueError(
            "frisson_ai.allowed_characters must cover each of the 26 launchable fighters exactly once"
        )
    return canonical


def _validate_slippi_release_config_contract(
    config: dict[str, Any],
    project_root: Path,
    request: FrissonSlippiMatchRequest,
) -> dict[str, Any]:
    release = slippi_ai_release_contract(request.player_2_slippi_release)
    if release.key == "medium-v2":
        result = _validate_slippi_config_contract(config, _slippi_request(request))
        return {**result, "release_contract": release.as_dict()}

    slippi = config.get("slippi_ai")
    integration = config.get("slippi_integration")
    if not isinstance(slippi, dict) or not isinstance(integration, dict):
        raise ValueError("config must contain [slippi_ai] and [slippi_integration]")
    required = {
        "repository_url": SLIPPI_AI_REPOSITORY_URL,
        "source_revision": SLIPPI_AI_SOURCE_REVISION,
        "compile": True,
        "async_inference": True,
    }
    mismatches = {
        name: {"observed": slippi.get(name), "required": expected}
        for name, expected in required.items()
        if slippi.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"Slippi-AI shared config contract mismatch: {mismatches}")
    if integration.get("exact_mode_only") is not True:
        raise ValueError("Slippi-AI integration must be exact-mode-only")
    policy_config = _slippi_config(config, project_root, request)
    policy_config.validate()
    return {
        "exact_mode_only": True,
        "inference_mode": EXACT_INFERENCE_MODE,
        "checkpoint_delay_frames": release.policy_delay_frames,
        "mixed_runtime_console_delay_frames": SLIPPI_CONSOLE_DELAY_FRAMES,
        "effective_policy_delay_frames": (release.policy_delay_frames - SLIPPI_CONSOLE_DELAY_FRAMES),
        "release_contract": release.as_dict(),
    }


def _validate_match_contract(
    config: dict[str, Any],
    project_root: Path,
    request: FrissonSlippiMatchRequest,
    checkpoint_identity: dict[str, Any],
) -> dict[str, Any]:
    request.validate()
    release = slippi_ai_release_contract(request.player_2_slippi_release)
    frisson_allowed_characters = _validate_frisson_character_allowlist(config)
    frisson_contract = _validate_frisson_config_contract(
        config,
        _frisson_request(request),
        checkpoint_identity=checkpoint_identity,
    )
    configured_slippi_contract = _validate_slippi_release_config_contract(
        config,
        project_root,
        request,
    )
    selected = _final_checkpoint_contract(
        checkpoint_identity,
        cast(Path, request.player_1_checkpoint),
    )
    if configured_slippi_contract["checkpoint_delay_frames"] != release.policy_delay_frames:
        raise ValueError(f"{release.key} checkpoint delay contract mismatch")
    effective_policy_delay_frames = release.policy_delay_frames - SLIPPI_CONSOLE_DELAY_FRAMES
    return {
        "schema_version": SCHEMA_VERSION,
        "inference_mode": EXACT_INFERENCE_MODE,
        "same_current_state": True,
        "controller_transaction": "opens after both current-state calls complete",
        "frisson": {
            "selected_checkpoint": selected,
            "controlled_character": canonical_frisson_character(request.player_1_character),
            "allowed_characters": list(frisson_allowed_characters),
            "character_selection": "physical CSS selection on configured Frisson port",
            "state_frame": frisson_contract["state_frame"],
            "command_frame": frisson_contract["command_frame"],
            "action_offset_frames": frisson_contract["action_offset_frames"],
            "delay_frames": frisson_contract["delay_frames"],
            "sample_temperature": frisson_contract["sample_temperature"],
            "codec": "custom_v1",
            "observation_filter": None,
            "slippi_ai_policy_fifo": False,
        },
        "slippi_ai": {
            "release": release.key,
            "release_contract": release.as_dict(),
            "source_revision": SLIPPI_AI_SOURCE_REVISION,
            "checkpoint_sha256": release.checkpoint_sha256,
            "checkpoint_byte_length": release.checkpoint_bytes,
            "parameter_count": release.parameter_count,
            "player_name": DEFAULT_PLAYER_NAME,
            "sample_temperature": SAMPLE_TEMPERATURE,
            "checkpoint_policy_delay_frames": release.policy_delay_frames,
            "policy_delay_frames": release.policy_delay_frames,
            "console_delay_frames": SLIPPI_CONSOLE_DELAY_FRAMES,
            "effective_policy_delay_frames": effective_policy_delay_frames,
            "configured_mixed_runtime": configured_slippi_contract,
            "async_inference": True,
            "compile": True,
            "native_parser_observation_filter_recurrent_state_and_decoder": True,
            "requested_character": request.player_2_character,
            "checkpoint_declared_characters": list(release.supported_characters),
            "requested_character_covered_by_training": (
                request.player_2_character in release.supported_characters
            ),
            "forced_ood_character_transfer": (
                request.allow_player_2_ood_character
                and request.player_2_character not in release.supported_characters
            ),
            "ood_provenance": (
                {
                    "explicit_opt_in": "allow_player_2_ood_character",
                    "physical_port": SLIPPI_PORT,
                    "checkpoint_roster_unchanged": True,
                    "playing_strength_interpretation": "out-of-training-roster transfer",
                }
                if request.allow_player_2_ood_character
                and request.player_2_character not in release.supported_characters
                else None
            ),
        },
    }


def _frisson_config(
    config: dict[str, Any],
    project_root: Path,
    request: FrissonSlippiMatchRequest,
) -> Any:
    return _frisson_policy_config(config, project_root, _frisson_request(request))


def _slippi_config(
    config: dict[str, Any],
    project_root: Path,
    request: FrissonSlippiMatchRequest,
) -> SlippiAIPolicyConfig:
    release = slippi_ai_release_contract(request.player_2_slippi_release)
    slippi = cast(dict[str, Any], config["slippi_ai"])
    result = SlippiAIPolicyConfig(
        source_directory=(project_root / str(slippi["source_directory"])).resolve(),
        checkpoint_path=cast(Path, request.player_2_checkpoint).resolve(),
        port=SLIPPI_PORT,
        opponent_port=FRISSON_PORT,
        release=release.key,
        name=DEFAULT_PLAYER_NAME,
        sample_temperature=SAMPLE_TEMPERATURE,
        policy_delay_frames=release.policy_delay_frames,
        console_delay_frames=SLIPPI_CONSOLE_DELAY_FRAMES,
        async_inference=True,
        compile=True,
        tf_jit_compile=False,
        batch_steps=0,
        mirror=False,
        requested_character=request.player_2_character,
        allow_ood_character=request.allow_player_2_ood_character,
    )
    result.validate()
    if result.port != SLIPPI_PORT or result.opponent_port != FRISSON_PORT:
        raise ValueError("Slippi-AI perspective is not bound to physical port 2 against port 1")
    return result


def _timed_step(session: Any, gamestate: Any) -> _PolicyStepResult:
    source_frame = int(gamestate.frame)
    observation_identity = id(gamestate)
    started = time.perf_counter()
    command = session.step(gamestate)
    elapsed = time.perf_counter() - started
    command.validate()
    return _PolicyStepResult(
        source_frame=source_frame,
        observation_identity=observation_identity,
        command=command,
        inference_seconds=elapsed,
    )


def _run_exact_frame(
    *,
    gamestate: Any,
    processed_frames: int,
    frisson_session: FrissonPolicySession,
    slippi_session: SlippiAIPolicySession,
    executor: ThreadPoolExecutor,
    controllers: dict[int, Any],
    transport: _ControllerPipeLockstep,
    slippi_effective_delay_frames: int = SLIPPI_EFFECTIVE_POLICY_DELAY_FRAMES,
) -> _ExactFrameResult:
    """Run both current-state policy calls before opening the pipe boundary."""
    game_frame = int(gamestate.frame)
    observation_identity = id(gamestate)
    barrier_started = time.perf_counter()
    futures = {
        FRISSON_PORT: executor.submit(_timed_step, frisson_session, gamestate),
        SLIPPI_PORT: executor.submit(_timed_step, slippi_session, gamestate),
    }
    both_submitted_before_wait = len(futures) == 2
    completed, pending = wait(tuple(futures.values()))
    if pending or len(completed) != 2:
        raise RuntimeError("both current-state policy calls did not complete")
    results = {port: futures[port].result() for port in (FRISSON_PORT, SLIPPI_PORT)}
    for port, result in results.items():
        if result.source_frame != game_frame:
            raise RuntimeError(
                f"policy port {port} returned source frame {result.source_frame} for {game_frame}"
            )
        if result.observation_identity != observation_identity:
            raise RuntimeError(f"policy port {port} did not receive the shared GameState object")
    barrier_seconds = time.perf_counter() - barrier_started

    transport.begin_boundary(reason="frisson-vs-slippi-gameplay", game_frame=game_frame)
    dispatches: dict[int, dict[str, Any]] = {}
    reasons = {
        FRISSON_PORT: "frisson-zero-delay-next-frame-command",
        SLIPPI_PORT: "slippi-ai-native-delayed-command",
    }
    for port in (FRISSON_PORT, SLIPPI_PORT):
        dispatch = send_canonical_controller(
            controllers[port],
            results[port].command,
            flush=False,
        ).as_dict()
        dispatch.update(
            {
                "called": True,
                "queued_after_both_current_state_calls": True,
                "console_step_preamble_flush_scheduled": True,
                "equivalent_next_step_boundary_flush": True,
            }
        )
        dispatches[port] = dispatch
        transport.schedule_next_boundary(port, reason=reasons[port], game_frame=game_frame)
    transport.commit_boundary()

    delayed_source = (
        None
        if processed_frames < slippi_effective_delay_frames
        else game_frame - slippi_effective_delay_frames
    )
    return _ExactFrameResult(
        game_frame=game_frame,
        frisson=results[FRISSON_PORT],
        slippi=results[SLIPPI_PORT],
        frisson_dispatch=dispatches[FRISSON_PORT],
        slippi_dispatch=dispatches[SLIPPI_PORT],
        slippi_delayed_source_frame=delayed_source,
        barrier_seconds=barrier_seconds,
        both_submitted_before_wait=both_submitted_before_wait,
    )


def _trace_row(
    gamestate: Any,
    request: FrissonSlippiMatchRequest,
    result: _ExactFrameResult,
    *,
    frisson_display_name: str = FRISSON_DISPLAY_NAME,
) -> dict[str, Any]:
    frame = result.game_frame
    delayed_source = result.slippi_delayed_source_frame
    release = slippi_ai_release_contract(request.player_2_slippi_release)
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "game_frame": frame,
        "barrier": {
            "source_frame": frame,
            "shared_gamestate_object": True,
            "both_submitted_before_wait": result.both_submitted_before_wait,
            "frisson_current_state_call_complete": True,
            "slippi_current_recurrent_call_complete": True,
            "controller_transaction_opened_after_both": True,
            "seconds": result.barrier_seconds,
        },
        "slots": {
            "p1": {
                "port": FRISSON_PORT,
                "model": "frisson-ai",
                "display_model": frisson_display_name,
                "requested_character": canonical_frisson_character(request.player_1_character),
                "player_state": _player_state(gamestate.players[FRISSON_PORT]),
                "command": result.frisson.command.as_dict(),
                "inference": {
                    "called": True,
                    "source_frame": frame,
                    "current_recurrent_source_frame": frame,
                    "command_age_frames": 0,
                    "seconds": result.frisson.inference_seconds,
                    "state_frame": "t",
                    "command_frame": "t+1",
                    "action_offset_frames": 1,
                    "delay_frames": 0,
                    "native_dummy_prefix": False,
                },
                "controller_dispatch": result.frisson_dispatch,
            },
            "p2": {
                "port": SLIPPI_PORT,
                "model": "slippi-ai",
                "display_model": release.display_name,
                "requested_character": request.player_2_character,
                "player_state": _player_state(gamestate.players[SLIPPI_PORT]),
                "command": result.slippi.command.as_dict(),
                "inference": {
                    "called": True,
                    "source_frame": delayed_source,
                    "current_recurrent_source_frame": frame,
                    "current_recurrent_command_age_frames": 0,
                    "command_age_frames": (None if delayed_source is None else frame - delayed_source),
                    "seconds": result.slippi.inference_seconds,
                    "native_dummy_prefix": delayed_source is None,
                    "policy_delay_frames": release.policy_delay_frames,
                    "console_delay_frames": SLIPPI_CONSOLE_DELAY_FRAMES,
                },
                "controller_dispatch": result.slippi_dispatch,
            },
        },
    }


def _first_context(gamestate: Any, request: FrissonSlippiMatchRequest) -> dict[str, Any]:
    stage = str(getattr(gamestate.stage, "name", gamestate.stage)).split(".")[-1]
    characters = {
        f"p{port}": str(
            getattr(gamestate.players[port].character, "name", gamestate.players[port].character)
        ).split(".")[-1]
        for port in (FRISSON_PORT, SLIPPI_PORT)
    }
    checks = {
        "first_frame_minus_123": int(gamestate.frame) == FIRST_POLICY_FRAME,
        "final_destination": stage == MATCH_STAGE,
        "frisson_requested_character": (
            characters["p1"] == canonical_frisson_character(request.player_1_character)
        ),
        "slippi_requested_character": characters["p2"] == request.player_2_character,
    }
    if not all(checks.values()):
        raise RuntimeError(f"first Frisson-versus-Slippi game context mismatch: {checks}")
    return {
        "frame": int(gamestate.frame),
        "stage": stage,
        "requested_characters": {
            "p1": canonical_frisson_character(request.player_1_character),
            "p2": request.player_2_character,
        },
        "characters": characters,
        "costumes": {
            f"p{port}": int(gamestate.players[port].costume) for port in (FRISSON_PORT, SLIPPI_PORT)
        },
        "checks": checks,
    }


def _selected_controller_audit(
    trace_path: Path,
    replay_paths: list[Path],
    replay_records: list[dict[str, Any]],
    project_root: Path,
    game_start_transport_proof: dict[str, Any],
) -> dict[str, Any]:
    selected = [
        index for index, record in enumerate(replay_records) if record.get("tournament_result_replay") is True
    ]
    if len(selected) != 1 or selected[0] >= len(replay_paths):
        return _unavailable_controller_boundary(
            trace_path,
            project_root,
            "controller audit requires exactly one trace-covering result replay",
        )
    audit = _audit_controller_boundary_candidate(
        trace_path,
        replay_paths[selected[0]],
        project_root,
        lag_frames=CONTROLLER_REPLAY_LAG_FRAMES,
        game_start_transport_proof=game_start_transport_proof,
    )
    audit["classification"] = (
        "Frisson and native Slippi-AI decoded commands verified at the shared physical boundary"
    )
    with trace_path.open("r", encoding="utf-8") as stream:
        trace_rows = sum(1 for _line in stream)
    audit["trace"] = {**_file_identity(trace_path, project_root), "rows": trace_rows}
    audit["replay"] = _file_identity(replay_paths[selected[0]], project_root)
    return audit


def _controller_gate_checks(audit: dict[str, Any]) -> dict[str, bool]:
    gate = cast(dict[str, Any], audit.get("gate", {}))
    checks = cast(dict[str, Any], gate.get("checks", {}))
    return {
        "controller_boundary_gate_pass": gate.get("decision") == "pass",
        "controller_boundary_full_causal_overlap": checks.get("full_causal_overlap") is True,
        "controller_boundary_both_slots_aligned": checks.get("both_slots_aligned") is True,
        "controller_boundary_physical_buttons_exact": (
            checks.get("both_slots_physical_buttons_exact") is True
        ),
        "controller_boundary_processed_buttons_exact": (
            checks.get("both_slots_processed_upstream_buttons_exact") is True
        ),
        "controller_boundary_raw_main_exact": (
            checks.get("both_slots_intended_raw_main_stick_exact") is True
        ),
        "controller_boundary_processed_c_stick_exact": (
            checks.get("both_slots_processed_c_stick_within_tolerance") is True
        ),
        "controller_boundary_physical_shoulders_exact": (
            checks.get("both_slots_physical_analog_shoulders_within_tolerance") is True
        ),
    }


def _selected_replay_identity_audit(
    replay_paths: list[Path],
    replay_records: list[dict[str, Any]],
    request: FrissonSlippiMatchRequest,
    project_root: Path,
) -> dict[str, Any]:
    """Audit the selected physical replay against both requested CSS slots."""
    selected = [
        index for index, record in enumerate(replay_records) if record.get("tournament_result_replay") is True
    ]
    required_checks = (
        "replay_parsed",
        "peppi_parser_version_exact",
        "exact_expected_ports",
        "human_slots",
        "exact_expected_stage",
        "exact_expected_characters",
    )
    expected = {
        "stage": request.stage,
        "characters": {
            "p1": canonical_frisson_character(request.player_1_character),
            "p2": request.player_2_character,
        },
        "replay_character_names": {
            "p1": _replay_character_name(request.player_1_character),
            "p2": _replay_character_name(request.player_2_character),
        },
    }
    if len(selected) != 1 or selected[0] >= len(replay_paths):
        return {
            "decision": "fail",
            "expected": expected,
            "checks": {name: False for name in required_checks},
            "error": "replay identity audit requires exactly one trace-covering result replay",
        }
    path = replay_paths[selected[0]]
    try:
        from melee_policy.integration.replay_result import audit_replay

        audit = audit_replay(
            path,
            expected_stage=request.stage,
            expected_characters={
                FRISSON_PORT: _replay_character_name(request.player_1_character),
                SLIPPI_PORT: _replay_character_name(request.player_2_character),
            },
        )
        raw_checks = audit.get("checks")
        audit_checks = raw_checks if isinstance(raw_checks, dict) else {}
        checks = {name: audit_checks.get(name) is True for name in required_checks}
        return {
            "decision": "pass" if all(checks.values()) else "fail",
            "expected": expected,
            "checks": checks,
            "replay": _file_identity(path, project_root),
            "audit": audit,
            "error": None,
        }
    except Exception as error:
        return {
            "decision": "fail",
            "expected": expected,
            "checks": {name: False for name in required_checks},
            "replay": _file_identity(path, project_root),
            "error": f"{type(error).__name__}: {error}",
        }


def _replay_identity_gate_checks(audit: dict[str, Any]) -> dict[str, bool]:
    checks = audit.get("checks")
    observed = checks if isinstance(checks, dict) else {}
    return {
        "replay_identity_audit_pass": audit.get("decision") == "pass",
        **{
            f"replay_identity.{name}": observed.get(name) is True
            for name in (
                "replay_parsed",
                "peppi_parser_version_exact",
                "exact_expected_ports",
                "human_slots",
                "exact_expected_stage",
                "exact_expected_characters",
            )
        },
    }


def _css_characters(melee: Any, request: FrissonSlippiMatchRequest) -> dict[int, Any]:
    """Resolve both canonical requested fighters through libmelee's enum."""
    return {
        FRISSON_PORT: melee.Character[canonical_frisson_character(request.player_1_character)],
        SLIPPI_PORT: melee.Character[request.player_2_character],
    }


def _run_console(
    *,
    config: dict[str, Any],
    project_root: Path,
    iso_path: Path,
    request: FrissonSlippiMatchRequest,
    contract: dict[str, Any],
    frisson_session: FrissonPolicySession,
    slippi_session: SlippiAIPolicySession,
    tensorflow_setup: dict[str, Any],
    reproducibility: dict[str, Any],
) -> dict[str, Any]:
    import melee

    selected_checkpoint = cast(dict[str, Any], contract["frisson"]["selected_checkpoint"])
    selected_checkpoint_spec = _checkpoint_spec_for_path(
        project_root,
        cast(Path, request.player_1_checkpoint),
    )
    slippi_release = slippi_ai_release_contract(request.player_2_slippi_release)
    slippi_effective_delay = slippi_release.policy_delay_frames - SLIPPI_CONSOLE_DELAY_FRAMES
    frisson_display_name = str(selected_checkpoint["display_name"])
    frisson_character = canonical_frisson_character(request.player_1_character)
    output_directory = project_root / str(config["frisson_ai"]["output_directory"])
    if request.artifact_label is not None:
        output_directory /= request.artifact_label
    _require_unused_artifact_label(output_directory, request.artifact_label)
    replay_directory = output_directory / "replays"
    trace_path = output_directory / "controller_trace.jsonl"
    summary_path = output_directory / "summary.json"
    output_directory.mkdir(parents=True, exist_ok=True)
    replay_directory.mkdir(parents=True, exist_ok=True)
    existing_replays = {path.resolve() for path in replay_directory.rglob("*.slp")}

    emulator_application = _emulator_application_identity(config, project_root)
    emulator_version = _attested_emulator_release(emulator_application)

    udp_port = int(config["emulator"]["slippi_port"])
    _assert_udp_port_available(udp_port)
    console = _create_attested_dolphin_console(
        config,
        project_root,
        emulator_application,
        **_console_options(replay_directory, udp_port),
    )
    raw_controllers = {
        port: melee.Controller(
            console=console,
            port=port,
            type=melee.ControllerType.STANDARD,
        )
        for port in (FRISSON_PORT, SLIPPI_PORT)
    }
    menu_helpers = {port: melee.MenuHelper() for port in (FRISSON_PORT, SLIPPI_PORT)}
    characters = _css_characters(melee, request)
    max_game_frames = (
        int(config["integration"]["max_game_frames"])
        if request.max_game_frames is None
        else request.max_game_frames
    )
    menu_timeout = float(config["integration"]["menu_timeout_seconds"])
    started_at = time.time()
    trace_stream: TextIO | None = None
    transport: _ControllerPipeLockstep | None = None
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="frisson-slippi-policy")
    in_game = False
    natural_game_end = False
    sudden_death_transition_observed = False
    termination = "not-started"
    exception: BaseException | None = None
    shutdown_method = "not-started"
    processed_frames = 0
    first_game_frame: int | None = None
    last_game_frame: int | None = None
    previous_game_frame: int | None = None
    first_context: dict[str, Any] | None = None
    frame_delta_counts: Counter[int] = Counter()
    inference_counts = {"frisson-ai": 0, "slippi-ai": 0}
    dispatch_counts = {"frisson-ai": 0, "slippi-ai": 0}
    inference_seconds: dict[str, list[float]] = {"frisson-ai": [], "slippi-ai": []}
    barrier_seconds: list[float] = []
    both_submitted_before_wait: list[bool] = []
    slippi_delayed_sources: list[int | None] = []
    stocks = {"frisson-ai": 4, "slippi-ai": 4}
    menu_flushes = {FRISSON_PORT: 0, SLIPPI_PORT: 0}
    no_frame_watchdog = InGameNoFrameWatchdog()

    def interrupt(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGINT, interrupt)
    try:
        trace_stream = trace_path.open("w", encoding="utf-8")
        if not _launch_and_connect_attested_dolphin(console, iso_path):
            raise RuntimeError("libmelee could not connect to Slippi Dolphin")
        if not all(controller.connect() for controller in raw_controllers.values()):
            raise RuntimeError("libmelee could not connect both virtual controllers")
        transport = _ControllerPipeLockstep.install(console, raw_controllers)
        controllers = transport.controllers
        transport.prime()
        print(
            f"P1={frisson_display_name.upper()} {frisson_character} | "
            f"P2={slippi_release.display_name.upper()} {request.player_2_character} | "
            f"stage={MATCH_STAGE} | seed={request.seed} | inference=exact-concurrent",
            flush=True,
        )

        while processed_frames < max_game_frames:
            gamestate = transport.step()
            no_frame_watchdog.observe_step_result(gamestate, gameplay_started=in_game)
            if gamestate is None:
                if not in_game and time.time() - started_at > menu_timeout:
                    raise TimeoutError("Slippi did not provide a menu or game state before timeout")
                continue
            if in_game and gamestate.menu_state == melee.Menu.SUDDEN_DEATH:
                natural_game_end = True
                sudden_death_transition_observed = True
                termination = "natural-game-end"
                break
            if gamestate.menu_state not in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
                if in_game:
                    natural_game_end = True
                    termination = "natural-game-end"
                    break
                if time.time() - started_at > menu_timeout:
                    raise TimeoutError("automatic menu navigation did not start a match before timeout")
                transport.begin_boundary(reason="menu", game_frame=None)
                for port in (FRISSON_PORT, SLIPPI_PORT):
                    menu_helpers[port].menu_helper_simple(
                        gamestate,
                        controllers[port],
                        characters[port],
                        melee.Stage.FINAL_DESTINATION,
                        cpu_level=0,
                        autostart=port == SLIPPI_PORT,
                        frozen_stadium=True,
                    )
                    controllers[port].flush()
                    menu_flushes[port] += 1
                transport.commit_boundary()
                continue

            in_game = True
            game_frame = int(gamestate.frame)
            _require_exact_player_ports(gamestate, game_frame)
            if first_game_frame is None:
                first_game_frame = game_frame
                first_context = _first_context(gamestate, request)
            if previous_game_frame is not None:
                delta = game_frame - previous_game_frame
                frame_delta_counts[delta] += 1
                if delta != 1:
                    raise RuntimeError(
                        f"rendered policy frames are not consecutive: {previous_game_frame} to {game_frame}"
                    )
            previous_game_frame = game_frame
            last_game_frame = game_frame

            frame_result = _run_exact_frame(
                gamestate=gamestate,
                processed_frames=processed_frames,
                frisson_session=frisson_session,
                slippi_session=slippi_session,
                executor=executor,
                controllers=controllers,
                transport=transport,
                slippi_effective_delay_frames=slippi_effective_delay,
            )
            trace_stream.write(
                json.dumps(
                    _trace_row(
                        gamestate,
                        request,
                        frame_result,
                        frisson_display_name=frisson_display_name,
                    ),
                    sort_keys=True,
                )
                + "\n"
            )
            trace_stream.flush()
            inference_counts["frisson-ai"] += 1
            inference_counts["slippi-ai"] += 1
            dispatch_counts["frisson-ai"] += 1
            dispatch_counts["slippi-ai"] += 1
            inference_seconds["frisson-ai"].append(frame_result.frisson.inference_seconds)
            inference_seconds["slippi-ai"].append(frame_result.slippi.inference_seconds)
            barrier_seconds.append(frame_result.barrier_seconds)
            both_submitted_before_wait.append(frame_result.both_submitted_before_wait)
            slippi_delayed_sources.append(frame_result.slippi_delayed_source_frame)
            stocks = {
                "frisson-ai": int(gamestate.players[FRISSON_PORT].stock),
                "slippi-ai": int(gamestate.players[SLIPPI_PORT].stock),
            }
            processed_frames += 1
            if processed_frames % 60 == 0:
                print(
                    f"frame {game_frame}: Frisson {stocks['frisson-ai']} stocks, "
                    f"Slippi-AI {stocks['slippi-ai']} stocks",
                    flush=True,
                )
            if has_decisive_zero_stock(gamestate, (FRISSON_PORT, SLIPPI_PORT)):
                natural_game_end = True
                termination = "natural-game-end"
                break
        if in_game and processed_frames >= max_game_frames:
            termination = "frame-limit-before-natural-end"
    except BaseException as caught:
        exception = caught
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        executor.shutdown(wait=True, cancel_futures=True)
        if trace_stream is not None:
            trace_stream.close()
        shutdown_method = _stop_console(
            console,
            float(config["integration"]["replay_finalize_timeout_seconds"]),
        )

    replay_paths, replay_records = _collect_replays(
        replay_directory,
        existing_replays,
        project_root,
        first_game_frame,
        last_game_frame,
        sudden_death_transition_observed,
    )
    replay_checks = _legacy_replay_gate_checks(
        replay_records,
        sudden_death_transition_observed=sudden_death_transition_observed,
    )
    transport_record = transport.audit_record() if transport is not None else {"installed": False}
    game_start_transport_proof = _game_start_transport_proof(transport_record)
    controller_audit = _selected_controller_audit(
        trace_path,
        replay_paths,
        replay_records,
        project_root,
        game_start_transport_proof,
    )
    controller_checks = _controller_gate_checks(controller_audit)
    replay_identity_audit = _selected_replay_identity_audit(
        replay_paths,
        replay_records,
        request,
        project_root,
    )
    replay_identity_checks = _replay_identity_gate_checks(replay_identity_audit)
    frisson_diagnostics = frisson_session.diagnostics()
    frisson_metadata = frisson_session.metadata()
    slippi_diagnostics = slippi_session.diagnostics()
    slippi_metadata = slippi_session.metadata()
    frisson_runtime = cast(dict[str, Any], frisson_metadata["runtime_contract"])
    frisson_action = cast(dict[str, Any], frisson_metadata["action_contract"])
    slippi_runtime = cast(dict[str, Any], slippi_metadata["runtime_contract"])
    slippi_source = cast(dict[str, Any], slippi_metadata["source"])
    slippi_checkpoint = cast(dict[str, Any], slippi_metadata["checkpoint"])
    slippi_upstream = cast(dict[str, Any], slippi_metadata["upstream_runtime"])
    expected_slippi_sources = [
        None if offset < slippi_effective_delay else FIRST_POLICY_FRAME + offset - slippi_effective_delay
        for offset in range(processed_frames)
    ]
    transport_checks = (
        transport.gate_checks() if transport is not None else {"controller_pipe_lockstep_installed": False}
    )
    gate_checks: dict[str, bool] = {
        "exact_final_frisson_p1_vs_selected_slippi_release_p2": (
            frisson_character in FRISSON_LAUNCHABLE_CHARACTERS
            and (
                request.player_2_character in slippi_release.supported_characters
                or request.allow_player_2_ood_character
            )
            and contract["slippi_ai"]["release"] == slippi_release.key
            and contract["frisson"]["controlled_character"] == frisson_character
        ),
        "slippi_character_semantics_explicit": (
            request.player_2_character in slippi_release.supported_characters
            or (
                request.allow_player_2_ood_character
                and contract["slippi_ai"]["forced_ood_character_transfer"] is True
                and contract["slippi_ai"]["requested_character_covered_by_training"] is False
            )
        ),
        "final_destination": request.stage == MATCH_STAGE,
        "evaluation_seed_valid": _validate_evaluation_seed(request.seed) == request.seed,
        "natural_game_end_observed": natural_game_end,
        "entered_gameplay": in_game,
        "processed_at_least_one_frame": processed_frames > 0,
        "first_policy_frame_minus_123": first_game_frame == FIRST_POLICY_FRAME,
        "first_game_context_exact": (
            first_context is not None and all(cast(dict[str, bool], first_context["checks"]).values())
        ),
        "strict_consecutive_policy_frames": set(frame_delta_counts) <= {1},
        "one_inference_per_model_per_frame": all(
            count == processed_frames for count in inference_counts.values()
        ),
        "one_dispatch_per_model_per_frame": all(
            count == processed_frames for count in dispatch_counts.values()
        ),
        "both_submitted_before_wait_every_frame": (
            len(both_submitted_before_wait) == processed_frames and all(both_submitted_before_wait)
        ),
        "both_current_state_calls_complete_before_every_boundary": (len(barrier_seconds) == processed_frames),
        "frisson_session_frame_count_exact": (frisson_diagnostics.get("frames_total") == processed_frames),
        "frisson_current_frame_barrier_exact": (
            frisson_diagnostics.get("current_frame_inference_barriers") == processed_frames
            and frisson_diagnostics.get("current_frame_inference_barrier_every_frame") is True
        ),
        "frisson_final_checkpoint_exact": (
            contract["frisson"]["selected_checkpoint"]["sha256"] == selected_checkpoint_spec.sha256
            and contract["frisson"]["selected_checkpoint"]["step"] == selected_checkpoint_spec.step
        ),
        "frisson_full_roster_allowlist_exact": (
            tuple(contract["frisson"]["allowed_characters"])
            == tuple(_validate_frisson_character_allowlist(config))
        ),
        "frisson_action_offset_one": frisson_action["action_offset_frames"] == 1,
        "frisson_delay_zero": frisson_runtime["delay_frames"] == 0,
        "frisson_temperature_one": frisson_runtime["sample_temperature"] == 1.0,
        "frisson_custom_v1_codec": frisson_action["codec"]
        == {"name": "custom_v1", "vocab_sizes": {"buttons": 728, "main_stick": 85}},
        "frisson_no_slippi_fifo": frisson_action["slippi_ai_21_frame_fifo"] is False,
        "slippi_session_frame_count_exact": (slippi_diagnostics.get("frames_total") == processed_frames),
        "slippi_current_recurrent_barrier_exact": (
            slippi_diagnostics.get("current_frame_inference_barriers") == processed_frames
            and slippi_diagnostics.get("current_frame_inference_barrier_every_frame") is True
        ),
        "slippi_native_decoder_capture_exact": (
            slippi_diagnostics.get("capture_decoder_assertions") == processed_frames
            and slippi_diagnostics.get("capture_decoder_mismatches") == 0
        ),
        "slippi_native_release_fifo_schedule_exact": (slippi_delayed_sources == expected_slippi_sources),
        "slippi_source_revision_exact": (
            slippi_source.get("revision") == SLIPPI_AI_SOURCE_REVISION
            and slippi_source.get("tracked_tree_clean") is True
        ),
        "slippi_checkpoint_exact": (
            slippi_checkpoint.get("release") == slippi_release.key
            and slippi_checkpoint.get("sha256") == slippi_release.checkpoint_sha256
            and slippi_checkpoint.get("byte_length") == slippi_release.checkpoint_bytes
            and slippi_upstream.get("parameter_count") == slippi_release.parameter_count
        ),
        "slippi_master_player_exact": (
            slippi_runtime.get("requested_player_name") == DEFAULT_PLAYER_NAME
            and slippi_runtime.get("effective_player_name") == DEFAULT_PLAYER_NAME
            and slippi_upstream.get("effective_name") == DEFAULT_PLAYER_NAME
        ),
        "slippi_temperature_one": slippi_runtime.get("sample_temperature") == SAMPLE_TEMPERATURE,
        "slippi_native_async_compile_exact": (
            slippi_runtime.get("async_inference") is True
            and slippi_runtime.get("compile") is True
            and slippi_runtime.get("tf_jit_compile") is False
        ),
        "slippi_delay_contract_exact": (
            slippi_runtime.get("policy_delay_frames") == slippi_release.policy_delay_frames
            and slippi_runtime.get("console_delay_frames") == SLIPPI_CONSOLE_DELAY_FRAMES
            and slippi_runtime.get("effective_policy_delay_frames") == slippi_effective_delay
        ),
        "tensorflow_cpu_only": (
            tensorflow_setup.get("physical_gpu_count") == 0 and tensorflow_setup.get("logical_gpu_count") == 0
        ),
        **replay_checks,
        **replay_identity_checks,
        **controller_checks,
        **transport_checks,
    }
    if exception is None and not all(gate_checks.values()):
        exception = RuntimeError(
            "Frisson-versus-Slippi integration gate failed: "
            f"{[name for name, passed in gate_checks.items() if not passed]}"
        )
    result = "complete" if exception is None and all(gate_checks.values()) else "failed"
    winner = (
        "frisson-ai"
        if stocks["frisson-ai"] > stocks["slippi-ai"]
        else "slippi-ai"
        if stocks["slippi-ai"] > stocks["frisson-ai"]
        else None
    )
    trace_artifact: dict[str, Any] = dict(_file_identity(trace_path, project_root))
    trace_artifact["rows"] = processed_frames
    summary = {
        "schema_version": SCHEMA_VERSION,
        "classification": (
            f"frame-exact final Frisson {selected_checkpoint_spec.profile.upper()} "
            f"{frisson_character} versus pinned vladfi1 Slippi-AI {slippi_release.key} "
            f"{request.player_2_character}"
        ),
        "result": result,
        "error": None if exception is None else f"{type(exception).__name__}: {exception}",
        "gate": {
            "decision": "pass" if result == "complete" else "fail",
            "checks": gate_checks,
        },
        "configuration": {
            "player_1": {
                "model": "frisson-ai",
                "display_model": frisson_display_name,
                "character": frisson_character,
                "port": FRISSON_PORT,
            },
            "player_2": {
                "model": "slippi-ai",
                "display_model": slippi_release.display_name,
                "release": slippi_release.key,
                "runtime_name": DEFAULT_PLAYER_NAME,
                "character": request.player_2_character,
                "port": SLIPPI_PORT,
            },
            "stage": MATCH_STAGE,
            "seed": request.seed,
            "require_natural_end": True,
            "maximum_game_frames": max_game_frames,
            "blocking_input": True,
            "online_delay_frames": 0,
            "controller_replay_lag_frames": CONTROLLER_REPLAY_LAG_FRAMES,
            "inference_mode": EXACT_INFERENCE_MODE,
        },
        "contract": contract,
        "policies": {
            "p1": {"metadata": frisson_metadata, "diagnostics": frisson_diagnostics},
            "p2": {"metadata": slippi_metadata, "diagnostics": slippi_diagnostics},
        },
        "reproducibility": reproducibility,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "libmelee_module": str(Path(melee.__file__).resolve()),
            "slippi_version": emulator_version,
            "tensorflow": tensorflow_setup,
        },
        "emulator_application": emulator_application,
        "game_image": _game_image_identity(iso_path),
        "execution": {
            "processed_policy_frames": processed_frames,
            "first_game_frame": first_game_frame,
            "last_game_frame": last_game_frame,
            "first_game_context": first_context,
            "frame_delta_counts": {str(delta): count for delta, count in sorted(frame_delta_counts.items())},
            "inference_counts": inference_counts,
            "dispatch_counts": dispatch_counts,
            "mean_inference_seconds": {
                name: sum(values) / len(values) if values else None
                for name, values in inference_seconds.items()
            },
            "maximum_inference_seconds": {
                name: max(values) if values else None for name, values in inference_seconds.items()
            },
            "mean_barrier_seconds": (
                sum(barrier_seconds) / len(barrier_seconds) if barrier_seconds else None
            ),
            "slippi_native_dummy_prefix_frames": min(
                processed_frames,
                slippi_effective_delay,
            ),
            "last_stocks": stocks,
            "winner": winner,
            "game_end_observed": natural_game_end,
            "natural_game_end": natural_game_end,
            "sudden_death_transition_observed": sudden_death_transition_observed,
            "termination": termination,
            "wall_seconds": time.time() - started_at,
            "shutdown_method": shutdown_method,
            "controller_transport": {
                **transport_record,
                "menu_flushes": {f"p{port}": menu_flushes[port] for port in (1, 2)},
            },
        },
        "controller_boundary_audit": controller_audit,
        "replay_identity_audit": replay_identity_audit,
        "artifacts": {
            "trace": trace_artifact,
            "replays": replay_records,
            "summary": _display_path(summary_path, project_root),
        },
    }
    _write_json(summary_path, summary)
    if exception is not None:
        raise RuntimeError(cast(str, summary["error"]))
    return summary


def run_frisson_slippi_match(
    config_path: Path,
    iso_path: Path | None = None,
    request: FrissonSlippiMatchRequest | None = None,
) -> dict[str, Any]:
    """Load one exact final Frisson winner and selected Slippi-AI release."""
    config, project_root = _load_config(config_path)
    match_request = FrissonSlippiMatchRequest() if request is None else request
    match_request = _canonicalize_request(project_root, config, match_request)
    selected_spec = _checkpoint_spec_for_path(
        project_root,
        cast(Path, match_request.player_1_checkpoint),
    )
    if match_request.artifact_label is None:
        match_request = replace(
            match_request,
            artifact_label=(
                f"final-{selected_spec.artifact_checkpoint_slug}-"
                f"{character_slug(match_request.player_1_character)}-"
                f"v-slippi-{match_request.player_2_slippi_release}-"
                f"{character_slug(match_request.player_2_character)}-s{match_request.seed}"
            ),
        )
    match_request.validate()
    checkpoint = cast(Path, match_request.player_1_checkpoint)
    _resolve_checkpoint(cast(dict[str, Any], config["frisson_ai"]), project_root, checkpoint)
    checkpoint_identity = inspect_frisson_checkpoint(checkpoint)
    contract = _validate_match_contract(
        config,
        project_root,
        match_request,
        checkpoint_identity,
    )
    frisson_config = _frisson_config(config, project_root, match_request)
    slippi_config = _slippi_config(config, project_root, match_request)

    from melee_policy.integration.slippi_compatibility import configure_cpu_tensorflow

    tensorflow_setup = configure_cpu_tensorflow(match_request.seed)
    random.seed(match_request.seed)
    np.random.seed(match_request.seed)
    torch.manual_seed(match_request.seed)
    torch.set_num_threads(1)
    image_path = _resolve_game_image_path(config, project_root, iso_path)
    reproducibility = _runtime_reproducibility_record(
        project_root,
        config_path.resolve(),
        "requirements-e010.lock",
        (
            "src/melee_policy/integration/frisson_slippi_match.py",
            "src/melee_policy/integration/frisson_policy.py",
            "src/melee_policy/integration/frisson_match.py",
            "src/melee_policy/integration/slippi_ai_policy.py",
            "src/melee_policy/integration/slippi_match.py",
            "src/melee_policy/integration/match_runtime.py",
            "src/melee_policy/integration/play.py",
            "src/melee_policy/integration/game_bundle.py",
            "patches/slippi-dolphin-two-pipe-frame-sync.patch",
            "scripts/run_final_75m_vs_slippi_ai_character_sweep.py",
        ),
    )
    frisson_session = FrissonPolicySession(frisson_config)
    slippi_session = SlippiAIPolicySession(slippi_config)
    try:
        frisson_session.start()
        slippi_session.start()
        summary = _run_console(
            config=config,
            project_root=project_root,
            iso_path=image_path,
            request=match_request,
            contract=contract,
            frisson_session=frisson_session,
            slippi_session=slippi_session,
            tensorflow_setup=tensorflow_setup,
            reproducibility=reproducibility,
        )
    finally:
        slippi_session.close()
        frisson_session.close()
    if match_request.save_slp or match_request.save_video:
        from melee_policy.integration.game_bundle import finalize_game_bundle

        summary = finalize_game_bundle(
            summary,
            project_root=project_root,
            config=config,
            iso_path=image_path,
            save_slp=match_request.save_slp,
            save_video=match_request.save_video,
        )
    return summary


__all__ = [
    "FINAL_10M_CHECKPOINT_BYTES",
    "FINAL_10M_CHECKPOINT_FRAMES",
    "FINAL_10M_CHECKPOINT_RELATIVE_PATH",
    "FINAL_10M_CHECKPOINT_SHA256",
    "FINAL_10M_CHECKPOINT_STEP",
    "FINAL_CHECKPOINT_BYTES",
    "FINAL_CHECKPOINT_FRAMES",
    "FINAL_CHECKPOINT_RELATIVE_PATH",
    "FINAL_CHECKPOINT_SHA256",
    "FINAL_CHECKPOINT_STEP",
    "FRISSON_LAUNCHABLE_CHARACTERS",
    "SCHEMA_VERSION",
    "FrissonSlippiMatchRequest",
    "canonical_frisson_character",
    "character_slug",
    "run_frisson_slippi_match",
]
