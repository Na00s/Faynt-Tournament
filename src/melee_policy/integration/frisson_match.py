"""Run a compatible Frisson policy against a released MIMIC character.

This is deliberately a separate rendered-match boundary.  Frisson receives
every current libmelee frame in order, MIMIC receives the same frame, and the
two-port controller transaction is opened only after both inference passes
have completed.  Frisson predicts the next controller boundary directly.  No
vladfi1 Slippi-AI policy FIFO, dummy prefix, observation filter, or TensorFlow
runtime participates in this module.
"""

from __future__ import annotations

import copy
import json
import platform
import random
import signal
import sys
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TextIO, cast

import numpy as np
import torch
from melee_policy.integration.post_rl_checkpoints import CHECKPOINTS as POST_RL_CHECKPOINTS, FORMAT as POST_RL_FORMAT, require_identity as require_post_rl_identity

from melee_policy.integration.frame_watchdog import InGameNoFrameWatchdog
from melee_policy.integration.frisson_policy import (
    EXPECTED_ACTOR_CONTEXT_FRAMES,
    EXPECTED_ACTOR_SEED,
    EXPECTED_MODEL_CONTEXT_LENGTH,
    EXPECTED_PARAMETER_COUNT,
    FRISSON_BC_CHECKPOINT_FORMAT,
    FRISSON_CHECKPOINT_FORMAT,
    FRISSON_FAMILY_CHECKPOINT_FORMAT,
    FRISSON_POSTTRAINING_CHECKPOINT_FORMAT,
    FRISSON_RUNTIME_SOURCE_REVISION,
    FRISSON_SOURCE_REVISION,
    SLIPPI_AI_SOURCE_REVISION,
    FrissonPolicyConfig,
    FrissonPolicySession,
    inspect_frisson_checkpoint,
    materialize_pinned_model_source,
)
from melee_policy.integration.mimic_bundle_manifest import MIMIC_NATIVE_BUNDLES
from melee_policy.integration.match_runtime import (
    LAUNCHABLE_CHARACTERS,
    MIMIC_DECODE_STRATEGY,
    MIMIC_SOURCE_REPOSITORY,
    MIMIC_SOURCE_REVISION,
    MimicLivePolicy,
    MimicRuntime,
    _assert_udp_port_available,
    _attested_emulator_release,
    _classify_trace_covering_replays,
    _ControllerPipeLockstep,
    _create_attested_dolphin_console,
    _dispatch_mimic_controller_inputs,
    _display_path,
    _emulator_application_identity,
    _file_identity,
    _finite_prediction,
    _game_image_identity,
    _git_output,
    _LatestInferenceWorker,
    _launch_and_connect_attested_dolphin,
    _legacy_replay_gate_checks,
    _load_config,
    _neutral_mimic_command,
    _require_exact_player_ports,
    _require_unused_artifact_label,
    _resolve_game_image_path,
    _runtime_reproducibility_record,
    _sha256_file,
    _stop_console,
    _validate_evaluation_seed,
    _validate_saved_replay,
    _write_json,
    load_mimic_runtime,
)
from melee_policy.integration.natural_game_end import has_decisive_zero_stock
from melee_policy.integration.slippi_ai_policy import (
    CanonicalControllerCommand,
    send_canonical_controller,
)

SCHEMA_VERSION = "integration.frisson_vs_mimic.v1"
TRACE_SCHEMA_VERSION = "integration.frisson_vs_mimic.controller_trace.v1"
FIRST_POLICY_FRAME = -123
NEXT_FRAME_CONTROLLER_BOUNDARY_FRAMES = 1
CONTROLLER_REPLAY_LAG_FRAMES = 1
FRISSON_PORT = 1
MIMIC_PORT = 2
# Fox remains the backwards-compatible default. The final family checkpoints
# consume the actual physical character ID and cover every independently
# launchable primary fighter in the pinned libmelee enum.
FRISSON_CHARACTER = "FOX"
FRISSON_SUPPORTED_CHARACTERS = tuple(LAUNCHABLE_CHARACTERS)
MIMIC_CHARACTER = "FOX"
MIMIC_SUPPORTED_CHARACTERS = tuple(
    str(bundle["character"]) for bundle in MIMIC_NATIVE_BUNDLES.values()
)
MIMIC_RELEASED_ASSET_DIRECTORIES = {
    str(bundle["character"]): str(bundle["name"])
    for bundle in MIMIC_NATIVE_BUNDLES.values()
}
_MIMIC_RELEASED_ASSET_ROOTS = {
    str(bundle["asset_directory"]) for bundle in MIMIC_NATIVE_BUNDLES.values()
}
if len(_MIMIC_RELEASED_ASSET_ROOTS) != 1:
    raise RuntimeError("current MIMIC native bundles must share one exact asset root")
MIMIC_RELEASED_ASSET_ROOT = Path(next(iter(_MIMIC_RELEASED_ASSET_ROOTS)))
MIMIC_OOD_CHARACTER_TRANSFER_SCHEMA_VERSION = (
    "integration.frisson_vs_mimic.mimic_ood_character_transfer.v1"
)
MIMIC_OOD_CHARACTER_TRANSFER_MODE = "released-checkpoint-on-ood-physical-character"
MATCH_STAGE = "FINAL_DESTINATION"
MATCH_SEED = 0
EXACT_INFERENCE_MODE = "synchronous-concurrent"
_REPLAY_CHARACTER_ALIASES = {"POPO": "ICE_CLIMBERS"}
_REPLAY_IDENTITY_REQUIRED_CHECKS = (
    "replay_parsed",
    "peppi_parser_version_exact",
    "exact_expected_ports",
    "human_slots",
    "exact_expected_stage",
    "exact_expected_characters",
)

_INSPECTED_CHECKPOINT_PARAMETER_COUNTS = {
    (POST_RL_FORMAT, "10m"): 10_163_629,
    (POST_RL_FORMAT, "75m"): 75_305_709,
    (FRISSON_CHECKPOINT_FORMAT, "20m"): EXPECTED_PARAMETER_COUNT,
    (FRISSON_BC_CHECKPOINT_FORMAT, "20m"): EXPECTED_PARAMETER_COUNT,
    (FRISSON_FAMILY_CHECKPOINT_FORMAT, "10m"): 10_163_629,
    (FRISSON_FAMILY_CHECKPOINT_FORMAT, "75m"): 75_305_709,
    (FRISSON_POSTTRAINING_CHECKPOINT_FORMAT, "10m"): 10_163_629,
    (FRISSON_POSTTRAINING_CHECKPOINT_FORMAT, "75m"): 75_305_709,
}

_POSTTRAINING_FINAL_WINNER_IDENTITIES: dict[str, dict[str, Any]] = {
    "10m": {
        "relative_path": (".e013-cache/posttraining-winners/frisson-melee-10m-posttrained-best-val.pt"),
        "sha256": "63b5ff05ef30476c4f41590c478f4a7218f3b72a8b5ef24a0e336eeb2b7c287b",
        "byte_length": 96_452_075,
        "step": 195_248,
        "processed_target_frames": 12_795_772_928,
    },
    "75m": {
        "relative_path": (".e013-cache/posttraining-winners/frisson-melee-75m-posttrained-best-val.pt"),
        "sha256": "8211f1832198646e9f4e3bacde26f181614f93320e68bd326dd5942c4e0f4077",
        "byte_length": 644_291_987,
        "step": 127_214,
        "processed_target_frames": 8_337_096_704,
    },
}


def _model_name(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    return "frisson-ai" if normalized == "frisson" else normalized


@dataclass(frozen=True, slots=True)
class FrissonMatchRequest:
    """Benchmark boundary: one supported Frisson P1 versus a released MIMIC P2."""

    player_1_model: str = "frisson-ai"
    player_2_model: str = "mimic"
    player_1_character: str = FRISSON_CHARACTER
    player_2_character: str = MIMIC_CHARACTER
    stage: str = MATCH_STAGE
    player_1_checkpoint: Path | None = None
    player_2_checkpoint: Path | None = None
    player_1_assets: Path | None = None
    player_2_assets: Path | None = None
    player_1_name: str | None = None
    player_2_name: str | None = None
    player_1_temperature: float | None = None
    player_2_temperature: float | None = None
    max_game_frames: int | None = None
    inference_mode: str = EXACT_INFERENCE_MODE
    artifact_label: str | None = None
    seed: int = MATCH_SEED
    require_natural_end: bool = True
    save_slp: bool = False
    save_video: bool = False
    allow_player_2_ood_character: bool = False

    def validate(self) -> None:
        fixed = {
            "player_1_model": (_model_name(self.player_1_model), "frisson-ai"),
            "player_2_model": (_model_name(self.player_2_model), "mimic"),
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
            raise ValueError(f"Frisson-versus-MIMIC fixed match contract mismatch: {mismatches}")
        if self.player_1_character not in FRISSON_SUPPORTED_CHARACTERS:
            raise ValueError(
                "Frisson-versus-MIMIC requires an independently launchable Frisson character, "
                f"got {self.player_1_character!r}; allowed={FRISSON_SUPPORTED_CHARACTERS}"
            )
        if not isinstance(self.allow_player_2_ood_character, bool):
            raise TypeError("allow_player_2_ood_character must be a boolean")
        if self.player_2_character not in MIMIC_SUPPORTED_CHARACTERS:
            if not self.allow_player_2_ood_character:
                raise ValueError(
                    "Frisson-versus-MIMIC requires a released MIMIC character, got "
                    f"{self.player_2_character!r}; allowed={MIMIC_SUPPORTED_CHARACTERS}; "
                    "explicit OOD transfer requires allow_player_2_ood_character=True"
                )
            if self.player_2_character not in FRISSON_SUPPORTED_CHARACTERS:
                raise ValueError(
                    "MIMIC OOD transfer requires an independently launchable physical player 2 "
                    f"character, got {self.player_2_character!r}; "
                    f"allowed={FRISSON_SUPPORTED_CHARACTERS}"
                )
            if self.player_2_checkpoint is None or self.player_2_assets is None:
                raise ValueError(
                    "MIMIC OOD transfer requires explicit player_2_checkpoint and "
                    "player_2_assets paths"
                )
        if self.player_1_assets is not None:
            raise ValueError("Frisson does not accept a separate asset directory")
        if self.player_1_name is not None or self.player_2_name is not None:
            raise ValueError("Frisson and MIMIC do not accept Slippi-AI player names")
        if self.player_1_temperature not in (None, 1, 1.0):
            raise ValueError("Frisson sample temperature is fixed at 1.0")
        if self.player_2_temperature is not None:
            raise ValueError("MIMIC sampling settings come from its pinned bundle contract")
        if self.max_game_frames is not None and self.max_game_frames < 1:
            raise ValueError("max_game_frames must be positive")
        if not isinstance(self.save_slp, bool) or not isinstance(self.save_video, bool):
            raise TypeError("save_slp and save_video must be booleans")
        _validate_evaluation_seed(self.seed)
        if self.artifact_label is not None and (
            not self.artifact_label
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in self.artifact_label
            )
        ):
            raise ValueError(
                "artifact_label must contain only lowercase letters, digits, underscores, or hyphens"
            )


@dataclass(frozen=True, slots=True)
class _ExactFrameResult:
    game_frame: int
    frisson_command: CanonicalControllerCommand
    frisson_dispatch: dict[str, Any]
    frisson_inference_seconds: float
    mimic_sent: dict[str, Any]
    mimic_pressed: list[str]
    mimic_inference_seconds: float
    barrier_seconds: float
    mimic_source_frame: int


def _selected_checkpoint_contract(checkpoint_identity: dict[str, Any]) -> dict[str, Any]:
    """Bind the match contract to the checkpoint that the policy loader inspected."""

    checkpoint_format = checkpoint_identity.get("format")
    profile = checkpoint_identity.get("profile")
    parameter_count = checkpoint_identity.get("parameter_count")
    if not isinstance(checkpoint_format, str) or not isinstance(profile, str):
        raise ValueError(
            "Frisson-versus-MIMIC inspected checkpoint format and profile must be strings"
        )
    expected_parameter_count = _INSPECTED_CHECKPOINT_PARAMETER_COUNTS.get((checkpoint_format, profile))
    if expected_parameter_count is None:
        raise ValueError(
            "Frisson-versus-MIMIC inspected checkpoint format/profile mismatch: "
            f"format={checkpoint_format!r}, profile={profile!r}"
        )
    if isinstance(parameter_count, bool) or parameter_count != expected_parameter_count:
        raise ValueError(
            "Frisson-versus-MIMIC inspected checkpoint parameter count mismatch: "
            f"observed={parameter_count!r}, required={expected_parameter_count}"
        )

    sha256 = checkpoint_identity.get("sha256")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise ValueError("Frisson-versus-MIMIC inspected checkpoint SHA-256 is invalid")
    byte_length = checkpoint_identity.get("byte_length")
    if isinstance(byte_length, bool) or not isinstance(byte_length, int) or byte_length <= 0:
        raise ValueError("Frisson-versus-MIMIC inspected checkpoint byte length is invalid")
    checkpoint_path = checkpoint_identity.get("path")
    if not isinstance(checkpoint_path, str) or not Path(checkpoint_path).is_absolute():
        raise ValueError("Frisson-versus-MIMIC inspected checkpoint path must be absolute")

    if checkpoint_format == POST_RL_FORMAT:
        require_post_rl_identity(checkpoint_identity)
    if checkpoint_format == FRISSON_POSTTRAINING_CHECKPOINT_FORMAT:
        winner = _POSTTRAINING_FINAL_WINNER_IDENTITIES[profile]
        project_root = Path(__file__).resolve().parents[3]
        expected_path = (project_root / str(winner["relative_path"])).resolve()
        exact_identity = {
            "sha256": winner["sha256"],
            "byte_length": winner["byte_length"],
            "step": winner["step"],
            "processed_target_frames": winner["processed_target_frames"],
        }
        posttraining_mismatches = {
            name: {"observed": checkpoint_identity.get(name), "required": required}
            for name, required in exact_identity.items()
            if checkpoint_identity.get(name) != required
        }
        observed_path = Path(checkpoint_path).expanduser().resolve()
        if observed_path != expected_path:
            posttraining_mismatches["path"] = {
                "observed": str(observed_path),
                "required": str(expected_path),
            }
        if posttraining_mismatches:
            raise ValueError(
                "Frisson-versus-MIMIC post-training final-winner identity mismatch: "
                f"{posttraining_mismatches}"
            )

    codec = checkpoint_identity.get("codec")
    representation = {
        "slippi_ai_commit": (
            checkpoint_identity.get("slippi_ai_commit"),
            SLIPPI_AI_SOURCE_REVISION,
        ),
        "codec.name": (
            codec.get("name") if isinstance(codec, dict) else None,
            "custom_v1",
        ),
        "codec.vocab_sizes": (
            codec.get("vocab_sizes") if isinstance(codec, dict) else None,
            {"buttons": 728, "main_stick": 85},
        ),
    }
    representation_mismatches = {
        name: {"observed": observed, "required": required}
        for name, (observed, required) in representation.items()
        if observed != required
    }
    if representation_mismatches:
        raise ValueError(
            f"Frisson-versus-MIMIC inspected checkpoint representation mismatch: {representation_mismatches}"
        )

    return {
        "inspected": True,
        "format": checkpoint_format,
        "profile": profile,
        "parameter_count": parameter_count,
        "sha256": sha256,
        "byte_length": byte_length,
        "path": checkpoint_path,
        "step": checkpoint_identity.get("step"),
        "processed_target_frames": checkpoint_identity.get("processed_target_frames"),
    }


def _validate_config_contract(
    config: dict[str, Any],
    request: FrissonMatchRequest,
    *,
    checkpoint_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request.validate()
    frisson = config.get("frisson_ai")
    mimic = config.get("mimic")
    integration = config.get("integration")
    if not isinstance(frisson, dict) or not isinstance(mimic, dict) or not isinstance(integration, dict):
        raise ValueError("config must contain [integration], [mimic], and [frisson_ai]")
    required = {
        "frisson.source_revision": (frisson.get("source_revision"), FRISSON_SOURCE_REVISION),
        "frisson.runtime_source_revision": (
            frisson.get("runtime_source_revision"),
            FRISSON_RUNTIME_SOURCE_REVISION,
        ),
        "frisson.slippi_ai_commit": (
            frisson.get("slippi_ai_commit"),
            SLIPPI_AI_SOURCE_REVISION,
        ),
        "frisson.checkpoint_format": (
            frisson.get("checkpoint_format"),
            FRISSON_CHECKPOINT_FORMAT,
        ),
        "frisson.actor_seed": (frisson.get("actor_seed"), EXPECTED_ACTOR_SEED),
        "frisson.sample_temperature": (frisson.get("sample_temperature"), 1.0),
        "frisson.action_offset_frames": (frisson.get("action_offset_frames"), 1),
        "frisson.delay_frames": (frisson.get("delay_frames"), 0),
        "frisson.context_mode": (frisson.get("context_mode"), "ring"),
        "frisson.context_length": (
            frisson.get("context_length"),
            EXPECTED_MODEL_CONTEXT_LENGTH,
        ),
        "frisson.controller_buttons_vocab": (frisson.get("controller_buttons_vocab"), 728),
        "frisson.controller_main_stick_vocab": (
            frisson.get("controller_main_stick_vocab"),
            85,
        ),
        "frisson.parameter_count": (frisson.get("parameter_count"), EXPECTED_PARAMETER_COUNT),
        "frisson.allowed_characters": (
            frisson.get("allowed_characters"),
            list(FRISSON_SUPPORTED_CHARACTERS),
        ),
        "mimic.character": (mimic.get("character"), MIMIC_CHARACTER),
        "mimic.decode_strategy": (mimic.get("decode_strategy"), MIMIC_DECODE_STRATEGY),
        "mimic.temperature": (mimic.get("temperature"), 1.0),
        "mimic.top_k": (mimic.get("top_k"), 0),
        "mimic.top_p": (mimic.get("top_p"), 0.0),
        "integration.device": (integration.get("device"), "cpu"),
        "integration.inference_mode": (
            integration.get("inference_mode"),
            EXACT_INFERENCE_MODE,
        ),
        "integration.maximum_pending_snapshots_per_model": (
            integration.get("maximum_pending_snapshots_per_model"),
            1,
        ),
    }
    mismatches = {
        name: {"observed": observed, "required": expected}
        for name, (observed, expected) in required.items()
        if observed != expected
    }
    if mismatches:
        raise ValueError(f"Frisson-versus-MIMIC config contract mismatch: {mismatches}")
    selected_checkpoint = (
        {
            "inspected": False,
            "format": FRISSON_CHECKPOINT_FORMAT,
            "profile": "20m",
            "parameter_count": EXPECTED_PARAMETER_COUNT,
        }
        if checkpoint_identity is None
        else _selected_checkpoint_contract(checkpoint_identity)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "exact_mode_only": True,
        "frisson_character": request.player_1_character,
        "frisson_supported_characters": list(FRISSON_SUPPORTED_CHARACTERS),
        "frisson_character_selection": "actual-live-physical-character-id",
        "mimic_character": request.player_2_character,
        "mimic_bundle_selection": "released-character-bundle",
        "state_frame": "t",
        "command_frame": "t+1",
        "action_offset_frames": 1,
        "delay_frames": 0,
        "sample_temperature": 1.0,
        "actor_trajectory_context_frames": EXPECTED_ACTOR_CONTEXT_FRAMES,
        "kv_cache_capacity_frames": EXPECTED_MODEL_CONTEXT_LENGTH,
        "controller_replay_lag_frames": CONTROLLER_REPLAY_LAG_FRAMES,
        "observation_filter": None,
        "tensorflow_runtime": False,
        "slippi_ai_policy_fifo": False,
        "native_dummy_prefix": False,
        "selected_checkpoint": selected_checkpoint,
    }


def _activate_mimic_source(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    source = (project_root / config["mimic"]["source_directory"]).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"pinned MIMIC source is missing: {source}")
    identity = {
        "repository_url": _git_output(source, "remote", "get-url", "origin"),
        "revision": _git_output(source, "rev-parse", "HEAD"),
        "tree_clean": not bool(_git_output(source, "status", "--short", "--untracked-files=all")),
    }
    checks = {
        "repository_url_exact": identity["repository_url"] == MIMIC_SOURCE_REPOSITORY,
        "revision_exact": identity["revision"] == MIMIC_SOURCE_REVISION,
        "tree_clean": identity["tree_clean"] is True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"pinned MIMIC source identity mismatch: {identity}")
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    return {**identity, "checks": checks, "directory": source_text}


def _resolve_mimic_bundle_selection(
    config: dict[str, Any],
    project_root: Path,
    request: FrissonMatchRequest,
) -> tuple[Path | None, Path | None]:
    """Select one complete bundle from the canonical current native snapshot."""

    if request.player_2_character not in MIMIC_SUPPORTED_CHARACTERS:
        request.validate()
        # OOD transfer takes both bundle paths directly from the request.
        explicit_checkpoint = cast(Path, request.player_2_checkpoint).expanduser().resolve()
        explicit_assets = cast(Path, request.player_2_assets).expanduser().resolve()
        return explicit_checkpoint, explicit_assets

    checkpoint = request.player_2_checkpoint
    assets = request.player_2_assets
    if checkpoint is not None and assets is None:
        assets = checkpoint.expanduser().resolve().parent
    elif checkpoint is None and assets is not None:
        checkpoint = assets.expanduser().resolve() / "model.pt"
    elif checkpoint is None and assets is None:
        assets = (
            project_root
            / MIMIC_RELEASED_ASSET_ROOT
            / MIMIC_RELEASED_ASSET_DIRECTORIES[request.player_2_character]
        ).resolve()
        checkpoint = assets / "model.pt"
    return checkpoint, assets


def _mimic_native_ood_inference_contract() -> dict[str, Any]:
    """Describe the unchanged released MIMIC execution path used for OOD control."""

    return {
        "live_policy_class": "MimicLivePolicy",
        "observation_builder": "tools.inference_utils.build_frame_p2",
        "model_forward": "MimicRuntime.model",
        "sampler_decoder": "tools.inference_utils.decode_and_press",
        "decode_strategy": MIMIC_DECODE_STRATEGY,
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 0.0,
        "online_delay_frames": 0,
        "inference_mode": EXACT_INFERENCE_MODE,
        "current_frame_barrier_before_controller_transaction": True,
    }


def _mimic_ood_character_transfer_provenance(
    request: FrissonMatchRequest,
    mimic_runtime: MimicRuntime,
    project_root: Path,
) -> dict[str, Any] | None:
    """Bind an opted-in physical fighter to the exact released checkpoint bundle."""

    request.validate()
    if request.player_2_character in MIMIC_SUPPORTED_CHARACTERS:
        return None

    requested_checkpoint = cast(Path, request.player_2_checkpoint).expanduser().resolve()
    requested_assets = cast(Path, request.player_2_assets).expanduser().resolve()
    bundle_identity = mimic_runtime.bundle_identity
    bundle_checks_value = bundle_identity.get("checks")
    bundle_checks = (
        cast(Mapping[str, Any], bundle_checks_value)
        if isinstance(bundle_checks_value, Mapping)
        else {}
    )
    state_dictionary_value = bundle_identity.get("state_dictionary")
    state_dictionary = (
        cast(Mapping[str, Any], state_dictionary_value)
        if isinstance(state_dictionary_value, Mapping)
        else {}
    )
    checkpoint_character = str(mimic_runtime.controlled_character)
    checks = {
        "explicit_opt_in": request.allow_player_2_ood_character is True,
        "physical_character_launchable": (
            request.player_2_character in FRISSON_SUPPORTED_CHARACTERS
        ),
        "physical_character_outside_released_mimic_roster": (
            request.player_2_character not in MIMIC_SUPPORTED_CHARACTERS
        ),
        "explicit_checkpoint_and_assets_supplied": (
            request.player_2_checkpoint is not None and request.player_2_assets is not None
        ),
        "requested_checkpoint_loaded_exactly": (
            mimic_runtime.checkpoint_path.resolve() == requested_checkpoint
        ),
        "requested_assets_loaded_exactly": (
            mimic_runtime.asset_directory.resolve() == requested_assets
        ),
        "checkpoint_sha256_bound_to_released_bundle": (
            bundle_identity.get("checkpoint_sha256") == mimic_runtime.checkpoint_sha256
            and bundle_checks.get("checkpoint_matches_asset_model") is True
        ),
        "released_bundle_semantics_validated": (
            bool(bundle_checks) and all(value is True for value in bundle_checks.values())
        ),
        "checkpoint_character_matches_bundle": (
            bundle_identity.get("character") == checkpoint_character
        ),
        "checkpoint_character_has_released_mimic_bundle": (
            checkpoint_character in MIMIC_SUPPORTED_CHARACTERS
        ),
        "checkpoint_character_separate_from_physical_character": (
            checkpoint_character != request.player_2_character
        ),
        "state_dictionary_loaded_strictly": (
            state_dictionary.get("strict") is True
            and state_dictionary.get("missing_keys") == []
            and state_dictionary.get("unexpected_keys") == []
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"MIMIC OOD character-transfer provenance failed: {failed}")

    return {
        "schema_version": MIMIC_OOD_CHARACTER_TRANSFER_SCHEMA_VERSION,
        "mode": MIMIC_OOD_CHARACTER_TRANSFER_MODE,
        "forced_ood_character_transfer": True,
        "explicit_opt_in_field": "allow_player_2_ood_character",
        "physical_port": MIMIC_PORT,
        "physical_character": request.player_2_character,
        "checkpoint_character": checkpoint_character,
        "physical_character_covered_by_released_checkpoint": False,
        "checkpoint_roster_unchanged": True,
        "checkpoint": {
            "path": _display_path(mimic_runtime.checkpoint_path, project_root),
            "sha256": mimic_runtime.checkpoint_sha256,
        },
        "assets": {
            "directory": _display_path(mimic_runtime.asset_directory, project_root),
            "released_bundle_name": bundle_identity.get("name"),
            "released_run_name": bundle_identity.get("run_name"),
        },
        "native_inference": _mimic_native_ood_inference_contract(),
        "scientific_interpretation": (
            "out-of-distribution character transfer by a released per-character MIMIC "
            "checkpoint; released-policy coverage remains limited to the checkpoint character"
        ),
        "checks": checks,
    }


def _bind_mimic_runtime_contract(
    contract: dict[str, Any],
    request: FrissonMatchRequest,
    mimic_runtime: MimicRuntime,
    project_root: Path,
) -> dict[str, Any]:
    """Bind checkpoint identity while preserving the released-character contract."""

    if request.player_2_character in MIMIC_SUPPORTED_CHARACTERS:
        if mimic_runtime.controlled_character != request.player_2_character:
            raise RuntimeError(
                f"MIMIC checkpoint controls {mimic_runtime.controlled_character}, "
                f"expected {request.player_2_character}"
            )
        return contract

    provenance = _mimic_ood_character_transfer_provenance(
        request,
        mimic_runtime,
        project_root,
    )
    if provenance is None:
        raise RuntimeError("MIMIC OOD character-transfer provenance is missing")
    return {
        **contract,
        "mimic_bundle_selection": "explicit-released-bundle-ood-character-transfer",
        "mimic_checkpoint_character": mimic_runtime.controlled_character,
        "mimic_physical_character": request.player_2_character,
        "mimic_requested_character_covered_by_released_checkpoint": False,
        "mimic_forced_ood_character_transfer": True,
        "allow_player_2_ood_character": True,
        "mimic_ood_character_transfer": provenance,
    }


def _mimic_character_gate_checks(
    request: FrissonMatchRequest,
    contract: dict[str, Any],
    mimic_runtime: MimicRuntime,
    mimic_policy: MimicLivePolicy,
) -> dict[str, bool]:
    """Return character and provenance checks for released and OOD match modes."""

    if request.player_2_character in MIMIC_SUPPORTED_CHARACTERS:
        return {
            "requested_frisson_p1_vs_released_mimic_p2_characters_exact": (
                request.player_1_character in FRISSON_SUPPORTED_CHARACTERS
                and request.player_2_character in MIMIC_SUPPORTED_CHARACTERS
                and mimic_runtime.controlled_character == request.player_2_character
            )
        }

    provenance_value = contract.get("mimic_ood_character_transfer")
    provenance = (
        cast(Mapping[str, Any], provenance_value)
        if isinstance(provenance_value, Mapping)
        else {}
    )
    provenance_checks_value = provenance.get("checks")
    provenance_checks = (
        cast(Mapping[str, Any], provenance_checks_value)
        if isinstance(provenance_checks_value, Mapping)
        else {}
    )
    return {
        "requested_frisson_p1_vs_explicit_mimic_ood_p2_character_exact": (
            request.player_1_character in FRISSON_SUPPORTED_CHARACTERS
            and request.allow_player_2_ood_character is True
            and request.player_2_character in FRISSON_SUPPORTED_CHARACTERS
            and request.player_2_character not in MIMIC_SUPPORTED_CHARACTERS
            and mimic_runtime.controlled_character in MIMIC_SUPPORTED_CHARACTERS
            and mimic_runtime.controlled_character != request.player_2_character
        ),
        "mimic_ood_character_provenance_exact": (
            provenance.get("schema_version")
            == MIMIC_OOD_CHARACTER_TRANSFER_SCHEMA_VERSION
            and provenance.get("mode") == MIMIC_OOD_CHARACTER_TRANSFER_MODE
            and provenance.get("physical_character") == request.player_2_character
            and provenance.get("checkpoint_character") == mimic_runtime.controlled_character
            and provenance.get("forced_ood_character_transfer") is True
            and provenance.get("physical_character_covered_by_released_checkpoint") is False
            and provenance.get("checkpoint_roster_unchanged") is True
            and bool(provenance_checks)
            and all(value is True for value in provenance_checks.values())
        ),
        "mimic_ood_native_builder_model_sampler_decoder_preserved": (
            provenance.get("native_inference") == _mimic_native_ood_inference_contract()
            and isinstance(mimic_policy, MimicLivePolicy)
            and mimic_policy.runtime is mimic_runtime
            and mimic_policy.port == MIMIC_PORT
            and mimic_policy.online_delay_frames == 0
        ),
    }


def _resolve_checkpoint(
    configured: dict[str, Any],
    project_root: Path,
    override: Path | None,
) -> Path:
    path = (
        (project_root / str(configured["checkpoint"])) if override is None else override.expanduser()
    ).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Frisson checkpoint is missing: {path}")
    observed = {"sha256": _sha256_file(path), "byte_length": path.stat().st_size}
    default: dict[str, str | int] = {
        "name": "configured-default",
        "sha256": str(configured["checkpoint_sha256"]),
        "byte_length": int(configured["checkpoint_byte_length"]),
    }
    expected_identities: list[dict[str, str | int]] = [default]
    explicit = configured.get("explicit_checkpoint_identities", [])
    if not isinstance(explicit, list):
        raise ValueError("frisson_ai.explicit_checkpoint_identities must be an array of tables")
    for index, value in enumerate(explicit):
        if not isinstance(value, dict):
            raise ValueError(f"frisson_ai.explicit_checkpoint_identities[{index}] must be a table")
        name = value.get("name")
        sha256 = value.get("sha256")
        byte_length = value.get("byte_length")
        if not isinstance(name, str) or not name:
            raise ValueError(f"explicit Frisson checkpoint identity {index} needs a name")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ValueError(f"explicit Frisson checkpoint identity {name!r} has an invalid SHA-256")
        if isinstance(byte_length, bool) or not isinstance(byte_length, int) or byte_length <= 0:
            raise ValueError(f"explicit Frisson checkpoint identity {name!r} has an invalid byte length")
        expected_identities.append({"name": name, "sha256": sha256, "byte_length": byte_length})

    names = [str(identity["name"]) for identity in expected_identities]
    identities = [(str(identity["sha256"]), int(identity["byte_length"])) for identity in expected_identities]
    if len(names) != len(set(names)):
        raise ValueError("Frisson checkpoint identity names must be unique")
    if len(identities) != len(set(identities)):
        raise ValueError("Frisson checkpoint identities must be unique")

    allowed = expected_identities if override is not None else [default]
    if override is not None:
        for profile, identity in POST_RL_CHECKPOINTS.items():
            if path == (project_root / identity["relative_path"]).resolve():
                allowed = [*allowed, {"name": f"p21-{profile}-rl-step{identity['step']}",
                                      "sha256": identity["sha256"], "byte_length": identity["byte_length"]}]
    if not any(
        observed == {"sha256": identity["sha256"], "byte_length": identity["byte_length"]}
        for identity in allowed
    ):
        expected = [
            {
                "name": identity["name"],
                "sha256": identity["sha256"],
                "byte_length": identity["byte_length"],
            }
            for identity in allowed
        ]
        raise RuntimeError(f"Frisson checkpoint identity mismatch: {observed} not in {expected}")
    return path


def _frisson_policy_config(
    config: dict[str, Any], project_root: Path, request: FrissonMatchRequest
) -> FrissonPolicyConfig:
    frisson = cast(dict[str, Any], config["frisson_ai"])
    checkpoint = _resolve_checkpoint(frisson, project_root, request.player_1_checkpoint)
    source_repository = (project_root / str(frisson["source_repository"])).resolve()
    source_subdirectory = str(frisson["source_subdirectory"])
    source_revision = str(frisson["source_revision"])
    source_directory = materialize_pinned_model_source(
        repository=source_repository,
        revision=source_revision,
        source_subdirectory=source_subdirectory,
        destination=(project_root / str(frisson["source_directory"])).resolve(),
    )
    base = FrissonPolicyConfig.from_project_root(
        project_root,
        port=FRISSON_PORT,
        opponent_port=MIMIC_PORT,
    )
    policy_config = replace(
        base,
        model_source_directory=source_directory,
        model_source_repository=source_repository,
        model_source_revision=source_revision,
        model_source_subdirectory=source_subdirectory,
        slippi_ai_source_directory=(project_root / str(frisson["parser_source_directory"])).resolve(),
        checkpoint_path=checkpoint,
        sample_temperature=float(frisson["sample_temperature"]),
        evaluation_seed=request.seed,
        device=str(config["integration"]["device"]),
        context_mode=str(frisson["context_mode"]),
        model_context_length=int(frisson["context_length"]),
        delay_frames=int(frisson["delay_frames"]),
    )
    policy_config.validate()
    return policy_config


def _console_options(replay_directory: Path, slippi_port: int) -> dict[str, Any]:
    return {
        "is_dolphin": True,
        "tmp_home_directory": True,
        "copy_home_directory": False,
        "blocking_input": True,
        "polling_mode": True,
        "polling_timeout": 1.0,
        "online_delay": 0,
        "setup_gecko_codes": True,
        "fullscreen": False,
        "gfx_backend": "",
        "disable_audio": False,
        "use_exi_inputs": False,
        "enable_ffw": False,
        "save_replays": True,
        "replay_dir": str(replay_directory),
        "replay_monthly_folders": False,
        "slippi_port": slippi_port,
    }


def _menu_character_selection(melee_module: Any, request: FrissonMatchRequest) -> dict[int, Any]:
    """Bind each requested character to the exact libmelee CSS enum."""

    request.validate()
    try:
        return {
            FRISSON_PORT: melee_module.Character[request.player_1_character],
            MIMIC_PORT: melee_module.Character[request.player_2_character],
        }
    except KeyError as error:
        raise ValueError(f"requested character lacks a pinned libmelee enum: {error}") from error


def _run_exact_frame(
    *,
    gamestate: Any,
    frisson_session: FrissonPolicySession,
    mimic_policy: MimicLivePolicy,
    mimic_worker: _LatestInferenceWorker,
    frisson_controller: Any,
    mimic_controller: Any,
    transport: _ControllerPipeLockstep,
    mimic_held: dict[str, Any],
    mimic_held_pressed: list[str],
    mimic_temperature: float,
    mimic_top_k: int,
    mimic_top_p: float,
) -> _ExactFrameResult:
    """Finish both current-frame passes before opening either controller transaction."""
    game_frame = int(gamestate.frame)
    barrier_started = time.perf_counter()
    if not mimic_policy.observe(gamestate):
        raise RuntimeError(f"MIMIC rejected required current frame {game_frame}")
    mimic_worker.submit(game_frame, mimic_policy.snapshot())

    frisson_started = time.perf_counter()
    frisson_command = frisson_session.step(gamestate)
    frisson_seconds = time.perf_counter() - frisson_started
    frisson_command.validate()

    mimic_source_frame, mimic_prediction, mimic_seconds = mimic_worker.wait_completed(game_frame)
    if mimic_source_frame != game_frame:
        raise RuntimeError(f"MIMIC returned source frame {mimic_source_frame} for current frame {game_frame}")
    if not _finite_prediction(mimic_prediction):
        raise RuntimeError(f"MIMIC returned nonfinite output at frame {game_frame}")
    barrier_seconds = time.perf_counter() - barrier_started

    # This is the first controller-side operation in the function.  Both model
    # passes above have returned, so neither port can release Dolphin early.
    transport.begin_boundary(reason="frisson-vs-mimic-gameplay", game_frame=game_frame)
    frisson_dispatch = send_canonical_controller(
        frisson_controller,
        frisson_command,
        flush=False,
    ).as_dict()
    frisson_dispatch["called"] = True
    frisson_dispatch["queued_after_both_current_frame_inferences"] = True
    transport.schedule_next_boundary(
        FRISSON_PORT,
        reason="frisson-zero-delay-next-frame-command",
        game_frame=game_frame,
    )
    mimic_sent, mimic_pressed = _dispatch_mimic_controller_inputs(
        mimic_controller,
        mimic_policy,
        cast(dict[str, torch.Tensor], mimic_prediction),
        mimic_policy.runtime.state.prev_sent,
        mimic_held,
        mimic_held_pressed,
        temperature=mimic_temperature,
        top_k=mimic_top_k,
        top_p=mimic_top_p,
    )
    mimic_policy.record_decoded_command(game_frame, mimic_sent)
    transport.commit_boundary()
    return _ExactFrameResult(
        game_frame=game_frame,
        frisson_command=frisson_command,
        frisson_dispatch=frisson_dispatch,
        frisson_inference_seconds=frisson_seconds,
        mimic_sent=mimic_sent,
        mimic_pressed=mimic_pressed,
        mimic_inference_seconds=mimic_seconds,
        barrier_seconds=barrier_seconds,
        mimic_source_frame=mimic_source_frame,
    )


def _player_state(player: Any) -> dict[str, Any]:
    return {
        "character": str(getattr(player.character, "name", player.character)).split(".")[-1],
        "action": int(player.action.value),
        "stocks": int(player.stock),
        "percent": float(player.percent),
        "position": [float(player.position.x), float(player.position.y)],
    }


def _trace_row(
    gamestate: Any,
    result: _ExactFrameResult,
    *,
    frisson_character: str = FRISSON_CHARACTER,
    mimic_character: str = MIMIC_CHARACTER,
    mimic_checkpoint_character: str | None = None,
) -> dict[str, Any]:
    frame = result.game_frame
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "game_frame": frame,
        "barrier": {
            "source_frame": frame,
            "frisson_current_frame_inference_complete": True,
            "mimic_current_frame_inference_complete": True,
            "controller_transaction_opened_after_both": True,
            "seconds": result.barrier_seconds,
        },
        "slots": {
            "p1": {
                "port": FRISSON_PORT,
                "model": "frisson-ai",
                "requested_character": frisson_character,
                "player_state": _player_state(gamestate.players[FRISSON_PORT]),
                "command": result.frisson_command.as_dict(),
                "inference": {
                    "source_frame": frame,
                    "command_age_frames": 0,
                    "seconds": result.frisson_inference_seconds,
                    "state_frame": "t",
                    "command_frame": "t+1",
                    "action_offset_frames": 1,
                    "delay_frames": 0,
                    "native_dummy_prefix": False,
                },
                "controller_dispatch": result.frisson_dispatch,
            },
            "p2": {
                "port": MIMIC_PORT,
                "model": "mimic",
                "requested_character": mimic_character,
                **(
                    {"checkpoint_character": mimic_checkpoint_character}
                    if mimic_checkpoint_character is not None
                    and mimic_checkpoint_character != mimic_character
                    else {}
                ),
                "player_state": _player_state(gamestate.players[MIMIC_PORT]),
                "command": result.mimic_sent,
                "pressed": list(result.mimic_pressed),
                "inference": {
                    "source_frame": result.mimic_source_frame,
                    "command_age_frames": frame - result.mimic_source_frame,
                    "seconds": result.mimic_inference_seconds,
                },
                "controller_dispatch": {
                    "called": True,
                    "native_complete_frame_flush_deferred_to_lockstep": True,
                    "queued_after_both_current_frame_inferences": True,
                },
            },
        },
    }


def _validate_first_context(
    gamestate: Any,
    *,
    frisson_character: str = FRISSON_CHARACTER,
    mimic_character: str = MIMIC_CHARACTER,
) -> dict[str, Any]:
    observed_stage = str(getattr(gamestate.stage, "name", gamestate.stage)).split(".")[-1]
    observed_characters = {
        f"p{port}": str(
            getattr(gamestate.players[port].character, "name", gamestate.players[port].character)
        ).split(".")[-1]
        for port in (FRISSON_PORT, MIMIC_PORT)
    }
    checks = {
        "first_frame_minus_123": int(gamestate.frame) == FIRST_POLICY_FRAME,
        "final_destination": observed_stage == MATCH_STAGE,
        "frisson_requested_character": observed_characters["p1"] == frisson_character,
        "mimic_requested_character": observed_characters["p2"] == mimic_character,
    }
    if not all(checks.values()):
        raise RuntimeError(f"first Frisson-versus-MIMIC game context mismatch: {checks}")
    return {
        "frame": int(gamestate.frame),
        "stage": observed_stage,
        "characters": observed_characters,
        "costumes": {f"p{port}": int(gamestate.players[port].costume) for port in (FRISSON_PORT, MIMIC_PORT)},
        "checks": checks,
    }


def _adapt_trace_rows_for_shared_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use the canonical-command audit without importing Slippi-AI runtime behavior."""
    adapted = copy.deepcopy(rows)
    for row in adapted:
        slot = cast(dict[str, Any], row["slots"])["p1"]
        if _model_name(str(slot.get("model"))) != "frisson-ai":
            raise ValueError("Frisson controller audit requires Frisson in trace slot p1")
        # The shared audit's canonical branch is named slippi-ai.  Frisson uses
        # the same CanonicalControllerCommand transport but never its runtime.
        slot["model"] = "slippi-ai"
        inference = slot.get("inference")
        if isinstance(inference, dict):
            inference["native_dummy_prefix"] = False
    return adapted


def _game_start_transport_proof(transport: object) -> dict[str, Any]:
    from melee_policy.integration.slippi_match import _game_start_transport_proof as shared_proof

    return shared_proof(transport)


def _audit_controller_boundary(
    trace_path: Path,
    replay_path: Path,
    project_root: Path,
    *,
    game_start_transport_proof: Mapping[str, Any],
) -> dict[str, Any]:
    from melee_policy.integration.slippi_match import (
        _audit_controller_boundary_records,
        _read_replay_controller_states,
        _read_trace_rows,
    )

    rows = _read_trace_rows(trace_path)
    audit = _audit_controller_boundary_records(
        _adapt_trace_rows_for_shared_audit(rows),
        _read_replay_controller_states(replay_path),
        lag_frames=CONTROLLER_REPLAY_LAG_FRAMES,
        game_start_transport_proof=game_start_transport_proof,
    )
    slots = cast(dict[str, Any], audit["slots"])
    if "p1" in slots:
        slots["p1"]["model"] = "frisson-ai"
        processed_c = slots["p1"].get("processed_c_stick")
        if isinstance(processed_c, dict):
            processed_c.pop("native_slippi_ai_dummy_prefix_expected", None)
    adapter = cast(dict[str, Any], audit.get("adapter", {}))
    adapter.pop("slippi_ai_explicit_flush", None)
    adapter.pop("slippi_ai_dispatch_boundary", None)
    adapter.update(
        {
            "frisson_explicit_flush": False,
            "frisson_dispatch_boundary": (
                "zero-delay command queued after both current-frame inferences and committed "
                "at the next two-port controller boundary"
            ),
            "frisson_native_dummy_prefix": False,
            "frisson_observation_filter": None,
        }
    )
    audit["classification"] = "Frisson and MIMIC decoded commands verified at the shared boundary"
    audit["trace"] = _file_identity(trace_path, project_root)
    audit["trace"]["rows"] = len(rows)
    audit["replay"] = _file_identity(replay_path, project_root)
    return audit


def _unavailable_controller_audit(
    trace_path: Path,
    project_root: Path,
    reason: str,
) -> dict[str, Any]:
    from melee_policy.integration.slippi_match import _unavailable_controller_boundary

    return _unavailable_controller_boundary(trace_path, project_root, reason)


def _selected_controller_audit(
    trace_path: Path,
    replay_paths: list[Path],
    replay_records: list[dict[str, Any]],
    project_root: Path,
    game_start_transport_proof: Mapping[str, Any],
) -> dict[str, Any]:
    selected = [
        index for index, record in enumerate(replay_records) if record.get("tournament_result_replay") is True
    ]
    if len(selected) != 1 or selected[0] >= len(replay_paths):
        return _unavailable_controller_audit(
            trace_path,
            project_root,
            "controller audit requires exactly one trace-covering result replay",
        )
    try:
        return _audit_controller_boundary(
            trace_path,
            replay_paths[selected[0]],
            project_root,
            game_start_transport_proof=game_start_transport_proof,
        )
    except Exception as error:
        return _unavailable_controller_audit(
            trace_path,
            project_root,
            f"{type(error).__name__}: {error}",
        )


def _collect_replays(
    replay_directory: Path,
    existing_replays: set[Path],
    project_root: Path,
    first_game_frame: int | None,
    last_game_frame: int | None,
    sudden_death_transition_observed: bool,
) -> tuple[list[Path], list[dict[str, Any]]]:
    paths = sorted(
        path
        for path in replay_directory.rglob("*.slp")
        if path.is_file() and path.resolve() not in existing_replays
    )
    records: list[dict[str, Any]] = []
    for path in paths:
        validation = (
            _validate_saved_replay(path)
            if first_game_frame is None or last_game_frame is None
            else _validate_saved_replay(
                path,
                required_first_frame=first_game_frame,
                required_last_frame=last_game_frame,
            )
        )
        records.append(
            {
                "path": _display_path(path, project_root),
                "byte_length": path.stat().st_size,
                "sha256": _sha256_file(path),
                "validation": validation,
            }
        )
    _classify_trace_covering_replays(
        records,
        sudden_death_transition_observed=sudden_death_transition_observed,
    )
    return paths, records


def _replay_character_expectation(character: str) -> str:
    """Translate the one libmelee CSS alias used by the replay parser."""

    return _REPLAY_CHARACTER_ALIASES.get(character, character)


def _selected_replay_identity_audit(
    replay_paths: list[Path],
    replay_records: list[dict[str, Any]],
    project_root: Path,
    request: FrissonMatchRequest,
) -> dict[str, Any]:
    """Prove that the selected saved replay contains the requested live matchup."""

    expected_characters = {
        "p1": {
            "requested": request.player_1_character,
            "replay_name": _replay_character_expectation(request.player_1_character),
        },
        "p2": {
            "requested": request.player_2_character,
            "replay_name": _replay_character_expectation(request.player_2_character),
        },
    }
    checks = {
        "selected_result_replay_exactly_one": False,
        "selected_result_replay_identity_exact": False,
        **{name: False for name in _REPLAY_IDENTITY_REQUIRED_CHECKS},
    }
    selected = [
        index
        for index, record in enumerate(replay_records)
        if record.get("tournament_result_replay") is True
    ]
    if len(selected) != 1 or selected[0] >= len(replay_paths):
        return {
            "schema_version": "integration.frisson_vs_mimic.replay_identity.v1",
            "decision": "fail",
            "checks": checks,
            "expected": {
                "stage": MATCH_STAGE,
                "characters": expected_characters,
            },
            "selected_replay": None,
            "audit": None,
            "error": "replay identity audit requires exactly one selected result replay",
        }

    index = selected[0]
    replay_path = replay_paths[index]
    selected_record = replay_records[index]
    checks["selected_result_replay_exactly_one"] = True
    try:
        from melee_policy.integration.replay_result import audit_replay

        audit = audit_replay(
            replay_path,
            expected_stage=MATCH_STAGE,
            expected_characters={
                FRISSON_PORT: expected_characters["p1"]["replay_name"],
                MIMIC_PORT: expected_characters["p2"]["replay_name"],
            },
        )
        audit_checks = audit.get("checks")
        if not isinstance(audit_checks, dict):
            raise TypeError("replay identity audit did not return a checks object")
        for name in _REPLAY_IDENTITY_REQUIRED_CHECKS:
            checks[name] = audit_checks.get(name) is True
        replay_identity = audit.get("replay")
        checks["selected_result_replay_identity_exact"] = bool(
            isinstance(replay_identity, dict)
            and replay_identity.get("sha256") == selected_record.get("sha256")
            and replay_identity.get("raw_byte_length") == selected_record.get("byte_length")
        )
        return {
            "schema_version": "integration.frisson_vs_mimic.replay_identity.v1",
            "decision": "pass" if all(checks.values()) else "fail",
            "checks": checks,
            "expected": {
                "stage": MATCH_STAGE,
                "characters": expected_characters,
            },
            "selected_replay": {
                "path": _display_path(replay_path, project_root),
                "sha256": selected_record.get("sha256"),
                "byte_length": selected_record.get("byte_length"),
            },
            "audit": audit,
            "error": None,
        }
    except Exception as error:
        return {
            "schema_version": "integration.frisson_vs_mimic.replay_identity.v1",
            "decision": "fail",
            "checks": checks,
            "expected": {
                "stage": MATCH_STAGE,
                "characters": expected_characters,
            },
            "selected_replay": {
                "path": _display_path(replay_path, project_root),
                "sha256": selected_record.get("sha256"),
                "byte_length": selected_record.get("byte_length"),
            },
            "audit": None,
            "error": f"{type(error).__name__}: {error}",
        }


def _run_console(
    config: dict[str, Any],
    project_root: Path,
    iso_path: Path,
    request: FrissonMatchRequest,
    contract: dict[str, Any],
    source_checks: dict[str, Any],
    mimic_runtime: MimicRuntime,
    frisson_session: FrissonPolicySession,
    reproducibility: dict[str, Any],
) -> dict[str, Any]:
    import melee

    menu_characters = _menu_character_selection(melee, request)
    output_directory = project_root / str(config["frisson_ai"]["output_directory"])
    if request.artifact_label is not None:
        output_directory = output_directory / request.artifact_label
    _require_unused_artifact_label(output_directory, request.artifact_label)
    replay_directory = output_directory / "replays"
    trace_path = output_directory / "controller_trace.jsonl"
    summary_path = output_directory / "summary.json"
    output_directory.mkdir(parents=True, exist_ok=True)
    replay_directory.mkdir(parents=True, exist_ok=True)
    existing_replays = {path.resolve() for path in replay_directory.rglob("*.slp")}

    emulator_application = _emulator_application_identity(config, project_root)
    version = _attested_emulator_release(emulator_application)

    udp_port = int(config["emulator"]["slippi_port"])
    _assert_udp_port_available(udp_port)
    console = _create_attested_dolphin_console(
        config,
        project_root,
        emulator_application,
        **_console_options(replay_directory, udp_port),
    )
    frisson_controller = melee.Controller(
        console=console,
        port=FRISSON_PORT,
        type=melee.ControllerType.STANDARD,
    )
    mimic_controller = melee.Controller(
        console=console,
        port=MIMIC_PORT,
        type=melee.ControllerType.STANDARD,
    )
    raw_controllers = {FRISSON_PORT: frisson_controller, MIMIC_PORT: mimic_controller}
    mimic_policy = MimicLivePolicy(
        mimic_runtime,
        MIMIC_PORT,
        online_delay_frames=0,
        evaluation_seed=request.seed,
    )
    mimic_worker = _LatestInferenceWorker("MIMIC", mimic_policy.infer_snapshot)
    menu_p1 = melee.MenuHelper()
    menu_p2 = melee.MenuHelper()
    max_game_frames = (
        int(config["integration"]["max_game_frames"])
        if request.max_game_frames is None
        else request.max_game_frames
    )
    menu_timeout = float(config["integration"]["menu_timeout_seconds"])
    started_at = time.time()
    trace_stream: TextIO | None = None
    transport: _ControllerPipeLockstep | None = None
    in_game = False
    game_end_observed = False
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
    strict_frame_order = True
    inference_counts = {"frisson-ai": 0, "mimic": 0}
    dispatch_counts = {"frisson-ai": 0, "mimic": 0}
    inference_seconds: dict[str, list[float]] = {"frisson-ai": [], "mimic": []}
    command_ages: dict[str, list[int]] = {"frisson-ai": [], "mimic": []}
    barrier_seconds: list[float] = []
    mimic_sent = _neutral_mimic_command()
    mimic_pressed: list[str] = []
    stocks = {"frisson-ai": 4, "mimic": 4}
    menu_transport_flushes = {1: 0, 2: 0}
    no_frame_watchdog = InGameNoFrameWatchdog()

    def interrupt(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGINT, interrupt)
    try:
        trace_stream = trace_path.open("w", encoding="utf-8")
        if not _launch_and_connect_attested_dolphin(console, iso_path):
            raise RuntimeError("libmelee could not connect to Slippi Dolphin")
        if not frisson_controller.connect() or not mimic_controller.connect():
            raise RuntimeError("libmelee could not connect both virtual controllers")
        transport = _ControllerPipeLockstep.install(console, raw_controllers)
        frisson_controller = transport.controllers[FRISSON_PORT]
        mimic_controller = transport.controllers[MIMIC_PORT]
        transport.prime()
        print(
            f"P1=FRISSON-AI {request.player_1_character} | "
            f"P2=MIMIC {request.player_2_character} | "
            f"stage={MATCH_STAGE} | seed={request.seed} | inference=exact",
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
                game_end_observed = True
                sudden_death_transition_observed = True
                termination = "natural-game-end"
                break
            if gamestate.menu_state not in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
                if in_game:
                    game_end_observed = True
                    termination = "natural-game-end"
                    break
                if time.time() - started_at > menu_timeout:
                    raise TimeoutError("automatic menu navigation did not start a match before timeout")
                transport.begin_boundary(reason="menu", game_frame=None)
                menu_p1.menu_helper_simple(
                    gamestate,
                    frisson_controller,
                    menu_characters[FRISSON_PORT],
                    melee.Stage.FINAL_DESTINATION,
                    cpu_level=0,
                    autostart=False,
                    frozen_stadium=True,
                )
                menu_p2.menu_helper_simple(
                    gamestate,
                    mimic_controller,
                    menu_characters[MIMIC_PORT],
                    melee.Stage.FINAL_DESTINATION,
                    cpu_level=0,
                    autostart=True,
                    frozen_stadium=True,
                )
                frisson_controller.flush()
                mimic_controller.flush()
                transport.commit_boundary()
                menu_transport_flushes[1] += 1
                menu_transport_flushes[2] += 1
                continue

            in_game = True
            game_frame = int(gamestate.frame)
            _require_exact_player_ports(gamestate, game_frame)
            if first_game_frame is None:
                first_game_frame = game_frame
                first_context = _validate_first_context(
                    gamestate,
                    frisson_character=request.player_1_character,
                    mimic_character=request.player_2_character,
                )
            if previous_game_frame is not None:
                delta = game_frame - previous_game_frame
                frame_delta_counts[delta] += 1
                if delta != 1:
                    strict_frame_order = False
                    raise RuntimeError(
                        f"rendered policy frames are not consecutive: {previous_game_frame} to {game_frame}"
                    )
            previous_game_frame = game_frame
            last_game_frame = game_frame

            frame_result = _run_exact_frame(
                gamestate=gamestate,
                frisson_session=frisson_session,
                mimic_policy=mimic_policy,
                mimic_worker=mimic_worker,
                frisson_controller=frisson_controller,
                mimic_controller=mimic_controller,
                transport=transport,
                mimic_held=mimic_sent,
                mimic_held_pressed=mimic_pressed,
                mimic_temperature=float(config["mimic"]["temperature"]),
                mimic_top_k=int(config["mimic"]["top_k"]),
                mimic_top_p=float(config["mimic"]["top_p"]),
            )
            mimic_sent = frame_result.mimic_sent
            mimic_pressed = frame_result.mimic_pressed
            inference_counts["frisson-ai"] += 1
            inference_counts["mimic"] += 1
            dispatch_counts["frisson-ai"] += 1
            dispatch_counts["mimic"] += 1
            inference_seconds["frisson-ai"].append(frame_result.frisson_inference_seconds)
            inference_seconds["mimic"].append(frame_result.mimic_inference_seconds)
            command_ages["frisson-ai"].append(0)
            command_ages["mimic"].append(game_frame - frame_result.mimic_source_frame)
            barrier_seconds.append(frame_result.barrier_seconds)
            row = _trace_row(
                gamestate,
                frame_result,
                frisson_character=request.player_1_character,
                mimic_character=request.player_2_character,
                mimic_checkpoint_character=mimic_runtime.controlled_character,
            )
            trace_stream.write(json.dumps(row, sort_keys=True) + "\n")
            trace_stream.flush()
            stocks = {
                "frisson-ai": int(gamestate.players[FRISSON_PORT].stock),
                "mimic": int(gamestate.players[MIMIC_PORT].stock),
            }
            processed_frames += 1
            if processed_frames % 60 == 0:
                print(
                    f"frame {game_frame}: Frisson {stocks['frisson-ai']} stocks, "
                    f"MIMIC {stocks['mimic']} stocks",
                    flush=True,
                )
            if has_decisive_zero_stock(gamestate, (FRISSON_PORT, MIMIC_PORT)):
                game_end_observed = True
                termination = "natural-game-end"
                break
        if in_game and processed_frames >= max_game_frames:
            termination = "frame-limit-before-natural-end"
    except BaseException as caught:
        exception = caught
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        try:
            mimic_worker.close()
        except BaseException as worker_error:
            if exception is None:
                exception = worker_error
        if mimic_worker.failure is not None and exception is None:
            exception = RuntimeError(f"MIMIC inference worker failed: {mimic_worker.failure}")
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
    replay_identity_audit = _selected_replay_identity_audit(
        replay_paths,
        replay_records,
        project_root,
        request,
    )
    replay_identity_checks = cast(dict[str, bool], replay_identity_audit["checks"])
    transport_record = transport.audit_record() if transport is not None else {"installed": False}
    game_start_transport_proof = _game_start_transport_proof(transport_record)
    controller_audit = _selected_controller_audit(
        trace_path,
        replay_paths,
        replay_records,
        project_root,
        game_start_transport_proof,
    )
    controller_checks = cast(dict[str, bool], controller_audit["gate"]["checks"])
    diagnostics = frisson_session.diagnostics()
    metadata = frisson_session.metadata()
    character_gate_checks = _mimic_character_gate_checks(
        request,
        contract,
        mimic_runtime,
        mimic_policy,
    )
    transport_checks = (
        transport.gate_checks()
        if transport is not None
        else {
            "controller_pipe_lockstep_installed": False,
            "controller_pipe_registration_exact": False,
            "controller_pipe_primed_exactly_once": False,
            "controller_pipe_device_order_complete": False,
            "controller_pipe_no_pending_boundaries_after_shutdown": False,
            "controller_pipe_every_boundary_committed_exactly_once": False,
            "controller_pipe_every_scheduled_boundary_consumed_exactly_once": False,
            "controller_pipe_every_group_flushes_both_ports": False,
            "controller_pipe_one_preamble_decision_per_step": False,
        }
    )
    gate_checks: dict[str, bool] = {
        **character_gate_checks,
        "final_destination": request.stage == MATCH_STAGE,
        "evaluation_seed_valid": _validate_evaluation_seed(request.seed) == request.seed,
        "natural_game_end_observed": game_end_observed,
        "entered_gameplay": in_game,
        "processed_at_least_one_frame": processed_frames > 0,
        "first_policy_frame_minus_123": first_game_frame == FIRST_POLICY_FRAME,
        "strict_consecutive_policy_frames": strict_frame_order,
        "one_frisson_inference_per_frame": inference_counts["frisson-ai"] == processed_frames,
        "one_mimic_inference_per_frame": inference_counts["mimic"] == processed_frames,
        "both_inferences_barrier_per_frame": len(barrier_seconds) == processed_frames,
        "all_commands_current_frame": all(age == 0 for ages in command_ages.values() for age in ages)
        and all(len(ages) == processed_frames for ages in command_ages.values()),
        "one_frisson_dispatch_per_frame": dispatch_counts["frisson-ai"] == processed_frames,
        "one_mimic_dispatch_per_frame": dispatch_counts["mimic"] == processed_frames,
        "frisson_session_frame_count_exact": diagnostics.get("frames_total") == processed_frames,
        "frisson_session_barrier_count_exact": (
            diagnostics.get("current_frame_inference_barriers") == processed_frames
            and diagnostics.get("current_frame_inference_barrier_every_frame") is True
        ),
        "frisson_delay_zero": metadata["runtime_contract"]["delay_frames"] == 0,
        "frisson_action_offset_one": metadata["action_contract"]["action_offset_frames"] == 1,
        "frisson_temperature_one": (metadata["runtime_contract"]["sample_temperature"] == 1.0),
        "frisson_rolling_256_kv_cache": (
            metadata["runtime_contract"]["kv_cache_capacity_frames"] == EXPECTED_MODEL_CONTEXT_LENGTH
            and metadata["runtime_contract"]["periodic_128_frame_reset"] is False
        ),
        "frisson_has_no_observation_filter": (metadata["runtime_contract"]["observation_filter"] is None),
        "frisson_has_no_slippi_ai_fifo": (metadata["action_contract"]["slippi_ai_21_frame_fifo"] is False),
        **replay_checks,
        **{
            f"replay_identity_audit.{name}": passed
            for name, passed in replay_identity_checks.items()
        },
        **{f"controller_audit.{name}": passed for name, passed in controller_checks.items()},
        **transport_checks,
    }
    if exception is None and not all(gate_checks.values()):
        exception = RuntimeError(
            "Frisson-versus-MIMIC integration gate failed: "
            f"{[name for name, passed in gate_checks.items() if not passed]}"
        )
    result = "complete" if exception is None and all(gate_checks.values()) else "failed"
    winner = (
        "frisson-ai"
        if stocks["frisson-ai"] > stocks["mimic"]
        else "mimic"
        if stocks["mimic"] > stocks["frisson-ai"]
        else None
    )
    trace_artifact: dict[str, Any] = {
        **_file_identity(trace_path, project_root),
        "rows": processed_frames,
    }
    ood_transfer = request.player_2_character not in MIMIC_SUPPORTED_CHARACTERS
    classification = (
        "frame-exact Frisson "
        f"{contract['selected_checkpoint']['profile']} {request.player_1_character} versus "
        f"released MIMIC {request.player_2_character} integration"
    )
    if ood_transfer:
        classification = (
            "frame-exact Frisson "
            f"{contract['selected_checkpoint']['profile']} {request.player_1_character} versus "
            f"released MIMIC {mimic_runtime.controlled_character} checkpoint on OOD physical "
            f"{request.player_2_character} integration"
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "classification": classification,
        "result": result,
        "error": None if exception is None else f"{type(exception).__name__}: {exception}",
        "gate": {
            "decision": "pass" if result == "complete" else "fail",
            "checks": gate_checks,
        },
        "configuration": {
            "player_1": {
                "model": "frisson-ai",
                "character": request.player_1_character,
                "port": 1,
            },
            "player_2": {
                "model": "mimic",
                "character": request.player_2_character,
                "port": 2,
                **(
                    {
                        "physical_character": request.player_2_character,
                        "checkpoint_character": mimic_runtime.controlled_character,
                        "character_transfer_mode": MIMIC_OOD_CHARACTER_TRANSFER_MODE,
                    }
                    if ood_transfer
                    else {}
                ),
            },
            "stage": MATCH_STAGE,
            "seed": request.seed,
            "require_natural_end": True,
            "maximum_game_frames": max_game_frames,
            "blocking_input": True,
            "online_delay_frames": 0,
            "controller_replay_lag_frames": CONTROLLER_REPLAY_LAG_FRAMES,
            "inference_mode": EXACT_INFERENCE_MODE,
            **({"allow_player_2_ood_character": True} if ood_transfer else {}),
        },
        "contract": contract,
        "frisson": {"metadata": metadata, "diagnostics": diagnostics},
        "mimic": {
            "checkpoint_sha256": mimic_runtime.checkpoint_sha256,
            "checkpoint_filename": mimic_runtime.checkpoint_path.name,
            "asset_directory": _display_path(mimic_runtime.asset_directory, project_root),
            "bundle_identity": mimic_runtime.bundle_identity,
            "controlled_character": mimic_runtime.controlled_character,
            "policy_rng": mimic_policy.policy_rng,
            **(
                {"ood_character_transfer": contract["mimic_ood_character_transfer"]}
                if ood_transfer
                else {}
            ),
        },
        "sources": source_checks,
        "reproducibility": reproducibility,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "libmelee_module": str(Path(melee.__file__).resolve()),
            "slippi_version": version,
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
            "last_stocks": stocks,
            "winner": winner,
            "game_end_observed": game_end_observed,
            "sudden_death_transition_observed": sudden_death_transition_observed,
            "termination": termination,
            "wall_seconds": time.time() - started_at,
            "shutdown_method": shutdown_method,
            "controller_transport": {
                **transport_record,
                "menu_flushes": {f"p{port}": menu_transport_flushes[port] for port in (1, 2)},
            },
        },
        "replay_identity_audit": replay_identity_audit,
        "controller_boundary_audit": controller_audit,
        "artifacts": {
            "trace": trace_artifact,
            "replays": replay_records,
            "summary": _display_path(summary_path, project_root),
        },
    }
    _write_json(summary_path, summary)
    if exception is not None:
        raise RuntimeError(summary["error"])
    return summary


def run_frisson_match(
    config_path: Path,
    iso_path: Path | None = None,
    request: FrissonMatchRequest | None = None,
) -> dict[str, Any]:
    """Load one supported Frisson character and one released MIMIC character."""
    match_request = FrissonMatchRequest() if request is None else request
    if match_request.artifact_label is None and (match_request.save_slp or match_request.save_video):
        from melee_policy.integration.game_bundle import export_artifact_label

        match_request = replace(
            match_request,
            artifact_label=export_artifact_label(
                None,
                save_slp=match_request.save_slp,
                save_video=match_request.save_video,
            ),
        )
    match_request.validate()
    config, project_root = _load_config(config_path)
    _validate_config_contract(config, match_request)
    policy_config = _frisson_policy_config(config, project_root, match_request)
    checkpoint_identity = inspect_frisson_checkpoint(policy_config.checkpoint_path)
    contract = _validate_config_contract(
        config,
        match_request,
        checkpoint_identity=checkpoint_identity,
    )
    random.seed(match_request.seed)
    np.random.seed(match_request.seed)
    torch.manual_seed(match_request.seed)
    torch.set_num_threads(1)
    image_path = _resolve_game_image_path(config, project_root, iso_path)
    mimic_checkpoint, mimic_assets = _resolve_mimic_bundle_selection(
        config,
        project_root,
        match_request,
    )
    mimic_runtime = load_mimic_runtime(
        config,
        project_root,
        checkpoint_override=mimic_checkpoint,
        asset_directory_override=mimic_assets,
    )
    mimic_source = mimic_runtime.source_identity
    contract = _bind_mimic_runtime_contract(
        contract,
        match_request,
        mimic_runtime,
        project_root,
    )
    reproducibility = _runtime_reproducibility_record(
        project_root,
        config_path.resolve(),
        "requirements-e010.lock",
        (
            "src/melee_policy/integration/frisson_policy.py",
            "src/melee_policy/integration/frisson_match.py",
            "src/melee_policy/integration/game_bundle.py",
            "src/melee_policy/integration/match_runtime.py",
            "src/melee_policy/integration/native/macos_mux_replay_audio.m",
            "src/melee_policy/integration/native/macos_replay_recorder.m",
            "src/melee_policy/integration/replay_video.py",
            "patches/slippi-dolphin-two-pipe-frame-sync.patch",
        ),
    )
    session = FrissonPolicySession(policy_config)
    session.start()
    try:
        summary = _run_console(
            config,
            project_root,
            image_path,
            match_request,
            contract,
            {"mimic": mimic_source},
            mimic_runtime,
            session,
            reproducibility,
        )
    finally:
        session.close()
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
    "CONTROLLER_REPLAY_LAG_FRAMES",
    "FRISSON_SUPPORTED_CHARACTERS",
    "MATCH_SEED",
    "MATCH_STAGE",
    "MIMIC_RELEASED_ASSET_DIRECTORIES",
    "MIMIC_SUPPORTED_CHARACTERS",
    "FrissonMatchRequest",
    "run_frisson_match",
]
