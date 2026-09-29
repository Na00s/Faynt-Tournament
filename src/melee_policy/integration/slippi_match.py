"""Run frame-exact Slippi-AI matches against MIMIC or another Slippi-AI policy."""

from __future__ import annotations

import json
import math
import os
import platform
import random
import signal
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TextIO, cast

import numpy as np
import torch

from melee_policy.integration.frame_watchdog import InGameNoFrameWatchdog
from melee_policy.integration.mimic_bundle_manifest import (
    MIMIC_NATIVE_BUNDLE_CHARACTERS,
    mimic_native_bundle,
)
from melee_policy.integration.match_runtime import (
    LAUNCHABLE_CHARACTERS,
    MIMIC_DECODE_STRATEGY,
    MIMIC_SOURCE_REPOSITORY,
    MIMIC_SOURCE_REVISION,
    MimicLivePolicy,
    MimicRuntime,
    _assert_udp_port_available,
    _attested_emulator_release,
    _ControllerPipeLockstep,
    _create_attested_dolphin_console,
    _display_path,
    _effective_evaluation_seed,
    _emulator_application_identity,
    _finite_prediction,
    _git_output,
    _LatestInferenceWorker,
    _launch_and_connect_attested_dolphin,
    _load_config,
    _natural_end_requirement_met,
    _neutral_mimic_command,
    _require_exact_player_ports,
    _require_unused_artifact_label,
    _resolve_game_image_path,
    _runtime_reproducibility_record,
    _send_mimic_controller_inputs,
    _sha256_file,
    _stop_console,
    _validate_evaluation_seed,
    _validate_saved_replay,
    _write_json,
    load_mimic_runtime,
)
from melee_policy.integration.natural_game_end import has_decisive_zero_stock
from melee_policy.integration.slippi_ai_policy import (
    DEFAULT_PLAYER_NAME,
    MEDIUM_V2_CHECKPOINT_BYTES,
    MEDIUM_V2_CHECKPOINT_SHA256,
    MEDIUM_V2_SUPPORTED_CHARACTERS,
    MEDIUM_V2_SUPPORTED_STAGES,
    POLICY_DELAY_FRAMES,
    SAMPLE_TEMPERATURE,
    SLIPPI_AI_RELEASE_CONTRACTS,
    SLIPPI_AI_REPOSITORY_URL,
    SLIPPI_AI_SOURCE_REVISION,
    UPSTREAM_EVAL_TWO_CONSOLE_DELAY_FRAMES,
    CanonicalControllerCommand,
    SlippiAIPolicyConfig,
    SlippiAIPolicySession,
    send_canonical_controller,
    slippi_ai_release_capabilities,
    slippi_ai_release_contract,
)

SCHEMA_VERSION = "integration.slippi_ai.v6"
RUNTIME_IMPLEMENTATION_PATHS = (
    "src/melee_policy/integration/game_bundle.py",
    "src/melee_policy/integration/match_runtime.py",
    "src/melee_policy/integration/native/macos_mux_replay_audio.m",
    "src/melee_policy/integration/native/macos_replay_recorder.m",
    "src/melee_policy/integration/replay_video.py",
    "src/melee_policy/integration/slippi_ai_policy.py",
    "src/melee_policy/integration/slippi_compatibility.py",
    "src/melee_policy/integration/slippi_match.py",
    "src/melee_policy/integration/play.py",
    "src/melee_policy/integration/runtime_identity.py",
    "src/melee_policy/integration/state_identity.py",
    "patches/slippi-dolphin-two-pipe-frame-sync.patch",
    "scripts/play",
)
CONTROLLER_BOUNDARY_SCHEMA_VERSION = "integration.controller_boundary.v12"
FIRST_POLICY_FRAME = -123
NEXT_FRAME_CONTROLLER_BOUNDARY_FRAMES = 1
FIRST_GAMEPLAY_CONTROLLER_LATCH_UNOBSERVABLE_FRAMES = 1
FIRST_GAMEPLAY_COMMAND_CARRYOVER_TRACE_FRAME = FIRST_POLICY_FRAME + 1
FIRST_GAMEPLAY_COMMAND_CARRYOVER_UNOBSERVABLE_FRAMES = 1
GAME_START_CONTROLLER_LATCH_TRACE_FRAME = FIRST_POLICY_FRAME + 4
GAME_START_CONTROLLER_LATCH_UNOBSERVABLE_FRAMES = 1
GAME_START_CONTROLLER_LATCH_LATEST_TRACE_FRAME = FIRST_POLICY_FRAME + 5
GAME_START_PLAYER_ACTIONS = frozenset((322, 323))
PROCESSED_C_STICK_TOLERANCE = 1e-6
RAW_AXIS_DEADZONE = 23
PHYSICAL_ANALOG_SHOULDER_TOLERANCE = 1e-6
MIXED_RUNTIME_CONSOLE_DELAY_FRAMES = 0
MIXED_EFFECTIVE_POLICY_DELAY_FRAMES = POLICY_DELAY_FRAMES - MIXED_RUNTIME_CONSOLE_DELAY_FRAMES
CONTROLLER_REPLAY_LAG_FRAMES = MIXED_RUNTIME_CONSOLE_DELAY_FRAMES + NEXT_FRAME_CONTROLLER_BOUNDARY_FRAMES
FORMAL_GAME_END_DRAIN_SCHEMA_VERSION = "integration.slippi_ai.formal_game_end_drain.v1"
FORMAL_GAME_END_DRAIN_MAX_STEP_CALLS = 600
REPLAY_RESULT_BINDING_SCHEMA_VERSION = "integration.slippi_ai.replay_result_binding.v1"
_REPLAY_CHARACTER_ALIASES = {"POPO": "ICE_CLIMBERS"}
_REPLAY_RESULT_BINDING_REQUIRED_CHECKS = (
    "selected_result_replay_exactly_one",
    "selected_result_replay_identity_exact",
    "replay_audit_passed",
    "replay_tournament_result_ready",
    "replay_outcome_complete",
    "replay_outcome_conclusive",
    "replay_outcome_is_decisive_win",
    "replay_winner_port_valid",
    "live_terminal_stocks_are_decisive",
    "replay_winner_matches_live_terminal_stocks",
)
LEGAL_STAGES = frozenset(
    {
        "BATTLEFIELD",
        "DREAMLAND",
        "FINAL_DESTINATION",
        "FOUNTAIN_OF_DREAMS",
        "POKEMON_STADIUM",
        "YOSHIS_STORY",
    }
)
def _reproducibility_inputs_hashed(record: object) -> bool:
    """Require exact input coverage using the same paths as the launch recorder."""
    if not isinstance(record, Mapping):
        return False
    files = record.get("implementation_files")
    if not isinstance(files, list) or len(files) != len(RUNTIME_IMPLEMENTATION_PATHS):
        return False
    identities = [*files, record.get("configuration"), record.get("dependency_lock")]
    for identity in identities:
        if not isinstance(identity, Mapping):
            return False
        path, digest, size = (
            identity.get("path"),
            identity.get("sha256"),
            identity.get("byte_length"),
        )
        if not (
            isinstance(path, str)
            and path
            and isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            and type(size) is int
            and size >= 0
        ):
            return False
    return {identity["path"] for identity in files} == set(RUNTIME_IMPLEMENTATION_PATHS)


_ALLOWED_WINDOWED_MODELS = frozenset(("mimic",))
_STREAMING_MODELS = frozenset(("slippi-ai",))
_EXACT_INFERENCE_MODES = frozenset(("exact", "synchronous-concurrent"))
# Nana cannot be selected as an independent primary character.
# Special, wireframe, and sentinel values are excluded too.
LAUNCHABLE_PRIMARY_CHARACTERS = frozenset(LAUNCHABLE_CHARACTERS)


def _model_name(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    if normalized == "slippi":
        return "slippi-ai"
    if normalized == "frisson":
        return "frisson-ai"
    return normalized


@dataclass(frozen=True, slots=True)
class SlippiMatchRequest:
    """Slot assignments and optional runtime overrides for one exact match."""

    player_1_model: str = "slippi-ai"
    player_2_model: str = "mimic"
    player_1_character: str = "FOX"
    player_2_character: str = "FOX"
    stage: str = "BATTLEFIELD"
    player_1_checkpoint: Path | None = None
    player_2_checkpoint: Path | None = None
    player_1_assets: Path | None = None
    player_2_assets: Path | None = None
    player_1_name: str | None = None
    player_2_name: str | None = None
    player_1_temperature: float | None = None
    player_2_temperature: float | None = None
    player_1_slippi_release: str = "medium-v2"
    player_2_slippi_release: str = "medium-v2"
    max_game_frames: int | None = None
    inference_mode: str | None = None
    artifact_label: str | None = None
    seed: int | None = None
    require_natural_end: bool = False
    save_slp: bool = False
    save_video: bool = False
    require_formal_game_end: bool = False
    allow_player_1_ood_character: bool = False
    allow_player_2_ood_character: bool = False

    def validate(self) -> None:
        models = (self.model_for_port(1), self.model_for_port(2))
        mixed_match = models.count("slippi-ai") == 1 and len(
            set(models) & _ALLOWED_WINDOWED_MODELS
        ) == 1
        dual_slippi_match = models == ("slippi-ai", "slippi-ai")
        if not (mixed_match or dual_slippi_match):
            raise ValueError(
                "this runner requires two 'slippi-ai' sides or exactly one 'slippi-ai' side "
                "and one 'mimic' side"
            )
        for label, character in (
            ("player 1", self.player_1_character),
            ("player 2", self.player_2_character),
        ):
            if not character or character != character.upper():
                raise ValueError(f"{label} character must be an uppercase libmelee enum name")
            if character not in LAUNCHABLE_PRIMARY_CHARACTERS:
                raise ValueError(
                    f"{label} character {character!r} is not an independently launchable "
                    "primary Melee character"
                )
        if self.stage not in LEGAL_STAGES:
            raise ValueError(f"stage must be one of {sorted(LEGAL_STAGES)}")
        for port in (1, 2):
            model = self.model_for_port(port)
            if model != "mimic" and self.assets_for_port(port) is not None:
                raise ValueError(f"{model} does not accept a separate asset directory")
            if model != "slippi-ai" and self.name_for_port(port) is not None:
                raise ValueError(f"{model} does not accept a Slippi-AI player name")
            temperature = self.temperature_for_port(port)
            if model not in _STREAMING_MODELS and temperature is not None:
                raise ValueError(f"{model} does not accept a Slippi-AI sample temperature")
            if temperature is not None and (not math.isfinite(temperature) or temperature <= 0.0):
                raise ValueError("Slippi-AI sample temperature must be finite and greater than zero")
        if self.max_game_frames is not None and self.max_game_frames < 1:
            raise ValueError("max_game_frames must be positive")
        if self.seed is not None:
            _validate_evaluation_seed(self.seed)
        if self.inference_mode is not None and self.inference_mode not in _EXACT_INFERENCE_MODES:
            if self.inference_mode in ("watch", "asynchronous-latest"):
                raise ValueError(
                    f"{self.inference_mode!r} is forbidden because recurrent Slippi-AI frames cannot drop"
                )
            raise ValueError(f"unsupported inference mode: {self.inference_mode}")
        if not all(
            isinstance(value, bool)
            for value in (self.save_slp, self.save_video, self.require_formal_game_end)
        ):
            raise TypeError("save_slp, save_video, and require_formal_game_end must be booleans")
        for port in (1, 2):
            release = self.slippi_release_for_port(port)
            if release not in SLIPPI_AI_RELEASE_CONTRACTS:
                raise ValueError(
                    f"unsupported Slippi-AI release {release!r}; "
                    f"allowed={tuple(SLIPPI_AI_RELEASE_CONTRACTS)}"
                )
            if self.model_for_port(port) != "slippi-ai" and release != "medium-v2":
                raise ValueError(
                    f"player {port} Slippi-AI release selection requires Slippi-AI on that port"
                )
            if (
                self.model_for_port(port) == "slippi-ai"
                and self.character_for_port(port)
                not in slippi_ai_release_contract(release).supported_characters
                and not self.allow_ood_for_port(port)
            ):
                raise ValueError(
                    f"Slippi-AI {release} does not cover physical "
                    f"{self.character_for_port(port)} on P{port}; an explicit P{port} OOD "
                    "opt-in is required"
                )
        if dual_slippi_match and any(
            self.slippi_release_for_port(port) != "medium-v2" for port in (1, 2)
        ):
            raise ValueError("dual Slippi-AI matches currently require medium-v2 on both ports")
        if not isinstance(self.allow_player_1_ood_character, bool):
            raise TypeError("allow_player_1_ood_character must be a boolean")
        if not isinstance(self.allow_player_2_ood_character, bool):
            raise TypeError("allow_player_2_ood_character must be a boolean")
        if self.allow_player_1_ood_character and self.model_for_port(1) != "slippi-ai":
            raise ValueError("P1 OOD character transfer is scoped to Slippi-AI")
        if self.allow_player_2_ood_character and self.model_for_port(2) not in {
            "slippi-ai",
            "mimic",
        }:
            raise ValueError("P2 OOD character transfer is scoped to Slippi-AI or MIMIC")
        if (
            self.model_for_port(2) == "mimic"
            and self.player_2_character not in MIMIC_NATIVE_BUNDLE_CHARACTERS
        ):
            if not self.allow_player_2_ood_character:
                raise ValueError(
                    "MIMIC P2 physical characters outside its native bundle family require "
                    "an explicit P2 OOD opt-in"
                )
            if self.player_2_checkpoint is None or self.player_2_assets is None:
                raise ValueError(
                    "MIMIC P2 OOD transfer requires explicit checkpoint and asset-directory paths"
                )
        if self.artifact_label is not None and (
            not self.artifact_label
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in self.artifact_label
            )
        ):
            raise ValueError(
                "artifact_label must contain only lowercase letters, digits, underscores, or hyphens"
            )

    def model_for_port(self, port: int) -> str:
        if port not in (1, 2):
            raise ValueError(f"match port must be 1 or 2, got {port}")
        return _model_name(self.player_1_model if port == 1 else self.player_2_model)

    def port_for(self, model_name: str) -> int:
        expected = _model_name(model_name)
        matches = [port for port in (1, 2) if self.model_for_port(port) == expected]
        if len(matches) != 1:
            raise ValueError(f"request does not contain exactly one {expected!r} side")
        return matches[0]

    def character_for_port(self, port: int) -> str:
        return self.player_1_character if port == 1 else self.player_2_character

    def checkpoint_for_port(self, port: int) -> Path | None:
        return self.player_1_checkpoint if port == 1 else self.player_2_checkpoint

    def assets_for_port(self, port: int) -> Path | None:
        return self.player_1_assets if port == 1 else self.player_2_assets

    def name_for_port(self, port: int) -> str | None:
        return self.player_1_name if port == 1 else self.player_2_name

    def temperature_for_port(self, port: int) -> float | None:
        return self.player_1_temperature if port == 1 else self.player_2_temperature

    def slippi_release_for_port(self, port: int) -> str:
        if port not in (1, 2):
            raise ValueError(f"match port must be 1 or 2, got {port}")
        return self.player_1_slippi_release if port == 1 else self.player_2_slippi_release

    def allow_ood_for_port(self, port: int) -> bool:
        if port not in (1, 2):
            raise ValueError(f"match port must be 1 or 2, got {port}")
        return self.allow_player_1_ood_character if port == 1 else self.allow_player_2_ood_character

    @property
    def windowed_model(self) -> str:
        return next(
            model
            for model in (self.model_for_port(1), self.model_for_port(2))
            if model not in _STREAMING_MODELS
        )

    @property
    def streaming_model(self) -> str:
        matches = [
            model for model in (self.model_for_port(1), self.model_for_port(2)) if model in _STREAMING_MODELS
        ]
        if len(matches) != 1:
            raise ValueError("request does not contain exactly one streaming policy side")
        return matches[0]


def _validate_config_contract(config: dict[str, Any], request: SlippiMatchRequest) -> dict[str, Any]:
    request.validate()
    slippi = config.get("slippi_ai")
    integration = config.get("slippi_integration")
    if not isinstance(slippi, dict) or not isinstance(integration, dict):
        raise ValueError("config must contain [slippi_ai] and [slippi_integration]")
    shared_required: dict[str, Any] = {
        "repository_url": SLIPPI_AI_REPOSITORY_URL,
        "source_revision": SLIPPI_AI_SOURCE_REVISION,
        "console_delay": MIXED_RUNTIME_CONSOLE_DELAY_FRAMES,
        "upstream_eval_two_console_delay": UPSTREAM_EVAL_TWO_CONSOLE_DELAY_FRAMES,
        "compile": True,
        "async_inference": True,
    }
    mismatches = {
        name: {"observed": slippi.get(name), "required": expected}
        for name, expected in shared_required.items()
        if slippi.get(name) != expected
    }
    slippi_ports = [port for port in (1, 2) if request.model_for_port(port) == "slippi-ai"]
    releases = {
        port: slippi_ai_release_contract(request.slippi_release_for_port(port))
        for port in slippi_ports
    }
    if any(release.key == "medium-v2" for release in releases.values()):
        medium_required = {
            "checkpoint_sha256": MEDIUM_V2_CHECKPOINT_SHA256,
            "checkpoint_byte_length": MEDIUM_V2_CHECKPOINT_BYTES,
            "checkpoint_delay": POLICY_DELAY_FRAMES,
            "effective_output_delay": MIXED_EFFECTIVE_POLICY_DELAY_FRAMES,
        }
        mismatches.update(
            {
                name: {"observed": slippi.get(name), "required": expected}
                for name, expected in medium_required.items()
                if slippi.get(name) != expected
            }
        )
    if mismatches:
        raise ValueError(f"Slippi-AI config contract mismatch: {mismatches}")
    if any(release.key == "medium-v2" for release in releases.values()) and tuple(
        slippi.get("allowed_characters", ())
    ) != MEDIUM_V2_SUPPORTED_CHARACTERS:
        raise ValueError("Slippi-AI character allowlist does not match the pinned checkpoint contract")
    if frozenset(MEDIUM_V2_SUPPORTED_STAGES) != LEGAL_STAGES:
        raise AssertionError("mixed-runtime stage allowlist differs from the pinned checkpoint contract")
    if integration.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Slippi-AI integration schema version mismatch")
    if integration.get("exact_mode_only") is not True:
        raise ValueError("Slippi-AI integration must be exact-mode-only")
    mode = request.inference_mode or str(config["integration"]["inference_mode"])
    if mode not in _EXACT_INFERENCE_MODES:
        raise ValueError(f"inference mode {mode!r} can drop recurrent frames and is forbidden")
    configured_console_delay = int(slippi["console_delay"])
    selected_release = releases[slippi_ports[0]]
    configured_effective_delay = selected_release.policy_delay_frames - configured_console_delay
    if configured_effective_delay < 0:
        raise AssertionError("selected policy delay precedes the mixed-runtime console delay")
    return {
        "exact_mode_only": True,
        "inference_mode": mode,
        "checkpoint_delay_frames": selected_release.policy_delay_frames,
        "mixed_runtime_console_delay_frames": configured_console_delay,
        "effective_policy_delay_frames": configured_effective_delay,
        "upstream_eval_two_console_delay_frames": UPSTREAM_EVAL_TWO_CONSOLE_DELAY_FRAMES,
        "release_contract": selected_release.as_dict(),
        "release_contracts_by_port": {
            f"p{port}": release.as_dict() for port, release in releases.items()
        },
    }


def _pinned_source_identity(
    source: Path,
    *,
    expected_repository: str,
    expected_revision: str,
) -> dict[str, Any]:
    """Fail closed unless a cached upstream tree has the exact clean identity."""
    identity = {
        "repository_url": _git_output(source, "remote", "get-url", "origin"),
        "revision": _git_output(source, "rev-parse", "HEAD"),
        "tracked_tree_clean": not bool(_git_output(source, "status", "--short", "--untracked-files=all")),
    }
    checks = {
        "repository_url_exact": identity["repository_url"] == expected_repository,
        "revision_exact": identity["revision"] == expected_revision,
        "tracked_tree_clean": identity["tracked_tree_clean"] is True,
    }
    identity["checks"] = checks
    if not all(checks.values()):
        raise RuntimeError(f"pinned source identity mismatch for {source}: {identity}")
    return identity


def _activate_windowed_source(config: dict[str, Any], project_root: Path, model_name: str) -> dict[str, Any]:
    """Activate the requested MIMIC source tree."""
    model = _model_name(model_name)
    if model == "mimic":
        source = (project_root / config["mimic"]["source_directory"]).resolve()
        revision = _git_output(source, "rev-parse", "HEAD")
        if revision != MIMIC_SOURCE_REVISION:
            raise RuntimeError(f"pinned MIMIC source mismatch: {revision} != {MIMIC_SOURCE_REVISION}")
        identity = _pinned_source_identity(
            source,
            expected_repository=MIMIC_SOURCE_REPOSITORY,
            expected_revision=MIMIC_SOURCE_REVISION,
        )
        source_text = str(source)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        return {
            "loaded_policy": "mimic",
            "source_directory": str(source),
            "source_revision": revision,
            "repositories": {"mimic": identity},
        }
    raise ValueError(f"unsupported windowed policy: {model}")


def _load_windowed_runtime(
    config: dict[str, Any], project_root: Path, request: SlippiMatchRequest
) -> MimicRuntime:
    model = request.windowed_model
    port = request.port_for(model)
    checkpoint = request.checkpoint_for_port(port)
    if model != "mimic":
        raise ValueError(f"unsupported windowed policy: {model}")
    assets = request.assets_for_port(port)
    if port == 2 and request.character_for_port(port) in MIMIC_NATIVE_BUNDLE_CHARACTERS:
        bundle = mimic_native_bundle(request.character_for_port(port))
        expected_assets = (
            project_root
            / str(bundle["asset_directory"])
            / str(bundle["name"])
        ).resolve()
        expected_checkpoint = expected_assets / "model.pt"
        if checkpoint is None and assets is None:
            if request.character_for_port(port) != str(config["mimic"]["character"]):
                checkpoint = expected_checkpoint
                assets = expected_assets
        else:
            selected_assets = (
                checkpoint.expanduser().resolve().parent
                if assets is None and checkpoint is not None
                else cast(Path, assets).expanduser().resolve()
            )
            selected_checkpoint = (
                selected_assets / "model.pt"
                if checkpoint is None
                else checkpoint.expanduser().resolve()
            )
            if selected_assets != expected_assets or selected_checkpoint != expected_checkpoint:
                raise ValueError(
                    f"MIMIC physical {request.character_for_port(port)} requires its exact "
                    f"native bundle at {expected_assets}"
                )
            checkpoint = selected_checkpoint
            assets = selected_assets
    elif port == 2 and request.allow_ood_for_port(port):
        if checkpoint is None or assets is None:
            raise ValueError(
                "MIMIC P2 OOD transfer requires explicit checkpoint and asset-directory paths"
            )
    if checkpoint is not None and assets is None:
        assets = checkpoint.expanduser().resolve().parent
    return load_mimic_runtime(
        config,
        project_root,
        checkpoint_override=checkpoint,
        asset_directory_override=assets,
    )


def _validate_character_contracts(
    request: SlippiMatchRequest,
    windowed_runtime: MimicRuntime | Any,
) -> dict[str, dict[str, Any]]:
    contracts: dict[str, dict[str, Any]] = {}
    for port in (1, 2):
        slot = f"p{port}"
        model = request.model_for_port(port)
        requested = request.character_for_port(port)
        if model == "slippi-ai":
            release = slippi_ai_release_contract(request.slippi_release_for_port(port))
            slippi_capabilities = slippi_ai_release_capabilities(release.key)
            covered_by_training = requested in release.supported_characters
            forced_ood = request.allow_ood_for_port(port) and not covered_by_training
            supported = covered_by_training or forced_ood
            contract = {
                "model": model,
                "release": release.key,
                "release_contract": release.as_dict(),
                "requested_character": requested,
                "supported_characters": list(release.supported_characters),
                "capability": "released-checkpoint multi-character control",
                "training_evidence": {
                    "kind": "released-checkpoint",
                    "scope": f"{release.key} declared character roster",
                    "requested_character_covered": covered_by_training,
                },
                "requested_character_covered_by_training": covered_by_training,
                "forced_ood_character_transfer": forced_ood,
                "ood_provenance": (
                    {
                        "opt_in": f"allow_player_{port}_ood_character",
                        "physical_port": port,
                        "checkpoint_roster_unchanged": True,
                        "semantics": (
                            f"native {release.key} inference on an out-of-training-roster fighter"
                        ),
                    }
                    if forced_ood
                    else None
                ),
                "supported_stages": sorted(LEGAL_STAGES),
                "checkpoint_capabilities": slippi_capabilities,
                "perspective_binding": {
                    "physical_port": port,
                    "checkpoint_p0": port,
                    "checkpoint_p1": 3 - port,
                },
                "evidence": f"pinned official {release.key} release contract",
                "match": supported,
            }
        elif model == "mimic":
            checkpoint_character = str(windowed_runtime.controlled_character)
            covered_by_training = requested == checkpoint_character
            forced_ood = (
                request.allow_ood_for_port(port)
                and not covered_by_training
                and requested not in MIMIC_NATIVE_BUNDLE_CHARACTERS
                and checkpoint_character in MIMIC_NATIVE_BUNDLE_CHARACTERS
                and request.checkpoint_for_port(port) is not None
                and request.assets_for_port(port) is not None
            )
            supported = covered_by_training or forced_ood
            contract = {
                "model": model,
                "requested_character": requested,
                "checkpoint_character": checkpoint_character,
                "supported_characters": [checkpoint_character],
                "capability": "released-checkpoint character-specific control",
                "training_evidence": {
                    "kind": "released-checkpoint",
                    "scope": f"MIMIC {checkpoint_character} checkpoint and matching asset bundle",
                    "requested_character_covered": covered_by_training,
                },
                "requested_character_covered_by_training": covered_by_training,
                "forced_ood_character_transfer": forced_ood,
                "ood_provenance": (
                    {
                        "opt_in": f"allow_player_{port}_ood_character",
                        "physical_port": port,
                        "checkpoint_character_unchanged": True,
                        "explicit_checkpoint_and_assets_supplied": True,
                        "physical_character_outside_native_bundle_family": True,
                        "semantics": (
                            f"native MIMIC {checkpoint_character} checkpoint inference while "
                            f"controlling physical {requested}"
                        ),
                    }
                    if forced_ood
                    else None
                ),
                "supported_stages": sorted(LEGAL_STAGES),
                "perspective_binding": {
                    "physical_port": port,
                    "native_builder": "build_frame" if port == 1 else "build_frame_p2",
                },
                "evidence": "MIMIC asset metadata.json melee_enum",
                "match": supported,
            }
        else:
            raise ValueError(f"unsupported policy: {model}")
        contracts[slot] = contract
        if not supported:
            if model == "slippi-ai":
                release = slippi_ai_release_contract(request.slippi_release_for_port(port))
                raise ValueError(
                    f"Slippi-AI does not support requested character {requested} with "
                    f"{release.key}; "
                    f"allowed: {list(release.supported_characters)}; use the physical-port OOD "
                    "flag only for an explicitly stratified transfer diagnostic"
                )
            if model == "mimic":
                raise ValueError(
                    f"MIMIC requires {contract['checkpoint_character']} but received {requested}; "
                    "use --allow-p2-ood-character only for an explicitly "
                    "stratified transfer diagnostic"
                )
    return contracts


def _enum_name(value: object) -> str:
    return str(getattr(value, "name", value)).split(".")[-1]


def _observed_match_context(gamestate: Any, request: SlippiMatchRequest) -> dict[str, Any]:
    """Record and compare the first real in-game stage and character values."""
    observed_stage = _enum_name(gamestate.stage)
    slots: dict[str, dict[str, Any]] = {}
    for port in (1, 2):
        player = gamestate.players.get(port)
        observed_character = None if player is None else _enum_name(player.character)
        observed_costume = None if player is None else int(player.costume)
        requested_character = request.character_for_port(port)
        slots[f"p{port}"] = {
            "port": port,
            "requested_character": requested_character,
            "observed_character": observed_character,
            "observed_costume": observed_costume,
            "match": observed_character == requested_character,
            "evidence": "first libmelee IN_GAME GameState",
        }
    observed_ports = sorted(int(port) for port in gamestate.players)
    checks = {
        "two_standard_ports_exact": observed_ports == [1, 2],
        "stage_exact": observed_stage == request.stage,
        "both_characters_exact": all(bool(slot["match"]) for slot in slots.values()),
    }
    return {
        "requested_stage": request.stage,
        "observed_stage": observed_stage,
        "observed_ports": observed_ports,
        "stage_match": checks["stage_exact"],
        "slots": slots,
        "evidence": "first libmelee IN_GAME GameState",
        "checks": checks,
        "match": all(checks.values()),
    }


def _slippi_policy_config(
    config: dict[str, Any],
    project_root: Path,
    request: SlippiMatchRequest,
    *,
    port_override: int | None = None,
    console_delay_override: int | None = None,
) -> SlippiAIPolicyConfig:
    slippi = config["slippi_ai"]
    port = request.port_for("slippi-ai") if port_override is None else port_override
    if request.model_for_port(port) != "slippi-ai":
        raise ValueError(f"port {port} is not assigned to Slippi-AI")
    release = slippi_ai_release_contract(request.slippi_release_for_port(port))
    checkpoint = request.checkpoint_for_port(port)
    checkpoint_path = (
        (
            (project_root / slippi["checkpoint"]).resolve()
            if release.key == "medium-v2"
            else release.checkpoint_path(project_root)
        )
        if checkpoint is None
        else checkpoint.expanduser().resolve()
    )
    requested_temperature = request.temperature_for_port(port)
    sample_temperature = (
        float(slippi.get("sample_temperature", SAMPLE_TEMPERATURE))
        if requested_temperature is None
        else requested_temperature
    )
    policy_config = SlippiAIPolicyConfig(
        source_directory=(project_root / slippi["source_directory"]).resolve(),
        checkpoint_path=checkpoint_path,
        port=port,
        opponent_port=3 - port,
        release=release.key,
        name=request.name_for_port(port) or str(slippi.get("default_name", DEFAULT_PLAYER_NAME)),
        sample_temperature=sample_temperature,
        policy_delay_frames=release.policy_delay_frames,
        console_delay_frames=(
            int(slippi["console_delay"]) if console_delay_override is None else console_delay_override
        ),
        async_inference=bool(slippi["async_inference"]),
        compile=bool(slippi["compile"]),
        tf_jit_compile=False,
        batch_steps=0,
        mirror=False,
        requested_character=request.character_for_port(port),
        allow_ood_character=request.allow_ood_for_port(port),
    )
    policy_config.validate()
    return policy_config


def _windowed_identity(
    model: str,
    runtime: MimicRuntime,
    source_checks: dict[str, Any],
) -> dict[str, Any]:
    if model == "mimic":
        mimic = cast(MimicRuntime, runtime)
        return {
            "source": source_checks,
            "checkpoint": {
                "path": str(mimic.checkpoint_path),
                "sha256": mimic.checkpoint_sha256,
                "byte_length": mimic.checkpoint_path.stat().st_size,
            },
            "asset_directory": str(mimic.asset_directory),
            "bundle_identity": mimic.bundle_identity,
            "controlled_character": mimic.controlled_character,
            "sequence_length": int(mimic.model_config.max_seq_len),
            "decoder": {
                "implementation": "released MIMIC tools.inference_utils.decode_and_press",
                "strategy": MIMIC_DECODE_STRATEGY,
            },
            "runtime_fidelity": {
                "classification": "native pinned inference path",
                "observation": "released build_frame/build_frame_p2 selected by physical port",
                "previous_controller": "released frame-builder encoding of the executed prior command",
                "decoder": "released decode_and_press at its declared sampling settings",
                "controller_transport": (
                    "released decoder flush request retained semantically; the project wrapper "
                    "neutralizes the command's declared analog R axis, then commits both ports "
                    "as one ordered pipe transaction"
                ),
            },
        }
    raise ValueError(f"unsupported windowed policy: {model}")


def _make_windowed_policy(
    model: str,
    runtime: MimicRuntime,
    port: int,
    *,
    online_delay_frames: int,
    evaluation_seed: int = 42,
) -> MimicLivePolicy:
    """Bind one native live policy to its physical port and timing contract."""
    if model == "mimic":
        return MimicLivePolicy(
            cast(MimicRuntime, runtime),
            port,
            online_delay_frames=online_delay_frames,
            evaluation_seed=evaluation_seed,
        )
    raise ValueError(f"unsupported windowed policy: {model}")


def _observe_windowed_policy(
    model: str,
    policy: MimicLivePolicy,
    gamestate: Any,
) -> tuple[bool, None]:
    """Observe the current frame through the native MIMIC policy."""
    if model != "mimic":
        raise ValueError(f"unsupported windowed policy: {model}")
    return policy.observe(gamestate), None





def _windowed_exact_timing_checks(
    model: str,
    *,
    processed_frames: int,
    inference_count: int,
    command_ages: list[int],
) -> dict[str, bool]:
    """Require one exact current-frame inference whenever the native policy is ready."""
    if model != "mimic":
        raise ValueError(f"unsupported windowed policy: {model}")
    expected = processed_frames
    return {
        "windowed_exact_inference_count": inference_count == expected,
        "windowed_exact_command_age_zero": len(command_ages) == expected
        and all(age == 0 for age in command_ages),
    }


def _dispatch_slippi_command(controller: Any, command: Any) -> dict[str, Any]:
    """Queue the native decoded command for the next Console.step() flush."""
    dispatch = send_canonical_controller(controller, command, flush=False).as_dict()
    dispatch["called"] = True
    return dispatch


def _player_state(player: Any) -> dict[str, Any]:
    return {
        "character": _enum_name(player.character),
        "action": int(player.action.value),
        "stocks": int(player.stock),
        "percent": float(player.percent),
        "position": [float(player.position.x), float(player.position.y)],
    }


def _build_trace_row(
    game_frame: int,
    request: SlippiMatchRequest,
    slot_payloads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if set(slot_payloads) != {"p1", "p2"}:
        raise ValueError("trace payloads must be keyed exactly by p1 and p2")
    slots: dict[str, dict[str, Any]] = {}
    for port in (1, 2):
        slot = f"p{port}"
        payload = dict(slot_payloads[slot])
        if "state" in payload:
            if "player_state" in payload:
                raise ValueError(f"{slot} trace payload supplies both state field names")
            payload["player_state"] = payload.pop("state")
        payload.update(
            {
                "port": port,
                "model": request.model_for_port(port),
                "requested_character": request.character_for_port(port),
            }
        )
        slots[slot] = payload
    return {"schema_version": SCHEMA_VERSION, "game_frame": game_frame, "slots": slots}


def _button_name(value: object) -> str:
    name = str(getattr(value, "name", value)).split(".")[-1]
    return name.removeprefix("BUTTON_")


def _pair(value: object, field_name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field_name} must contain exactly two axes")
    return float(value[0]), float(value[1])


def _normalize_trace_command(model_name: str, command: Mapping[str, Any]) -> dict[str, Any]:
    """Project all three policy command schemas onto one controller boundary."""
    model = _model_name(model_name)
    if model == "mimic":
        main_stick = (float(command["main_x"]), float(command["main_y"]))
        c_stick = (float(command["c_x"]), float(command["c_y"]))
        analog_l = float(command["l_shldr"])
        analog_r = float(command["r_shldr"])
        buttons = tuple(
            sorted(
                _button_name(key.removeprefix("btn_"))
                for key, value in command.items()
                if key.startswith("btn_BUTTON_") and bool(int(float(value)))
            )
        )
    elif model in ("slippi-ai", "frisson-ai"):
        main_stick = _pair(command["main_stick"], "main_stick")
        c_stick = _pair(command["c_stick"], "c_stick")
        analog_l = float(command.get("analog_l", command.get("shoulder", 0.0)))
        analog_r = float(command.get("analog_r", 0.0))
        supplied = command.get("buttons", ())
        if not isinstance(supplied, (list, tuple)):
            raise ValueError("buttons must be a list or tuple")
        buttons = tuple(sorted(name for value in supplied if (name := _button_name(value)) != "NO_BUTTON"))
    else:
        raise ValueError(f"unsupported trace command model: {model}")
    return {
        "main_stick": main_stick,
        "c_stick": c_stick,
        "analog_l": analog_l,
        "analog_r": analog_r,
        "buttons": buttons,
    }


def _expected_raw_main_axis(value: float) -> int:
    """Return libmelee's intended raw axis before its +0.1 pipe-input fudge."""
    return round((value - 0.5) * 160)


def _expected_processed_c_stick(
    model_name: str,
    trace_slot: Mapping[str, Any],
    intended_c: tuple[float, float],
) -> tuple[float, float]:
    """Project a canonical C-stick command through Melee's input processing."""
    inference = trace_slot.get("inference")
    if (
        _model_name(model_name) == "slippi-ai"
        and isinstance(inference, dict)
        and inference.get("native_dummy_prefix") is True
    ):
        return (-0.7, -0.7)

    raw_x, raw_y = (_expected_raw_main_axis(axis) for axis in intended_c)
    magnitude = math.hypot(raw_x, raw_y)
    if magnitude > 80.0:
        scale = 80.0 / magnitude
        # Melee truncates the radially clamped coordinates toward zero before
        # Slippi records each processed axis in units of 1 / 80.
        raw_x = math.trunc(raw_x * scale)
        raw_y = math.trunc(raw_y * scale)
    # The pinned Slippi-AI controller contract treats each nonzero raw axis
    # below 23 as deadzone input. Slippi's processed pre-frame C-stick columns
    # therefore record that component as zero even when the other component
    # remains outside the deadzone.
    raw_x = 0 if 0 < abs(raw_x) < RAW_AXIS_DEADZONE else raw_x
    raw_y = 0 if 0 < abs(raw_y) < RAW_AXIS_DEADZONE else raw_y
    return raw_x / 80.0, raw_y / 80.0


def _expected_processed_buttons(buttons: tuple[str, ...]) -> tuple[str, ...]:
    """Apply Melee's observable Z-to-A+Z processed-button expansion."""
    expected = set(buttons)
    if "Z" in expected:
        expected.add("A")
    return tuple(sorted(expected))


def _component_statistics(
    samples: Iterable[tuple[int, int, str, int | float, int | float]],
) -> dict[str, Any]:
    total = 0
    exact = 0
    mismatch_occurrences: Counter[tuple[str, int | float, int | float]] = Counter()
    for _trace_frame, _replay_frame, component, expected, observed in samples:
        total += 1
        if expected == observed:
            exact += 1
        else:
            mismatch_occurrences[(component, expected, observed)] += 1
    categories = [
        {
            "component": component,
            "expected": expected,
            "observed": observed,
            "count": count,
        }
        for (component, expected, observed), count in sorted(
            mismatch_occurrences.items(), key=lambda item: repr(item[0])
        )
    ]
    return {
        "components_compared": total,
        "exact_components": exact,
        "mismatch_components": total - exact,
        "mismatch_categories": categories,
    }


def _tolerant_component_statistics(
    samples: Iterable[tuple[int, int, str, float, float]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    total = 0
    within_tolerance = 0
    mismatches: list[dict[str, Any]] = []
    for trace_frame, replay_frame, component, expected, observed in samples:
        total += 1
        absolute_error = abs(expected - observed)
        if absolute_error <= tolerance:
            within_tolerance += 1
        else:
            mismatches.append(
                {
                    "trace_frame": trace_frame,
                    "replay_frame": replay_frame,
                    "component": component,
                    "expected": expected,
                    "observed": observed,
                    "absolute_error": absolute_error,
                }
            )
    return {
        "components_compared": total,
        "components_within_tolerance": within_tolerance,
        "mismatch_components": total - within_tolerance,
        "tolerance": tolerance,
        "mismatches": mismatches,
    }


def _normalize_trace_row(row: dict[str, Any], *, line_number: int) -> dict[str, Any]:
    """Validate current two-slot match traces."""
    frame = int(row["game_frame"])
    raw_slots = row.get("slots")
    if not isinstance(raw_slots, dict) or set(raw_slots) != {"p1", "p2"}:
        raise ValueError(f"trace line {line_number}, frame {frame} is not keyed exactly by p1 and p2")
    return row


def _read_trace_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_frames: set[int] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"trace line {line_number} is not an object")
            row = _normalize_trace_row(cast(dict[str, Any], value), line_number=line_number)
            frame = int(row["game_frame"])
            if frame in seen_frames:
                raise ValueError(f"trace contains duplicate game frame {frame}")
            seen_frames.add(frame)
            rows.append(row)
    return rows


def _read_replay_controller_states(path: Path) -> dict[int, dict[int, dict[str, Any]]]:
    import melee
    import peppi_py

    game = peppi_py.read_slippi(str(path))
    peppi_frame_ids = [int(value) for value in game.frames.id.to_pylist()]
    peppi_analogs: dict[int, dict[int, tuple[float, float, float, float]]] = {}
    for index, start_player in enumerate(game.start.players):
        if start_player is None:
            continue
        port = int(start_player.port.value) + 1
        pre = game.frames.ports[index].leader.pre
        processed_c_x = pre.cstick.x.to_pylist()
        processed_c_y = pre.cstick.y.to_pylist()
        physical_l = pre.triggers_physical.l.to_pylist()
        physical_r = pre.triggers_physical.r.to_pylist()
        if any(
            len(column) != len(peppi_frame_ids)
            for column in (processed_c_x, processed_c_y, physical_l, physical_r)
        ):
            raise ValueError(f"Peppi controller columns have inconsistent length for port {port}")
        for frame, c_x, c_y, analog_l, analog_r in zip(
            peppi_frame_ids,
            processed_c_x,
            processed_c_y,
            physical_l,
            physical_r,
            strict=True,
        ):
            if c_x is None or c_y is None or analog_l is None or analog_r is None:
                raise ValueError(f"Peppi controller analogs are unavailable at frame {frame}, port {port}")
            peppi_analogs.setdefault(frame, {})[port] = (
                float(c_x),
                float(c_y),
                float(analog_l),
                float(analog_r),
            )

    console = melee.Console(path=str(path), is_dolphin=False, allow_old_version=True)
    states: dict[int, dict[int, dict[str, Any]]] = {}
    try:
        if not console.connect():
            raise RuntimeError("libmelee did not connect to saved replay for boundary audit")
        while (gamestate := console.step()) is not None:
            frame = int(gamestate.frame)
            slots: dict[int, dict[str, Any]] = {}
            for port in (1, 2):
                player = gamestate.players.get(port)
                if player is None:
                    continue
                controller = player.controller_state
                peppi_controller_analogs = peppi_analogs.get(frame, {}).get(port)
                if peppi_controller_analogs is None:
                    raise ValueError(
                        f"Peppi controller analogs are unavailable at frame {frame}, port {port}"
                    )
                slots[port] = {
                    "buttons_physical": tuple(
                        sorted(
                            _button_name(button) for button, pressed in controller.button.items() if pressed
                        )
                    ),
                    "buttons_processed": tuple(
                        sorted(
                            _button_name(button)
                            for button, pressed in controller.processed_button.items()
                            if pressed
                        )
                    ),
                    "raw_main_stick": tuple(int(value) for value in controller.raw_main_stick),
                    "main_stick": tuple(float(value) for value in controller.main_stick),
                    "c_stick": peppi_controller_analogs[:2],
                    "libmelee_c_stick": tuple(float(value) for value in controller.c_stick),
                    "analog_l": float(controller.l_shoulder),
                    "physical_analog_l": peppi_controller_analogs[2],
                    "physical_analog_r": peppi_controller_analogs[3],
                }
            states[frame] = slots
    finally:
        with suppress(AssertionError):
            console.stop()
    return states


def _game_start_transport_proof(transport: object) -> dict[str, Any]:
    """Prove the one paired internal commit that changes startup alignment."""

    record = transport if isinstance(transport, Mapping) else {}
    frame_sync_value = record.get("frame_sync")
    frame_sync = frame_sync_value if isinstance(frame_sync_value, Mapping) else {}
    kinds_value = frame_sync.get("kind_commits")
    kinds = kinds_value if isinstance(kinds_value, Mapping) else {}
    reasons_value = record.get("group_commit_reasons")
    reasons = reasons_value if isinstance(reasons_value, Mapping) else {}
    ports_value = record.get("ports")
    ports = ports_value if isinstance(ports_value, Mapping) else {}
    p1_value = ports.get("p1")
    p1 = p1_value if isinstance(p1_value, Mapping) else {}
    p2_value = ports.get("p2")
    p2 = p2_value if isinstance(p2_value, Mapping) else {}
    scope_value = record.get("benchmark_scope")
    scope = scope_value if isinstance(scope_value, Mapping) else {}
    checks = {
        "transport_schema_v7": record.get("schema_version")
        == "integration.controller_pipe_lockstep.v7",
        "exactly_one_internal_group_commit": kinds.get("internal") == 1,
        "exactly_one_console_internal_reason": reasons.get("console-internal") == 1,
        "p1_exactly_one_later_internal_flush": p1.get("later_internal_flush_requests") == 1,
        "p2_exactly_one_later_internal_flush": p2.get("later_internal_flush_requests") == 1,
        "both_internal_ports_clear": record.get("pending_internal_ports") == [],
        "both_unscoped_ports_clear": record.get("pending_unscoped_ports") == [],
        "sealed_before_shutdown_drain": scope.get("sealed_before_shutdown_drain") is True,
    }
    return {
        "decision": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "frame_sync_internal_commits": kinds.get("internal"),
        "console_internal_group_commits": reasons.get("console-internal"),
        "later_internal_flush_requests": {
            "p1": p1.get("later_internal_flush_requests"),
            "p2": p2.get("later_internal_flush_requests"),
        },
    }


def _piecewise_game_start_replay_frames(
    trace_by_frame: Mapping[int, Mapping[str, Any]],
    replay_states: Mapping[int, Mapping[int, Mapping[str, Any]]],
    *,
    lag_frames: int,
    transport_proof: Mapping[str, Any],
    first_policy_replay_frame: int | None = None,
    first_policy_trace_unobservable: bool = False,
    fixed_normal_boundary: bool = False,
    neutral_overwrite_trace_frame: int | None = None,
) -> tuple[dict[int, int], list[int]]:
    """Build the exact startup-to-replay mapping around libmelee GAME_START.

    libmelee 0.47.3 sends ``release_all(); flush()`` for both registered
    controllers when it parses GAME_START. Complete two-port evidence can prove
    a fixed one-frame boundary, a piecewise boundary, or an unobservable first
    command. One transport-proved paired neutral overwrite is also permitted
    during the bounded ENTRY startup window. Callers provide the evidence-selected
    classification.
    """

    if lag_frames != NEXT_FRAME_CONTROLLER_BOUNDARY_FRAMES:
        raise ValueError("piecewise GAME_START proof requires the zero-online-delay boundary")
    required_trace_frames = set(
        range(FIRST_POLICY_FRAME, GAME_START_CONTROLLER_LATCH_TRACE_FRAME + 2)
    )
    if min(trace_by_frame) != FIRST_POLICY_FRAME or not required_trace_frames <= set(trace_by_frame):
        raise ValueError("piecewise GAME_START proof lacks the pinned -123..-118 trace window")
    proof_checks = transport_proof.get("checks")
    if (
        transport_proof.get("decision") != "pass"
        or not isinstance(proof_checks, Mapping)
        or not proof_checks
        or not all(value is True for value in proof_checks.values())
    ):
        raise ValueError("piecewise GAME_START proof lacks one exact paired internal commit")

    failures: list[str] = []
    for frame in range(FIRST_POLICY_FRAME, GAME_START_CONTROLLER_LATCH_TRACE_FRAME + 1):
        slots = trace_by_frame[frame].get("slots")
        if not isinstance(slots, Mapping) or set(slots) != {"p1", "p2"}:
            failures.append(f"trace-{frame}.two-slots")
            continue
        for port in (1, 2):
            slot = slots.get(f"p{port}")
            if not isinstance(slot, Mapping):
                failures.append(f"trace-{frame}.p{port}.slot")
                continue
            player = slot.get("player_state")
            dispatch = slot.get("controller_dispatch")
            if not isinstance(player, Mapping):
                failures.append(f"trace-{frame}.p{port}.player-state")
            else:
                if player.get("action") != 322:
                    failures.append(f"trace-{frame}.p{port}.entry-action")
                if player.get("stocks") != 4:
                    failures.append(f"trace-{frame}.p{port}.four-stocks")
                percent = player.get("percent")
                if (
                    isinstance(percent, bool)
                    or not isinstance(percent, (int, float))
                    or percent != 0.0
                ):
                    failures.append(f"trace-{frame}.p{port}.zero-percent")
            if not isinstance(dispatch, Mapping) or dispatch.get("called") is not True:
                failures.append(f"trace-{frame}.p{port}.dispatch-called")
    if failures:
        raise ValueError(
            "piecewise GAME_START proof failed its exact startup signature: "
            f"{failures}"
        )

    early_replay_frame = FIRST_POLICY_FRAME + lag_frames
    delayed_replay_frame = early_replay_frame + 1
    if first_policy_trace_unobservable and first_policy_replay_frame is not None:
        raise ValueError(
            "an unobservable first policy trace frame cannot select a replay frame"
        )
    if fixed_normal_boundary and first_policy_trace_unobservable:
        raise ValueError("a fixed normal boundary cannot hide the first policy trace frame")
    if neutral_overwrite_trace_frame is not None and not fixed_normal_boundary:
        raise ValueError("a neutral GAME_START overwrite requires the fixed normal boundary")
    selected_first_replay_frame = None
    if not first_policy_trace_unobservable:
        selected_first_replay_frame = (
            delayed_replay_frame
            if first_policy_replay_frame is None
            else int(first_policy_replay_frame)
        )
    if (
        not first_policy_trace_unobservable
        and selected_first_replay_frame not in {early_replay_frame, delayed_replay_frame}
    ):
        raise ValueError(
            "piecewise GAME_START first policy replay frame must be one of "
            f"{[early_replay_frame, delayed_replay_frame]}, got {selected_first_replay_frame}"
        )

    mapping: dict[int, int] = {}
    for frame in sorted(trace_by_frame):
        if fixed_normal_boundary:
            replay_frame = frame + lag_frames
        elif first_policy_trace_unobservable:
            if frame == FIRST_POLICY_FRAME:
                continue
            replay_frame = frame + lag_frames
        elif frame == FIRST_POLICY_FRAME:
            assert selected_first_replay_frame is not None
            replay_frame = selected_first_replay_frame
        elif FIRST_POLICY_FRAME < frame < GAME_START_CONTROLLER_LATCH_TRACE_FRAME:
            replay_frame = frame + lag_frames + 1
        elif frame == GAME_START_CONTROLLER_LATCH_TRACE_FRAME:
            continue
        else:
            replay_frame = frame + lag_frames
        replay = replay_states.get(replay_frame)
        if replay is not None and all(port in replay for port in (1, 2)):
            mapping[frame] = replay_frame
    game_start_unobservable = (
        [neutral_overwrite_trace_frame]
        if neutral_overwrite_trace_frame is not None
        else []
        if first_policy_trace_unobservable or fixed_normal_boundary
        else [GAME_START_CONTROLLER_LATCH_TRACE_FRAME]
    )
    return mapping, game_start_unobservable


def _replay_controller_state_is_exact_neutral(observed: object) -> bool:
    """Require neutral values across every replay dimension used as a hard gate."""

    return bool(
        isinstance(observed, Mapping)
        and observed.get("buttons_physical") == ()
        and observed.get("buttons_processed") == ()
        and observed.get("raw_main_stick") == (0, 0)
        and observed.get("c_stick") == (0.0, 0.0)
        and observed.get("physical_analog_l") == 0.0
        and observed.get("physical_analog_r") == 0.0
    )


def _normal_next_boundary_two_port_evidence(
    trace_by_frame: Mapping[int, Mapping[str, Any]],
    replay_states: Mapping[int, Mapping[int, Mapping[str, Any]]],
    *,
    lag_frames: int,
    include_first_policy_frame: bool = False,
) -> dict[str, Any]:
    """Prove the requested observable range at the normal two-port boundary."""

    trace_frames = sorted(trace_by_frame)
    replay_frames = sorted(replay_states)
    trace_consecutive = trace_frames == list(range(trace_frames[0], trace_frames[-1] + 1))
    candidate_trace_frames = (
        trace_frames if include_first_policy_frame else trace_frames[1:]
    )
    terminal_trace_frames = [
        frame
        for frame in candidate_trace_frames
        if frame + lag_frames > replay_frames[-1]
    ]
    terminal_tail_exact = (
        len(terminal_trace_frames) <= lag_frames
        and terminal_trace_frames
        == trace_frames[len(trace_frames) - len(terminal_trace_frames) :]
    )
    observable_trace_frames = [
        frame for frame in candidate_trace_frames if frame not in terminal_trace_frames
    ]
    structural_mismatches: list[dict[str, Any]] = []
    command_mismatches: list[dict[str, Any]] = []
    exact_pairs_by_port = {"p1": 0, "p2": 0}
    for trace_frame in observable_trace_frames:
        replay_frame = trace_frame + lag_frames
        trace_slots = trace_by_frame[trace_frame].get("slots")
        replay_slots = replay_states.get(replay_frame)
        if (
            not isinstance(trace_slots, Mapping)
            or set(trace_slots) != {"p1", "p2"}
            or not isinstance(replay_slots, Mapping)
            or set(replay_slots) != {1, 2}
        ):
            structural_mismatches.append(
                {
                    "trace_frame": trace_frame,
                    "replay_frame": replay_frame,
                    "trace_slots": (
                        sorted(str(slot) for slot in trace_slots)
                        if isinstance(trace_slots, Mapping)
                        else None
                    ),
                    "replay_ports": (
                        sorted(str(port) for port in replay_slots)
                        if isinstance(replay_slots, Mapping)
                        else None
                    ),
                }
            )
            continue
        for port in (1, 2):
            slot_name = f"p{port}"
            trace_slot = trace_slots.get(slot_name)
            observed = replay_slots.get(port)
            dispatch = trace_slot.get("controller_dispatch") if isinstance(trace_slot, Mapping) else None
            dispatch_exact = isinstance(dispatch, Mapping) and dispatch.get("called") is True
            try:
                command_exact = bool(
                    isinstance(trace_slot, Mapping)
                    and isinstance(observed, Mapping)
                    and _trace_command_matches_replay(trace_slot, observed)
                )
            except (KeyError, TypeError, ValueError):
                command_exact = False
            if dispatch_exact and command_exact:
                exact_pairs_by_port[slot_name] += 1
            else:
                command_mismatches.append(
                    {
                        "trace_frame": trace_frame,
                        "replay_frame": replay_frame,
                        "port": port,
                        "dispatch_called": (
                            dispatch.get("called") if isinstance(dispatch, Mapping) else None
                        ),
                        "all_hard_gated_dimensions_exact": command_exact,
                    }
                )

    pair_count = len(observable_trace_frames)
    checks = {
        "trace_frames_consecutive": trace_consecutive,
        "terminal_unobservable_tail_bounded_by_lag": terminal_tail_exact,
        "at_least_one_observable_subsequent_pair": pair_count > 0,
        "exactly_two_trace_and_replay_ports_per_pair": not structural_mismatches,
        "both_ports_dispatched_and_all_dimensions_exact": (
            not command_mismatches
            and all(count == pair_count for count in exact_pairs_by_port.values())
        ),
    }
    return {
        "decision": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "equation": "replay_frame = trace_frame + 1",
        "includes_first_policy_frame": include_first_policy_frame,
        "evidence_ports": [1, 2],
        "trace_frame_range": (
            [observable_trace_frames[0], observable_trace_frames[-1]]
            if observable_trace_frames
            else None
        ),
        "first_pair": (
            [observable_trace_frames[0], observable_trace_frames[0] + lag_frames]
            if observable_trace_frames
            else None
        ),
        "last_pair": (
            [observable_trace_frames[-1], observable_trace_frames[-1] + lag_frames]
            if observable_trace_frames
            else None
        ),
        "pair_count": pair_count,
        "exact_pairs_by_port": exact_pairs_by_port,
        "terminal_unobservable_trace_frames": terminal_trace_frames,
        "structural_mismatch_count": len(structural_mismatches),
        "command_mismatch_count": len(command_mismatches),
        "structural_mismatches": structural_mismatches[:8],
        "command_mismatches": command_mismatches[:8],
    }


def _single_game_start_neutral_overwrite_evidence(
    trace_by_frame: Mapping[int, Mapping[str, Any]],
    replay_states: Mapping[int, Mapping[int, Mapping[str, Any]]],
    normal_boundary_evidence: Mapping[str, Any],
    transport_proof: Mapping[str, Any],
    *,
    lag_frames: int,
) -> dict[str, Any]:
    """Prove one paired neutral GAME_START transaction inside a fixed-lag stream."""

    mismatches_value = normal_boundary_evidence.get("command_mismatches")
    mismatches = (
        list(mismatches_value)
        if isinstance(mismatches_value, list)
        else []
    )
    mismatch_frames = sorted(
        {
            int(item["trace_frame"])
            for item in mismatches
            if isinstance(item, Mapping) and isinstance(item.get("trace_frame"), int)
        }
    )
    mismatch_ports = [
        int(item["port"])
        for item in mismatches
        if isinstance(item, Mapping) and isinstance(item.get("port"), int)
    ]
    trace_frame = mismatch_frames[0] if len(mismatch_frames) == 1 else None
    replay_frame = trace_frame + lag_frames if trace_frame is not None else None
    trace_row = trace_by_frame.get(trace_frame) if trace_frame is not None else None
    next_row = trace_by_frame.get(trace_frame + 1) if trace_frame is not None else None
    trace_slots = trace_row.get("slots") if isinstance(trace_row, Mapping) else None
    next_slots = next_row.get("slots") if isinstance(next_row, Mapping) else None
    replay_slots = replay_states.get(replay_frame) if replay_frame is not None else None
    next_replay_slots = (
        replay_states.get(replay_frame + 1) if replay_frame is not None else None
    )

    transport_checks = transport_proof.get("checks")
    transport_exact = bool(
        transport_proof.get("decision") == "pass"
        and isinstance(transport_checks, Mapping)
        and transport_checks
        and all(value is True for value in transport_checks.values())
    )
    normal_checks_value = normal_boundary_evidence.get("checks")
    normal_checks = (
        normal_checks_value if isinstance(normal_checks_value, Mapping) else {}
    )
    normal_structure_exact = bool(
        normal_checks
        and normal_checks.get("trace_frames_consecutive") is True
        and normal_checks.get("terminal_unobservable_tail_bounded_by_lag") is True
        and normal_checks.get("at_least_one_observable_subsequent_pair") is True
        and normal_checks.get("exactly_two_trace_and_replay_ports_per_pair") is True
        and normal_boundary_evidence.get("structural_mismatch_count") == 0
    )
    one_two_port_boundary = bool(
        normal_boundary_evidence.get("command_mismatch_count") == len(mismatches)
        and 1 <= len(mismatches) <= 2
        and len(mismatch_frames) == 1
        and len(set(mismatch_ports)) == len(mismatch_ports)
        and set(mismatch_ports) <= {1, 2}
    )
    startup_window = bool(
        trace_frame is not None
        and FIRST_POLICY_FRAME <= trace_frame <= GAME_START_CONTROLLER_LATCH_LATEST_TRACE_FRAME
    )
    needs_dynamic_overwrite_mapping = bool(
        trace_frame is not None
        and trace_frame != GAME_START_CONTROLLER_LATCH_TRACE_FRAME
    )
    exact_two_port_records = bool(
        isinstance(trace_slots, Mapping)
        and set(trace_slots) == {"p1", "p2"}
        and isinstance(next_slots, Mapping)
        and set(next_slots) == {"p1", "p2"}
        and isinstance(replay_slots, Mapping)
        and set(replay_slots) == {1, 2}
        and isinstance(next_replay_slots, Mapping)
        and set(next_replay_slots) == {1, 2}
    )
    startup_states_exact = exact_two_port_records
    dispatches_exact = exact_two_port_records
    replay_boundary_exact_neutral = exact_two_port_records
    next_boundary_restored = exact_two_port_records
    if exact_two_port_records:
        assert isinstance(trace_slots, Mapping)
        assert isinstance(next_slots, Mapping)
        assert isinstance(replay_slots, Mapping)
        assert isinstance(next_replay_slots, Mapping)
        for port in (1, 2):
            for slots in (trace_slots, next_slots):
                slot = slots[f"p{port}"]
                if not isinstance(slot, Mapping):
                    startup_states_exact = False
                    dispatches_exact = False
                    next_boundary_restored = False
                    continue
                player = slot.get("player_state")
                dispatch = slot.get("controller_dispatch")
                startup_states_exact = bool(
                    startup_states_exact
                    and isinstance(player, Mapping)
                    and player.get("action") in GAME_START_PLAYER_ACTIONS
                    and player.get("stocks") == 4
                    and player.get("percent") == 0.0
                )
                dispatches_exact = bool(
                    dispatches_exact
                    and isinstance(dispatch, Mapping)
                    and dispatch.get("called") is True
                )
            replay_boundary_exact_neutral = bool(
                replay_boundary_exact_neutral
                and _replay_controller_state_is_exact_neutral(replay_slots[port])
            )
            next_slot = next_slots[f"p{port}"]
            try:
                restored = bool(
                    isinstance(next_slot, Mapping)
                    and _trace_command_matches_replay(
                        next_slot,
                        next_replay_slots[port],
                    )
                )
            except (KeyError, TypeError, ValueError):
                restored = False
            next_boundary_restored = bool(
                next_boundary_restored
                and restored
            )

    pair_count = normal_boundary_evidence.get("pair_count")
    exact_pairs_value = normal_boundary_evidence.get("exact_pairs_by_port")
    exact_pairs = exact_pairs_value if isinstance(exact_pairs_value, Mapping) else {}
    all_other_boundaries_exact = bool(
        isinstance(pair_count, int)
        and all(
            exact_pairs.get(f"p{port}")
            == pair_count - (1 if port in mismatch_ports else 0)
            for port in (1, 2)
        )
    )
    checks = {
        "transport_proves_one_paired_internal_commit": transport_exact,
        "normal_boundary_structure_exact": normal_structure_exact,
        "exactly_one_mismatched_two_port_boundary": one_two_port_boundary,
        "mismatch_inside_entry_startup_window": startup_window,
        "mismatch_requires_dynamic_game_start_mapping": needs_dynamic_overwrite_mapping,
        "exact_two_port_current_and_next_records": exact_two_port_records,
        "both_players_remain_in_entry_startup": startup_states_exact,
        "both_ports_dispatched_current_and_next": dispatches_exact,
        "overwritten_replay_boundary_exactly_neutral_on_both_ports": (
            replay_boundary_exact_neutral
        ),
        "next_fixed_lag_boundary_restored_on_both_ports": next_boundary_restored,
        "every_other_fixed_lag_boundary_exact": all_other_boundaries_exact,
    }
    return {
        "decision": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "trace_frame": trace_frame,
        "replay_frame": replay_frame,
        "mismatched_ports": sorted(mismatch_ports),
        "command_mismatches": mismatches,
    }


def _first_policy_frame_mapping_evidence(
    trace_by_frame: Mapping[int, Mapping[str, Any]],
    replay_states: Mapping[int, Mapping[int, Mapping[str, Any]]],
    *,
    lag_frames: int,
    transport_proof: Mapping[str, Any],
) -> dict[str, Any]:
    """Select the first startup boundary from exact two-slot replay evidence."""

    trace_row = trace_by_frame.get(FIRST_POLICY_FRAME)
    if not isinstance(trace_row, Mapping):
        raise ValueError("piecewise GAME_START mapping lacks trace frame -123")
    slots = trace_row.get("slots")
    if not isinstance(slots, Mapping) or set(slots) != {"p1", "p2"}:
        raise ValueError("piecewise GAME_START trace frame -123 lacks exactly two slots")

    candidate_frames = {
        "early-latched": FIRST_POLICY_FRAME + lag_frames,
        "delayed-first": FIRST_POLICY_FRAME + lag_frames + 1,
    }
    candidate_matches: dict[str, dict[str, bool]] = {}
    candidate_neutrality: dict[str, dict[str, bool]] = {}
    candidate_exact_two_ports: dict[str, bool] = {}
    for candidate, replay_frame in candidate_frames.items():
        replay = replay_states.get(replay_frame)
        candidate_exact_two_ports[candidate] = bool(
            isinstance(replay, Mapping) and set(replay) == {1, 2}
        )
        port_matches: dict[str, bool] = {}
        port_neutrality: dict[str, bool] = {}
        for port in (1, 2):
            trace_slot = slots.get(f"p{port}")
            observed = replay.get(port) if isinstance(replay, Mapping) else None
            try:
                port_matches[f"p{port}"] = bool(
                    isinstance(trace_slot, Mapping)
                    and isinstance(observed, Mapping)
                    and _trace_command_matches_replay(trace_slot, observed)
                )
            except (KeyError, TypeError, ValueError):
                port_matches[f"p{port}"] = False
            port_neutrality[f"p{port}"] = _replay_controller_state_is_exact_neutral(
                observed
            )
        port_matches["both_slots"] = all(port_matches.values())
        port_neutrality["both_slots"] = all(port_neutrality.values())
        candidate_matches[candidate] = port_matches
        candidate_neutrality[candidate] = port_neutrality

    early_exact = candidate_matches["early-latched"]["both_slots"]
    delayed_exact = candidate_matches["delayed-first"]["both_slots"]
    normal_next_boundary_evidence: dict[str, Any] | None = None
    fixed_normal_boundary_evidence: dict[str, Any] | None = None
    neutral_game_start_overwrite_evidence: dict[str, Any] | None = None
    matching_candidates = [
        candidate
        for candidate in ("early-latched", "delayed-first")
        if candidate_matches[candidate]["both_slots"]
    ]
    both_boundaries_neutral = all(
        candidate_exact_two_ports[candidate]
        and candidate_neutrality[candidate]["both_slots"]
        for candidate in ("early-latched", "delayed-first")
    )
    # A repeated initial command can also match replay -121 because trace -122
    # already uses the normal next-frame boundary. Prove the complete subsequent
    # two-port stream before distinguishing that case from delayed-first startup.
    repeated_initial_command_candidate = bool(
        not early_exact
        and delayed_exact
        and candidate_exact_two_ports["early-latched"]
        and candidate_neutrality["early-latched"]["both_slots"]
    )
    if both_boundaries_neutral or repeated_initial_command_candidate:
        normal_next_boundary_evidence = _normal_next_boundary_two_port_evidence(
            trace_by_frame,
            replay_states,
            lag_frames=lag_frames,
        )
    normal_boundary_exact = bool(
        normal_next_boundary_evidence is not None
        and normal_next_boundary_evidence["decision"] == "pass"
    )
    if early_exact:
        fixed_normal_boundary_evidence = _normal_next_boundary_two_port_evidence(
            trace_by_frame,
            replay_states,
            lag_frames=lag_frames,
            include_first_policy_frame=True,
        )
    fixed_normal_boundary_exact = bool(
        fixed_normal_boundary_evidence is not None
        and fixed_normal_boundary_evidence["decision"] == "pass"
    )
    if fixed_normal_boundary_evidence is not None and not fixed_normal_boundary_exact:
        neutral_game_start_overwrite_evidence = (
            _single_game_start_neutral_overwrite_evidence(
                trace_by_frame,
                replay_states,
                fixed_normal_boundary_evidence,
                transport_proof,
                lag_frames=lag_frames,
            )
        )
    neutral_game_start_overwrite_exact = bool(
        neutral_game_start_overwrite_evidence is not None
        and neutral_game_start_overwrite_evidence["decision"] == "pass"
    )
    if normal_boundary_exact:
        decision = "first-gameplay-latch-unobservable"
        selection_reason = (
            "replay -122 is exactly neutral for both ports; the first command matches "
            "replay -121 and is repeated by trace -122; every subsequent two-port command "
            "is exact at the normal one-frame boundary"
            if repeated_initial_command_candidate
            else
            "both permitted replay boundaries are exactly neutral for both ports, and every "
            "observable command from trace -122 onward is exact at the normal one-frame "
            "boundary"
        )
    elif fixed_normal_boundary_exact:
        decision = "fixed-normal-boundary"
        selection_reason = (
            "replay -122 uniquely matches the first command, and every observable two-port "
            "command is exact at the normal one-frame boundary"
        )
    elif neutral_game_start_overwrite_exact:
        decision = "fixed-normal-boundary-with-neutral-game-start-overwrite"
        selection_reason = (
            "the complete two-port stream uses the normal one-frame boundary, with one "
            "transport-proved paired neutral GAME_START overwrite during ENTRY startup"
        )
    elif early_exact and not delayed_exact:
        decision = "early-latched"
        selection_reason = (
            "only replay -122 matches the first command across every hard-gated dimension "
            "for both ports"
        )
    elif delayed_exact and not early_exact:
        decision = "delayed-first"
        selection_reason = (
            "only replay -121 matches the first command across every hard-gated dimension "
            "for both ports"
        )
    elif early_exact and delayed_exact:
        decision = "observationally-equivalent-delayed-first"
        selection_reason = (
            "both permitted boundaries are controller-observationally equivalent; select the "
            "canonical delayed-first mapping"
        )
    else:
        raise ValueError(
            "piecewise GAME_START first policy command matches neither transport-permitted "
            "replay boundary and the exact first-gameplay-latch classification failed: "
            f"candidate_matches={candidate_matches}, "
            f"candidate_neutrality={candidate_neutrality}, "
            f"candidate_exact_two_ports={candidate_exact_two_ports}, "
            f"normal_next_boundary={normal_next_boundary_evidence}"
        )

    if decision in {
        "early-latched",
        "fixed-normal-boundary",
        "fixed-normal-boundary-with-neutral-game-start-overwrite",
    }:
        selected_candidate = "early-latched"
    elif decision in {"delayed-first", "observationally-equivalent-delayed-first"}:
        selected_candidate = "delayed-first"
    else:
        selected_candidate = None
    return {
        "trace_frame": FIRST_POLICY_FRAME,
        "comparison": "every hard-gated controller dimension across both ports",
        "evidence_ports": [1, 2],
        "candidate_replay_frames": candidate_frames,
        "candidate_matches": candidate_matches,
        "candidate_neutrality": candidate_neutrality,
        "candidate_exact_two_ports": candidate_exact_two_ports,
        "matching_candidates": matching_candidates,
        "decision": decision,
        "selection_reason": selection_reason,
        "selected_replay_frame": (
            candidate_frames[selected_candidate] if selected_candidate is not None else None
        ),
        "first_policy_trace_unobservable": (
            decision == "first-gameplay-latch-unobservable"
        ),
        "repeated_initial_command_normal_boundary_proved": (
            repeated_initial_command_candidate and normal_boundary_exact
        ),
        "normal_next_boundary_evidence": normal_next_boundary_evidence,
        "fixed_normal_boundary_evidence": fixed_normal_boundary_evidence,
        "neutral_game_start_overwrite_evidence": neutral_game_start_overwrite_evidence,
    }


def _trace_command_matches_replay(
    trace_slot: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> bool:
    """Compare one trace command with every replay dimension used as a hard gate."""

    model = str(trace_slot.get("model"))
    command = trace_slot.get("command")
    if not isinstance(command, Mapping):
        return False
    intended = _normalize_trace_command(model, command)
    expected_buttons = tuple(cast(tuple[str, ...], intended["buttons"]))
    expected_processed_buttons = _expected_processed_buttons(expected_buttons)
    intended_main = cast(tuple[float, float], intended["main_stick"])
    intended_c = cast(tuple[float, float], intended["c_stick"])
    expected_c = _expected_processed_c_stick(model, trace_slot, intended_c)
    expected_raw_main = tuple(_expected_raw_main_axis(axis) for axis in intended_main)
    expected_triggers = tuple(
        max(
            float(intended[f"analog_{side}"]),
            1.0 if side.upper() in expected_buttons else 0.0,
        )
        for side in ("l", "r")
    )
    observed_c = cast(tuple[float, float], observed.get("c_stick"))
    observed_triggers = (
        float(observed["physical_analog_l"]),
        float(observed["physical_analog_r"]),
    )
    return (
        expected_buttons == observed.get("buttons_physical")
        and expected_processed_buttons == observed.get("buttons_processed")
        and expected_raw_main == observed.get("raw_main_stick")
        and all(
            abs(expected - actual) <= PROCESSED_C_STICK_TOLERANCE
            for expected, actual in zip(expected_c, observed_c, strict=True)
        )
        and all(
            abs(expected - actual) <= PHYSICAL_ANALOG_SHOULDER_TOLERANCE
            for expected, actual in zip(expected_triggers, observed_triggers, strict=True)
        )
    )


def _first_gameplay_command_carryover_trace_frames(
    trace_by_frame: Mapping[int, Mapping[str, Any]],
    replay_states: Mapping[int, Mapping[int, Mapping[str, Any]]],
    aligned_trace_frames: list[int],
    *,
    lag_frames: int,
) -> list[int]:
    """Prove a first-command carryover before excluding the next startup boundary.

    The first policy command can miss replay frame -122 because Dolphin still owns
    the final menu latch.  When that exact command is then observed one boundary
    later, trace frame -122 cannot be compared to replay frame -121.  This proof is
    deliberately limited to the fixed ENTRY startup state and requires the replay
    controller state for both ports to match the preceding trace command across
    every hard-gated controller dimension.
    """

    frame = FIRST_GAMEPLAY_COMMAND_CARRYOVER_TRACE_FRAME
    prior_frame = FIRST_POLICY_FRAME
    if (
        lag_frames != NEXT_FRAME_CONTROLLER_BOUNDARY_FRAMES
        or min(trace_by_frame) != FIRST_POLICY_FRAME
        or prior_frame not in aligned_trace_frames
        or frame not in aligned_trace_frames
    ):
        return []
    prior_row = trace_by_frame[prior_frame]
    current_row = trace_by_frame[frame]
    prior_slots = prior_row.get("slots")
    current_slots = current_row.get("slots")
    replay = replay_states.get(frame + lag_frames)
    if (
        not isinstance(prior_slots, Mapping)
        or not isinstance(current_slots, Mapping)
        or set(prior_slots) != {"p1", "p2"}
        or set(current_slots) != {"p1", "p2"}
        or not isinstance(replay, Mapping)
        or set(replay) < {1, 2}
    ):
        return []

    current_mismatch_proved = False
    for port in (1, 2):
        prior_slot = prior_slots.get(f"p{port}")
        current_slot = current_slots.get(f"p{port}")
        if not isinstance(prior_slot, Mapping) or not isinstance(current_slot, Mapping):
            return []
        for slot in (prior_slot, current_slot):
            player = slot.get("player_state")
            dispatch = slot.get("controller_dispatch")
            if (
                not isinstance(player, Mapping)
                or player.get("action") != 322
                or player.get("stocks") != 4
                or player.get("percent") != 0.0
                or not isinstance(dispatch, Mapping)
                or dispatch.get("called") is not True
            ):
                return []
        observed = replay[port]
        if not _trace_command_matches_replay(prior_slot, observed):
            return []
        if not _trace_command_matches_replay(current_slot, observed):
            current_mismatch_proved = True
    return [frame] if current_mismatch_proved else []


def _audit_controller_boundary_records(
    trace_rows: list[dict[str, Any]],
    replay_states: Mapping[int, Mapping[int, Mapping[str, Any]]],
    *,
    lag_frames: int = CONTROLLER_REPLAY_LAG_FRAMES,
    game_start_transport_proof: Mapping[str, Any] | None = None,
    classify_first_gameplay_command_carryover: bool = False,
) -> dict[str, Any]:
    """Compare intended commands with the delayed controller events saved by Slippi."""
    if not trace_rows:
        raise ValueError("controller boundary audit requires at least one trace row")
    if not replay_states:
        raise ValueError("controller boundary audit requires at least one replay frame")
    if lag_frames < NEXT_FRAME_CONTROLLER_BOUNDARY_FRAMES:
        raise ValueError("controller boundary lag cannot precede the next-frame dispatch boundary")
    trace_by_frame = {int(row["game_frame"]): row for row in trace_rows}
    if len(trace_by_frame) != len(trace_rows):
        raise ValueError("controller boundary audit trace frames must be unique")
    trace_frames = sorted(trace_by_frame)
    replay_frames = sorted(int(frame) for frame in replay_states)
    expected_trace_sequence = list(range(trace_frames[0], trace_frames[-1] + 1))
    trace_is_consecutive = trace_frames == expected_trace_sequence
    replay_covers_observed_trace_states = all(
        frame in replay_states and all(port in replay_states[frame] for port in (1, 2))
        for frame in trace_frames
    )
    first_policy_mapping_evidence: dict[str, Any] | None = None
    first_policy_trace_unobservable = False
    fixed_normal_game_start_boundary = False
    neutral_game_start_overwrite_trace_frame: int | None = None
    if game_start_transport_proof is not None:
        piecewise_game_start = True
        first_policy_mapping_evidence = _first_policy_frame_mapping_evidence(
            trace_by_frame,
            replay_states,
            lag_frames=lag_frames,
            transport_proof=game_start_transport_proof,
        )
        first_policy_trace_unobservable = bool(
            first_policy_mapping_evidence["first_policy_trace_unobservable"]
        )
        fixed_normal_game_start_boundary = first_policy_mapping_evidence["decision"] in {
            "fixed-normal-boundary",
            "fixed-normal-boundary-with-neutral-game-start-overwrite",
        }
        if (
            first_policy_mapping_evidence["decision"]
            == "fixed-normal-boundary-with-neutral-game-start-overwrite"
        ):
            overwrite_evidence = cast(
                Mapping[str, Any],
                first_policy_mapping_evidence["neutral_game_start_overwrite_evidence"],
            )
            neutral_game_start_overwrite_trace_frame = int(
                overwrite_evidence["trace_frame"]
            )
        selected_first_replay_frame = first_policy_mapping_evidence[
            "selected_replay_frame"
        ]
        replay_frame_by_trace_frame, game_start_controller_latch_trace_frames = (
            _piecewise_game_start_replay_frames(
                trace_by_frame,
                replay_states,
                lag_frames=lag_frames,
                transport_proof=game_start_transport_proof,
                first_policy_replay_frame=(
                    int(selected_first_replay_frame)
                    if selected_first_replay_frame is not None
                    else None
                ),
                first_policy_trace_unobservable=first_policy_trace_unobservable,
                fixed_normal_boundary=fixed_normal_game_start_boundary,
                neutral_overwrite_trace_frame=neutral_game_start_overwrite_trace_frame,
            )
        )
    else:
        piecewise_game_start = False
        replay_frame_by_trace_frame = {
            frame: frame + lag_frames
            for frame in trace_frames
            if frame + lag_frames in replay_states
            and all(port in replay_states[frame + lag_frames] for port in (1, 2))
        }
        game_start_controller_latch_trace_frames = []
    terminal_candidates = [
        frame
        for frame in trace_frames
        if frame not in game_start_controller_latch_trace_frames
        and frame not in replay_frame_by_trace_frame
        and frame + lag_frames > replay_frames[-1]
    ]
    terminal_unobservable_trace_frames = terminal_candidates
    terminal_unobservable_is_bounded_suffix = (
        len(terminal_unobservable_trace_frames) <= lag_frames
        and terminal_unobservable_trace_frames
        == trace_frames[len(trace_frames) - len(terminal_unobservable_trace_frames) :]
    )
    expected_trace_frames = [
        frame
        for frame in trace_frames
        if frame not in terminal_unobservable_trace_frames
        and frame not in game_start_controller_latch_trace_frames
        and not (
            first_policy_trace_unobservable and frame == FIRST_POLICY_FRAME
        )
    ]
    aligned_trace_frames = [
        frame
        for frame in expected_trace_frames
        if frame in trace_by_frame
        and frame in replay_frame_by_trace_frame
    ]
    missing_pairs = sorted(set(expected_trace_frames) - set(aligned_trace_frames))
    full_causal_overlap = (
        bool(expected_trace_frames)
        and trace_is_consecutive
        and replay_covers_observed_trace_states
        and terminal_unobservable_is_bounded_suffix
        and not missing_pairs
    )
    first_gameplay_latch_trace_frames = (
        [FIRST_POLICY_FRAME]
        if (
            first_policy_trace_unobservable
            or (
                not piecewise_game_start
                and trace_frames[0] == FIRST_POLICY_FRAME
                and FIRST_POLICY_FRAME in aligned_trace_frames
            )
        )
        else []
    )
    first_gameplay_command_carryover_trace_frames = (
        _first_gameplay_command_carryover_trace_frames(
            trace_by_frame,
            replay_states,
            aligned_trace_frames,
            lag_frames=lag_frames,
        )
        if classify_first_gameplay_command_carryover and not piecewise_game_start
        else []
    )
    unobservable_startup_trace_frames = sorted(
        set(first_gameplay_latch_trace_frames)
        | set(first_gameplay_command_carryover_trace_frames)
        | set(game_start_controller_latch_trace_frames)
    )
    latch_comparable_trace_frames = [
        frame for frame in aligned_trace_frames if frame not in unobservable_startup_trace_frames
    ]

    slots: dict[str, dict[str, Any]] = {}
    for port in (1, 2):
        slot = f"p{port}"
        slot_models = {
            _model_name(str(cast(dict[str, Any], trace_by_frame[frame]["slots"])[slot]["model"]))
            for frame in aligned_trace_frames
        }
        if len(slot_models) != 1:
            raise ValueError(f"{slot} trace does not contain exactly one model: {sorted(slot_models)}")
        model_name = next(iter(slot_models))
        dispatch_called_trace_frames: list[int] = []
        dispatch_by_frame: dict[int, bool] = {}
        for frame in aligned_trace_frames:
            trace_slot = cast(dict[str, Any], trace_by_frame[frame]["slots"])[slot]
            dispatch = trace_slot.get("controller_dispatch")
            if not isinstance(dispatch, dict) or not isinstance(dispatch.get("called"), bool):
                raise ValueError(f"{slot} trace frame {frame} lacks explicit controller-dispatch evidence")
            dispatch_called = bool(dispatch["called"])
            dispatch_by_frame[frame] = dispatch_called
            if dispatch_called:
                dispatch_called_trace_frames.append(frame)
        controller_trace_frames = [
            frame for frame in dispatch_called_trace_frames if frame not in unobservable_startup_trace_frames
        ]
        excluded_no_dispatch_frames = [
            frame for frame in aligned_trace_frames if not dispatch_by_frame[frame]
        ]
        expected_no_dispatch_frames: list[int] = []
        no_dispatch_rule_exact = excluded_no_dispatch_frames == expected_no_dispatch_frames
        excluded_latch_frames = [
            frame for frame in first_gameplay_latch_trace_frames if frame in trace_by_frame
        ]
        for frame in excluded_latch_frames:
            if frame in dispatch_by_frame:
                continue
            trace_slot = cast(dict[str, Any], trace_by_frame[frame]["slots"])[slot]
            dispatch = trace_slot.get("controller_dispatch")
            if not isinstance(dispatch, dict) or not isinstance(dispatch.get("called"), bool):
                raise ValueError(
                    f"{slot} first-gameplay latch frame {frame} lacks exact dispatch evidence"
                )
            dispatch_by_frame[frame] = bool(dispatch["called"])
        excluded_game_start_frames = [
            frame
            for frame in game_start_controller_latch_trace_frames
            if frame in trace_by_frame
        ]
        for frame in excluded_game_start_frames:
            trace_slot = cast(dict[str, Any], trace_by_frame[frame]["slots"])[slot]
            dispatch = trace_slot.get("controller_dispatch")
            if not isinstance(dispatch, dict) or dispatch.get("called") is not True:
                raise ValueError(
                    f"{slot} GAME_START trace frame {frame} lacks exact dispatch evidence"
                )
            dispatch_by_frame[frame] = True
        excluded_first_command_carryover_frames = [
            frame
            for frame in first_gameplay_command_carryover_trace_frames
            if frame in aligned_trace_frames
        ]
        excluded_startup_frames = sorted(
            set(excluded_no_dispatch_frames)
            | set(excluded_latch_frames)
            | set(excluded_first_command_carryover_frames)
            | set(excluded_game_start_frames)
        )
        startup_exclusions = [
            {
                "trace_frame": frame,
                "dispatch_called": dispatch_by_frame[frame],
                "classifications": [
                    classification
                    for applies, classification in (
                        (
                            frame in excluded_latch_frames,
                            "first-gameplay controller latch unavailable",
                        ),
                        (
                            frame in excluded_first_command_carryover_frames,
                            "first gameplay command carried across the next startup boundary",
                        ),
                        (
                            frame in excluded_game_start_frames,
                            "libmelee GAME_START controller transaction unavailable",
                        ),
                        (frame in excluded_no_dispatch_frames, "controller dispatch not called"),
                    )
                    if applies
                ],
            }
            for frame in excluded_startup_frames
        ]
        physical_button_mismatches: list[dict[str, Any]] = []
        processed_button_mismatches: list[dict[str, Any]] = []
        direct_processed_button_mismatches: list[dict[str, Any]] = []
        raw_samples: list[tuple[int, int, str, int | float, int | float]] = []
        main_samples: list[tuple[int, int, str, int | float, int | float]] = []
        c_samples: list[tuple[int, int, str, int | float, int | float]] = []
        trigger_samples: list[tuple[int, int, str, float, float]] = []
        for trace_frame in controller_trace_frames:
            replay_frame = replay_frame_by_trace_frame[trace_frame]
            trace_slot = cast(dict[str, Any], trace_by_frame[trace_frame]["slots"])[slot]
            model = str(trace_slot["model"])
            intended = _normalize_trace_command(model, cast(dict[str, Any], trace_slot["command"]))
            observed = replay_states[replay_frame][port]
            expected_buttons = tuple(cast(tuple[str, ...], intended["buttons"]))
            expected_processed_buttons = _expected_processed_buttons(expected_buttons)
            observed_physical_buttons = tuple(cast(tuple[str, ...], observed["buttons_physical"]))
            observed_processed_buttons = tuple(cast(tuple[str, ...], observed["buttons_processed"]))
            if expected_buttons != observed_physical_buttons:
                physical_button_mismatches.append(
                    {
                        "trace_frame": trace_frame,
                        "replay_frame": replay_frame,
                        "expected": list(expected_buttons),
                        "observed": list(observed_physical_buttons),
                    }
                )
            if expected_processed_buttons != observed_processed_buttons:
                processed_button_mismatches.append(
                    {
                        "trace_frame": trace_frame,
                        "replay_frame": replay_frame,
                        "expected": list(expected_processed_buttons),
                        "observed": list(observed_processed_buttons),
                    }
                )
            if expected_buttons != observed_processed_buttons:
                direct_processed_button_mismatches.append(
                    {
                        "trace_frame": trace_frame,
                        "replay_frame": replay_frame,
                        "decoded_physical": list(expected_buttons),
                        "observed_processed": list(observed_processed_buttons),
                    }
                )
            intended_main = cast(tuple[float, float], intended["main_stick"])
            observed_raw_main = cast(tuple[int, int], observed["raw_main_stick"])
            observed_main = cast(tuple[float, float], observed["main_stick"])
            intended_c = cast(tuple[float, float], intended["c_stick"])
            expected_c = _expected_processed_c_stick(model, trace_slot, intended_c)
            observed_c = cast(tuple[float, float], observed["c_stick"])
            for axis_index, axis_name in enumerate(("x", "y")):
                raw_samples.append(
                    (
                        trace_frame,
                        replay_frame,
                        axis_name,
                        _expected_raw_main_axis(intended_main[axis_index]),
                        observed_raw_main[axis_index],
                    )
                )
                main_samples.append(
                    (
                        trace_frame,
                        replay_frame,
                        axis_name,
                        intended_main[axis_index],
                        observed_main[axis_index],
                    )
                )
                c_samples.append(
                    (
                        trace_frame,
                        replay_frame,
                        axis_name,
                        expected_c[axis_index],
                        observed_c[axis_index],
                    )
                )
            for side in ("l", "r"):
                expected_trigger = max(
                    float(intended[f"analog_{side}"]),
                    1.0 if side.upper() in expected_buttons else 0.0,
                )
                trigger_samples.append(
                    (
                        trace_frame,
                        replay_frame,
                        f"analog_{side}",
                        expected_trigger,
                        float(observed[f"physical_analog_{side}"]),
                    )
                )
        raw_main = _component_statistics(raw_samples)
        processed_c_stick = _tolerant_component_statistics(
            c_samples,
            tolerance=PROCESSED_C_STICK_TOLERANCE,
        )
        physical_analog_shoulders = _tolerant_component_statistics(
            trigger_samples,
            tolerance=PHYSICAL_ANALOG_SHOULDER_TOLERANCE,
        )
        comparison_evidence = bool(controller_trace_frames)
        slots[slot] = {
            "port": port,
            "model": model_name,
            "aligned_frame_pairs": len(aligned_trace_frames),
            "controller_frame_pairs_compared": len(controller_trace_frames),
            "excluded_startup_trace_frames": excluded_startup_frames,
            "excluded_first_gameplay_latch_trace_frames": excluded_latch_frames,
            "excluded_first_gameplay_command_carryover_trace_frames": (
                excluded_first_command_carryover_frames
            ),
            "excluded_game_start_controller_latch_trace_frames": excluded_game_start_frames,
            "excluded_no_dispatch_trace_frames": excluded_no_dispatch_frames,
            "expected_no_dispatch_trace_frames": expected_no_dispatch_frames,
            "no_dispatch_rule": (
                "Every supported policy must dispatch on every aligned trace frame"
            ),
            "startup_exclusions": startup_exclusions,
            "startup_boundary_rule": (
                "The Frisson two-pipe runtime uses a transport-proved startup mapping. Full "
                "two-slot controller evidence selects a fixed one-frame or piecewise boundary, "
                "or proves one tightly bounded unobservable startup transaction."
            ),
            "digital_buttons": {
                "physical": {
                    "frames_compared": len(controller_trace_frames),
                    "exact_frames": len(controller_trace_frames) - len(physical_button_mismatches),
                    "mismatch_frames": len(physical_button_mismatches),
                    "mismatches": physical_button_mismatches,
                },
                "processed_upstream_observation": {
                    "frames_compared": len(controller_trace_frames),
                    "exact_frames": len(controller_trace_frames) - len(processed_button_mismatches),
                    "mismatch_frames": len(processed_button_mismatches),
                    "mismatches": processed_button_mismatches,
                    "expected_mapping": "physical Z is observed by Melee and Slippi-AI as A+Z",
                    "upstream_source": "slippi_db.parse_libmelee.get_controller(processed_button)",
                },
                "decoded_physical_vs_processed_diagnostic": {
                    "hard_gate": False,
                    "frames_compared": len(controller_trace_frames),
                    "exact_frames": (len(controller_trace_frames) - len(direct_processed_button_mismatches)),
                    "mismatch_frames": len(direct_processed_button_mismatches),
                    "mismatches": direct_processed_button_mismatches,
                    "classification": "expected game-level button processing, including Z to A+Z",
                },
            },
            "intended_raw_main_stick": {
                **raw_main,
                "formula": "round((decoded_axis - 0.5) * 160)",
                "source": "libmelee.controller.fix_analog_stick before its +0.1 pipe-input fudge",
            },
            "processed_c_stick": {
                **processed_c_stick,
                "hard_gate": True,
                "formula": (
                    "canonical axes to raw [-80, 80], circular radius-80 clamp with "
                    "toward-zero truncation, per-axis nonzero abs(raw) < 23 deadzone, "
                    "then raw / 80"
                ),
                "raw_axis_deadzone": {
                    "threshold": RAW_AXIS_DEADZONE,
                    "rule": "raw != 0 and abs(raw) < 23 becomes zero",
                    "source": "pinned slippi_ai.controller_lib",
                },
                "native_slippi_ai_dummy_prefix_expected": [-0.7, -0.7],
                "source": "pinned Peppi pre-frame cstick.x and .y from the saved replay",
            },
            "physical_analog_shoulders": {
                **physical_analog_shoulders,
                "hard_gate": True,
                "formula": (
                    "max(canonical analog side, 1.0 when the corresponding digital L/R button is pressed)"
                ),
                "source": "pinned Peppi pre-frame triggers_physical.l and .r",
            },
            "processed_diagnostics": {
                "hard_gate": False,
                "reason": (
                    "Dolphin and game calibration change processed main-stick values; the raw "
                    "main-stick wire values provide the corresponding hard gate"
                ),
                "main_stick": _component_statistics(main_samples),
            },
            "gate": {
                "full_causal_overlap": full_causal_overlap,
                "comparison_evidence": comparison_evidence,
                "no_dispatch_rule_exact": no_dispatch_rule_exact,
                "physical_buttons_exact": not physical_button_mismatches,
                "processed_upstream_buttons_exact": not processed_button_mismatches,
                "intended_raw_main_stick_exact": raw_main["mismatch_components"] == 0,
                "processed_c_stick_within_tolerance": (processed_c_stick["mismatch_components"] == 0),
                "physical_analog_shoulders_within_tolerance": (
                    physical_analog_shoulders["mismatch_components"] == 0
                ),
            },
        }

    checks = {
        "full_causal_overlap": full_causal_overlap,
        "trace_frames_consecutive": trace_is_consecutive,
        "replay_covers_every_observed_trace_state": replay_covers_observed_trace_states,
        "terminal_unobservable_tail_bounded_by_lag": terminal_unobservable_is_bounded_suffix,
        "both_slots_aligned": all(
            slot["aligned_frame_pairs"] == len(expected_trace_frames) for slot in slots.values()
        ),
        "both_slots_have_controller_evidence": all(
            slot["gate"]["comparison_evidence"] for slot in slots.values()
        ),
        "both_slots_physical_buttons_exact": all(
            slot["gate"]["physical_buttons_exact"] for slot in slots.values()
        ),
        "both_slots_processed_upstream_buttons_exact": all(
            slot["gate"]["processed_upstream_buttons_exact"] for slot in slots.values()
        ),
        "both_slots_intended_raw_main_stick_exact": all(
            slot["gate"]["intended_raw_main_stick_exact"] for slot in slots.values()
        ),
        "both_slots_processed_c_stick_within_tolerance": all(
            slot["gate"]["processed_c_stick_within_tolerance"] for slot in slots.values()
        ),
        "both_slots_physical_analog_shoulders_within_tolerance": all(
            slot["gate"]["physical_analog_shoulders_within_tolerance"] for slot in slots.values()
        ),
        "both_slots_first_gameplay_latch_rule_exact": all(
            slot["excluded_first_gameplay_latch_trace_frames"] == first_gameplay_latch_trace_frames
            for slot in slots.values()
        ),
        "both_slots_first_gameplay_command_carryover_rule_exact": all(
            slot["excluded_first_gameplay_command_carryover_trace_frames"]
            == first_gameplay_command_carryover_trace_frames
            for slot in slots.values()
        ),
        "both_slots_game_start_controller_latch_rule_exact": all(
            slot["excluded_game_start_controller_latch_trace_frames"]
            == game_start_controller_latch_trace_frames
            for slot in slots.values()
        ),
        "first_policy_frame_mapping_evidence_exact": (
            not piecewise_game_start
            or (
                first_policy_mapping_evidence is not None
                and first_policy_mapping_evidence["decision"]
                in {
                    "early-latched",
                    "delayed-first",
                    "fixed-normal-boundary",
                    "fixed-normal-boundary-with-neutral-game-start-overwrite",
                    "observationally-equivalent-delayed-first",
                    "first-gameplay-latch-unobservable",
                }
                and (
                    not first_policy_trace_unobservable
                    or (
                        first_policy_mapping_evidence["selected_replay_frame"] is None
                        and first_policy_mapping_evidence["normal_next_boundary_evidence"][
                            "decision"
                        ]
                        == "pass"
                    )
                )
                and (
                    first_policy_mapping_evidence["decision"]
                    != "fixed-normal-boundary-with-neutral-game-start-overwrite"
                    or first_policy_mapping_evidence[
                        "neutral_game_start_overwrite_evidence"
                    ]["decision"]
                    == "pass"
                )
            )
        ),
        "game_start_transport_proof_exact": (
            not piecewise_game_start
            or (
                game_start_transport_proof is not None
                and game_start_transport_proof.get("decision") == "pass"
            )
        ),
        "both_slots_no_dispatch_rule_exact": all(
            slot["gate"]["no_dispatch_rule_exact"] for slot in slots.values()
        ),
    }
    return {
        "schema_version": CONTROLLER_BOUNDARY_SCHEMA_VERSION,
        "classification": "deliberate decoded-command to libmelee controller adaptation",
        "alignment": {
            "mode": (
                "normal-boundary-after-unobservable-first-gameplay-latch"
                if first_policy_trace_unobservable
                else "fixed-lag-with-game-start-neutral-overwrite"
                if neutral_game_start_overwrite_trace_frame is not None
                else "fixed-lag-through-game-start"
                if fixed_normal_game_start_boundary
                else "piecewise-game-start"
                if piecewise_game_start
                else "fixed-lag"
            ),
            "steady_state_trace_to_replay_lag_frames": lag_frames,
            "first_policy_trace_to_replay_lag_frames": (
                None
                if first_policy_trace_unobservable
                else int(first_policy_mapping_evidence["selected_replay_frame"])
                - FIRST_POLICY_FRAME
                if first_policy_mapping_evidence is not None
                else lag_frames
            ),
            "startup_trace_to_replay_lag_frames": (
                lag_frames
                if first_policy_trace_unobservable or fixed_normal_game_start_boundary
                else lag_frames + 1
                if piecewise_game_start
                else lag_frames
            ),
            "online_delay_frames": lag_frames - NEXT_FRAME_CONTROLLER_BOUNDARY_FRAMES,
            "next_frame_controller_boundary_frames": NEXT_FRAME_CONTROLLER_BOUNDARY_FRAMES,
            "equation": (
                (
                    "trace -123 is unavailable at the universal first-gameplay latch; "
                    "replay = trace + 1 from -122"
                    if first_policy_trace_unobservable
                    else (
                        "replay_frame = trace_frame + 1 except one transport-proved paired "
                        "neutral GAME_START overwrite"
                    )
                    if neutral_game_start_overwrite_trace_frame is not None
                    else "replay_frame = trace_frame + 1 across the complete startup and game"
                    if fixed_normal_game_start_boundary
                    else "replay = trace + 1 at -123; replay -121 is the GAME_START internal "
                    "transaction; replay = trace + 2 for -122..-120; trace -119 unavailable; "
                    "replay = trace + 1 from -118"
                    if first_policy_mapping_evidence is not None
                    and first_policy_mapping_evidence["selected_replay_frame"]
                    == FIRST_POLICY_FRAME + lag_frames
                    else "replay = trace + 2 for -123..-120; trace -119 unavailable at "
                    "GAME_START; replay = trace + 1 from -118"
                )
                if piecewise_game_start
                else "replay_frame = trace_frame + online_delay_frames + 1"
            ),
            "trace_frame_range": [trace_frames[0], trace_frames[-1]],
            "replay_frame_range": [replay_frames[0], replay_frames[-1]],
            "expected_overlap_pairs": len(expected_trace_frames),
            "actual_overlap_pairs": len(aligned_trace_frames),
            "first_aligned_pair": (
                [
                    aligned_trace_frames[0],
                    replay_frame_by_trace_frame[aligned_trace_frames[0]],
                ]
                if aligned_trace_frames
                else None
            ),
            "first_gameplay_controller_latch_unobservable_trace_frames": (first_gameplay_latch_trace_frames),
            "first_gameplay_controller_latch_unobservable_frames": (
                FIRST_GAMEPLAY_CONTROLLER_LATCH_UNOBSERVABLE_FRAMES
            ),
            "first_gameplay_command_carryover_unobservable_trace_frames": (
                first_gameplay_command_carryover_trace_frames
            ),
            "first_gameplay_command_carryover_unobservable_frames": (
                len(first_gameplay_command_carryover_trace_frames)
            ),
            "game_start_controller_latch_unobservable_trace_frames": (
                game_start_controller_latch_trace_frames
            ),
            "game_start_controller_latch_unobservable_frames": len(
                game_start_controller_latch_trace_frames
            ),
            "first_comparable_controller_pair": (
                [
                    latch_comparable_trace_frames[0],
                    replay_frame_by_trace_frame[latch_comparable_trace_frames[0]],
                ]
                if latch_comparable_trace_frames
                else None
            ),
            "last_aligned_pair": (
                [
                    aligned_trace_frames[-1],
                    replay_frame_by_trace_frame[aligned_trace_frames[-1]],
                ]
                if aligned_trace_frames
                else None
            ),
            "missing_internal_pairs": missing_pairs,
            "permitted_terminal_unobservable_trace_frames": terminal_unobservable_trace_frames,
            "maximum_permitted_terminal_tail_frames": lag_frames,
            "unpaired_trace_boundary_frames": sorted(set(trace_frames) - set(aligned_trace_frames)),
            "unpaired_replay_boundary_frames": [
                frame for frame in replay_frames if frame not in set(replay_frame_by_trace_frame.values())
            ],
            "game_start_transport_proof": game_start_transport_proof,
            "first_policy_frame_mapping_evidence": first_policy_mapping_evidence,
        },
        "adapter": {
            "slippi_ai_explicit_flush": False,
            "slippi_ai_dispatch_boundary": (
                "queued into the current two-port transaction before the next Console.step()"
            ),
            "first_gameplay_controller_latch": (
                "The first queued gameplay command is not guaranteed to replace the final menu "
                "input in replay frame -122; source-exact dispatch is retained and this single "
                "universal boundary is excluded."
            ),
            "first_gameplay_command_carryover": (
                "When the first policy command is proved across every hard-gated dimension "
                "at replay frame -121, trace frame -122 is excluded as the single exact "
                "carryover boundary while both players remain in the ENTRY startup state."
            ),
            "game_start_controller_latch": (
                "The pinned two-pipe transport proves one paired libmelee GAME_START internal "
                "commit. Exact two-slot replay evidence selects the complete boundary mapping. "
                "A dynamic neutral overwrite requires both replay ports neutral, ENTRY startup, "
                "the next boundary restored, and every other controller value hard-gated."
            ),
            "no_dispatch_exclusion": (
                "Every supported policy must dispatch on every aligned trace frame"
            ),
            "windowed_native_sender_single_flush_retained": True,
            "windowed_transport_neutralizes_analog_r": True,
            "fix_analog_inputs": True,
            "processed_main_stick_comparison_is_diagnostic_only": True,
            "processed_c_stick_is_hard_gated": True,
            "peppi_physical_analog_shoulders_are_hard_gated": True,
        },
        "slots": slots,
        "gate": {"decision": "pass" if all(checks.values()) else "fail", "checks": checks},
    }


def _trace_identity(path: Path, project_root: Path, rows: int) -> dict[str, Any]:
    return {
        "path": _display_path(path, project_root),
        "sha256": _sha256_file(path),
        "byte_length": path.stat().st_size,
        "rows": rows,
    }


def _audit_controller_boundary(
    trace_path: Path,
    replay_path: Path,
    project_root: Path,
    *,
    lag_frames: int = CONTROLLER_REPLAY_LAG_FRAMES,
    game_start_transport_proof: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows = _read_trace_rows(trace_path)
    audit = _audit_controller_boundary_records(
        rows,
        _read_replay_controller_states(replay_path),
        lag_frames=lag_frames,
        game_start_transport_proof=game_start_transport_proof,
    )
    audit["trace"] = _trace_identity(trace_path, project_root, len(rows))
    audit["replay"] = {
        "path": _display_path(replay_path, project_root),
        "sha256": _sha256_file(replay_path),
        "byte_length": replay_path.stat().st_size,
    }
    return audit


def _unavailable_controller_boundary(trace_path: Path, project_root: Path, reason: str) -> dict[str, Any]:
    rows = _read_trace_rows(trace_path) if trace_path.is_file() else []
    checks = {
        "full_causal_overlap": False,
        "trace_frames_consecutive": False,
        "replay_covers_every_observed_trace_state": False,
        "terminal_unobservable_tail_bounded_by_lag": False,
        "both_slots_aligned": False,
        "both_slots_have_controller_evidence": False,
        "both_slots_physical_buttons_exact": False,
        "both_slots_processed_upstream_buttons_exact": False,
        "both_slots_intended_raw_main_stick_exact": False,
        "both_slots_processed_c_stick_within_tolerance": False,
        "both_slots_physical_analog_shoulders_within_tolerance": False,
        "both_slots_first_gameplay_latch_rule_exact": False,
        "both_slots_first_gameplay_command_carryover_rule_exact": False,
        "both_slots_game_start_controller_latch_rule_exact": False,
        "first_policy_frame_mapping_evidence_exact": False,
        "game_start_transport_proof_exact": False,
        "both_slots_no_dispatch_rule_exact": False,
    }
    return {
        "schema_version": CONTROLLER_BOUNDARY_SCHEMA_VERSION,
        "classification": "deliberate decoded-command to libmelee controller adaptation",
        "error": reason,
        "trace": (
            _trace_identity(trace_path, project_root, len(rows))
            if trace_path.is_file()
            else {"path": _display_path(trace_path, project_root), "missing": True, "rows": 0}
        ),
        "alignment": None,
        "slots": {},
        "gate": {"decision": "fail", "checks": checks},
    }


def _audit_controller_boundary_candidate(
    trace_path: Path,
    replay_path: Path,
    project_root: Path,
    *,
    lag_frames: int | None = None,
    game_start_transport_proof: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Represent an invalid auxiliary replay as a failed candidate instead of aborting."""

    try:
        if lag_frames is None:
            return _audit_controller_boundary(
                trace_path,
                replay_path,
                project_root,
                game_start_transport_proof=game_start_transport_proof,
            )
        return _audit_controller_boundary(
            trace_path,
            replay_path,
            project_root,
            lag_frames=lag_frames,
            game_start_transport_proof=game_start_transport_proof,
        )
    except Exception as error:
        return _unavailable_controller_boundary(
            trace_path,
            project_root,
            f"{type(error).__name__}: {error}",
        )


def _audit_mixed_controller_boundary_candidates(
    trace_path: Path,
    replay_paths: Iterable[Path],
    project_root: Path,
    transport_record: Mapping[str, Any],
) -> list[tuple[int, dict[str, Any]]]:
    """Bind every mixed-policy replay candidate to the sealed startup transport proof."""

    game_start_transport_proof = _game_start_transport_proof(transport_record)
    return [
        (
            index,
            _audit_controller_boundary_candidate(
                trace_path,
                path,
                project_root,
                game_start_transport_proof=game_start_transport_proof,
            ),
        )
        for index, path in enumerate(replay_paths)
    ]


def _replay_character_expectation(character: str) -> str:
    """Translate the one libmelee CSS alias used by the replay parser."""

    return _REPLAY_CHARACTER_ALIASES.get(character, character)


def _live_decisive_stock_winner(stocks: Mapping[str, Any]) -> str | None:
    """Return the live winning physical slot for an observed decisive stock-out."""

    try:
        p1_stocks = int(stocks["p1"])
        p2_stocks = int(stocks["p2"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if p1_stocks > 0 and p2_stocks == 0:
        return "p1"
    if p2_stocks > 0 and p1_stocks == 0:
        return "p2"
    return None


def _selected_replay_result_binding(
    replay_paths: list[Path],
    replay_records: list[dict[str, Any]],
    selected_replay_index: int | None,
    project_root: Path,
    request: SlippiMatchRequest,
    live_stocks: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the selected replay's conclusive result to the live terminal stocks."""

    expected_characters = {
        port: _replay_character_expectation(request.character_for_port(port))
        for port in (1, 2)
    }
    checks = {name: False for name in _REPLAY_RESULT_BINDING_REQUIRED_CHECKS}
    live_winner = _live_decisive_stock_winner(live_stocks)
    checks["live_terminal_stocks_are_decisive"] = live_winner is not None
    selected_indices = [
        index
        for index, record in enumerate(replay_records)
        if record.get("tournament_result_replay") is True
    ]
    selected_valid = (
        selected_replay_index is not None
        and selected_indices == [selected_replay_index]
        and 0 <= selected_replay_index < len(replay_paths)
        and selected_replay_index < len(replay_records)
    )
    checks["selected_result_replay_exactly_one"] = selected_valid
    base = {
        "schema_version": REPLAY_RESULT_BINDING_SCHEMA_VERSION,
        "decision": "fail",
        "checks": checks,
        "expected": {
            "stage": request.stage,
            "characters": {f"p{port}": value for port, value in expected_characters.items()},
        },
        "live_terminal": {
            "stocks": {"p1": live_stocks.get("p1"), "p2": live_stocks.get("p2")},
            "winner": live_winner,
        },
        "selected_replay": None,
        "audit": None,
        "outcome": None,
        "winner_port": None,
        "winner": None,
        "error": None,
    }
    if not selected_valid or selected_replay_index is None:
        base["error"] = "replay result binding requires exactly one selected result replay"
        return base

    replay_path = replay_paths[selected_replay_index]
    selected_record = replay_records[selected_replay_index]
    base["selected_replay"] = {
        "path": _display_path(replay_path, project_root),
        "sha256": selected_record.get("sha256"),
        "byte_length": selected_record.get("byte_length"),
    }
    try:
        from melee_policy.integration.replay_result import audit_replay

        audit = audit_replay(
            replay_path,
            expected_stage=request.stage,
            expected_characters=expected_characters,
        )
        audit_identity = audit.get("replay")
        checks["selected_result_replay_identity_exact"] = bool(
            isinstance(audit_identity, Mapping)
            and audit_identity.get("sha256") == selected_record.get("sha256")
            and audit_identity.get("raw_byte_length") == selected_record.get("byte_length")
        )
        checks["replay_audit_passed"] = audit.get("audit_passed") is True
        checks["replay_tournament_result_ready"] = (
            audit.get("tournament_result_ready") is True
        )
        raw_outcome = audit.get("outcome")
        outcome = dict(raw_outcome) if isinstance(raw_outcome, Mapping) else None
        winner_port = outcome.get("winner_port") if outcome is not None else None
        winner = "p1" if winner_port == 1 else "p2" if winner_port == 2 else None
        checks["replay_outcome_complete"] = bool(
            outcome is not None and outcome.get("game_complete") is True
        )
        checks["replay_outcome_conclusive"] = bool(
            outcome is not None and outcome.get("conclusive") is True
        )
        checks["replay_outcome_is_decisive_win"] = bool(
            outcome is not None
            and outcome.get("status") == "win"
            and outcome.get("draw") is False
        )
        checks["replay_winner_port_valid"] = winner is not None
        checks["replay_winner_matches_live_terminal_stocks"] = (
            winner is not None and winner == live_winner
        )
        base.update(
            {
                "decision": "pass" if all(checks.values()) else "fail",
                "audit": audit,
                "outcome": outcome,
                "winner_port": winner_port,
                "winner": winner,
            }
        )
        return base
    except Exception as error:
        base["error"] = f"{type(error).__name__}: {error}"
        return base


def _console_observed_formal_game_end(console: Any) -> bool:
    """Read libmelee's exact event ledger for the most recent step call."""

    from melee.slippstream import EventType

    return EventType.GAME_END in getattr(console, "_events_this_frame", ())


def _formal_game_end_drain_record(
    *,
    terminal_policy_frame: int | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    return {
        "schema_version": FORMAL_GAME_END_DRAIN_SCHEMA_VERSION,
        "attempted": True,
        "decision": "fail",
        "reason": "drain-not-completed",
        "terminal_policy_frame": terminal_policy_frame,
        "timeout_seconds": timeout_seconds,
        "maximum_step_calls": FORMAL_GAME_END_DRAIN_MAX_STEP_CALLS,
        "step_calls": 0,
        "none_step_results": 0,
        "returned_states": 0,
        "neutral_boundaries_started": 0,
        "neutral_boundaries_committed": 0,
        "neutral_dispatches_by_port": {"p1": 0, "p2": 0},
        "last_returned_game_frame": None,
        "last_returned_menu_state": None,
        "formal_game_end_observed": False,
        "formal_game_end_evidence": "libmelee-console-events-this-frame",
        "policy_inference_calls": 0,
        "policy_inference_calls_by_model": {},
        "policy_inference_counters_observed": False,
        "controller_trace_rows_written": 0,
        "controller_trace_counter_observed": False,
        "game_scope_transport_audit_sealed_before_drain": False,
        "game_scope_transport_gate_passed_before_drain": False,
        "failure_phase": None,
        "error": None,
        "wall_seconds": 0.0,
    }


def _drain_formal_game_end(
    *,
    transport: _ControllerPipeLockstep,
    controllers: Mapping[int, Any],
    terminal_policy_frame: int,
    in_game_menu_states: tuple[Any, ...],
    timeout_seconds: float,
    clock: Callable[[], float] = time.monotonic,
    formal_game_end_probe: Callable[[Any], bool] = _console_observed_formal_game_end,
) -> dict[str, Any]:
    """Reach formal Game End using paired neutral commands and zero inference."""

    if set(controllers) != {1, 2}:
        raise ValueError("formal Game End drain requires physical controllers 1 and 2")
    if timeout_seconds <= 0:
        raise ValueError("formal Game End drain timeout must be positive")
    record = _formal_game_end_drain_record(
        terminal_policy_frame=terminal_policy_frame,
        timeout_seconds=timeout_seconds,
    )
    started_at = clock()
    phase = "seal-game-scope-controller-transport-audit"
    try:
        transport_record, transport_checks = transport.seal_benchmark_audit()
        benchmark_scope = transport_record.get("benchmark_scope", {})
        scope_sealed = (
            isinstance(benchmark_scope, Mapping)
            and benchmark_scope.get("sealed_before_shutdown_drain") is True
        )
        transport_gate_passed = all(transport_checks.values())
        record["game_scope_transport_audit_sealed_before_drain"] = scope_sealed
        record["game_scope_transport_gate_passed_before_drain"] = transport_gate_passed
        if not scope_sealed or not transport_gate_passed:
            record["reason"] = "game-scope-controller-transport-gate-failed-before-drain"
            record["wall_seconds"] = clock() - started_at
            return record

        deadline = started_at + timeout_seconds
        neutral = CanonicalControllerCommand.neutral()
        while record["step_calls"] < FORMAL_GAME_END_DRAIN_MAX_STEP_CALLS:
            if clock() >= deadline:
                record["reason"] = "formal-game-end-drain-timeout"
                break
            phase = "console-step"
            gamestate = transport.step()
            record["step_calls"] += 1
            phase = "formal-game-end-event-probe"
            if formal_game_end_probe(transport.console):
                record["formal_game_end_observed"] = True
                record["decision"] = "pass"
                record["reason"] = "formal-game-end-observed"
                break
            if gamestate is None:
                record["none_step_results"] += 1
                continue
            record["returned_states"] += 1
            menu_state = getattr(gamestate, "menu_state", None)
            record["last_returned_menu_state"] = str(getattr(menu_state, "name", menu_state))
            game_frame = int(getattr(gamestate, "frame", terminal_policy_frame))
            record["last_returned_game_frame"] = game_frame
            if menu_state not in in_game_menu_states:
                record["reason"] = "left-gameplay-without-formal-game-end-event"
                break
            phase = "neutral-boundary-begin"
            transport.begin_boundary(
                reason="mixed-post-stock-out-formal-game-end-drain",
                game_frame=game_frame,
            )
            record["neutral_boundaries_started"] += 1
            for port in (1, 2):
                phase = f"p{port}-neutral-dispatch"
                send_canonical_controller(controllers[port], neutral, flush=False)
                record["neutral_dispatches_by_port"][f"p{port}"] += 1
                phase = f"p{port}-neutral-schedule"
                transport.schedule_next_boundary(
                    port,
                    reason=f"p{port}-post-stock-out-neutral-finalization",
                    game_frame=game_frame,
                )
            phase = "neutral-boundary-commit"
            transport.commit_boundary()
            record["neutral_boundaries_committed"] += 1
        else:
            record["reason"] = "formal-game-end-drain-step-limit"
    except Exception as error:
        record["decision"] = "fail"
        record["reason"] = "formal-game-end-drain-exception"
        record["failure_phase"] = phase
        record["error"] = f"{type(error).__name__}: {error}"
    record["wall_seconds"] = clock() - started_at
    return record


def _collect_replay_records(
    replay_paths: list[Path],
    project_root: Path,
    first_game_frame: int | None,
    last_game_frame: int | None,
) -> list[dict[str, Any]]:
    records = []
    for path in replay_paths:
        validation = (
            _validate_saved_replay(
                path,
                required_first_frame=first_game_frame,
                required_last_frame=last_game_frame,
            )
            if first_game_frame is not None and last_game_frame is not None
            else _validate_saved_replay(path)
        )
        records.append(
            {
                "path": _display_path(path, project_root),
                "byte_length": path.stat().st_size,
                "sha256": _sha256_file(path),
                "validation": validation,
            }
        )
    return records


def _run_console(
    config: dict[str, Any],
    project_root: Path,
    iso_path: Path,
    request: SlippiMatchRequest,
    contract: dict[str, Any],
    source_checks: dict[str, Any],
    windowed_runtime: MimicRuntime,
    session: SlippiAIPolicySession,
    tensorflow_setup: dict[str, Any],
    effective_seed: int,
) -> dict[str, Any]:
    import melee

    windowed_model = request.windowed_model
    slippi_port = request.port_for("slippi-ai")
    windowed_port = 3 - slippi_port
    character_contracts = _validate_character_contracts(request, windowed_runtime)

    output_directory = project_root / config["slippi_integration"]["output_directory"]
    if request.artifact_label is not None:
        output_directory = output_directory / request.artifact_label
    _require_unused_artifact_label(output_directory, request.artifact_label)
    trace_path = output_directory / "controller_trace.jsonl"
    summary_path = output_directory / "summary.json"
    replay_directory = output_directory / "replays"
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
        is_dolphin=True,
        tmp_home_directory=True,
        copy_home_directory=False,
        blocking_input=True,
        polling_mode=True,
        polling_timeout=1.0,
        online_delay=int(contract["mixed_runtime_console_delay_frames"]),
        setup_gecko_codes=True,
        fullscreen=False,
        gfx_backend="",
        disable_audio=False,
        use_exi_inputs=False,
        enable_ffw=False,
        save_replays=True,
        replay_dir=str(replay_directory),
        replay_monthly_folders=False,
        slippi_port=udp_port,
    )
    controllers: dict[int, Any] = {
        port: melee.Controller(console=console, port=port, type=melee.ControllerType.STANDARD)
        for port in (1, 2)
    }
    windowed_policy = _make_windowed_policy(
        windowed_model,
        windowed_runtime,
        windowed_port,
        online_delay_frames=int(contract["mixed_runtime_console_delay_frames"]),
        evaluation_seed=effective_seed,
    )
    windowed_worker = _LatestInferenceWorker(windowed_model.upper(), windowed_policy.infer_snapshot)
    menu_helpers = {1: melee.MenuHelper(), 2: melee.MenuHelper()}
    try:
        characters = {port: melee.Character[request.character_for_port(port)] for port in (1, 2)}
        stage = melee.Stage[request.stage]
    except KeyError as error:
        windowed_worker.close()
        raise ValueError(f"unknown character or stage enum: {error}") from error
    maximum_frames = (
        int(config["integration"]["max_game_frames"])
        if request.max_game_frames is None
        else request.max_game_frames
    )
    menu_timeout = float(config["integration"]["menu_timeout_seconds"])
    replay_finalize_timeout = float(config["integration"]["replay_finalize_timeout_seconds"])
    started_at = time.time()
    trace_stream: TextIO | None = None
    exception: BaseException | None = None
    in_game = False
    natural_game_end = False
    formal_game_end_observed = False
    sudden_death_transition_observed = False
    termination = "not-started"
    processed_frames = 0
    first_game_frame: int | None = None
    last_game_frame: int | None = None
    previous_game_frame: int | None = None
    frame_delta_counts: dict[int, int] = {}
    strict_frame_order = True
    all_outputs_finite = True
    shutdown_method = "not-started"
    windowed_inference_count = 0
    windowed_observe_count = 0
    windowed_inference_seconds: list[float] = []
    command_ages: list[int] = []
    windowed_not_ready_frames: list[int] = []
    first_windowed_ready_frame: int | None = None
    slippi_step_seconds: list[float] = []
    slippi_dispatch_records = 0
    slippi_dispatch_contract_exact = True
    stocks = {"p1": 4, "p2": 4}
    mimic_sent = _neutral_mimic_command()
    mimic_pressed: list[str] = []
    session_metadata: dict[str, Any] = {}
    diagnostics_before_close: dict[str, Any] = {}
    diagnostics_after_close: dict[str, Any] = {}
    observed_context: dict[str, Any] = {
        "requested_stage": request.stage,
        "observed_stage": None,
        "stage_match": False,
        "slots": {},
        "evidence": "first libmelee IN_GAME GameState was not observed",
        "checks": {"stage_exact": False, "both_characters_exact": False},
        "match": False,
    }
    transport: _ControllerPipeLockstep | None = None
    menu_transport_flushes = {1: 0, 2: 0}
    slippi_boundary_schedule_frames: list[int] = []
    no_frame_watchdog = InGameNoFrameWatchdog()
    formal_game_end_drain = _formal_game_end_drain_record(
        terminal_policy_frame=None,
        timeout_seconds=replay_finalize_timeout,
    )
    formal_game_end_drain.update(
        {
            "attempted": False,
            "decision": "not-required",
            "reason": "formal-game-end-drain-not-requested",
        }
    )

    def interrupt(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGINT, interrupt)
    try:
        trace_stream = trace_path.open("w", encoding="utf-8")
        session.start()
        session_metadata = session.metadata()
        if not _launch_and_connect_attested_dolphin(console, iso_path):
            raise RuntimeError("libmelee could not connect to Slippi Dolphin")
        if not all(controller.connect() for controller in controllers.values()):
            raise RuntimeError("libmelee could not connect both virtual controllers")
        transport = _ControllerPipeLockstep.install(console, controllers)
        controllers = dict(transport.controllers)
        transport.prime()
        print(
            f"P1={request.model_for_port(1).upper()} {request.player_1_character} | "
            f"P2={request.model_for_port(2).upper()} {request.player_2_character} | "
            f"stage={request.stage} | inference=frame-exact",
            flush=True,
        )

        while processed_frames < maximum_frames:
            gamestate = transport.step()
            formal_game_end_observed = (
                formal_game_end_observed or _console_observed_formal_game_end(console)
            )
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
                for port in (1, 2):
                    menu_helpers[port].menu_helper_simple(
                        gamestate,
                        controllers[port],
                        characters[port],
                        stage,
                        cpu_level=0,
                        autostart=port == 2,
                        frozen_stadium=True,
                    )
                for controller in controllers.values():
                    controller.flush()
                transport.commit_boundary()
                for port in (1, 2):
                    menu_transport_flushes[port] += 1
                continue

            if formal_game_end_observed:
                natural_game_end = True
                termination = "natural-game-end"
                break

            in_game = True
            game_frame = int(gamestate.frame)
            if first_game_frame is None:
                first_game_frame = game_frame
                if game_frame != FIRST_POLICY_FRAME:
                    raise RuntimeError(
                        f"first Slippi-AI policy frame must be {FIRST_POLICY_FRAME}, got {game_frame}"
                    )
            if previous_game_frame is not None:
                delta = game_frame - previous_game_frame
                frame_delta_counts[delta] = frame_delta_counts.get(delta, 0) + 1
                if delta != 1:
                    strict_frame_order = False
                    raise RuntimeError(
                        f"rendered policy frame stream is not exact: {previous_game_frame} to {game_frame}"
                    )
            previous_game_frame = game_frame
            last_game_frame = game_frame
            _require_exact_player_ports(gamestate, game_frame)
            if processed_frames == 0:
                observed_context = _observed_match_context(gamestate, request)
                if not observed_context["match"]:
                    raise RuntimeError(
                        f"first-frame requested/observed match context mismatch: {observed_context}"
                    )

            ready, warmup_rng_unchanged = _observe_windowed_policy(
                windowed_model,
                windowed_policy,
                gamestate,
            )
            windowed_observe_count += 1
            if ready and first_windowed_ready_frame is None:
                first_windowed_ready_frame = game_frame
            if not ready:
                windowed_not_ready_frames.append(game_frame)
            if ready:
                windowed_worker.submit(game_frame, windowed_policy.snapshot())

            slippi_started = time.perf_counter()
            slippi_command = session.step(gamestate)
            slippi_step_seconds.append(time.perf_counter() - slippi_started)
            windowed_result = windowed_worker.wait_completed(game_frame) if ready else None
            windowed_source_frame: int | None = None
            windowed_prediction: Any = None
            if windowed_result is not None:
                windowed_source_frame, windowed_prediction, duration = windowed_result
                if not _finite_prediction(windowed_prediction):
                    all_outputs_finite = False
                    raise RuntimeError(f"non-finite {windowed_model} output at game frame {game_frame}")
                windowed_inference_count += 1
                windowed_inference_seconds.append(duration)
                command_ages.append(game_frame - windowed_source_frame)

            slippi_source_frame = (
                None
                if processed_frames < int(contract["effective_policy_delay_frames"])
                else game_frame - int(contract["effective_policy_delay_frames"])
            )
            slippi_dispatch: dict[str, Any] | None = None
            dispatch_called: dict[int, bool] = {1: False, 2: False}
            boundary_flush_ports: list[int] = []
            transport.begin_boundary(reason="gameplay", game_frame=game_frame)
            for port in (1, 2):
                model = request.model_for_port(port)
                if model == "slippi-ai":
                    slippi_dispatch = _dispatch_slippi_command(controllers[port], slippi_command)
                    dispatch_called[port] = True
                    boundary_flush_ports.append(port)
                    slippi_dispatch_records += 1
                    flush = cast(dict[str, Any], slippi_dispatch["flush"])
                    slippi_dispatch_contract_exact = slippi_dispatch_contract_exact and (
                        flush.get("upstream_native_sender_count") == 0
                        and flush.get("project_adapter_count") == 0
                        and flush.get("project_boundary_deliberately_flushes_once") is False
                    )
                elif model == "mimic":
                    mimic_runtime = cast(MimicRuntime, windowed_runtime)
                    if windowed_result is not None:
                        mimic_sent, mimic_pressed, _button_names = cast(
                            MimicLivePolicy, windowed_policy
                        ).decode_and_press(
                            controllers[port],
                            windowed_prediction,
                            mimic_runtime.state.prev_sent,
                            temperature=float(config["mimic"]["temperature"]),
                            top_k=int(config["mimic"]["top_k"]),
                            top_p=float(config["mimic"]["top_p"]),
                        )
                    else:
                        _send_mimic_controller_inputs(controllers[port], mimic_sent)
                    cast(MimicLivePolicy, windowed_policy).record_decoded_command(game_frame, mimic_sent)
                    dispatch_called[port] = True
                else:
                    raise ValueError(f"unsupported policy: {model}")

            for port in boundary_flush_ports:
                if request.model_for_port(port) == "slippi-ai":
                    transport.schedule_next_boundary(
                        port,
                        reason="slippi-ai-queued-command",
                        game_frame=game_frame,
                    )
                    slippi_boundary_schedule_frames.append(game_frame)
            transport.commit_boundary()

            slot_payloads: dict[str, dict[str, Any]] = {}
            for port in (1, 2):
                model = request.model_for_port(port)
                player = gamestate.players[port]
                stocks[f"p{port}"] = int(player.stock)
                command: dict[str, Any] | None
                if model == "slippi-ai":
                    command = slippi_command.as_dict()
                    inference = {
                        "called": True,
                        "source_frame": slippi_source_frame,
                        "command_age_frames": (
                            None if slippi_source_frame is None else game_frame - slippi_source_frame
                        ),
                        "native_dummy_prefix": slippi_source_frame is None,
                    }
                elif model == "mimic":
                    mimic_policy = cast(MimicLivePolicy, windowed_policy)
                    command = {str(key): float(value) for key, value in mimic_sent.items()}
                    command["pressed"] = list(mimic_pressed)
                    inference = {
                        "called": windowed_result is not None,
                        "source_frame": windowed_source_frame,
                        "command_age_frames": (
                            None if windowed_source_frame is None else game_frame - windowed_source_frame
                        ),
                        "previous_executed_controller_frame": mimic_policy.previous_executed_frame,
                    }
                else:
                    raise ValueError(f"unsupported policy: {model}")
                slot_payloads[f"p{port}"] = {
                    "player_state": _player_state(player),
                    "command": command,
                    "inference": inference,
                }
                if model == "slippi-ai":
                    if slippi_dispatch is None:
                        raise AssertionError("Slippi-AI command has no controller dispatch record")
                    slippi_dispatch["console_step_preamble_flush_scheduled"] = True
                    slippi_dispatch["equivalent_next_step_boundary_flush"] = True
                    slot_payloads[f"p{port}"]["controller_dispatch"] = slippi_dispatch
                elif model == "mimic":
                    slot_payloads[f"p{port}"]["controller_dispatch"] = {
                        "called": dispatch_called[port],
                        "sender": "released MIMIC decode_and_press or exact repeat sender",
                        "explicit_flush": True,
                        "implicit_console_step_flush_suppressed": True,
                    }
                else:
                    raise ValueError(f"unsupported policy: {model}")
            trace_row = _build_trace_row(game_frame, request, slot_payloads)
            trace_stream.write(json.dumps(trace_row, sort_keys=True) + "\n")
            trace_stream.flush()
            processed_frames += 1
            if processed_frames % 60 == 0:
                print(
                    f"frame {game_frame}: P1 {stocks['p1']} stocks, P2 {stocks['p2']} stocks",
                    flush=True,
                )
            if has_decisive_zero_stock(gamestate, (1, 2)):
                natural_game_end = True
                if request.require_formal_game_end:
                    termination = "decisive-stock-out-awaiting-formal-game-end"
                    slippi_inference_before_drain = int(
                        session.diagnostics().get("frames_total", -1)
                    )
                    windowed_inference_before_drain = windowed_inference_count
                    trace_rows_before_drain = processed_frames
                    formal_game_end_drain = _formal_game_end_drain_record(
                        terminal_policy_frame=game_frame,
                        timeout_seconds=replay_finalize_timeout,
                    )
                    formal_game_end_drain["reason"] = "formal-game-end-drain-in-progress"
                    formal_game_end_drain = _drain_formal_game_end(
                        transport=transport,
                        controllers=controllers,
                        terminal_policy_frame=game_frame,
                        in_game_menu_states=(melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH),
                        timeout_seconds=replay_finalize_timeout,
                    )
                    slippi_inference_after_drain = int(
                        session.diagnostics().get("frames_total", -1)
                    )
                    slippi_inference_delta = (
                        slippi_inference_after_drain - slippi_inference_before_drain
                    )
                    windowed_inference_delta = (
                        windowed_inference_count - windowed_inference_before_drain
                    )
                    formal_game_end_drain["policy_inference_calls_by_model"] = {
                        "slippi-ai": slippi_inference_delta,
                        windowed_model: windowed_inference_delta,
                    }
                    formal_game_end_drain["policy_inference_counters_observed"] = (
                        slippi_inference_before_drain >= 0 and slippi_inference_after_drain >= 0
                    )
                    formal_game_end_drain["policy_inference_calls"] = (
                        slippi_inference_delta + windowed_inference_delta
                    )
                    formal_game_end_drain["controller_trace_counter_observed"] = True
                    formal_game_end_drain["controller_trace_rows_written"] = (
                        processed_frames - trace_rows_before_drain
                    )
                    formal_game_end_observed = bool(
                        formal_game_end_drain["formal_game_end_observed"]
                    )
                    if formal_game_end_drain["decision"] != "pass":
                        raise RuntimeError(
                            "decisive stock-out did not reach formal Slippi Game End before "
                            f"shutdown: {formal_game_end_drain['reason']}; "
                            f"{formal_game_end_drain['error']}"
                        )
                termination = "natural-game-end"
                break
        if in_game and processed_frames >= maximum_frames:
            termination = "frame-limit-graceful-stop"
    except BaseException as caught:
        exception = caught
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        try:
            diagnostics_before_close = session.diagnostics()
            session.close()
            diagnostics_after_close = session.diagnostics()
        except BaseException as session_error:
            if exception is None:
                exception = session_error
        try:
            windowed_worker.close()
        except BaseException as worker_error:
            if exception is None:
                exception = worker_error
        if windowed_worker.failure is not None and exception is None:
            exception = RuntimeError(f"{windowed_worker.name} inference failed: {windowed_worker.failure}")
        if trace_stream is not None:
            trace_stream.close()
        shutdown_method = _stop_console(
            console,
            replay_finalize_timeout,
        )

    replay_paths = sorted(
        path
        for path in replay_directory.rglob("*.slp")
        if path.is_file() and path.resolve() not in existing_replays
    )
    replay_records = _collect_replay_records(
        replay_paths, project_root, first_game_frame, last_game_frame
    )
    transport_record = transport.audit_record() if transport is not None else {"installed": False}
    boundary_candidates = _audit_mixed_controller_boundary_candidates(
        trace_path,
        replay_paths,
        project_root,
        transport_record,
    )
    passing_boundaries = [
        (index, boundary)
        for index, boundary in boundary_candidates
        if cast(dict[str, Any], boundary.get("gate", {})).get("decision") == "pass"
    ]
    selected_replay_index = passing_boundaries[0][0] if len(passing_boundaries) == 1 else None
    for index, record in enumerate(replay_records):
        selected = index == selected_replay_index
        record["tournament_result_replay"] = selected
        record["role"] = (
            "base-game-result"
            if selected
            else "sudden-death-transition-auxiliary"
            if sudden_death_transition_observed
            else "unexpected-auxiliary"
        )
    replay_parseable = selected_replay_index is not None and bool(
        replay_records[selected_replay_index]["validation"]["libmelee_parseable"]
    )
    replay_result_binding = _selected_replay_result_binding(
        replay_paths,
        replay_records,
        selected_replay_index,
        project_root,
        request,
        stocks,
    )
    replay_result_checks = cast(dict[str, bool], replay_result_binding["checks"])
    replay_outcome = replay_result_binding.get("outcome")
    replay_winner = replay_result_binding.get("winner")
    live_decisive_stock_winner = cast(dict[str, Any], replay_result_binding["live_terminal"])[
        "winner"
    ]
    if len(passing_boundaries) == 1:
        controller_boundary = passing_boundaries[0][1]
    else:
        controller_boundary = _unavailable_controller_boundary(
            trace_path,
            project_root,
            "controller boundary audit requires exactly one replay covering the base-game trace",
        )
    boundary_checks = cast(dict[str, bool], controller_boundary["gate"]["checks"])
    slippi_diagnostics = diagnostics_before_close
    controller_contract = cast(dict[str, Any], session_metadata.get("controller_contract", {}))
    windowed_timing_checks = _windowed_exact_timing_checks(
        windowed_model,
        processed_frames=processed_frames,
        inference_count=windowed_inference_count,
        command_ages=command_ages,
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
    gate_checks = {
        "exact_mode_only": contract["exact_mode_only"],
        "entered_gameplay": in_game,
        "processed_policy_frames": processed_frames > 0,
        "first_policy_frame_minus_123": first_game_frame == FIRST_POLICY_FRAME,
        "strict_consecutive_frame_order": strict_frame_order
        and all(delta == 1 for delta in frame_delta_counts),
        "slippi_policy_called_once_per_processed_frame": int(slippi_diagnostics.get("frames_total", -1))
        == processed_frames,
        "no_recurrent_frame_drop": int(slippi_diagnostics.get("frames_total", -1)) == processed_frames,
        "slippi_current_frame_inference_barrier": (
            int(slippi_diagnostics.get("current_frame_inference_barriers", -1)) == processed_frames
            and slippi_diagnostics.get("current_frame_inference_barrier_every_frame") is True
        ),
        "capture_matches_native_decoder": int(slippi_diagnostics.get("capture_decoder_mismatches", -1)) == 0
        and int(slippi_diagnostics.get("capture_decoder_assertions", -1)) == processed_frames,
        "mixed_console_delay_zero": contract["mixed_runtime_console_delay_frames"] == 0,
        "formal_game_end_requirement_met": (
            not request.require_formal_game_end or formal_game_end_observed
        ),
        "formal_game_end_drain_passed_if_attempted": (
            formal_game_end_drain["decision"] in {"pass", "not-required"}
        ),
        "formal_game_end_drain_no_policy_inference": (
            formal_game_end_drain["attempted"] is False
            or (
                formal_game_end_drain["policy_inference_counters_observed"] is True
                and formal_game_end_drain["policy_inference_calls"] == 0
            )
        ),
        "formal_game_end_drain_wrote_no_controller_trace_rows": (
            formal_game_end_drain["attempted"] is False
            or (
                formal_game_end_drain["controller_trace_counter_observed"] is True
                and formal_game_end_drain["controller_trace_rows_written"] == 0
            )
        ),
        "formal_game_end_drain_neutral_boundaries_exactly_paired": (
            formal_game_end_drain["attempted"] is False
            or (
                formal_game_end_drain["decision"] == "pass"
                and formal_game_end_drain["neutral_boundaries_started"]
                == formal_game_end_drain["neutral_boundaries_committed"]
                == formal_game_end_drain["neutral_dispatches_by_port"]["p1"]
                == formal_game_end_drain["neutral_dispatches_by_port"]["p2"]
            )
        ),
        "game_scope_transport_sealed_before_formal_game_end_drain": (
            formal_game_end_drain["attempted"] is False
            or formal_game_end_drain["game_scope_transport_audit_sealed_before_drain"] is True
        ),
        "selected_release_delay_contract_exact": (
            int(contract["checkpoint_delay_frames"])
            - int(contract["mixed_runtime_console_delay_frames"])
            == int(contract["effective_policy_delay_frames"])
        ),
        **(
            {
                "delay_contract_21_minus_0_equals_21": (
                    POLICY_DELAY_FRAMES
                    - int(contract["mixed_runtime_console_delay_frames"])
                    == int(contract["effective_policy_delay_frames"])
                    == MIXED_EFFECTIVE_POLICY_DELAY_FRAMES
                )
            }
            if int(contract["checkpoint_delay_frames"]) == POLICY_DELAY_FRAMES
            else {}
        ),
        "all_outputs_finite": all_outputs_finite,
        "character_contracts": all(bool(value["match"]) for value in character_contracts.values()),
        "first_frame_stage_exact": bool(observed_context["checks"]["stage_exact"]),
        "first_frame_two_standard_ports_exact": bool(observed_context["checks"]["two_standard_ports_exact"]),
        "first_frame_characters_exact": bool(observed_context["checks"]["both_characters_exact"]),
        "windowed_observed_once_per_processed_frame": windowed_observe_count == processed_frames,
        **windowed_timing_checks,
        "trace_slot_keyed": True,
        "saved_replay_parseable": replay_parseable,
        "exactly_one_tournament_result_replay": selected_replay_index is not None,
        **(
            {
                f"replay_result_binding.{name}": passed
                for name, passed in replay_result_checks.items()
            }
            if request.require_formal_game_end
            else {}
        ),
        "no_unexpected_auxiliary_replays": len(replay_paths) == 1
        or (sudden_death_transition_observed and len(replay_paths) == 2),
        "controller_boundary_full_causal_overlap": boundary_checks["full_causal_overlap"],
        "controller_boundary_gate_pass": controller_boundary["gate"]["decision"] == "pass",
        "controller_boundary_trace_consecutive": boundary_checks["trace_frames_consecutive"],
        "controller_boundary_replay_covers_trace_states": boundary_checks[
            "replay_covers_every_observed_trace_state"
        ],
        "controller_boundary_terminal_tail_bounded": boundary_checks[
            "terminal_unobservable_tail_bounded_by_lag"
        ],
        "controller_boundary_both_slots_aligned": boundary_checks["both_slots_aligned"],
        "controller_boundary_both_slots_have_evidence": boundary_checks[
            "both_slots_have_controller_evidence"
        ],
        "controller_boundary_physical_buttons_exact": boundary_checks["both_slots_physical_buttons_exact"],
        "controller_boundary_processed_upstream_buttons_exact": boundary_checks[
            "both_slots_processed_upstream_buttons_exact"
        ],
        "controller_boundary_raw_main_exact": boundary_checks["both_slots_intended_raw_main_stick_exact"],
        "slippi_source_exact_deferred_flush": (
            controller_contract.get("native_sender_flushes") is False
            and controller_contract.get("adapter_flush_mode") == "call-site-configurable"
            and controller_contract.get("source_exact_deferred_flush_supported") is True
        ),
        "slippi_dispatch_recorded_per_frame": slippi_dispatch_records == processed_frames,
        "slippi_dispatch_contract_exact": slippi_dispatch_contract_exact,
        **transport_checks,
        "slippi_next_step_boundary_schedule_exact": (
            slippi_boundary_schedule_frames
            == list(range(FIRST_POLICY_FRAME, FIRST_POLICY_FRAME + processed_frames))
        ),
        "tensorflow_cpu_only": tensorflow_setup.get("physical_gpu_count") == 0
        and tensorflow_setup.get("logical_gpu_count") == 0,
        "reproducibility_inputs_hashed": _reproducibility_inputs_hashed(
            contract.get("reproducibility")
        ),
        "runtime_environment_lock_gate_passed": cast(
            dict[str, Any],
            cast(dict[str, Any], contract["reproducibility"])["runtime_environment"],
        )["dependency_lock_validation"]["environment_lock_gate_passed"]
        is True,
        "natural_game_end_requirement_met": _natural_end_requirement_met(
            required=request.require_natural_end,
            observed=natural_game_end,
        ),
    }
    if exception is None and not all(gate_checks.values()):
        exception = RuntimeError(
            "Slippi-AI integration gate failed: "
            f"{[name for name, passed in gate_checks.items() if not passed]}"
        )

    slot_summaries: dict[str, dict[str, Any]] = {}
    windowed_identity = _windowed_identity(windowed_model, windowed_runtime, source_checks)
    windowed_identity["policy_rng"] = windowed_policy.policy_rng
    for port in (1, 2):
        slot = f"p{port}"
        model = request.model_for_port(port)
        if model == "slippi-ai":
            identity = session_metadata
            timing = {
                "count": len(slippi_step_seconds),
                "mean_seconds": (
                    sum(slippi_step_seconds) / len(slippi_step_seconds) if slippi_step_seconds else None
                ),
                "maximum_seconds": max(slippi_step_seconds) if slippi_step_seconds else None,
            }
            diagnostics: dict[str, Any] | None = {
                "before_close": diagnostics_before_close,
                "after_close": diagnostics_after_close,
            }
        else:
            identity = windowed_identity
            timing = {
                "count": windowed_inference_count,
                "mean_seconds": (
                    sum(windowed_inference_seconds) / len(windowed_inference_seconds)
                    if windowed_inference_seconds
                    else None
                ),
                "maximum_seconds": (max(windowed_inference_seconds) if windowed_inference_seconds else None),
                "mean_command_age_frames": (sum(command_ages) / len(command_ages) if command_ages else None),
                "maximum_command_age_frames": max(command_ages) if command_ages else None,
            }
            diagnostics = None
        slot_summaries[slot] = {
            "port": port,
            "model": model,
            "requested_character": request.character_for_port(port),
            "observed_character": cast(dict[str, Any], observed_context.get("slots", {}))
            .get(slot, {})
            .get("observed_character"),
            "observed_costume": cast(dict[str, Any], observed_context.get("slots", {}))
            .get(slot, {})
            .get("observed_costume"),
            "character_contract": character_contracts[slot],
            "observed_selection": cast(dict[str, Any], observed_context.get("slots", {})).get(slot),
            "identity": identity,
            "timing": timing,
            "diagnostics": diagnostics,
        }

    result = "pass" if exception is None and all(gate_checks.values()) else "fail"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "classification": (
            "frame-exact rendered integration infrastructure with a deliberate decoded-command "
            "controller adaptation, not E003 evidence"
        ),
        "result": result,
        "error": None if exception is None else f"{type(exception).__name__}: {exception}",
        "configuration": {
            "seed": effective_seed,
            "device": config["integration"]["device"],
            "stage": request.stage,
            "observed_match_context": observed_context,
            "maximum_game_frames": maximum_frames,
            "require_natural_end": request.require_natural_end,
            "require_formal_game_end": request.require_formal_game_end,
            "local_css_costume_assignment": {
                "mode": "console-auto-assigned",
                "controlled_by_launcher": False,
                "observed_first_game_state": {
                    slot: cast(dict[str, Any], observed_context.get("slots", {}))
                    .get(slot, {})
                    .get("observed_costume")
                    for slot in ("p1", "p2")
                },
            },
            "replay_finalize_timeout_seconds": float(
                config["integration"]["replay_finalize_timeout_seconds"]
            ),
            "inference_mode": contract["inference_mode"],
            "blocking_input": True,
            "mixed_runtime_console_delay_frames": contract["mixed_runtime_console_delay_frames"],
            "upstream_eval_two_console_delay_frames": contract["upstream_eval_two_console_delay_frames"],
            "checkpoint_policy_delay_frames": contract["checkpoint_delay_frames"],
            "effective_policy_delay_frames": contract["effective_policy_delay_frames"],
            "controller_replay_lag_frames": CONTROLLER_REPLAY_LAG_FRAMES,
            "controller_boundary": "deliberate decoded-command to libmelee adaptation",
            "slippi_ai_dispatch": "source-exact queued command; no explicit sender flush",
            "windowed_dispatch": (
                "native one-flush timing retained; project transport explicitly neutralizes analog R"
            ),
            "graphics": "rendered",
            "audio": "enabled",
        },
        "slots": slot_summaries,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "libmelee_module": str(Path(melee.__file__).resolve()),
            "emulator_version": emulator_version,
            "tensorflow": tensorflow_setup,
        },
        "reproducibility": contract["reproducibility"],
        "emulator_application": emulator_application,
        "game_image": _game_image_record(iso_path),
        "controller_boundary_audit": controller_boundary,
        "execution": {
            "entered_gameplay": in_game,
            "processed_policy_frames": processed_frames,
            "windowed_observe_count": windowed_observe_count,
            "windowed_not_ready_frames": windowed_not_ready_frames,
            "first_windowed_ready_frame": first_windowed_ready_frame,
            "first_game_frame": first_game_frame,
            "last_game_frame": last_game_frame,
            "formal_game_end_observed": formal_game_end_observed,
            "formal_game_end_drain": formal_game_end_drain,
            "frame_delta_counts": {str(delta): count for delta, count in sorted(frame_delta_counts.items())},
            "strict_frame_order": strict_frame_order,
            "all_outputs_finite": all_outputs_finite,
            "last_stocks": stocks,
            "winner": replay_winner,
            "winner_source": "selected-replay-derived-outcome",
            "replay_outcome": replay_outcome,
            "live_decisive_stock_winner": live_decisive_stock_winner,
            "replay_result_binding": replay_result_binding,
            "wall_seconds": time.time() - started_at,
            "natural_game_end": natural_game_end,
            "sudden_death_transition_observed": sudden_death_transition_observed,
            "termination": termination,
            "shutdown_method": shutdown_method,
            "controller_transport": {
                **transport_record,
                "menu_flushes": {f"p{port}": menu_transport_flushes[port] for port in (1, 2)},
                "slippi_next_step_boundary_scheduled_from_frames": (slippi_boundary_schedule_frames),
                "policy_for_auxiliary_poll": "no inference, dispatch, or synthetic reflush",
            },
        },
        "artifacts": {
            "trace": controller_boundary["trace"],
            "replays": replay_records,
        },
        "gate": {"decision": result, "checks": gate_checks},
    }
    _write_json(summary_path, summary)
    if exception is not None:
        raise RuntimeError(summary["error"])
    return summary


def _run_dual_slippi_console(
    config: dict[str, Any],
    project_root: Path,
    iso_path: Path,
    request: SlippiMatchRequest,
    contract: dict[str, Any],
    sessions: dict[int, SlippiAIPolicySession],
    tensorflow_setup: dict[str, Any],
    effective_seed: int,
) -> dict[str, Any]:
    """Run two independent pinned Slippi-AI sessions with a shared render barrier."""

    import melee

    if sessions[1] is sessions[2]:
        raise RuntimeError("dual Slippi-AI slots unexpectedly share one policy session")
    dual_console_delay_frames = UPSTREAM_EVAL_TWO_CONSOLE_DELAY_FRAMES
    dual_effective_policy_delay_frames = POLICY_DELAY_FRAMES - dual_console_delay_frames
    dual_controller_replay_lag_frames = dual_console_delay_frames + NEXT_FRAME_CONTROLLER_BOUNDARY_FRAMES

    output_directory = project_root / config["slippi_integration"]["output_directory"]
    if request.artifact_label is not None:
        output_directory = output_directory / request.artifact_label
    _require_unused_artifact_label(output_directory, request.artifact_label)
    trace_path = output_directory / "controller_trace.jsonl"
    summary_path = output_directory / "summary.json"
    replay_directory = output_directory / "replays"
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
        is_dolphin=True,
        tmp_home_directory=True,
        copy_home_directory=False,
        blocking_input=True,
        polling_mode=True,
        polling_timeout=1.0,
        online_delay=dual_console_delay_frames,
        setup_gecko_codes=True,
        fullscreen=False,
        gfx_backend="",
        disable_audio=False,
        use_exi_inputs=False,
        enable_ffw=False,
        save_replays=True,
        replay_dir=str(replay_directory),
        replay_monthly_folders=False,
        slippi_port=udp_port,
    )
    controllers: dict[int, Any] = {
        port: melee.Controller(console=console, port=port, type=melee.ControllerType.STANDARD)
        for port in (1, 2)
    }
    menu_helpers = {1: melee.MenuHelper(), 2: melee.MenuHelper()}
    try:
        characters = {port: melee.Character[request.character_for_port(port)] for port in (1, 2)}
        stage = melee.Stage[request.stage]
    except KeyError as error:
        raise ValueError(f"unknown character or stage enum: {error}") from error

    started_session_ports: list[int] = []
    try:
        for port in (1, 2):
            sessions[port].start()
            started_session_ports.append(port)
    except BaseException:
        for port in reversed(started_session_ports):
            sessions[port].close()
        raise
    session_workers = {
        port: _LatestInferenceWorker(f"SLIPPI-AI-P{port}", sessions[port].step) for port in (1, 2)
    }
    session_metadata = {port: sessions[port].metadata() for port in (1, 2)}
    maximum_frames = (
        int(config["integration"]["max_game_frames"])
        if request.max_game_frames is None
        else request.max_game_frames
    )
    menu_timeout = float(config["integration"]["menu_timeout_seconds"])
    started_at = time.time()
    trace_stream: TextIO | None = None
    exception: BaseException | None = None
    in_game = False
    natural_game_end = False
    sudden_death_transition_observed = False
    termination = "not-started"
    shutdown_method = "not-started"
    processed_frames = 0
    first_game_frame: int | None = None
    last_game_frame: int | None = None
    previous_game_frame: int | None = None
    frame_delta_counts: dict[int, int] = {}
    observed_context: dict[str, Any] | None = None
    dispatch_counts = {1: 0, 2: 0}
    session_step_counts = {1: 0, 2: 0}
    session_step_ages: dict[int, list[int]] = {1: [], 2: []}
    stocks = {"p1": 4, "p2": 4}
    diagnostics_before_close: dict[int, dict[str, Any]] = {}
    diagnostics_after_close: dict[int, dict[str, Any]] = {}
    transport: _ControllerPipeLockstep | None = None
    menu_transport_flushes = {1: 0, 2: 0}
    slippi_boundary_schedule_frames: list[int] = []
    no_frame_watchdog = InGameNoFrameWatchdog()

    def interrupt(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGINT, interrupt)
    try:
        trace_stream = trace_path.open("w", encoding="utf-8")
        if not _launch_and_connect_attested_dolphin(console, iso_path):
            raise RuntimeError("libmelee could not connect to Slippi Dolphin")
        if not all(controller.connect() for controller in controllers.values()):
            raise RuntimeError("libmelee could not connect both virtual controllers")
        transport = _ControllerPipeLockstep.install(console, controllers)
        controllers = dict(transport.controllers)
        transport.prime()

        while processed_frames < maximum_frames:
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
                for port in (1, 2):
                    menu_helpers[port].menu_helper_simple(
                        gamestate,
                        controllers[port],
                        characters[port],
                        stage,
                        cpu_level=0,
                        autostart=port == 2,
                        frozen_stadium=True,
                    )
                    controllers[port].flush()
                    menu_transport_flushes[port] += 1
                transport.commit_boundary()
                continue

            in_game = True
            game_frame = int(gamestate.frame)
            if first_game_frame is None:
                first_game_frame = game_frame
                if game_frame != FIRST_POLICY_FRAME:
                    raise RuntimeError(
                        f"first Slippi-AI policy frame must be {FIRST_POLICY_FRAME}, got {game_frame}"
                    )
                observed_context = _observed_match_context(gamestate, request)
                if not observed_context["match"]:
                    raise RuntimeError(
                        f"first-frame requested/observed match context mismatch: {observed_context}"
                    )
            if previous_game_frame is not None:
                delta = game_frame - previous_game_frame
                frame_delta_counts[delta] = frame_delta_counts.get(delta, 0) + 1
                if delta != 1:
                    raise RuntimeError(
                        f"rendered policy frame stream is not exact: {previous_game_frame} to {game_frame}"
                    )
            previous_game_frame = game_frame
            last_game_frame = game_frame
            _require_exact_player_ports(gamestate, game_frame)

            for port in (1, 2):
                session_workers[port].submit(game_frame, gamestate)
            results = {port: session_workers[port].wait_completed(game_frame) for port in (1, 2)}
            # Both native recurrent updates and both session barriers have now completed.
            transport.begin_boundary(reason="gameplay", game_frame=game_frame)
            slot_payloads: dict[str, dict[str, Any]] = {}
            for port in (1, 2):
                source_frame, command, _duration = results[port]
                session_step_counts[port] += 1
                session_step_ages[port].append(game_frame - source_frame)
                dispatch = _dispatch_slippi_command(controllers[port], command)
                dispatch["console_step_preamble_flush_scheduled"] = True
                dispatch["equivalent_next_step_boundary_flush"] = True
                dispatch_counts[port] += 1
                player = gamestate.players[port]
                stocks[f"p{port}"] = int(player.stock)
                slippi_source_frame = (
                    None
                    if processed_frames < dual_effective_policy_delay_frames
                    else game_frame - dual_effective_policy_delay_frames
                )
                slot_payloads[f"p{port}"] = {
                    "player_state": _player_state(player),
                    "command": command.as_dict(),
                    "inference": {
                        "called": True,
                        "source_frame": slippi_source_frame,
                        "command_age_frames": (
                            None if slippi_source_frame is None else game_frame - slippi_source_frame
                        ),
                        "native_dummy_prefix": slippi_source_frame is None,
                        "current_recurrent_source_frame": source_frame,
                        "current_recurrent_command_age_frames": game_frame - source_frame,
                    },
                    "controller_dispatch": dispatch,
                }
                transport.schedule_next_boundary(
                    port,
                    reason="slippi-ai-queued-command",
                    game_frame=game_frame,
                )
            transport.commit_boundary()
            slippi_boundary_schedule_frames.append(game_frame)
            trace_stream.write(
                json.dumps(_build_trace_row(game_frame, request, slot_payloads), sort_keys=True) + "\n"
            )
            trace_stream.flush()
            processed_frames += 1
            if has_decisive_zero_stock(gamestate, (1, 2)):
                natural_game_end = True
                termination = "natural-game-end"
                break
        if in_game and processed_frames >= maximum_frames:
            termination = "frame-limit-graceful-stop"
    except BaseException as caught:
        exception = caught
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        for port in (1, 2):
            try:
                session_workers[port].close()
            except BaseException as worker_error:
                if exception is None:
                    exception = worker_error
            if session_workers[port].failure is not None and exception is None:
                exception = RuntimeError(
                    f"Slippi-AI P{port} inference failed: {session_workers[port].failure}"
                )
            try:
                diagnostics_before_close[port] = sessions[port].diagnostics()
                sessions[port].close()
                diagnostics_after_close[port] = sessions[port].diagnostics()
            except BaseException as session_error:
                if exception is None:
                    exception = session_error
        if trace_stream is not None:
            trace_stream.close()
        shutdown_method = _stop_console(
            console,
            float(config["integration"]["replay_finalize_timeout_seconds"]),
        )

    replay_paths = sorted(
        path
        for path in replay_directory.rglob("*.slp")
        if path.is_file() and path.resolve() not in existing_replays
    )
    replay_records = _collect_replay_records(
        replay_paths, project_root, first_game_frame, last_game_frame
    )
    boundary_candidates = [
        (
            index,
            _audit_controller_boundary_candidate(
                trace_path,
                path,
                project_root,
                lag_frames=dual_controller_replay_lag_frames,
            ),
        )
        for index, path in enumerate(replay_paths)
    ]
    passing_boundaries = [
        (index, boundary)
        for index, boundary in boundary_candidates
        if cast(dict[str, Any], boundary.get("gate", {})).get("decision") == "pass"
    ]
    selected_replay_index = passing_boundaries[0][0] if len(passing_boundaries) == 1 else None
    for index, record in enumerate(replay_records):
        selected = index == selected_replay_index
        record["tournament_result_replay"] = selected
        record["role"] = (
            "base-game-result"
            if selected
            else "sudden-death-transition-auxiliary"
            if sudden_death_transition_observed
            else "unexpected-auxiliary"
        )
    controller_boundary = (
        passing_boundaries[0][1]
        if len(passing_boundaries) == 1
        else _unavailable_controller_boundary(
            trace_path,
            project_root,
            "controller boundary audit requires exactly one replay covering the base-game trace",
        )
    )
    boundary_gate = cast(dict[str, Any], controller_boundary["gate"])
    boundary_checks = cast(dict[str, bool], boundary_gate["checks"])
    replay_parseable = selected_replay_index is not None and bool(
        replay_records[selected_replay_index]["validation"]["libmelee_parseable"]
    )

    per_slot_checks: dict[str, bool] = {}
    for port in (1, 2):
        diagnostics = diagnostics_before_close.get(port, {})
        per_slot_checks.update(
            {
                f"p{port}_session_step_count_exact": session_step_counts[port] == processed_frames,
                f"p{port}_current_recurrent_source_frame_exact": (
                    len(session_step_ages[port]) == processed_frames
                    and all(age == 0 for age in session_step_ages[port])
                ),
                f"p{port}_native_frame_count_exact": diagnostics.get("frames_total") == processed_frames,
                f"p{port}_native_current_frame_barrier_exact": (
                    diagnostics.get("current_frame_inference_barriers") == processed_frames
                    and diagnostics.get("current_frame_inference_barrier_every_frame") is True
                ),
                f"p{port}_native_decoder_capture_exact": (
                    diagnostics.get("capture_decoder_assertions") == processed_frames
                    and diagnostics.get("capture_decoder_mismatches") == 0
                ),
                f"p{port}_dispatch_count_exact": dispatch_counts[port] == processed_frames,
            }
        )
    transport_record = transport.audit_record() if transport is not None else {"installed": False}
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
    gate_checks = {
        "exact_mode_only": contract["exact_mode_only"],
        "entered_gameplay": in_game,
        "processed_policy_frames": processed_frames > 0,
        "independent_policy_runtime_objects": sessions[1] is not sessions[2],
        "first_policy_frame_minus_123": first_game_frame == FIRST_POLICY_FRAME,
        "strict_consecutive_frame_order": all(delta == 1 for delta in frame_delta_counts),
        "both_policies_finish_current_step_before_advance": all(
            len(session_step_ages[port]) == processed_frames
            and all(age == 0 for age in session_step_ages[port])
            for port in (1, 2)
        ),
        "both_current_frame_inference_barriers_exact": all(
            diagnostics_before_close.get(port, {}).get("current_frame_inference_barriers") == processed_frames
            and diagnostics_before_close.get(port, {}).get("current_frame_inference_barrier_every_frame")
            is True
            for port in (1, 2)
        ),
        **per_slot_checks,
        "native_async_inference_preserved": all(
            cast(dict[str, Any], session_metadata[port]["runtime_contract"])["async_inference"] is True
            for port in (1, 2)
        ),
        "native_eval_two_19_frame_fifo_preserved": all(
            cast(dict[str, Any], session_metadata[port]["runtime_contract"])["effective_policy_delay_frames"]
            == dual_effective_policy_delay_frames
            for port in (1, 2)
        ),
        "saved_replay_parseable": replay_parseable,
        "exactly_one_tournament_result_replay": selected_replay_index is not None,
        "no_unexpected_auxiliary_replays": len(replay_paths) == 1
        or (sudden_death_transition_observed and len(replay_paths) == 2),
        "controller_boundary_gate_pass": boundary_gate.get("decision") == "pass",
        "controller_boundary_physical_buttons_exact": boundary_checks["both_slots_physical_buttons_exact"],
        "controller_boundary_processed_buttons_exact": boundary_checks[
            "both_slots_processed_upstream_buttons_exact"
        ],
        "controller_boundary_raw_main_exact": boundary_checks["both_slots_intended_raw_main_stick_exact"],
        "controller_boundary_processed_c_stick_exact": boundary_checks[
            "both_slots_processed_c_stick_within_tolerance"
        ],
        "controller_boundary_physical_shoulders_exact": boundary_checks[
            "both_slots_physical_analog_shoulders_within_tolerance"
        ],
        **transport_checks,
        "slippi_next_step_boundary_schedule_exact": (
            slippi_boundary_schedule_frames
            == list(range(FIRST_POLICY_FRAME, FIRST_POLICY_FRAME + processed_frames))
        ),
        "natural_game_end_requirement_met": _natural_end_requirement_met(
            required=request.require_natural_end,
            observed=natural_game_end,
        ),
        "tensorflow_cpu_only": tensorflow_setup.get("physical_gpu_count") == 0
        and tensorflow_setup.get("logical_gpu_count") == 0,
        "runtime_environment_lock_gate_passed": cast(dict[str, Any], contract["reproducibility"])[
            "runtime_environment"
        ]["dependency_lock_validation"]["environment_lock_gate_passed"]
        is True,
    }
    if exception is None and not all(gate_checks.values()):
        exception = RuntimeError(
            "dual Slippi-AI integration gate failed: "
            f"{[name for name, passed in gate_checks.items() if not passed]}"
        )

    slots = {
        f"p{port}": {
            "port": port,
            "model": "slippi-ai",
            "requested_character": request.character_for_port(port),
            "character": request.character_for_port(port),
            "identity": session_metadata[port],
            "timing": {
                "session_step_count": session_step_counts[port],
                "dispatch_count": dispatch_counts[port],
                "mean_current_recurrent_command_age_frames": (
                    sum(session_step_ages[port]) / len(session_step_ages[port])
                    if session_step_ages[port]
                    else None
                ),
                "maximum_current_recurrent_command_age_frames": (
                    max(session_step_ages[port]) if session_step_ages[port] else None
                ),
            },
            "diagnostics": {
                "before_close": diagnostics_before_close.get(port),
                "after_close": diagnostics_after_close.get(port),
            },
        }
        for port in (1, 2)
    }
    result_status = "pass" if exception is None and all(gate_checks.values()) else "fail"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "classification": (
            "frame-exact time-dilated Slippi-AI-versus-Slippi-AI integration with two "
            "independent pinned native sessions"
        ),
        "result": result_status,
        "error": None if exception is None else f"{type(exception).__name__}: {exception}",
        "configuration": {
            "seed": effective_seed,
            "stage": request.stage,
            "observed_match_context": observed_context,
            "inference_mode": contract["inference_mode"],
            "maximum_game_frames": maximum_frames,
            "require_natural_end": request.require_natural_end,
            "console_delay_frames": dual_console_delay_frames,
            "effective_policy_delay_frames": dual_effective_policy_delay_frames,
            "controller_replay_lag_frames": dual_controller_replay_lag_frames,
            "timing_recipe": "pinned upstream scripts/eval_two.py",
            "blocking_input": True,
        },
        "slots": slots,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "libmelee_module": str(Path(melee.__file__).resolve()),
            "emulator_version": emulator_version,
            "tensorflow": tensorflow_setup,
        },
        "reproducibility": contract["reproducibility"],
        "emulator_application": emulator_application,
        "game_image": _game_image_record(iso_path),
        "controller_boundary_audit": controller_boundary,
        "execution": {
            "entered_gameplay": in_game,
            "processed_policy_frames": processed_frames,
            "first_game_frame": first_game_frame,
            "last_game_frame": last_game_frame,
            "frame_delta_counts": {str(delta): count for delta, count in sorted(frame_delta_counts.items())},
            "last_stocks": stocks,
            "natural_game_end": natural_game_end,
            "sudden_death_transition_observed": sudden_death_transition_observed,
            "termination": termination,
            "shutdown_method": shutdown_method,
            "controller_transport": {
                **transport_record,
                "menu_flushes": {f"p{port}": menu_transport_flushes[port] for port in (1, 2)},
                "slippi_next_step_boundary_scheduled_from_frames": (slippi_boundary_schedule_frames),
                "policy_for_auxiliary_poll": "no inference, dispatch, or synthetic reflush",
            },
        },
        "artifacts": {
            "trace": controller_boundary["trace"],
            "replays": replay_records,
        },
        "gate": {"decision": result_status, "checks": gate_checks},
    }
    _write_json(summary_path, summary)
    if exception is not None:
        raise RuntimeError(summary["error"])
    return summary


def _game_image_record(path: Path) -> dict[str, Any]:
    from melee_policy.integration.match_runtime import _game_image_identity

    return _game_image_identity(path)


def run_slippi_match(
    config_path: Path,
    iso_path: Path | None,
    request: SlippiMatchRequest | None = None,
) -> dict[str, Any]:
    """Load only the selected two policies and execute one rendered exact match."""
    config, project_root = _load_config(config_path)
    match_request = SlippiMatchRequest() if request is None else request
    if match_request.artifact_label is None and (
        match_request.save_slp or match_request.save_video
    ):
        from melee_policy.integration.game_bundle import export_artifact_label

        match_request = replace(
            match_request,
            artifact_label=export_artifact_label(
                None,
                save_slp=match_request.save_slp,
                save_video=match_request.save_video,
            ),
        )
    contract = _validate_config_contract(config, match_request)
    contract["reproducibility"] = _runtime_reproducibility_record(
        project_root,
        config_path.resolve(),
        "requirements-e010.lock",
        RUNTIME_IMPLEMENTATION_PATHS,
    )
    seed = _effective_evaluation_seed(config, match_request.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    image_path = _resolve_game_image_path(config, project_root, iso_path)
    if all(match_request.model_for_port(port) == "slippi-ai" for port in (1, 2)):
        _validate_character_contracts(match_request, None)
        from melee_policy.integration.slippi_compatibility import configure_cpu_tensorflow

        tensorflow_setup = configure_cpu_tensorflow(seed)
        sessions = {
            port: SlippiAIPolicySession(
                _slippi_policy_config(
                    config,
                    project_root,
                    match_request,
                    port_override=port,
                    console_delay_override=UPSTREAM_EVAL_TWO_CONSOLE_DELAY_FRAMES,
                )
            )
            for port in (1, 2)
        }
        summary = _run_dual_slippi_console(
            config,
            project_root,
            image_path,
            match_request,
            contract,
            sessions,
            tensorflow_setup,
            seed,
        )
    else:
        if match_request.windowed_model == "mimic" and match_request.port_for("mimic") == 2:
            windowed_runtime = _load_windowed_runtime(config, project_root, match_request)
            mimic_runtime = cast(MimicRuntime, windowed_runtime)
            source_checks = {
                "loaded_policy": "mimic",
                "source_directory": str(mimic_runtime.source_identity.get("directory", "")),
                "repositories": {"mimic": mimic_runtime.source_identity},
            }
        else:
            source_checks = _activate_windowed_source(
                config,
                project_root,
                match_request.windowed_model,
            )
            windowed_runtime = _load_windowed_runtime(config, project_root, match_request)
        _validate_character_contracts(match_request, windowed_runtime)
        from melee_policy.integration.slippi_compatibility import configure_cpu_tensorflow

        tensorflow_setup = configure_cpu_tensorflow(seed)
        policy_config = _slippi_policy_config(config, project_root, match_request)
        session = SlippiAIPolicySession(policy_config)
        summary = _run_console(
            config,
            project_root,
            image_path,
            match_request,
            contract,
            source_checks,
            windowed_runtime,
            session,
            tensorflow_setup,
            seed,
        )
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


__all__ = ["SlippiMatchRequest", "run_slippi_match"]
