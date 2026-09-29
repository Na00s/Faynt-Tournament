"""Run a Frisson-AI character mirror against Melee's native level 9 CPU.

Frisson keeps its native zero-delay next-boundary action contract. The second
controller pipe remains present because the pinned Dolphin advances only after
receiving a matching two-port FRAME_SYNC transaction. During gameplay that
pipe receives a complete neutral controller command while Melee itself owns
the CPU player's decisions.
"""

from __future__ import annotations

import json
import platform
import random
import signal
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TextIO, cast

import numpy as np
import torch

from melee_policy.e000.enums import (
    CHARACTER_FOLDER_TO_INTERNAL,
    EXTERNAL_CHARACTER_TO_INTERNAL,
    SLIPPI_STAGE_TO_LIBMELEE,
)
from melee_policy.integration.frame_watchdog import InGameNoFrameWatchdog
from melee_policy.integration.frisson_match import (
    CONTROLLER_REPLAY_LAG_FRAMES,
    EXACT_INFERENCE_MODE,
    FIRST_POLICY_FRAME,
    FRISSON_CHARACTER,
    FRISSON_PORT,
    FRISSON_SUPPORTED_CHARACTERS,
    MATCH_SEED,
    MATCH_STAGE,
    MIMIC_PORT,
    _collect_replays,
    _console_options,
    _frisson_policy_config,
    _menu_character_selection,
    _player_state,
    _replay_character_expectation,
    _validate_config_contract,
)
from melee_policy.integration.frisson_policy import (
    EXPECTED_MODEL_CONTEXT_LENGTH,
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
    CanonicalControllerCommand,
    send_canonical_controller,
)
from melee_policy.integration.slippi_match import (
    _game_start_transport_proof,
    _piecewise_game_start_replay_frames,
)

SCHEMA_VERSION = "integration.frisson_vs_cpu9.v1"
TRACE_SCHEMA_VERSION = "integration.frisson_vs_cpu9.controller_trace.v1"
CONTROLLER_AUDIT_SCHEMA_VERSION = "integration.frisson_vs_cpu9.controller_audit.v3"
CPU_MODEL = "cpu"
CPU_LEVEL = 9
CPU_CHARACTER = "FOX"
CPU_PORT = MIMIC_PORT
POST_STOCK_OUT_DRAIN_SCHEMA_VERSION = "integration.frisson_vs_cpu9.post_stock_out_drain.v1"
POST_STOCK_OUT_DRAIN_MAX_STEP_CALLS = 600
FINAL_CPU_CHECKPOINT_IDENTITIES: dict[str, dict[str, Any]] = {
    "5a54ea4ecfa150198180dff4d06ac4fe6ecec5d9433803cc13b527e41f41d14e": {
        "relative_path": ".e011-cache/final-winners/remote-verification/10m-step-122064.pt",
        "byte_length": 96_451_691,
        "format": "melee_policy.final_pretraining_checkpoint.v1",
        "profile": "10m",
        "step": 122_064,
        "processed_target_frames": 7_999_586_304,
        "parameter_count": 10_163_629,
    },
    "8662de0c4c0deaae2879548d6373def792dbb52adec44f2b474adb7fe155c648": {
        "relative_path": ".e011-cache/final-winners/remote-verification/75m-step-86016.pt",
        "byte_length": 644_291_987,
        "format": "melee_policy.final_pretraining_checkpoint.v1",
        "profile": "75m",
        "step": 86_016,
        "processed_target_frames": 5_637_144_576,
        "parameter_count": 75_305_709,
    },
    "63b5ff05ef30476c4f41590c478f4a7218f3b72a8b5ef24a0e336eeb2b7c287b": {
        "relative_path": (
            ".e013-cache/posttraining-winners/frisson-melee-10m-posttrained-best-val.pt"
        ),
        "byte_length": 96_452_075,
        "format": "melee_policy.posttraining_curriculum_checkpoint.v1",
        "profile": "10m",
        "step": 195_248,
        "processed_target_frames": 12_795_772_928,
        "parameter_count": 10_163_629,
    },
    "8211f1832198646e9f4e3bacde26f181614f93320e68bd326dd5942c4e0f4077": {
        "relative_path": (
            ".e013-cache/posttraining-winners/frisson-melee-75m-posttrained-best-val.pt"
        ),
        "byte_length": 644_291_987,
        "format": "melee_policy.posttraining_curriculum_checkpoint.v1",
        "profile": "75m",
        "step": 127_214,
        "processed_target_frames": 8_337_096_704,
        "parameter_count": 75_305_709,
    },
}


from melee_policy.integration.post_rl_checkpoints import CHECKPOINTS as POST_RL_CHECKPOINTS

FINAL_CPU_CHECKPOINT_IDENTITIES.update({row["sha256"]: row for row in POST_RL_CHECKPOINTS.values()})


@dataclass(frozen=True, slots=True)
class FrissonCpuMatchRequest:
    """Benchmark boundary: one Frisson character mirrored by native CPU 9."""

    player_1_model: str = "frisson-ai"
    player_2_model: str = CPU_MODEL
    player_1_character: str = FRISSON_CHARACTER
    player_2_character: str = CPU_CHARACTER
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
    cpu_level: int = CPU_LEVEL

    def validate(self) -> None:
        fixed = {
            "player_1_model": (self.player_1_model.strip().lower().replace("_", "-"), "frisson-ai"),
            "player_2_model": (self.player_2_model.strip().lower(), CPU_MODEL),
            "stage": (self.stage, MATCH_STAGE),
            "inference_mode": (self.inference_mode, EXACT_INFERENCE_MODE),
            "require_natural_end": (self.require_natural_end, True),
            "cpu_level": (self.cpu_level, CPU_LEVEL),
        }
        mismatches = {
            name: {"observed": observed, "required": required}
            for name, (observed, required) in fixed.items()
            if observed != required
        }
        if mismatches:
            raise ValueError(f"Frisson-versus-CPU9 fixed match contract mismatch: {mismatches}")
        if self.player_1_character not in FRISSON_SUPPORTED_CHARACTERS:
            raise ValueError(
                "Frisson-versus-CPU9 requires an independently launchable character, got "
                f"{self.player_1_character!r}; allowed={FRISSON_SUPPORTED_CHARACTERS}"
            )
        if self.player_2_character not in FRISSON_SUPPORTED_CHARACTERS:
            raise ValueError(
                "Frisson-versus-CPU9 requires an independently launchable CPU character, got "
                f"{self.player_2_character!r}; allowed={FRISSON_SUPPORTED_CHARACTERS}"
            )
        if self.player_1_character != self.player_2_character:
            raise ValueError(
                "Frisson-versus-CPU9 requires a same-character mirror, got "
                f"{self.player_1_character} versus {self.player_2_character}"
            )
        if self.player_1_assets is not None:
            raise ValueError("Frisson does not accept a separate asset directory")
        if self.player_2_checkpoint is not None or self.player_2_assets is not None:
            raise ValueError("Melee's native CPU does not accept checkpoint or asset paths")
        if self.player_1_name is not None or self.player_2_name is not None:
            raise ValueError("Frisson and Melee's native CPU do not accept player names")
        if self.player_1_temperature not in (None, 1, 1.0):
            raise ValueError("Frisson sample temperature is fixed at 1.0")
        if self.player_2_temperature is not None:
            raise ValueError("Melee's native CPU does not accept a sampling temperature")
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
class _CpuFrameResult:
    game_frame: int
    frisson_command: CanonicalControllerCommand
    frisson_dispatch: dict[str, Any]
    frisson_inference_seconds: float
    cpu_pipe_dispatch: dict[str, Any]
    barrier_seconds: float


def _validate_final_cpu_checkpoint(
    checkpoint_identity: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    """Accept exactly one frozen pretraining or posttraining final winner."""

    sha256 = checkpoint_identity.get("sha256")
    expected = FINAL_CPU_CHECKPOINT_IDENTITIES.get(sha256) if isinstance(sha256, str) else None
    if expected is None:
        raise ValueError(
            "Frisson-versus-CPU9 requires one exact frozen final-winner checkpoint: "
            "pretraining 10M step-122064, pretraining 75M step-86016, posttraining 10M "
            f"step-195248, or posttraining 75M step-127214; got SHA-256 {sha256!r}"
        )
    expected_path = (project_root / str(expected["relative_path"])).resolve()
    observed_path = Path(str(checkpoint_identity.get("path", ""))).expanduser().resolve()
    required = {
        "byte_length": expected["byte_length"],
        "format": expected["format"],
        "profile": expected["profile"],
        "step": expected["step"],
        "processed_target_frames": expected["processed_target_frames"],
        "parameter_count": expected["parameter_count"],
        "all_state_tensors_finite": True,
    }
    mismatches = {
        name: {"observed": checkpoint_identity.get(name), "required": value}
        for name, value in required.items()
        if checkpoint_identity.get(name) != value
    }
    if observed_path != expected_path:
        mismatches["path"] = {"observed": str(observed_path), "required": str(expected_path)}
    if mismatches:
        raise ValueError(f"Frisson-versus-CPU9 final checkpoint identity mismatch: {mismatches}")
    return {
        "sha256": sha256,
        "path": str(expected_path),
        **required,
    }


def _cpu_menu_character_for_state(
    melee_module: Any,
    gamestate: Any,
    requested_character: str,
) -> Any:
    """Select Zelda on CSS, then hold A as CPU Sheik during stage loading."""

    requested = melee_module.Character[requested_character]
    character_select_states = {
        melee_module.Menu.CHARACTER_SELECT,
        melee_module.Menu.SLIPPI_ONLINE_CSS,
    }
    if requested_character == "SHEIK" and gamestate.menu_state in character_select_states:
        return melee_module.Character.ZELDA
    return requested


def _run_cpu_frame(
    *,
    gamestate: Any,
    frisson_session: FrissonPolicySession,
    frisson_controller: Any,
    cpu_pipe_controller: Any,
    transport: _ControllerPipeLockstep,
) -> _CpuFrameResult:
    """Complete Frisson inference, then release one paired pipe boundary."""
    game_frame = int(gamestate.frame)
    barrier_started = time.perf_counter()
    inference_started = time.perf_counter()
    frisson_command = frisson_session.step(gamestate)
    inference_seconds = time.perf_counter() - inference_started
    frisson_command.validate()
    barrier_seconds = time.perf_counter() - barrier_started

    transport.begin_boundary(reason="frisson-vs-cpu9-gameplay", game_frame=game_frame)
    frisson_dispatch = send_canonical_controller(
        frisson_controller,
        frisson_command,
        flush=False,
    ).as_dict()
    frisson_dispatch.update(
        {
            "called": True,
            "queued_after_current_frame_inference": True,
        }
    )
    transport.schedule_next_boundary(
        FRISSON_PORT,
        reason="frisson-zero-delay-next-frame-command",
        game_frame=game_frame,
    )

    neutral = CanonicalControllerCommand.neutral()
    cpu_pipe_dispatch = send_canonical_controller(
        cpu_pipe_controller,
        neutral,
        flush=False,
    ).as_dict()
    cpu_pipe_dispatch.update(
        {
            "called": True,
            "neutral": True,
            "gameplay_owner": "melee-native-cpu",
            "pipe_role": "paired-frame-sync-transaction-only",
        }
    )
    transport.schedule_next_boundary(
        CPU_PORT,
        reason="cpu9-neutral-pipe-frame-sync",
        game_frame=game_frame,
    )
    transport.commit_boundary()
    return _CpuFrameResult(
        game_frame=game_frame,
        frisson_command=frisson_command,
        frisson_dispatch=frisson_dispatch,
        frisson_inference_seconds=inference_seconds,
        cpu_pipe_dispatch=cpu_pipe_dispatch,
        barrier_seconds=barrier_seconds,
    )


def _console_observed_formal_game_end(console: Any) -> bool:
    """Read libmelee's exact event ledger for the most recent step call."""

    from melee.slippstream import EventType

    events = getattr(console, "_events_this_frame", ())
    return EventType.GAME_END in events


def _post_stock_out_drain_record(
    *,
    terminal_policy_frame: int | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    return {
        "schema_version": POST_STOCK_OUT_DRAIN_SCHEMA_VERSION,
        "attempted": True,
        "decision": "fail",
        "reason": "drain-not-completed",
        "terminal_policy_frame": terminal_policy_frame,
        "timeout_seconds": timeout_seconds,
        "maximum_step_calls": POST_STOCK_OUT_DRAIN_MAX_STEP_CALLS,
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
        "controller_trace_rows_written": 0,
        "game_scope_transport_audit_sealed_before_drain": False,
        "game_scope_transport_gate_passed_before_drain": False,
        "failure_phase": None,
        "error": None,
        "wall_seconds": 0.0,
    }


def _drain_cpu_formal_game_end(
    *,
    transport: _ControllerPipeLockstep,
    frisson_controller: Any,
    cpu_pipe_controller: Any,
    terminal_policy_frame: int,
    in_game_menu_states: tuple[Any, ...],
    timeout_seconds: float,
    clock: Callable[[], float] = time.monotonic,
    formal_game_end_probe: Callable[[Any], bool] = _console_observed_formal_game_end,
) -> dict[str, Any]:
    """Advance a completed stock-out through formal Game End without inference."""

    if timeout_seconds <= 0:
        raise ValueError("formal Game End drain timeout must be positive")
    record = _post_stock_out_drain_record(
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
        while record["step_calls"] < POST_STOCK_OUT_DRAIN_MAX_STEP_CALLS:
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
                reason="cpu9-post-stock-out-formal-game-end-drain",
                game_frame=game_frame,
            )
            record["neutral_boundaries_started"] += 1
            phase = "frisson-neutral-dispatch"
            send_canonical_controller(frisson_controller, neutral, flush=False)
            record["neutral_dispatches_by_port"]["p1"] += 1
            phase = "frisson-neutral-schedule"
            transport.schedule_next_boundary(
                FRISSON_PORT,
                reason="frisson-post-stock-out-neutral-finalization",
                game_frame=game_frame,
            )
            phase = "cpu-neutral-dispatch"
            send_canonical_controller(cpu_pipe_controller, neutral, flush=False)
            record["neutral_dispatches_by_port"]["p2"] += 1
            phase = "cpu-neutral-schedule"
            transport.schedule_next_boundary(
                CPU_PORT,
                reason="cpu9-post-stock-out-neutral-finalization",
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


def _cpu_trace_row(
    gamestate: Any,
    result: _CpuFrameResult,
    *,
    character: str = CPU_CHARACTER,
) -> dict[str, Any]:
    frame = result.game_frame
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "game_frame": frame,
        "barrier": {
            "source_frame": frame,
            "frisson_current_frame_inference_complete": True,
            "controller_transaction_opened_after_inference": True,
            "both_pipe_transactions_committed_together": True,
            "seconds": result.barrier_seconds,
        },
        "slots": {
            "p1": {
                "port": FRISSON_PORT,
                "model": "frisson-ai",
                "requested_character": character,
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
                "port": CPU_PORT,
                "model": CPU_MODEL,
                "requested_character": character,
                "player_state": _player_state(gamestate.players[CPU_PORT]),
                "command": CanonicalControllerCommand.neutral().as_dict(),
                "inference": {
                    "called": False,
                    "owner": "melee-native-cpu",
                    "cpu_level": CPU_LEVEL,
                },
                "controller_dispatch": result.cpu_pipe_dispatch,
            },
        },
    }


def _validate_cpu_first_context(
    gamestate: Any,
    *,
    character: str = CPU_CHARACTER,
) -> dict[str, Any]:
    observed_stage = str(getattr(gamestate.stage, "name", gamestate.stage)).split(".")[-1]
    observed_characters = {
        f"p{port}": str(
            getattr(gamestate.players[port].character, "name", gamestate.players[port].character)
        ).split(".")[-1]
        for port in (FRISSON_PORT, CPU_PORT)
    }
    checks = {
        "first_frame_minus_123": int(gamestate.frame) == FIRST_POLICY_FRAME,
        "final_destination": observed_stage == MATCH_STAGE,
        "frisson_requested_character": observed_characters["p1"] == character,
        "cpu_requested_character": observed_characters["p2"] == character,
        "frisson_live_cpu_level_zero": int(gamestate.players[FRISSON_PORT].cpu_level) == 0,
        "cpu_live_level_9": int(gamestate.players[CPU_PORT].cpu_level) == CPU_LEVEL,
    }
    if not all(checks.values()):
        raise RuntimeError(f"first Frisson-versus-CPU9 game context mismatch: {checks}")
    return {
        "frame": int(gamestate.frame),
        "stage": observed_stage,
        "characters": observed_characters,
        "costumes": {f"p{port}": int(gamestate.players[port].costume) for port in (FRISSON_PORT, CPU_PORT)},
        "cpu_levels": {
            f"p{port}": int(gamestate.players[port].cpu_level) for port in (FRISSON_PORT, CPU_PORT)
        },
        "checks": checks,
    }


def _validate_cpu_replay_start(
    path: Path,
    *,
    character: str = CPU_CHARACTER,
) -> dict[str, Any]:
    """Verify the replay declares P2 as Melee's native level 9 CPU."""
    import melee
    import peppi_py

    from melee_policy.integration.replay_result import audit_parsed_game

    try:
        game = peppi_py.read_slippi(str(path))
        players = {int(player.port.value) + 1: player for player in game.start.players if player is not None}
        details = {
            f"p{port}": {
                "type": str(getattr(player.type, "name", player.type)),
                "cpu_level": player.cpu_level,
                "external_character_id": int(player.character),
                "internal_character_id": EXTERNAL_CHARACTER_TO_INTERNAL.get(int(player.character)),
            }
            for port, player in sorted(players.items())
        }
        game_end = getattr(game, "end", None)
        end_players = {
            int(player.port.value) + 1: int(player.placement)
            for player in getattr(game_end, "players", ())
            if player is not None
        }
        winning_ports = [port for port, placement in end_players.items() if placement == 0]
        winner_port = winning_ports[0] if len(winning_ports) == 1 else None
        game_random_seed = getattr(game.start, "random_seed", None)
        end_method = getattr(game_end, "method", None)
        raw_stage_id = int(game.start.stage)
        libmelee_stage_id = SLIPPI_STAGE_TO_LIBMELEE.get(raw_stage_id)
        replay_character = _replay_character_expectation(character)
        requested_internal_ids = CHARACTER_FOLDER_TO_INTERNAL[replay_character]
        replay_audit = audit_parsed_game(
            game,
            expected_stage=MATCH_STAGE,
            expected_characters={1: replay_character, 2: replay_character},
        )
        outcome = cast(dict[str, Any], replay_audit["outcome"])
        checks = {
            "exact_ports_1_and_2": sorted(players) == [1, 2],
            "p1_declared_human": 1 in players
            and str(getattr(players[1].type, "name", players[1].type)) == "HUMAN",
            "p2_declared_cpu": 2 in players
            and str(getattr(players[2].type, "name", players[2].type)) == "CPU",
            "p2_cpu_level_9": 2 in players and players[2].cpu_level == CPU_LEVEL,
            "p1_requested_character": 1 in players
            and EXTERNAL_CHARACTER_TO_INTERNAL.get(int(players[1].character)) in requested_internal_ids,
            "p2_requested_character": 2 in players
            and EXTERNAL_CHARACTER_TO_INTERNAL.get(int(players[2].character)) in requested_internal_ids,
            "final_destination": libmelee_stage_id == int(melee.Stage.FINAL_DESTINATION.value),
            "game_random_seed_recorded": isinstance(game_random_seed, int),
            "replay_outcome_complete": outcome.get("game_complete") is True,
            "replay_outcome_conclusive": outcome.get("conclusive") is True,
            "replay_outcome_is_win_or_draw": outcome.get("status") in {"win", "draw"},
        }
        return {
            "decision": "pass" if all(checks.values()) else "fail",
            "checks": checks,
            "players": details,
            "requested_character": character,
            "replay_character_name": replay_character,
            "stage": {
                "raw_slippi_id": raw_stage_id,
                "libmelee_id": libmelee_stage_id,
                "name": MATCH_STAGE
                if libmelee_stage_id == int(melee.Stage.FINAL_DESTINATION.value)
                else None,
            },
            "game_random_seed": game_random_seed,
            "end": {
                "method": str(getattr(end_method, "name", end_method)),
                "placements": {f"p{port}": value for port, value in sorted(end_players.items())},
                "winner_port": winner_port,
            },
            "outcome": outcome,
            "terminal": replay_audit.get("terminal"),
            "error": None,
        }
    except Exception as error:
        return {
            "decision": "fail",
            "checks": {
                "exact_ports_1_and_2": False,
                "p1_declared_human": False,
                "p2_declared_cpu": False,
                "p2_cpu_level_9": False,
                "p1_requested_character": False,
                "p2_requested_character": False,
                "final_destination": False,
                "game_random_seed_recorded": False,
                "replay_outcome_complete": False,
                "replay_outcome_conclusive": False,
                "replay_outcome_is_win_or_draw": False,
            },
            "players": {},
            "requested_character": character,
            "replay_character_name": _replay_character_expectation(character),
            "stage": None,
            "game_random_seed": None,
            "end": None,
            "outcome": None,
            "terminal": None,
            "error": f"{type(error).__name__}: {error}",
        }


def _frisson_first_policy_frame_mapping_evidence(
    trace_by_frame: Mapping[int, Mapping[str, Any]],
    replay_states: Mapping[int, Mapping[int, Mapping[str, Any]]],
    *,
    lag_frames: int,
) -> dict[str, Any]:
    """Select Frisson's first replay boundary without treating native CPU input as policy input."""
    from melee_policy.integration.slippi_match import _trace_command_matches_replay

    trace_row = trace_by_frame.get(FIRST_POLICY_FRAME)
    if not isinstance(trace_row, Mapping):
        raise ValueError("Frisson CPU GAME_START mapping lacks trace frame -123")
    slots = trace_row.get("slots")
    if not isinstance(slots, Mapping) or set(slots) != {"p1", "p2"}:
        raise ValueError("Frisson CPU trace frame -123 lacks exactly two slots")
    trace_slot = slots.get("p1")
    if not isinstance(trace_slot, Mapping) or trace_slot.get("model") != "frisson-ai":
        raise ValueError("Frisson CPU trace frame -123 lacks the Frisson P1 slot")

    candidate_frames = {
        "early-latched": FIRST_POLICY_FRAME + lag_frames,
        "delayed-first": FIRST_POLICY_FRAME + lag_frames + 1,
    }
    candidate_matches: dict[str, dict[str, bool]] = {}
    for candidate, replay_frame in candidate_frames.items():
        replay = replay_states.get(replay_frame)
        observed = replay.get(FRISSON_PORT) if isinstance(replay, Mapping) else None
        exact = bool(
            isinstance(observed, Mapping)
            and _trace_command_matches_replay(trace_slot, observed)
        )
        candidate_matches[candidate] = {"p1": exact, "all_evidence_ports": exact}

    early_exact = candidate_matches["early-latched"]["all_evidence_ports"]
    delayed_exact = candidate_matches["delayed-first"]["all_evidence_ports"]
    if early_exact and not delayed_exact:
        decision = "early-latched"
    elif delayed_exact and not early_exact:
        decision = "delayed-first"
    elif early_exact and delayed_exact:
        decision = "observationally-equivalent-delayed-first"
    else:
        raise ValueError(
            "Frisson's first policy command matches neither transport-permitted replay "
            f"boundary: {candidate_matches}"
        )

    selected_candidate = "early-latched" if decision == "early-latched" else "delayed-first"
    return {
        "trace_frame": FIRST_POLICY_FRAME,
        "comparison": "every hard-gated controller dimension for Frisson P1",
        "evidence_ports": [FRISSON_PORT],
        "excluded_ports": {
            "p2": "Melee owns native CPU inputs; its pipe command is frame-sync only",
        },
        "candidate_replay_frames": candidate_frames,
        "candidate_matches": candidate_matches,
        "decision": decision,
        "selected_replay_frame": candidate_frames[selected_candidate],
    }


def _audit_frisson_only_controller_boundary(
    trace_path: Path,
    replay_path: Path,
    project_root: Path,
    *,
    game_start_transport_proof: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit Frisson's P1 commands while leaving native CPU inputs to Melee."""
    from melee_policy.integration.slippi_match import (
        PHYSICAL_ANALOG_SHOULDER_TOLERANCE,
        PROCESSED_C_STICK_TOLERANCE,
        _component_statistics,
        _expected_processed_buttons,
        _expected_processed_c_stick,
        _expected_raw_main_axis,
        _normalize_trace_command,
        _read_replay_controller_states,
        _read_trace_rows,
        _tolerant_component_statistics,
    )

    rows = _read_trace_rows(trace_path)
    replay_states = _read_replay_controller_states(replay_path)
    if not rows:
        raise ValueError("controller boundary audit requires at least one trace row")
    if not replay_states:
        raise ValueError("controller boundary audit requires at least one replay frame")
    trace_by_frame = {int(row["game_frame"]): row for row in rows}
    if len(trace_by_frame) != len(rows):
        raise ValueError("controller boundary audit trace frames must be unique")
    trace_frames = sorted(trace_by_frame)
    replay_frames = sorted(int(frame) for frame in replay_states)
    trace_is_consecutive = trace_frames == list(range(trace_frames[0], trace_frames[-1] + 1))
    replay_covers_observed_trace_states = all(
        frame in replay_states and FRISSON_PORT in replay_states[frame] for frame in trace_frames
    )
    first_policy_mapping_evidence = _frisson_first_policy_frame_mapping_evidence(
        trace_by_frame,
        replay_states,
        lag_frames=CONTROLLER_REPLAY_LAG_FRAMES,
    )
    replay_frame_by_trace_frame, game_start_unavailable = _piecewise_game_start_replay_frames(
        trace_by_frame,
        replay_states,
        lag_frames=CONTROLLER_REPLAY_LAG_FRAMES,
        transport_proof=game_start_transport_proof,
        first_policy_replay_frame=int(first_policy_mapping_evidence["selected_replay_frame"]),
    )
    terminal_unobservable = [
        frame
        for frame in trace_frames
        if frame not in game_start_unavailable
        and frame not in replay_frame_by_trace_frame
        and frame + CONTROLLER_REPLAY_LAG_FRAMES > replay_frames[-1]
    ]
    bounded_terminal_suffix = len(terminal_unobservable) <= CONTROLLER_REPLAY_LAG_FRAMES and (
        not terminal_unobservable
        or terminal_unobservable == trace_frames[len(trace_frames) - len(terminal_unobservable) :]
    )
    expected_trace_frames = [
        frame
        for frame in trace_frames
        if frame not in terminal_unobservable and frame not in game_start_unavailable
    ]
    aligned_trace_frames = [frame for frame in expected_trace_frames if frame in replay_frame_by_trace_frame]
    missing_pairs = sorted(set(expected_trace_frames) - set(aligned_trace_frames))
    expected_startup_pairs = [
        [FIRST_POLICY_FRAME, int(first_policy_mapping_evidence["selected_replay_frame"])],
        *[
            [frame, frame + CONTROLLER_REPLAY_LAG_FRAMES + 1]
            for frame in range(FIRST_POLICY_FRAME + 1, FIRST_POLICY_FRAME + 4)
        ],
    ]
    actual_startup_pairs = [
        [frame, replay_frame_by_trace_frame[frame]]
        for frame in range(FIRST_POLICY_FRAME, FIRST_POLICY_FRAME + 4)
        if frame in replay_frame_by_trace_frame
    ]
    steady_state_pairs_exact = FIRST_POLICY_FRAME + 5 in replay_frame_by_trace_frame and all(
        replay_frame == trace_frame + CONTROLLER_REPLAY_LAG_FRAMES
        for trace_frame, replay_frame in replay_frame_by_trace_frame.items()
        if trace_frame >= FIRST_POLICY_FRAME + 5
    )
    piecewise_alignment_exact = (
        actual_startup_pairs == expected_startup_pairs
        and game_start_unavailable == [FIRST_POLICY_FRAME + 4]
        and steady_state_pairs_exact
    )
    full_causal_overlap = (
        bool(expected_trace_frames)
        and trace_is_consecutive
        and replay_covers_observed_trace_states
        and bounded_terminal_suffix
        and piecewise_alignment_exact
        and not missing_pairs
    )
    dispatch_by_frame: dict[int, bool] = {}
    for frame in aligned_trace_frames:
        slot = cast(dict[str, Any], trace_by_frame[frame]["slots"])["p1"]
        dispatch = slot.get("controller_dispatch")
        if not isinstance(dispatch, dict) or not isinstance(dispatch.get("called"), bool):
            raise ValueError(f"p1 trace frame {frame} lacks explicit controller-dispatch evidence")
        dispatch_by_frame[frame] = bool(dispatch["called"])
    dispatch_frames = [frame for frame in aligned_trace_frames if dispatch_by_frame[frame]]
    controller_trace_frames = dispatch_frames
    excluded_no_dispatch = [frame for frame in aligned_trace_frames if not dispatch_by_frame[frame]]

    physical_button_mismatches: list[dict[str, Any]] = []
    processed_button_mismatches: list[dict[str, Any]] = []
    raw_samples: list[tuple[int, int, str, int | float, int | float]] = []
    c_samples: list[tuple[int, int, str, float, float]] = []
    trigger_samples: list[tuple[int, int, str, float, float]] = []
    for trace_frame in controller_trace_frames:
        replay_frame = replay_frame_by_trace_frame[trace_frame]
        trace_slot = cast(dict[str, Any], trace_by_frame[trace_frame]["slots"])["p1"]
        intended = _normalize_trace_command(
            "frisson-ai",
            cast(dict[str, Any], trace_slot["command"]),
        )
        observed = replay_states[replay_frame][FRISSON_PORT]
        expected_buttons = tuple(cast(tuple[str, ...], intended["buttons"]))
        expected_processed = _expected_processed_buttons(expected_buttons)
        observed_physical = tuple(cast(tuple[str, ...], observed["buttons_physical"]))
        observed_processed = tuple(cast(tuple[str, ...], observed["buttons_processed"]))
        if expected_buttons != observed_physical:
            physical_button_mismatches.append(
                {
                    "trace_frame": trace_frame,
                    "replay_frame": replay_frame,
                    "expected": list(expected_buttons),
                    "observed": list(observed_physical),
                }
            )
        if expected_processed != observed_processed:
            processed_button_mismatches.append(
                {
                    "trace_frame": trace_frame,
                    "replay_frame": replay_frame,
                    "expected": list(expected_processed),
                    "observed": list(observed_processed),
                }
            )
        intended_main = cast(tuple[float, float], intended["main_stick"])
        observed_raw_main = cast(tuple[int, int], observed["raw_main_stick"])
        intended_c = cast(tuple[float, float], intended["c_stick"])
        expected_c = _expected_processed_c_stick("frisson-ai", trace_slot, intended_c)
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
    processed_c = _tolerant_component_statistics(
        c_samples,
        tolerance=PROCESSED_C_STICK_TOLERANCE,
    )
    physical_shoulders = _tolerant_component_statistics(
        trigger_samples,
        tolerance=PHYSICAL_ANALOG_SHOULDER_TOLERANCE,
    )
    p1_gate = {
        "comparison_evidence": bool(controller_trace_frames),
        "physical_buttons_exact": not physical_button_mismatches,
        "processed_upstream_buttons_exact": not processed_button_mismatches,
        "intended_raw_main_stick_exact": raw_main["mismatch_components"] == 0,
        "processed_c_stick_within_tolerance": processed_c["mismatch_components"] == 0,
        "physical_analog_shoulders_within_tolerance": (physical_shoulders["mismatch_components"] == 0),
        "no_dispatch_rule_exact": not excluded_no_dispatch,
    }
    p1 = {
        "port": FRISSON_PORT,
        "model": "frisson-ai",
        "aligned_frame_pairs": len(aligned_trace_frames),
        "controller_frame_pairs_compared": len(controller_trace_frames),
        "all_mapped_commands_hard_gated": controller_trace_frames == aligned_trace_frames,
        "excluded_first_gameplay_latch_trace_frames": [],
        "excluded_game_start_controller_latch_trace_frames": game_start_unavailable,
        "excluded_no_dispatch_trace_frames": excluded_no_dispatch,
        "digital_buttons": {
            "physical": {
                "frames_compared": len(controller_trace_frames),
                "mismatch_frames": len(physical_button_mismatches),
                "mismatches": physical_button_mismatches,
            },
            "processed_upstream_observation": {
                "frames_compared": len(controller_trace_frames),
                "mismatch_frames": len(processed_button_mismatches),
                "mismatches": processed_button_mismatches,
            },
        },
        "intended_raw_main_stick": raw_main,
        "processed_c_stick": processed_c,
        "physical_analog_shoulders": physical_shoulders,
        "gate": p1_gate,
    }
    alignment = {
        "mode": "piecewise-game-start",
        "steady_state_trace_to_replay_lag_frames": CONTROLLER_REPLAY_LAG_FRAMES,
        "first_policy_trace_to_replay_lag_frames": (
            int(first_policy_mapping_evidence["selected_replay_frame"])
            - FIRST_POLICY_FRAME
        ),
        "startup_trace_to_replay_lag_frames": CONTROLLER_REPLAY_LAG_FRAMES + 1,
        "equation": (
            "replay = trace + 1 at -123; replay -121 is the GAME_START internal "
            "transaction; replay = trace + 2 for -122..-120; trace -119 unavailable; "
            "replay = trace + 1 from -118"
            if first_policy_mapping_evidence["selected_replay_frame"]
            == FIRST_POLICY_FRAME + CONTROLLER_REPLAY_LAG_FRAMES
            else "replay = trace + 2 for -123..-120; trace -119 unavailable at "
            "GAME_START; replay = trace + 1 from -118"
        ),
        "trace_frame_range": [trace_frames[0], trace_frames[-1]],
        "replay_frame_range": [replay_frames[0], replay_frames[-1]],
        "expected_overlap_pairs": len(expected_trace_frames),
        "actual_overlap_pairs": len(aligned_trace_frames),
        "startup_mapped_pairs": actual_startup_pairs,
        "first_aligned_pair": (
            [aligned_trace_frames[0], replay_frame_by_trace_frame[aligned_trace_frames[0]]]
            if aligned_trace_frames
            else None
        ),
        "last_aligned_pair": (
            [aligned_trace_frames[-1], replay_frame_by_trace_frame[aligned_trace_frames[-1]]]
            if aligned_trace_frames
            else None
        ),
        "first_gameplay_controller_latch_unobservable_trace_frames": [],
        "game_start_controller_latch_unobservable_trace_frames": game_start_unavailable,
        "missing_internal_pairs": missing_pairs,
        "permitted_terminal_unobservable_trace_frames": terminal_unobservable,
        "unpaired_trace_boundary_frames": sorted(set(trace_frames) - set(aligned_trace_frames)),
        "game_start_transport_proof": dict(game_start_transport_proof),
        "first_policy_frame_mapping_evidence": first_policy_mapping_evidence,
    }
    checks = {
        "full_causal_overlap": full_causal_overlap,
        "trace_frames_consecutive": trace_is_consecutive,
        "replay_covers_every_observed_trace_state": replay_covers_observed_trace_states,
        "terminal_unobservable_tail_bounded_by_lag": bounded_terminal_suffix,
        "game_start_piecewise_alignment_exact": piecewise_alignment_exact,
        "first_policy_frame_mapping_evidence_exact": (
            first_policy_mapping_evidence["decision"]
            in {
                "early-latched",
                "delayed-first",
                "observationally-equivalent-delayed-first",
            }
        ),
        "game_start_transport_proof_exact": (game_start_transport_proof.get("decision") == "pass"),
        "frisson_slot_aligned": p1["aligned_frame_pairs"] == alignment["expected_overlap_pairs"],
        "frisson_controller_evidence": bool(p1_gate["comparison_evidence"]),
        "frisson_all_mapped_commands_hard_gated": bool(p1["all_mapped_commands_hard_gated"]),
        "frisson_physical_buttons_exact": bool(p1_gate["physical_buttons_exact"]),
        "frisson_processed_upstream_buttons_exact": bool(p1_gate["processed_upstream_buttons_exact"]),
        "frisson_intended_raw_main_stick_exact": bool(p1_gate["intended_raw_main_stick_exact"]),
        "frisson_processed_c_stick_within_tolerance": bool(p1_gate["processed_c_stick_within_tolerance"]),
        "frisson_physical_analog_shoulders_within_tolerance": bool(
            p1_gate["physical_analog_shoulders_within_tolerance"]
        ),
        "frisson_game_start_controller_latch_rule_exact": (
            p1["excluded_game_start_controller_latch_trace_frames"]
            == alignment["game_start_controller_latch_unobservable_trace_frames"]
            == [FIRST_POLICY_FRAME + 4]
        ),
        "frisson_no_dispatch_rule_exact": bool(p1_gate["no_dispatch_rule_exact"]),
    }
    return {
        "schema_version": CONTROLLER_AUDIT_SCHEMA_VERSION,
        "classification": "Frisson P1 decoded commands verified against the saved replay",
        "alignment": alignment,
        "adapter": {
            "frisson_explicit_flush": False,
            "frisson_native_dummy_prefix": False,
            "game_start_controller_alignment": (
                "sealed two-pipe transport proves both permitted GAME_START boundaries; exact "
                "Frisson P1 replay evidence selects the first command boundary"
            ),
            "cpu_p2_controller_audit": "excluded because Melee owns native CPU inputs",
            "cpu_p2_pipe_transaction": "neutral paired FRAME_SYNC command",
        },
        "slots": {"p1": p1},
        "excluded_slots": {
            "p2": {
                "model": CPU_MODEL,
                "reason": "native CPU controller decisions are generated inside Melee",
            }
        },
        "gate": {"decision": "pass" if all(checks.values()) else "fail", "checks": checks},
        "trace": {**_file_identity(trace_path, project_root), "rows": len(rows)},
        "replay": _file_identity(replay_path, project_root),
    }


def _unavailable_frisson_cpu_audit(
    trace_path: Path,
    project_root: Path,
    reason: str,
    *,
    game_start_transport_proof: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checks = {
        "full_causal_overlap": False,
        "trace_frames_consecutive": False,
        "replay_covers_every_observed_trace_state": False,
        "terminal_unobservable_tail_bounded_by_lag": False,
        "game_start_piecewise_alignment_exact": False,
        "first_policy_frame_mapping_evidence_exact": False,
        "game_start_transport_proof_exact": False,
        "frisson_slot_aligned": False,
        "frisson_controller_evidence": False,
        "frisson_all_mapped_commands_hard_gated": False,
        "frisson_physical_buttons_exact": False,
        "frisson_processed_upstream_buttons_exact": False,
        "frisson_intended_raw_main_stick_exact": False,
        "frisson_processed_c_stick_within_tolerance": False,
        "frisson_physical_analog_shoulders_within_tolerance": False,
        "frisson_game_start_controller_latch_rule_exact": False,
        "frisson_no_dispatch_rule_exact": False,
    }
    return {
        "schema_version": CONTROLLER_AUDIT_SCHEMA_VERSION,
        "classification": "Frisson P1 decoded command replay audit unavailable",
        "error": reason,
        "trace": (
            _file_identity(trace_path, project_root)
            if trace_path.is_file()
            else {"path": _display_path(trace_path, project_root), "missing": True}
        ),
        "alignment": {
            "mode": "unavailable",
            "game_start_transport_proof": (
                dict(game_start_transport_proof) if game_start_transport_proof is not None else None
            ),
        },
        "slots": {},
        "excluded_slots": {"p2": {"model": CPU_MODEL, "reason": "native Melee CPU"}},
        "gate": {"decision": "fail", "checks": checks},
    }


def _selected_frisson_cpu_audit(
    trace_path: Path,
    replay_paths: list[Path],
    replay_records: list[dict[str, Any]],
    project_root: Path,
    *,
    game_start_transport_proof: Mapping[str, Any],
) -> dict[str, Any]:
    selected = [
        index for index, record in enumerate(replay_records) if record.get("tournament_result_replay") is True
    ]
    if len(selected) != 1 or selected[0] >= len(replay_paths):
        return _unavailable_frisson_cpu_audit(
            trace_path,
            project_root,
            "controller audit requires exactly one trace-covering result replay",
            game_start_transport_proof=game_start_transport_proof,
        )
    try:
        return _audit_frisson_only_controller_boundary(
            trace_path,
            replay_paths[selected[0]],
            project_root,
            game_start_transport_proof=game_start_transport_proof,
        )
    except Exception as error:
        return _unavailable_frisson_cpu_audit(
            trace_path,
            project_root,
            f"{type(error).__name__}: {error}",
            game_start_transport_proof=game_start_transport_proof,
        )


def _run_cpu_console(
    config: dict[str, Any],
    project_root: Path,
    iso_path: Path,
    request: FrissonCpuMatchRequest,
    contract: dict[str, Any],
    frisson_session: FrissonPolicySession,
    reproducibility: dict[str, Any],
) -> dict[str, Any]:
    import melee

    menu_characters = _menu_character_selection(melee, request)  # type: ignore[arg-type]
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
    cpu_pipe_controller = melee.Controller(
        console=console,
        port=CPU_PORT,
        type=melee.ControllerType.STANDARD,
    )
    raw_controllers = {
        FRISSON_PORT: frisson_controller,
        CPU_PORT: cpu_pipe_controller,
    }
    menu_p1 = melee.MenuHelper()
    menu_p2 = melee.MenuHelper()
    max_game_frames = (
        int(config["integration"]["max_game_frames"])
        if request.max_game_frames is None
        else request.max_game_frames
    )
    menu_timeout = float(config["integration"]["menu_timeout_seconds"])
    replay_finalize_timeout = float(config["integration"]["replay_finalize_timeout_seconds"])
    started_at = time.time()
    trace_stream: TextIO | None = None
    transport: _ControllerPipeLockstep | None = None
    in_game = False
    game_end_observed = False
    formal_game_end_observed = False
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
    inference_seconds: list[float] = []
    barrier_seconds: list[float] = []
    frisson_dispatch_count = 0
    cpu_neutral_pipe_dispatch_count = 0
    stocks = {"frisson-ai": 4, "cpu": 4}
    percents = {"frisson-ai": 0.0, "cpu": 0.0}
    menu_transport_flushes = {1: 0, 2: 0}
    no_frame_watchdog = InGameNoFrameWatchdog()
    post_stock_out_drain = _post_stock_out_drain_record(
        terminal_policy_frame=None,
        timeout_seconds=replay_finalize_timeout,
    )
    post_stock_out_drain.update(
        {
            "attempted": False,
            "decision": "not-required",
            "reason": "no-decisive-stock-out-observed",
        }
    )

    def interrupt(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGINT, interrupt)
    try:
        trace_stream = trace_path.open("w", encoding="utf-8")
        if not _launch_and_connect_attested_dolphin(console, iso_path):
            raise RuntimeError("libmelee could not connect to Slippi Dolphin")
        if not frisson_controller.connect() or not cpu_pipe_controller.connect():
            raise RuntimeError("libmelee could not connect both synchronized controller pipes")
        transport = _ControllerPipeLockstep.install(console, raw_controllers)
        frisson_controller = transport.controllers[FRISSON_PORT]
        cpu_pipe_controller = transport.controllers[CPU_PORT]
        transport.prime()
        print(
            f"P1=FRISSON-AI {request.player_1_character} | "
            f"P2=MELEE CPU {request.player_2_character} LEVEL {CPU_LEVEL} | "
            f"stage={MATCH_STAGE} | seed={request.seed} | inference=exact",
            flush=True,
        )

        while processed_frames < max_game_frames:
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
                    cpu_pipe_controller,
                    _cpu_menu_character_for_state(
                        melee,
                        gamestate,
                        request.player_2_character,
                    ),
                    melee.Stage.FINAL_DESTINATION,
                    cpu_level=CPU_LEVEL,
                    autostart=True,
                    frozen_stadium=True,
                )
                frisson_controller.flush()
                cpu_pipe_controller.flush()
                transport.commit_boundary()
                menu_transport_flushes[1] += 1
                menu_transport_flushes[2] += 1
                continue

            if formal_game_end_observed:
                game_end_observed = True
                termination = "natural-game-end"
                break

            in_game = True
            game_frame = int(gamestate.frame)
            _require_exact_player_ports(gamestate, game_frame)
            if first_game_frame is None:
                first_game_frame = game_frame
                first_context = _validate_cpu_first_context(
                    gamestate,
                    character=request.player_1_character,
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

            frame_result = _run_cpu_frame(
                gamestate=gamestate,
                frisson_session=frisson_session,
                frisson_controller=frisson_controller,
                cpu_pipe_controller=cpu_pipe_controller,
                transport=transport,
            )
            inference_seconds.append(frame_result.frisson_inference_seconds)
            barrier_seconds.append(frame_result.barrier_seconds)
            frisson_dispatch_count += 1
            cpu_neutral_pipe_dispatch_count += 1
            trace_stream.write(
                json.dumps(
                    _cpu_trace_row(
                        gamestate,
                        frame_result,
                        character=request.player_1_character,
                    ),
                    sort_keys=True,
                )
                + "\n"
            )
            trace_stream.flush()
            stocks = {
                "frisson-ai": int(gamestate.players[FRISSON_PORT].stock),
                "cpu": int(gamestate.players[CPU_PORT].stock),
            }
            percents = {
                "frisson-ai": float(gamestate.players[FRISSON_PORT].percent),
                "cpu": float(gamestate.players[CPU_PORT].percent),
            }
            processed_frames += 1
            if processed_frames % 60 == 0:
                print(
                    f"frame {game_frame}: Frisson {stocks['frisson-ai']} stocks, CPU9 {stocks['cpu']} stocks",
                    flush=True,
                )
            if has_decisive_zero_stock(gamestate, (FRISSON_PORT, CPU_PORT)):
                game_end_observed = True
                termination = "decisive-stock-out-awaiting-formal-game-end"
                post_stock_out_drain = _post_stock_out_drain_record(
                    terminal_policy_frame=game_frame,
                    timeout_seconds=replay_finalize_timeout,
                )
                post_stock_out_drain["reason"] = "formal-game-end-drain-in-progress"
                post_stock_out_drain = _drain_cpu_formal_game_end(
                    transport=transport,
                    frisson_controller=frisson_controller,
                    cpu_pipe_controller=cpu_pipe_controller,
                    terminal_policy_frame=game_frame,
                    in_game_menu_states=(melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH),
                    timeout_seconds=replay_finalize_timeout,
                )
                formal_game_end_observed = bool(
                    post_stock_out_drain["formal_game_end_observed"]
                )
                if post_stock_out_drain["decision"] != "pass":
                    raise RuntimeError(
                        "decisive stock-out did not reach formal Slippi Game End before shutdown: "
                        f"{post_stock_out_drain['reason']}; {post_stock_out_drain['error']}"
                    )
                termination = "natural-game-end"
                break
        if in_game and processed_frames >= max_game_frames:
            termination = "frame-limit-before-natural-end"
    except BaseException as caught:
        exception = caught
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        if trace_stream is not None:
            trace_stream.close()
        shutdown_method = _stop_console(
            console,
            replay_finalize_timeout,
        )

    replay_paths, replay_records = _collect_replays(
        replay_directory,
        existing_replays,
        project_root,
        first_game_frame,
        last_game_frame,
        sudden_death_transition_observed,
    )
    for replay_path, replay_record in zip(replay_paths, replay_records, strict=True):
        replay_record["cpu_contract"] = _validate_cpu_replay_start(
            replay_path,
            character=request.player_1_character,
        )
    replay_checks = _legacy_replay_gate_checks(
        replay_records,
        sudden_death_transition_observed=sudden_death_transition_observed,
    )
    selected_cpu_contracts = [
        record.get("cpu_contract")
        for record in replay_records
        if record.get("tournament_result_replay") is True
    ]
    cpu_replay_contract = (
        selected_cpu_contracts[0]
        if len(selected_cpu_contracts) == 1 and isinstance(selected_cpu_contracts[0], dict)
        else None
    )
    cpu_replay_checks = (
        cast(dict[str, bool], cpu_replay_contract["checks"])
        if cpu_replay_contract is not None
        else {
            "exact_ports_1_and_2": False,
            "p1_declared_human": False,
            "p2_declared_cpu": False,
            "p2_cpu_level_9": False,
            "p1_requested_character": False,
            "p2_requested_character": False,
            "final_destination": False,
            "game_random_seed_recorded": False,
            "replay_outcome_complete": False,
            "replay_outcome_conclusive": False,
            "replay_outcome_is_win_or_draw": False,
        }
    )
    transport_record = transport.audit_record() if transport is not None else {"installed": False}
    game_start_transport_proof = _game_start_transport_proof(transport_record)
    controller_audit = _selected_frisson_cpu_audit(
        trace_path,
        replay_paths,
        replay_records,
        project_root,
        game_start_transport_proof=game_start_transport_proof,
    )
    controller_checks = cast(dict[str, bool], controller_audit["gate"]["checks"])
    diagnostics = frisson_session.diagnostics()
    metadata = frisson_session.metadata()
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
    transport_ports = cast(dict[str, Any], transport_record.get("ports", {}))
    cpu_transport = cast(dict[str, Any], transport_ports.get("p2", {}))
    gate_checks: dict[str, bool] = {
        "same_character_frisson_p1_vs_native_cpu9_p2_mirror": (
            request.player_1_character == request.player_2_character
            and request.player_1_character in FRISSON_SUPPORTED_CHARACTERS
        ),
        "final_destination": request.stage == MATCH_STAGE,
        "evaluation_seed_valid": _validate_evaluation_seed(request.seed) == request.seed,
        "natural_game_end_observed": game_end_observed,
        "formal_game_end_observed_before_shutdown": formal_game_end_observed,
        "post_stock_out_drain_passed_if_attempted": (
            post_stock_out_drain["decision"] in {"pass", "not-required"}
        ),
        "post_stock_out_drain_no_policy_inference": (
            post_stock_out_drain["policy_inference_calls"] == 0
        ),
        "post_stock_out_drain_wrote_no_controller_trace_rows": (
            post_stock_out_drain["controller_trace_rows_written"] == 0
        ),
        "post_stock_out_drain_neutral_boundaries_exactly_paired": (
            post_stock_out_drain["attempted"] is False
            or (
                post_stock_out_drain["decision"] == "pass"
                and post_stock_out_drain["neutral_boundaries_started"]
                == post_stock_out_drain["neutral_boundaries_committed"]
                == post_stock_out_drain["neutral_dispatches_by_port"]["p1"]
                == post_stock_out_drain["neutral_dispatches_by_port"]["p2"]
            )
        ),
        "game_scope_transport_sealed_before_post_stock_out_drain": (
            post_stock_out_drain["attempted"] is False
            or post_stock_out_drain["game_scope_transport_audit_sealed_before_drain"] is True
        ),
        "entered_gameplay": in_game,
        "processed_at_least_one_frame": processed_frames > 0,
        "first_policy_frame_minus_123": first_game_frame == FIRST_POLICY_FRAME,
        "strict_consecutive_policy_frames": strict_frame_order,
        "one_frisson_inference_per_frame": len(inference_seconds) == processed_frames,
        "one_frisson_dispatch_per_frame": frisson_dispatch_count == processed_frames,
        "one_cpu_neutral_pipe_dispatch_per_frame": (cpu_neutral_pipe_dispatch_count == processed_frames),
        "one_neutral_cpu_pipe_transaction_per_frame": (
            cpu_transport.get("scheduled_boundaries_created") == processed_frames
        ),
        "frisson_session_frame_count_exact": diagnostics.get("frames_total") == processed_frames,
        "frisson_session_barrier_count_exact": (
            diagnostics.get("current_frame_inference_barriers") == processed_frames
            and diagnostics.get("current_frame_inference_barrier_every_frame") is True
        ),
        "frisson_delay_zero": metadata["runtime_contract"]["delay_frames"] == 0,
        "frisson_action_offset_one": metadata["action_contract"]["action_offset_frames"] == 1,
        "frisson_temperature_one": metadata["runtime_contract"]["sample_temperature"] == 1.0,
        "frisson_rolling_256_kv_cache": (
            metadata["runtime_contract"]["kv_cache_capacity_frames"] == EXPECTED_MODEL_CONTEXT_LENGTH
            and metadata["runtime_contract"]["periodic_128_frame_reset"] is False
        ),
        "frisson_has_no_observation_filter": metadata["runtime_contract"]["observation_filter"] is None,
        "frisson_has_no_slippi_ai_fifo": metadata["action_contract"]["slippi_ai_21_frame_fifo"] is False,
        **replay_checks,
        **{f"cpu_replay.{name}": passed for name, passed in cpu_replay_checks.items()},
        **{f"controller_audit.{name}": passed for name, passed in controller_checks.items()},
        **transport_checks,
    }
    if exception is None and not all(gate_checks.values()):
        exception = RuntimeError(
            "Frisson-versus-CPU9 integration gate failed: "
            f"{[name for name, passed in gate_checks.items() if not passed]}"
        )
    result = "complete" if exception is None and all(gate_checks.values()) else "failed"
    replay_outcome = cpu_replay_contract.get("outcome") if cpu_replay_contract is not None else None
    replay_winner_port = replay_outcome.get("winner_port") if isinstance(replay_outcome, dict) else None
    winner = "frisson-ai" if replay_winner_port == 1 else "cpu" if replay_winner_port == 2 else None
    trace_artifact = _file_identity(trace_path, project_root)
    trace_artifact["rows"] = processed_frames
    summary = {
        "schema_version": SCHEMA_VERSION,
        "classification": (
            f"frame-exact Frisson-AI {request.player_1_character} versus Melee native CPU "
            f"level 9 {request.player_2_character} mirror"
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
                "character": request.player_1_character,
                "port": 1,
            },
            "player_2": {
                "model": CPU_MODEL,
                "character": request.player_2_character,
                "port": 2,
                "cpu_level": CPU_LEVEL,
                "gameplay_owner": "melee-native-cpu",
            },
            "stage": MATCH_STAGE,
            "seed": request.seed,
            "seed_scope": "Frisson policy sampling",
            "cpu_randomness": {
                "source": "saved replay Game Start random_seed",
                "game_random_seed": (
                    cpu_replay_contract.get("game_random_seed") if cpu_replay_contract is not None else None
                ),
                "controlled_by_policy_evaluation_seed": False,
            },
            "require_natural_end": True,
            "maximum_game_frames": max_game_frames,
            "blocking_input": True,
            "online_delay_frames": 0,
            "controller_replay_lag_frames": CONTROLLER_REPLAY_LAG_FRAMES,
            "inference_mode": EXACT_INFERENCE_MODE,
        },
        "contract": {
            **contract,
            "opponent": {
                "implementation": "melee-native-cpu",
                "character": request.player_2_character,
                "cpu_level": CPU_LEVEL,
                "p2_pipe_command": "neutral",
                "p2_pipe_purpose": "paired strict FRAME_SYNC transaction",
                "sheik_css_selection": "select Zelda, then hold A during stage loading",
            },
        },
        "frisson": {"metadata": metadata, "diagnostics": diagnostics},
        "cpu": {
            "implementation": "melee-native-cpu",
            "level": CPU_LEVEL,
            "character": request.player_2_character,
            "replay_start_contract": cpu_replay_contract,
        },
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
            "inference_counts": {"frisson-ai": len(inference_seconds), "cpu": 0},
            "dispatch_counts": {
                "frisson-ai": frisson_dispatch_count,
                "cpu-neutral-pipe": cpu_neutral_pipe_dispatch_count,
            },
            "mean_inference_seconds": (
                sum(inference_seconds) / len(inference_seconds) if inference_seconds else None
            ),
            "maximum_inference_seconds": max(inference_seconds) if inference_seconds else None,
            "mean_barrier_seconds": (
                sum(barrier_seconds) / len(barrier_seconds) if barrier_seconds else None
            ),
            "last_stocks": stocks,
            "last_percents": percents,
            "winner": winner,
            "winner_source": "selected-replay-derived-outcome",
            "replay_outcome": replay_outcome,
            "game_end_observed": game_end_observed,
            "formal_game_end_observed": formal_game_end_observed,
            "post_stock_out_drain": post_stock_out_drain,
            "sudden_death_transition_observed": sudden_death_transition_observed,
            "termination": termination,
            "wall_seconds": time.time() - started_at,
            "shutdown_method": shutdown_method,
            "controller_transport": {
                **transport_record,
                "menu_flushes": {f"p{port}": menu_transport_flushes[port] for port in (1, 2)},
            },
        },
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


def run_frisson_cpu_match(
    config_path: Path,
    iso_path: Path | None = None,
    request: FrissonCpuMatchRequest | None = None,
) -> dict[str, Any]:
    """Run one exact same-character game against Melee's native CPU level 9."""
    match_request = FrissonCpuMatchRequest() if request is None else request
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
    _validate_config_contract(config, match_request)  # type: ignore[arg-type]
    policy_config = _frisson_policy_config(
        config,
        project_root,
        match_request,  # type: ignore[arg-type]
    )
    checkpoint_identity = inspect_frisson_checkpoint(policy_config.checkpoint_path)
    final_checkpoint = _validate_final_cpu_checkpoint(checkpoint_identity, project_root)
    contract = _validate_config_contract(
        config,
        match_request,  # type: ignore[arg-type]
        checkpoint_identity=checkpoint_identity,
    )
    contract = {
        **contract,
        "frisson_character": match_request.player_1_character,
        "frisson_supported_characters": list(FRISSON_SUPPORTED_CHARACTERS),
        "final_checkpoint_scope": {
            "allowed_profiles": ["10m", "75m"],
            "selected": final_checkpoint,
        },
    }
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
            "src/melee_policy/integration/frisson_policy.py",
            "src/melee_policy/integration/frisson_match.py",
            "src/melee_policy/integration/frisson_cpu_match.py",
            "src/melee_policy/integration/game_bundle.py",
            "src/melee_policy/integration/match_runtime.py",
            "src/melee_policy/integration/replay_video.py",
            "patches/slippi-dolphin-two-pipe-frame-sync.patch",
        ),
    )
    session = FrissonPolicySession(policy_config)
    session.start()
    try:
        summary = _run_cpu_console(
            config,
            project_root,
            image_path,
            match_request,
            contract,
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
    "CPU_CHARACTER",
    "CPU_LEVEL",
    "CPU_MODEL",
    "FINAL_CPU_CHECKPOINT_IDENTITIES",
    "FrissonCpuMatchRequest",
    "run_frisson_cpu_match",
]
