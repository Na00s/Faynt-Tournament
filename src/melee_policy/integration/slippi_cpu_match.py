"""Run one official Slippi-AI release against Melee's native level 9 CPU.

Slippi-AI keeps its released observation, recurrent inference, categorical
decoder, and action FIFO.  Dolphin runs at console delay zero, so medium-v2
retains its 21-frame FIFO while the DK and Dr. Mario specialists retain their
18-frame FIFOs.  The P2 controller pipe carries neutral FRAME_SYNC traffic;
Melee owns every CPU decision internally.
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

from melee_policy.integration.frame_watchdog import InGameNoFrameWatchdog
from melee_policy.integration.frisson_cpu_match import (
    CPU_LEVEL,
    CPU_MODEL,
    CPU_PORT,
    POST_STOCK_OUT_DRAIN_MAX_STEP_CALLS,
    _console_observed_formal_game_end,
    _cpu_menu_character_for_state,
    _validate_cpu_replay_start,
)
from melee_policy.integration.frisson_match import (
    FIRST_POLICY_FRAME,
    MATCH_SEED,
    MATCH_STAGE,
    _collect_replays,
    _console_options,
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
    MEDIUM_V2_SUPPORTED_STAGES,
    SAMPLE_TEMPERATURE,
    SLIPPI_AI_REPOSITORY_URL,
    SLIPPI_AI_SOURCE_REVISION,
    CanonicalControllerCommand,
    SlippiAIPolicyConfig,
    SlippiAIPolicySession,
    send_canonical_controller,
    slippi_ai_release_capabilities,
    slippi_ai_release_contract,
)
from melee_policy.integration.slippi_match import (
    CONTROLLER_REPLAY_LAG_FRAMES,
    LAUNCHABLE_PRIMARY_CHARACTERS,
    _game_start_transport_proof,
    _piecewise_game_start_replay_frames,
    _player_state,
    _read_replay_controller_states,
    _read_trace_rows,
    _trace_command_matches_replay,
)

SCHEMA_VERSION = "integration.slippi_ai_vs_cpu9.v1"
TRACE_SCHEMA_VERSION = "integration.slippi_ai_vs_cpu9.controller_trace.v1"
CONTROLLER_AUDIT_SCHEMA_VERSION = "integration.slippi_ai_vs_cpu9.controller_audit.v1"
POST_STOCK_OUT_DRAIN_SCHEMA_VERSION = "integration.slippi_ai_vs_cpu9.post_stock_out_drain.v1"
SLIPPI_PORT = 1
SLIPPI_CONSOLE_DELAY_FRAMES = 0
EXACT_INFERENCE_MODE = "synchronous-concurrent"


@dataclass(frozen=True, slots=True)
class SlippiCpuMatchRequest:
    """One official Slippi-AI release in a same-character CPU9 mirror."""

    player_1_model: str = "slippi-ai"
    player_2_model: str = CPU_MODEL
    player_1_character: str = "FOX"
    player_2_character: str = "FOX"
    player_1_slippi_release: str = "medium-v2"
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
    allow_player_1_ood_character: bool = False

    def validate(self) -> None:
        release = slippi_ai_release_contract(self.player_1_slippi_release)
        fixed = {
            "player_1_model": (
                self.player_1_model.strip().lower().replace("_", "-"),
                "slippi-ai",
            ),
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
            raise ValueError(f"Slippi-AI-versus-CPU9 fixed match contract mismatch: {mismatches}")
        for label, character in (
            ("player 1", self.player_1_character),
            ("player 2", self.player_2_character),
        ):
            if character not in LAUNCHABLE_PRIMARY_CHARACTERS:
                raise ValueError(f"{label} character is not independently launchable: {character!r}")
        if self.player_1_character != self.player_2_character:
            raise ValueError("Slippi-AI-versus-CPU9 requires a same-character mirror")
        if not isinstance(self.allow_player_1_ood_character, bool):
            raise TypeError("allow_player_1_ood_character must be a boolean")
        if (
            self.player_1_character not in release.supported_characters
            and not self.allow_player_1_ood_character
        ):
            raise ValueError(
                f"{release.key} does not cover {self.player_1_character!r}; "
                "an explicit P1 OOD opt-in is required for transfer diagnostics"
            )
        if self.player_1_assets is not None or self.player_2_assets is not None:
            raise ValueError("Slippi-AI and Melee's native CPU do not accept asset directories")
        if self.player_2_checkpoint is not None:
            raise ValueError("Melee's native CPU does not accept a checkpoint")
        if self.player_2_name is not None:
            raise ValueError("Melee's native CPU does not accept a player name")
        if self.player_2_temperature is not None:
            raise ValueError("Melee's native CPU does not accept a sampling temperature")
        if self.player_1_temperature is not None and self.player_1_temperature != SAMPLE_TEMPERATURE:
            raise ValueError("release-card Slippi-AI sampling temperature is fixed at 1.0")
        if self.max_game_frames is not None and self.max_game_frames < 1:
            raise ValueError("max_game_frames must be positive")
        if not isinstance(self.save_slp, bool) or not isinstance(self.save_video, bool):
            raise TypeError("save_slp and save_video must be booleans")
        _validate_evaluation_seed(self.seed)
        if self.artifact_label is not None and (
            not self.artifact_label
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
                for character in self.artifact_label
            )
        ):
            raise ValueError(
                "artifact_label must contain only lowercase letters, digits, underscores, or hyphens"
            )


@dataclass(frozen=True, slots=True)
class _SlippiCpuFrameResult:
    game_frame: int
    command: CanonicalControllerCommand
    slippi_dispatch: dict[str, Any]
    cpu_pipe_dispatch: dict[str, Any]
    inference_seconds: float
    barrier_seconds: float


def _validate_config_contract(
    config: dict[str, Any],
    project_root: Path,
    request: SlippiCpuMatchRequest,
) -> dict[str, Any]:
    request.validate()
    release = slippi_ai_release_contract(request.player_1_slippi_release)
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
    if frozenset(MEDIUM_V2_SUPPORTED_STAGES) != frozenset(
        {
            "BATTLEFIELD",
            "DREAMLAND",
            "FINAL_DESTINATION",
            "FOUNTAIN_OF_DREAMS",
            "POKEMON_STADIUM",
            "YOSHIS_STORY",
        }
    ):
        raise AssertionError("pinned rendered stage contract changed")
    policy_config = _slippi_policy_config(config, project_root, request)
    policy_config.validate()
    covered = request.player_1_character in release.supported_characters
    return {
        "exact_mode_only": True,
        "inference_mode": EXACT_INFERENCE_MODE,
        "release_contract": release.as_dict(),
        "checkpoint_delay_frames": release.policy_delay_frames,
        "console_delay_frames": SLIPPI_CONSOLE_DELAY_FRAMES,
        "effective_policy_delay_frames": release.policy_delay_frames,
        "requested_character": request.player_1_character,
        "requested_character_covered_by_training": covered,
        "forced_ood_character_transfer": not covered,
        "ood_opt_in": request.allow_player_1_ood_character,
        "checkpoint_capabilities": slippi_ai_release_capabilities(release.key),
    }


def _slippi_policy_config(
    config: dict[str, Any],
    project_root: Path,
    request: SlippiCpuMatchRequest,
) -> SlippiAIPolicyConfig:
    release = slippi_ai_release_contract(request.player_1_slippi_release)
    slippi = cast(dict[str, Any], config["slippi_ai"])
    if request.player_1_checkpoint is None:
        checkpoint_path = (
            (project_root / str(slippi["checkpoint"])).resolve()
            if release.key == "medium-v2"
            else release.checkpoint_path(project_root)
        )
    else:
        checkpoint_path = request.player_1_checkpoint.expanduser().resolve()
    policy_config = SlippiAIPolicyConfig(
        source_directory=(project_root / str(slippi["source_directory"])).resolve(),
        checkpoint_path=checkpoint_path,
        port=SLIPPI_PORT,
        opponent_port=CPU_PORT,
        release=release.key,
        name=request.player_1_name or str(slippi.get("default_name", DEFAULT_PLAYER_NAME)),
        sample_temperature=SAMPLE_TEMPERATURE,
        policy_delay_frames=release.policy_delay_frames,
        console_delay_frames=SLIPPI_CONSOLE_DELAY_FRAMES,
        async_inference=True,
        compile=True,
        tf_jit_compile=False,
        batch_steps=0,
        mirror=False,
        requested_character=request.player_1_character,
        allow_ood_character=request.allow_player_1_ood_character,
    )
    policy_config.validate()
    return policy_config


def _run_cpu_frame(
    *,
    gamestate: Any,
    session: SlippiAIPolicySession,
    slippi_controller: Any,
    cpu_pipe_controller: Any,
    transport: _ControllerPipeLockstep,
) -> _SlippiCpuFrameResult:
    """Finish the current recurrent update before releasing the paired boundary."""

    game_frame = int(gamestate.frame)
    barrier_started = time.perf_counter()
    inference_started = time.perf_counter()
    command = session.step(gamestate)
    inference_seconds = time.perf_counter() - inference_started
    command.validate()
    barrier_seconds = time.perf_counter() - barrier_started

    transport.begin_boundary(reason="slippi-ai-vs-cpu9-gameplay", game_frame=game_frame)
    slippi_dispatch = send_canonical_controller(
        slippi_controller,
        command,
        flush=False,
    ).as_dict()
    slippi_dispatch.update(
        {
            "called": True,
            "queued_after_current_frame_inference": True,
            "source_exact_deferred_flush": True,
        }
    )
    transport.schedule_next_boundary(
        SLIPPI_PORT,
        reason="slippi-ai-native-fifo-command",
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
    return _SlippiCpuFrameResult(
        game_frame=game_frame,
        command=command,
        slippi_dispatch=slippi_dispatch,
        cpu_pipe_dispatch=cpu_pipe_dispatch,
        inference_seconds=inference_seconds,
        barrier_seconds=barrier_seconds,
    )


def _trace_row(
    gamestate: Any,
    result: _SlippiCpuFrameResult,
    *,
    character: str,
    effective_policy_delay_frames: int,
    processed_frames: int,
) -> dict[str, Any]:
    game_frame = result.game_frame
    source_frame = (
        None
        if processed_frames < effective_policy_delay_frames
        else game_frame - effective_policy_delay_frames
    )
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "game_frame": game_frame,
        "barrier": {
            "source_frame": game_frame,
            "slippi_current_frame_inference_complete": True,
            "controller_transaction_opened_after_inference": True,
            "both_pipe_transactions_committed_together": True,
            "seconds": result.barrier_seconds,
        },
        "slots": {
            "p1": {
                "port": SLIPPI_PORT,
                "model": "slippi-ai",
                "requested_character": character,
                "player_state": _player_state(gamestate.players[SLIPPI_PORT]),
                "command": result.command.as_dict(),
                "inference": {
                    "called": True,
                    "source_frame": source_frame,
                    "command_age_frames": (
                        None if source_frame is None else game_frame - source_frame
                    ),
                    "native_dummy_prefix": source_frame is None,
                    "policy_delay_frames": effective_policy_delay_frames,
                },
                "controller_dispatch": result.slippi_dispatch,
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


def _post_stock_out_drain_record(
    *, terminal_policy_frame: int | None, timeout_seconds: float
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


def _drain_cpu_formal_game_end(
    *,
    transport: _ControllerPipeLockstep,
    slippi_controller: Any,
    cpu_pipe_controller: Any,
    terminal_policy_frame: int,
    in_game_menu_states: tuple[Any, ...],
    timeout_seconds: float,
    clock: Callable[[], float] = time.monotonic,
    formal_game_end_probe: Callable[[Any], bool] = _console_observed_formal_game_end,
) -> dict[str, Any]:
    """Advance formal Game End with paired neutral traffic and zero policy inference."""

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
                reason="slippi-ai-vs-cpu9-post-stock-out-game-end-drain",
                game_frame=game_frame,
            )
            record["neutral_boundaries_started"] += 1
            phase = "slippi-ai-neutral-dispatch"
            send_canonical_controller(slippi_controller, neutral, flush=False)
            record["neutral_dispatches_by_port"]["p1"] += 1
            phase = "slippi-ai-neutral-schedule"
            transport.schedule_next_boundary(
                SLIPPI_PORT,
                reason="slippi-ai-post-stock-out-neutral-finalization",
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


def _validate_first_context(gamestate: Any, *, character: str) -> dict[str, Any]:
    observed_stage = str(getattr(gamestate.stage, "name", gamestate.stage)).split(".")[-1]
    observed_characters = {
        f"p{port}": str(
            getattr(gamestate.players[port].character, "name", gamestate.players[port].character)
        ).split(".")[-1]
        for port in (SLIPPI_PORT, CPU_PORT)
    }
    checks = {
        "first_frame_minus_123": int(gamestate.frame) == FIRST_POLICY_FRAME,
        "final_destination": observed_stage == MATCH_STAGE,
        "slippi_requested_character": observed_characters["p1"] == character,
        "cpu_requested_character": observed_characters["p2"] == character,
        "slippi_live_cpu_level_zero": int(gamestate.players[SLIPPI_PORT].cpu_level) == 0,
        "cpu_live_level_9": int(gamestate.players[CPU_PORT].cpu_level) == CPU_LEVEL,
    }
    if not all(checks.values()):
        raise RuntimeError(f"first Slippi-AI-versus-CPU9 game context mismatch: {checks}")
    return {
        "frame": int(gamestate.frame),
        "stage": observed_stage,
        "characters": observed_characters,
        "costumes": {
            f"p{port}": int(gamestate.players[port].costume)
            for port in (SLIPPI_PORT, CPU_PORT)
        },
        "cpu_levels": {
            f"p{port}": int(gamestate.players[port].cpu_level)
            for port in (SLIPPI_PORT, CPU_PORT)
        },
        "checks": checks,
    }


def _audit_slippi_only_controller_boundary(
    trace_path: Path,
    replay_path: Path,
    project_root: Path,
    *,
    game_start_transport_proof: Mapping[str, Any],
) -> dict[str, Any]:
    """Hard-gate every observable Slippi-AI P1 controller dimension."""

    rows = _read_trace_rows(trace_path)
    replay_states = _read_replay_controller_states(replay_path)
    if not rows or not replay_states:
        raise ValueError("controller audit requires trace and replay controller evidence")
    trace_by_frame = {int(row["game_frame"]): row for row in rows}
    if len(trace_by_frame) != len(rows):
        raise ValueError("controller audit trace frames must be unique")
    trace_frames = sorted(trace_by_frame)
    replay_frames = sorted(int(frame) for frame in replay_states)
    replay_covers_observed_trace_states = all(
        frame in replay_states and SLIPPI_PORT in replay_states[frame]
        for frame in trace_frames
    )
    if trace_frames[0] != FIRST_POLICY_FRAME:
        raise ValueError("controller audit must begin at policy frame -123")
    first_slot = cast(Mapping[str, Any], trace_by_frame[FIRST_POLICY_FRAME]["slots"])["p1"]
    if not isinstance(first_slot, Mapping) or first_slot.get("model") != "slippi-ai":
        raise ValueError("controller audit lacks the Slippi-AI P1 slot")
    candidates = {
        "early-latched": FIRST_POLICY_FRAME + CONTROLLER_REPLAY_LAG_FRAMES,
        "delayed-first": FIRST_POLICY_FRAME + CONTROLLER_REPLAY_LAG_FRAMES + 1,
    }
    candidate_matches = {}
    for name, replay_frame in candidates.items():
        state = replay_states.get(replay_frame, {}).get(SLIPPI_PORT)
        candidate_matches[name] = bool(
            isinstance(state, Mapping) and _trace_command_matches_replay(first_slot, state)
        )
    if candidate_matches["early-latched"] and not candidate_matches["delayed-first"]:
        decision = "early-latched"
    elif candidate_matches["delayed-first"]:
        decision = (
            "observationally-equivalent-delayed-first"
            if candidate_matches["early-latched"]
            else "delayed-first"
        )
    else:
        raise ValueError(f"first Slippi-AI command has no permitted replay mapping: {candidate_matches}")
    selected_first_replay_frame = (
        candidates["early-latched"]
        if decision == "early-latched"
        else candidates["delayed-first"]
    )
    replay_frame_by_trace_frame, startup_unavailable = _piecewise_game_start_replay_frames(
        trace_by_frame,
        replay_states,
        lag_frames=CONTROLLER_REPLAY_LAG_FRAMES,
        transport_proof=game_start_transport_proof,
        first_policy_replay_frame=selected_first_replay_frame,
    )
    terminal_unobservable = [
        frame
        for frame in trace_frames
        if frame not in startup_unavailable
        and frame not in replay_frame_by_trace_frame
        and frame + CONTROLLER_REPLAY_LAG_FRAMES > replay_frames[-1]
    ]
    terminal_suffix_bounded = len(terminal_unobservable) <= CONTROLLER_REPLAY_LAG_FRAMES and (
        not terminal_unobservable
        or terminal_unobservable == trace_frames[-len(terminal_unobservable) :]
    )
    comparable_frames = [
        frame
        for frame in trace_frames
        if frame not in startup_unavailable and frame not in terminal_unobservable
    ]
    mismatches: list[dict[str, int]] = []
    missing_pairs: list[int] = []
    for trace_frame in comparable_frames:
        replay_frame = replay_frame_by_trace_frame.get(trace_frame)
        if replay_frame is None or SLIPPI_PORT not in replay_states.get(replay_frame, {}):
            missing_pairs.append(trace_frame)
            continue
        trace_slot = cast(Mapping[str, Any], trace_by_frame[trace_frame]["slots"])["p1"]
        if not isinstance(trace_slot, Mapping):
            missing_pairs.append(trace_frame)
            continue
        dispatch = trace_slot.get("controller_dispatch")
        if not isinstance(dispatch, Mapping) or dispatch.get("called") is not True:
            missing_pairs.append(trace_frame)
            continue
        if not _trace_command_matches_replay(
            trace_slot,
            replay_states[replay_frame][SLIPPI_PORT],
        ):
            mismatches.append({"trace_frame": trace_frame, "replay_frame": replay_frame})
    trace_consecutive = trace_frames == list(range(trace_frames[0], trace_frames[-1] + 1))
    expected_startup_pairs = [
        [FIRST_POLICY_FRAME, selected_first_replay_frame],
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
    steady_state_exact = FIRST_POLICY_FRAME + 5 in replay_frame_by_trace_frame and all(
        replay_frame == trace_frame + CONTROLLER_REPLAY_LAG_FRAMES
        for trace_frame, replay_frame in replay_frame_by_trace_frame.items()
        if trace_frame >= FIRST_POLICY_FRAME + 5
    )
    startup_pairs_exact = (
        startup_unavailable == [FIRST_POLICY_FRAME + 4]
        and actual_startup_pairs == expected_startup_pairs
        and steady_state_exact
    )
    checks = {
        "full_causal_overlap": bool(comparable_frames)
        and trace_consecutive
        and replay_covers_observed_trace_states
        and terminal_suffix_bounded
        and startup_pairs_exact
        and not missing_pairs,
        "trace_frames_consecutive": trace_consecutive,
        "replay_covers_every_observed_trace_state": replay_covers_observed_trace_states,
        "terminal_unobservable_tail_bounded_by_lag": terminal_suffix_bounded,
        "game_start_piecewise_alignment_exact": startup_pairs_exact,
        "game_start_transport_proof_exact": game_start_transport_proof.get("decision") == "pass",
        "slippi_slot_aligned": not missing_pairs,
        "slippi_controller_evidence": bool(comparable_frames),
        "slippi_all_mapped_commands_hard_gated": not missing_pairs,
        "slippi_controller_values_exact": not mismatches,
        "slippi_no_dispatch_rule_exact": not missing_pairs,
    }
    return {
        "schema_version": CONTROLLER_AUDIT_SCHEMA_VERSION,
        "classification": "Slippi-AI P1 decoded commands verified against the saved replay",
        "alignment": {
            "mode": "piecewise-game-start",
            "steady_state_trace_to_replay_lag_frames": CONTROLLER_REPLAY_LAG_FRAMES,
            "first_policy_frame_mapping": {
                "candidate_replay_frames": candidates,
                "candidate_matches": candidate_matches,
                "decision": decision,
                "selected_replay_frame": selected_first_replay_frame,
            },
            "game_start_controller_latch_unobservable_trace_frames": startup_unavailable,
            "startup_mapped_pairs": actual_startup_pairs,
            "permitted_terminal_unobservable_trace_frames": terminal_unobservable,
            "missing_internal_pairs": missing_pairs,
            "game_start_transport_proof": dict(game_start_transport_proof),
        },
        "slots": {
            "p1": {
                "port": SLIPPI_PORT,
                "model": "slippi-ai",
                "controller_frame_pairs_compared": len(comparable_frames) - len(missing_pairs),
                "mismatch_frames": mismatches,
                "native_dummy_prefix_hard_gated": True,
            }
        },
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


def _unavailable_controller_audit(
    trace_path: Path,
    project_root: Path,
    reason: str,
    *,
    game_start_transport_proof: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "full_causal_overlap": False,
        "trace_frames_consecutive": False,
        "replay_covers_every_observed_trace_state": False,
        "terminal_unobservable_tail_bounded_by_lag": False,
        "game_start_piecewise_alignment_exact": False,
        "game_start_transport_proof_exact": False,
        "slippi_slot_aligned": False,
        "slippi_controller_evidence": False,
        "slippi_all_mapped_commands_hard_gated": False,
        "slippi_controller_values_exact": False,
        "slippi_no_dispatch_rule_exact": False,
    }
    return {
        "schema_version": CONTROLLER_AUDIT_SCHEMA_VERSION,
        "classification": "Slippi-AI P1 decoded command replay audit unavailable",
        "error": reason,
        "trace": (
            _file_identity(trace_path, project_root)
            if trace_path.is_file()
            else {"path": _display_path(trace_path, project_root), "missing": True}
        ),
        "alignment": {"game_start_transport_proof": dict(game_start_transport_proof)},
        "slots": {},
        "excluded_slots": {"p2": {"model": CPU_MODEL, "reason": "native Melee CPU"}},
        "gate": {"decision": "fail", "checks": checks},
    }


def _selected_controller_audit(
    trace_path: Path,
    replay_paths: list[Path],
    replay_records: list[dict[str, Any]],
    project_root: Path,
    *,
    game_start_transport_proof: Mapping[str, Any],
) -> dict[str, Any]:
    selected = [
        index
        for index, record in enumerate(replay_records)
        if record.get("tournament_result_replay") is True
    ]
    if len(selected) != 1 or selected[0] >= len(replay_paths):
        return _unavailable_controller_audit(
            trace_path,
            project_root,
            "controller audit requires exactly one trace-covering result replay",
            game_start_transport_proof=game_start_transport_proof,
        )
    try:
        return _audit_slippi_only_controller_boundary(
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
            game_start_transport_proof=game_start_transport_proof,
        )


def _run_cpu_console(
    config: dict[str, Any],
    project_root: Path,
    iso_path: Path,
    request: SlippiCpuMatchRequest,
    contract: dict[str, Any],
    session: SlippiAIPolicySession,
    tensorflow_setup: dict[str, Any],
    reproducibility: dict[str, Any],
) -> dict[str, Any]:
    import melee

    output_directory = project_root / str(config["slippi_integration"]["output_directory"])
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
        port: melee.Controller(console=console, port=port, type=melee.ControllerType.STANDARD)
        for port in (SLIPPI_PORT, CPU_PORT)
    }
    menu_helpers = {port: melee.MenuHelper() for port in (SLIPPI_PORT, CPU_PORT)}
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
    inference_seconds: list[float] = []
    barrier_seconds: list[float] = []
    slippi_dispatch_count = 0
    cpu_neutral_dispatch_count = 0
    stocks = {"slippi-ai": 4, "cpu": 4}
    percents = {"slippi-ai": 0.0, "cpu": 0.0}
    menu_transport_flushes = {1: 0, 2: 0}
    no_frame_watchdog = InGameNoFrameWatchdog()
    diagnostics_before_close: dict[str, Any] = {}
    diagnostics_after_close: dict[str, Any] = {}
    session_metadata: dict[str, Any] = {}
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
        session.start()
        session_metadata = session.metadata()
        if not _launch_and_connect_attested_dolphin(console, iso_path):
            raise RuntimeError("libmelee could not connect to Slippi Dolphin")
        if not all(controller.connect() for controller in raw_controllers.values()):
            raise RuntimeError("libmelee could not connect both synchronized controller pipes")
        transport = _ControllerPipeLockstep.install(console, raw_controllers)
        controllers = dict(transport.controllers)
        transport.prime()
        print(
            f"P1=SLIPPI-AI {request.player_1_slippi_release} {request.player_1_character} | "
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
            if gamestate.menu_state not in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
                if in_game:
                    game_end_observed = True
                    termination = "natural-game-end"
                    break
                if time.time() - started_at > menu_timeout:
                    raise TimeoutError("automatic menu navigation did not start a match before timeout")
                transport.begin_boundary(reason="menu", game_frame=None)
                menu_helpers[SLIPPI_PORT].menu_helper_simple(
                    gamestate,
                    controllers[SLIPPI_PORT],
                    melee.Character[request.player_1_character],
                    melee.Stage.FINAL_DESTINATION,
                    cpu_level=0,
                    autostart=False,
                    frozen_stadium=True,
                )
                menu_helpers[CPU_PORT].menu_helper_simple(
                    gamestate,
                    controllers[CPU_PORT],
                    _cpu_menu_character_for_state(melee, gamestate, request.player_2_character),
                    melee.Stage.FINAL_DESTINATION,
                    cpu_level=CPU_LEVEL,
                    autostart=True,
                    frozen_stadium=True,
                )
                controllers[SLIPPI_PORT].flush()
                controllers[CPU_PORT].flush()
                transport.commit_boundary()
                menu_transport_flushes[1] += 1
                menu_transport_flushes[2] += 1
                continue
            if formal_game_end_observed:
                game_end_observed = True
                termination = "natural-game-end"
                break
            if in_game and gamestate.menu_state == melee.Menu.SUDDEN_DEATH:
                sudden_death_transition_observed = True

            in_game = True
            game_frame = int(gamestate.frame)
            _require_exact_player_ports(gamestate, game_frame)
            if first_game_frame is None:
                first_game_frame = game_frame
                first_context = _validate_first_context(
                    gamestate,
                    character=request.player_1_character,
                )
            if previous_game_frame is not None:
                delta = game_frame - previous_game_frame
                frame_delta_counts[delta] += 1
                if delta != 1:
                    raise RuntimeError(
                        f"rendered policy frames are not consecutive: {previous_game_frame} to {game_frame}"
                    )
            previous_game_frame = game_frame
            last_game_frame = game_frame

            frame_result = _run_cpu_frame(
                gamestate=gamestate,
                session=session,
                slippi_controller=controllers[SLIPPI_PORT],
                cpu_pipe_controller=controllers[CPU_PORT],
                transport=transport,
            )
            inference_seconds.append(frame_result.inference_seconds)
            barrier_seconds.append(frame_result.barrier_seconds)
            slippi_dispatch_count += 1
            cpu_neutral_dispatch_count += 1
            trace_stream.write(
                json.dumps(
                    _trace_row(
                        gamestate,
                        frame_result,
                        character=request.player_1_character,
                        effective_policy_delay_frames=int(
                            contract["effective_policy_delay_frames"]
                        ),
                        processed_frames=processed_frames,
                    ),
                    sort_keys=True,
                )
                + "\n"
            )
            trace_stream.flush()
            stocks = {
                "slippi-ai": int(gamestate.players[SLIPPI_PORT].stock),
                "cpu": int(gamestate.players[CPU_PORT].stock),
            }
            percents = {
                "slippi-ai": float(gamestate.players[SLIPPI_PORT].percent),
                "cpu": float(gamestate.players[CPU_PORT].percent),
            }
            processed_frames += 1
            if processed_frames % 60 == 0:
                print(
                    f"frame {game_frame}: Slippi-AI {stocks['slippi-ai']} stocks, "
                    f"CPU9 {stocks['cpu']} stocks",
                    flush=True,
                )
            if has_decisive_zero_stock(gamestate, (SLIPPI_PORT, CPU_PORT)):
                game_end_observed = True
                termination = "decisive-stock-out-awaiting-formal-game-end"
                inference_frames_before_drain = int(session.diagnostics().get("frames_total", -1))
                trace_rows_before_drain = processed_frames
                post_stock_out_drain = _post_stock_out_drain_record(
                    terminal_policy_frame=game_frame,
                    timeout_seconds=replay_finalize_timeout,
                )
                post_stock_out_drain["reason"] = "formal-game-end-drain-in-progress"
                post_stock_out_drain = _drain_cpu_formal_game_end(
                    transport=transport,
                    slippi_controller=controllers[SLIPPI_PORT],
                    cpu_pipe_controller=controllers[CPU_PORT],
                    terminal_policy_frame=game_frame,
                    in_game_menu_states=(melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH),
                    timeout_seconds=replay_finalize_timeout,
                )
                inference_frames_after_drain = int(session.diagnostics().get("frames_total", -1))
                inference_frame_delta = (
                    inference_frames_after_drain - inference_frames_before_drain
                )
                post_stock_out_drain["policy_inference_calls_by_model"] = {
                    "slippi-ai": inference_frame_delta,
                    "cpu": 0,
                }
                post_stock_out_drain["policy_inference_counters_observed"] = (
                    inference_frames_before_drain >= 0 and inference_frames_after_drain >= 0
                )
                post_stock_out_drain["policy_inference_calls"] = inference_frame_delta
                post_stock_out_drain["controller_trace_counter_observed"] = True
                post_stock_out_drain["controller_trace_rows_written"] = (
                    processed_frames - trace_rows_before_drain
                )
                formal_game_end_observed = bool(post_stock_out_drain["formal_game_end_observed"])
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
        try:
            diagnostics_before_close = session.diagnostics()
            session.close()
            diagnostics_after_close = session.diagnostics()
        except BaseException as session_error:
            if exception is None:
                exception = session_error
        if trace_stream is not None:
            trace_stream.close()
        shutdown_method = _stop_console(console, replay_finalize_timeout)

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
    transport_start_proof = _game_start_transport_proof(transport_record)
    controller_audit = _selected_controller_audit(
        trace_path,
        replay_paths,
        replay_records,
        project_root,
        game_start_transport_proof=transport_start_proof,
    )
    controller_checks = cast(dict[str, bool], controller_audit["gate"]["checks"])
    transport_checks = (
        transport.gate_checks()
        if transport is not None
        else {"controller_pipe_lockstep_installed": False}
    )
    diagnostics = diagnostics_before_close
    controller_contract = cast(dict[str, Any], session_metadata.get("controller_contract", {}))
    release = slippi_ai_release_contract(request.player_1_slippi_release)
    drain_paired = (
        post_stock_out_drain["attempted"] is False
        or (
            post_stock_out_drain["decision"] == "pass"
            and post_stock_out_drain["neutral_boundaries_started"]
            == post_stock_out_drain["neutral_boundaries_committed"]
            == post_stock_out_drain["neutral_dispatches_by_port"]["p1"]
            == post_stock_out_drain["neutral_dispatches_by_port"]["p2"]
        )
    )
    gate_checks: dict[str, bool] = {
        "same_character_slippi_p1_vs_native_cpu9_p2_mirror": (
            request.player_1_character == request.player_2_character
        ),
        "selected_release_identity_exact": (
            session_metadata.get("release_contract") == release.as_dict()
        ),
        "selected_release_character_stratum_exact": (
            contract["requested_character_covered_by_training"]
            == (request.player_1_character in release.supported_characters)
            and contract["forced_ood_character_transfer"]
            == (request.player_1_character not in release.supported_characters)
        ),
        "selected_release_delay_exact": (
            int(contract["checkpoint_delay_frames"]) == release.policy_delay_frames
            and int(contract["console_delay_frames"]) == 0
            and int(contract["effective_policy_delay_frames"]) == release.policy_delay_frames
        ),
        "entered_gameplay": in_game,
        "processed_at_least_one_frame": processed_frames > 0,
        "first_policy_frame_minus_123": first_game_frame == FIRST_POLICY_FRAME,
        "strict_consecutive_policy_frames": all(delta == 1 for delta in frame_delta_counts),
        "formal_game_end_observed_before_shutdown": formal_game_end_observed,
        "post_stock_out_drain_passed_if_attempted": (
            post_stock_out_drain["decision"] in {"pass", "not-required"}
        ),
        "post_stock_out_drain_no_policy_inference": (
            post_stock_out_drain["attempted"] is False
            or (
                post_stock_out_drain["policy_inference_counters_observed"] is True
                and post_stock_out_drain["policy_inference_calls"] == 0
            )
        ),
        "post_stock_out_drain_wrote_no_controller_trace_rows": (
            post_stock_out_drain["attempted"] is False
            or (
                post_stock_out_drain["controller_trace_counter_observed"] is True
                and post_stock_out_drain["controller_trace_rows_written"] == 0
            )
        ),
        "post_stock_out_drain_neutral_boundaries_exactly_paired": drain_paired,
        "game_scope_transport_sealed_before_post_stock_out_drain": (
            post_stock_out_drain["attempted"] is False
            or post_stock_out_drain["game_scope_transport_audit_sealed_before_drain"] is True
        ),
        "one_slippi_session_step_per_frame": diagnostics.get("frames_total") == processed_frames,
        "slippi_current_frame_inference_barrier": (
            diagnostics.get("current_frame_inference_barriers") == processed_frames
            and diagnostics.get("current_frame_inference_barrier_every_frame") is True
        ),
        "capture_matches_native_decoder": (
            diagnostics.get("capture_decoder_mismatches") == 0
            and diagnostics.get("capture_decoder_assertions") == processed_frames
        ),
        "one_slippi_dispatch_per_frame": slippi_dispatch_count == processed_frames,
        "one_cpu_neutral_pipe_dispatch_per_frame": cpu_neutral_dispatch_count == processed_frames,
        "slippi_source_exact_deferred_flush": (
            controller_contract.get("native_sender_flushes") is False
            and controller_contract.get("source_exact_deferred_flush_supported") is True
        ),
        "tensorflow_cpu_only": tensorflow_setup.get("physical_gpu_count") == 0
        and tensorflow_setup.get("logical_gpu_count") == 0,
        **replay_checks,
        **{f"cpu_replay.{name}": passed for name, passed in cpu_replay_checks.items()},
        **{f"controller_audit.{name}": passed for name, passed in controller_checks.items()},
        **transport_checks,
    }
    if exception is None and not all(gate_checks.values()):
        exception = RuntimeError(
            "Slippi-AI-versus-CPU9 integration gate failed: "
            f"{[name for name, passed in gate_checks.items() if not passed]}"
        )
    result = "complete" if exception is None and all(gate_checks.values()) else "failed"
    replay_outcome = cpu_replay_contract.get("outcome") if cpu_replay_contract is not None else None
    replay_winner_port = replay_outcome.get("winner_port") if isinstance(replay_outcome, dict) else None
    winner = "slippi-ai" if replay_winner_port == 1 else "cpu" if replay_winner_port == 2 else None
    trace_artifact = _file_identity(trace_path, project_root)
    trace_artifact["rows"] = processed_frames
    summary = {
        "schema_version": SCHEMA_VERSION,
        "classification": (
            f"frame-exact {release.display_name} {request.player_1_character} versus "
            f"Melee native CPU level 9 {request.player_2_character} mirror"
        ),
        "result": result,
        "error": None if exception is None else f"{type(exception).__name__}: {exception}",
        "gate": {"decision": "pass" if result == "complete" else "fail", "checks": gate_checks},
        "configuration": {
            "player_1": {
                "model": "slippi-ai",
                "release": release.key,
                "character": request.player_1_character,
                "port": SLIPPI_PORT,
            },
            "player_2": {
                "model": CPU_MODEL,
                "character": request.player_2_character,
                "port": CPU_PORT,
                "cpu_level": CPU_LEVEL,
                "gameplay_owner": "melee-native-cpu",
            },
            "stage": MATCH_STAGE,
            "seed": request.seed,
            "seed_scope": "Slippi-AI policy sampling",
            "require_natural_end": True,
            "maximum_game_frames": max_game_frames,
            "blocking_input": True,
            "console_delay_frames": SLIPPI_CONSOLE_DELAY_FRAMES,
            "checkpoint_policy_delay_frames": release.policy_delay_frames,
            "effective_policy_delay_frames": release.policy_delay_frames,
            "controller_replay_lag_frames": CONTROLLER_REPLAY_LAG_FRAMES,
            "inference_mode": EXACT_INFERENCE_MODE,
        },
        "contract": contract,
        "slippi_ai": {
            "metadata": session_metadata,
            "diagnostics": {
                "before_close": diagnostics_before_close,
                "after_close": diagnostics_after_close,
            },
        },
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
            "emulator_version": emulator_version,
            "tensorflow": tensorflow_setup,
        },
        "emulator_application": emulator_application,
        "game_image": _game_image_identity(iso_path),
        "execution": {
            "processed_policy_frames": processed_frames,
            "first_game_frame": first_game_frame,
            "last_game_frame": last_game_frame,
            "first_game_context": first_context,
            "frame_delta_counts": {
                str(delta): count for delta, count in sorted(frame_delta_counts.items())
            },
            "inference_counts": {"slippi-ai": len(inference_seconds), "cpu": 0},
            "dispatch_counts": {
                "slippi-ai": slippi_dispatch_count,
                "cpu-neutral-pipe": cpu_neutral_dispatch_count,
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
                "menu_flushes": {
                    f"p{port}": menu_transport_flushes[port] for port in (1, 2)
                },
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


def run_slippi_cpu_match(
    config_path: Path,
    iso_path: Path | None = None,
    request: SlippiCpuMatchRequest | None = None,
) -> dict[str, Any]:
    """Run one faithful official Slippi-AI release against native CPU level 9."""

    match_request = SlippiCpuMatchRequest() if request is None else request
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
    contract = _validate_config_contract(config, project_root, match_request)
    seed = _validate_evaluation_seed(match_request.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    from melee_policy.integration.slippi_compatibility import configure_cpu_tensorflow

    tensorflow_setup = configure_cpu_tensorflow(seed)
    image_path = _resolve_game_image_path(config, project_root, iso_path)
    reproducibility = _runtime_reproducibility_record(
        project_root,
        config_path.resolve(),
        "requirements-e010.lock",
        (
            "src/melee_policy/integration/slippi_ai_policy.py",
            "src/melee_policy/integration/slippi_cpu_match.py",
            "src/melee_policy/integration/slippi_match.py",
            "src/melee_policy/integration/frisson_cpu_match.py",
            "src/melee_policy/integration/frisson_match.py",
            "src/melee_policy/integration/game_bundle.py",
            "src/melee_policy/integration/match_runtime.py",
            "src/melee_policy/integration/replay_video.py",
            "patches/slippi-dolphin-two-pipe-frame-sync.patch",
            "scripts/play",
        ),
    )
    session = SlippiAIPolicySession(
        _slippi_policy_config(config, project_root, match_request)
    )
    summary = _run_cpu_console(
        config,
        project_root,
        image_path,
        match_request,
        contract,
        session,
        tensorflow_setup,
        reproducibility,
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


__all__ = ["SlippiCpuMatchRequest", "run_slippi_cpu_match"]
