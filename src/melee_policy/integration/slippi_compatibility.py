"""E010 compatibility canary for the released Slippi-AI medium-v2 policy.

The policy-facing path is deliberately narrow. Raw libmelee ``GameState``
objects are passed to :class:`SlippiAIPolicySession`, which retains the pinned
upstream parser, live observation filter, recurrent state, delay queue,
categorical sampler, and controller decoder. This module verifies identities,
runs one ordered replay trace, and records deterministic JSON artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import resource
import subprocess
import sys
import tempfile
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Self, cast

import numpy as np

from melee_policy.integration.slippi_ai_policy import (
    DIGITAL_BUTTON_ORDER,
    EFFECTIVE_POLICY_DELAY_FRAMES,
    MEDIUM_V2_CHECKPOINT_BYTES,
    MEDIUM_V2_CHECKPOINT_SHA256,
    MEDIUM_V2_PARAMETER_COUNT,
    MEDIUM_V2_VARIABLE_COUNT,
    POLICY_DELAY_FRAMES,
    SLIPPI_AI_REPOSITORY_URL,
    SLIPPI_AI_SOURCE_REVISION,
    CanonicalControllerCommand,
    SlippiAIPolicyConfig,
    SlippiAIPolicySession,
    verify_runtime_assets,
)

CHECKPOINT_MANIFEST_SCHEMA = "melee_policy.e010.slippi.checkpoint_manifest.v1"
COMPATIBILITY_SCHEMA = "melee_policy.e010.slippi.compatibility.v1"
SINGLE_RUN_SCHEMA = "melee_policy.e010.slippi.single_run.v1"
ARTIFACT_DIRECTORY = Path("artifacts/e010/slippi")
EXPECTED_RESET_FRAME = -123
AXIS_BUCKET_COUNT = 32
SHOULDER_BUCKET_COUNT = 10
IMPLEMENTATION_PATHS = (
    Path("src/melee_policy/integration/slippi_ai_policy.py"),
    Path("src/melee_policy/integration/slippi_compatibility.py"),
    Path("src/melee_policy/integration/slippi_match.py"),
)
E000_SAMPLE_MANIFEST_PATH = Path("artifacts/e000/sample_manifest.csv")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return cast(dict[str, Any], value)


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _local_path(project_root: Path, value: object) -> Path:
    candidate = Path(str(value)).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()


def _portable_project_paths(value: object, project_root: Path) -> object:
    """Replace checkout-local absolute paths with stable project-relative paths."""
    if isinstance(value, Mapping):
        return {str(key): _portable_project_paths(item, project_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_project_paths(item, project_root) for item in value]
    if isinstance(value, tuple):
        return [_portable_project_paths(item, project_root) for item in value]
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.is_absolute():
            with suppress(ValueError):
                return str(candidate.relative_to(project_root))
    return value


@dataclass(frozen=True, slots=True)
class E010CompatibilityConfig:
    """Validated projection of ``configs/e010_slippi.toml``."""

    config_path: Path
    project_root: Path
    raw: Mapping[str, Any]
    config_sha256: str
    experiment_id: str
    seed: int
    source_repository_url: str
    source_branch: str
    source_revision: str
    source_directory: Path
    checkpoint_name: str
    checkpoint_url: str
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_byte_length: int
    checkpoint_name_label: str
    sample_temperature: float
    policy_delay_frames: int
    console_delay_frames: int
    effective_output_delay_frames: int
    replay_path: Path
    replay_sha256: str
    controlled_port: int
    controlled_character: str
    opponent_port: int
    opponent_character: str
    frames: int

    @classmethod
    def load(cls, path: Path) -> Self:
        config_path = path.expanduser().resolve()
        raw_bytes = config_path.read_bytes()
        raw = tomllib.loads(raw_bytes.decode("utf-8"))
        project_root = config_path.parent.parent
        experiment = cast(dict[str, Any], raw["experiment"])
        source = cast(dict[str, Any], raw["source"])
        checkpoint = cast(dict[str, Any], raw["checkpoint"])
        runtime = cast(dict[str, Any], raw["native_runtime"])
        canary = cast(dict[str, Any], raw["offline_canary"])
        gate = cast(dict[str, Any], raw["gate"])

        result = cls(
            config_path=config_path,
            project_root=project_root,
            raw=raw,
            config_sha256=_sha256_bytes(raw_bytes),
            experiment_id=str(experiment["id"]),
            seed=int(experiment["seed"]),
            source_repository_url=str(source["repository_url"]),
            source_branch=str(source["branch"]),
            source_revision=str(source["revision"]),
            source_directory=_local_path(project_root, source["directory"]),
            checkpoint_name=str(checkpoint["name"]),
            checkpoint_url=str(checkpoint["url"]),
            checkpoint_path=_local_path(project_root, checkpoint["path"]),
            checkpoint_sha256=str(checkpoint["sha256"]),
            checkpoint_byte_length=int(checkpoint["byte_length"]),
            checkpoint_name_label=str(checkpoint["name_code_label"]),
            sample_temperature=float(checkpoint["sample_temperature"]),
            policy_delay_frames=int(checkpoint["policy_delay"]),
            console_delay_frames=int(runtime["console_delay"]),
            effective_output_delay_frames=int(runtime["effective_output_delay"]),
            replay_path=_local_path(project_root, canary["replay_path"]),
            replay_sha256=str(canary["replay_sha256"]),
            controlled_port=int(canary["controlled_port"]),
            controlled_character=str(canary["controlled_character"]),
            opponent_port=int(canary["opponent_port"]),
            opponent_character=str(canary["opponent_character"]),
            frames=int(canary["frames"]),
        )
        result.validate(experiment, runtime, gate)
        return result

    def validate(
        self,
        experiment: Mapping[str, Any],
        runtime: Mapping[str, Any],
        gate: Mapping[str, Any],
    ) -> None:
        exact_values = {
            "experiment.id": (self.experiment_id, "E010-SLIPPI"),
            "experiment.kind": (
                experiment.get("kind"),
                "public-checkpoint-compatibility-canary",
            ),
            "experiment.seed": (self.seed, 42),
            "experiment.device": (experiment.get("device"), "cpu"),
            "experiment.precision": (experiment.get("precision"), "float32"),
            "source.repository_url": (
                self.source_repository_url,
                SLIPPI_AI_REPOSITORY_URL,
            ),
            "source.branch": (self.source_branch, "main"),
            "source.revision": (self.source_revision, SLIPPI_AI_SOURCE_REVISION),
            "checkpoint.name": (self.checkpoint_name, "medium-v2"),
            "checkpoint.sha256": (
                self.checkpoint_sha256,
                MEDIUM_V2_CHECKPOINT_SHA256,
            ),
            "checkpoint.byte_length": (
                self.checkpoint_byte_length,
                MEDIUM_V2_CHECKPOINT_BYTES,
            ),
            "checkpoint.platform": (
                cast(Mapping[str, Any], self.raw["checkpoint"]).get("platform"),
                "tf",
            ),
            "checkpoint.name_code_label": (
                self.checkpoint_name_label,
                "Master Player",
            ),
            "checkpoint.sample_temperature": (self.sample_temperature, 1.0),
            "checkpoint.policy_delay": (
                self.policy_delay_frames,
                POLICY_DELAY_FRAMES,
            ),
            "native_runtime.console_delay": (self.console_delay_frames, 2),
            "native_runtime.effective_output_delay": (
                self.effective_output_delay_frames,
                EFFECTIVE_POLICY_DELAY_FRAMES,
            ),
            "native_runtime.compile": (runtime.get("compile"), True),
            "native_runtime.jit_compile": (runtime.get("jit_compile"), False),
            "native_runtime.async_inference": (runtime.get("async_inference"), True),
            "native_runtime.disable_gpus": (runtime.get("disable_gpus"), True),
            "native_runtime.reset_frame": (runtime.get("reset_frame"), -123),
            "native_runtime.live_observation_filter": (
                runtime.get("live_observation_filter"),
                "sequential",
            ),
        }
        mismatches = {
            name: {"observed": observed, "required": required}
            for name, (observed, required) in exact_values.items()
            if observed != required
        }
        if mismatches:
            raise ValueError(f"E010 frozen configuration mismatch: {mismatches}")
        if self.frames <= self.effective_output_delay_frames:
            raise ValueError("offline canary must include outputs after the native dummy prefix")
        if self.controlled_port == self.opponent_port:
            raise ValueError("controlled and opponent ports must differ")
        if len(self.replay_sha256) != 64:
            raise ValueError("offline replay SHA-256 must contain 64 hexadecimal characters")
        false_gates = sorted(name for name, value in gate.items() if value is not True)
        if false_gates:
            raise ValueError(f"all frozen E010 gate requirements must be true: {false_gates}")

    def policy_config(self) -> SlippiAIPolicyConfig:
        return SlippiAIPolicyConfig(
            source_directory=self.source_directory,
            checkpoint_path=self.checkpoint_path,
            port=self.controlled_port,
            opponent_port=self.opponent_port,
            name=self.checkpoint_name_label,
            sample_temperature=self.sample_temperature,
            policy_delay_frames=self.policy_delay_frames,
            console_delay_frames=self.console_delay_frames,
            async_inference=True,
            compile=True,
            tf_jit_compile=False,
            batch_steps=0,
            mirror=False,
        )


def _distribution_version(*names: str) -> str | None:
    for name in names:
        with suppress(importlib.metadata.PackageNotFoundError):
            return importlib.metadata.version(name)
    return None


def configure_cpu_tensorflow(seed: int) -> dict[str, Any]:
    """Disable accelerators and seed all RNGs before the policy is loaded."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    random.seed(seed)
    np.random.seed(seed)

    import tensorflow as tf  # type: ignore[import-untyped]

    tf.config.set_visible_devices([], "GPU")
    tf.random.set_seed(seed)
    physical_gpus = tf.config.list_physical_devices("GPU")
    logical_gpus = tf.config.list_logical_devices("GPU")
    if physical_gpus or logical_gpus:
        raise RuntimeError(
            f"CPU-only TensorFlow gate failed: physical_gpus={physical_gpus!r}, logical_gpus={logical_gpus!r}"
        )
    return {
        "tensorflow": str(tf.__version__),
        "seed": seed,
        "python_random_seeded": True,
        "numpy_random_seeded": True,
        "tensorflow_random_seeded": True,
        "physical_gpu_count": 0,
        "logical_gpu_count": 0,
    }


class ReplayConsole(Protocol):
    def connect(self) -> bool: ...

    def step(self) -> Any | None: ...

    def stop(self) -> None: ...


class PolicySession(Protocol):
    def start(self) -> None: ...

    def step(self, gamestate: Any) -> CanonicalControllerCommand: ...

    def metadata(self) -> dict[str, Any]: ...

    def diagnostics(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


ConsoleFactory = Callable[[Path], ReplayConsole]
SessionFactory = Callable[[SlippiAIPolicyConfig], PolicySession]
AssetVerifier = Callable[[SlippiAIPolicyConfig], dict[str, Any]]
TensorFlowSetup = Callable[[int], dict[str, Any]]
ManifestVerifier = Callable[[E010CompatibilityConfig], dict[str, Any]]


def _default_console_factory(replay_path: Path) -> ReplayConsole:
    import melee

    return cast(
        ReplayConsole,
        melee.Console(path=str(replay_path), is_dolphin=False, allow_old_version=True),
    )


def _default_session_factory(config: SlippiAIPolicyConfig) -> PolicySession:
    return SlippiAIPolicySession(config)


def _enum_fact(value: Any) -> dict[str, Any]:
    return {
        "name": str(getattr(value, "name", value)),
        "value": int(getattr(value, "value", value)),
    }


def _on_grid(value: float, denominator: int) -> bool:
    scaled = value * denominator
    return math.isfinite(value) and abs(scaled - round(scaled)) <= 1e-6


def command_vocabulary_checks(command: CanonicalControllerCommand) -> dict[str, bool]:
    """Check the exact released independent-axis and button vocabularies."""
    command.validate()
    values = (*command.main_stick, *command.c_stick, command.analog_l, command.analog_r)
    return {
        "finite": all(math.isfinite(value) for value in values),
        "main_stick_axis_vocabulary": all(_on_grid(value, AXIS_BUCKET_COUNT) for value in command.main_stick),
        "c_stick_axis_vocabulary": all(_on_grid(value, AXIS_BUCKET_COUNT) for value in command.c_stick),
        "analog_l_vocabulary": _on_grid(command.analog_l, SHOULDER_BUCKET_COUNT),
        "analog_r_adapter_value": command.analog_r == 0.0,
        "digital_button_vocabulary": set(command.buttons).issubset(DIGITAL_BUTTON_ORDER),
        "start_forbidden": "START" not in command.buttons,
    }


def _expected_dummy_command() -> dict[str, Any]:
    return CanonicalControllerCommand(
        main_stick=(0.0, 0.0),
        c_stick=(0.0, 0.0),
        analog_l=0.0,
        analog_r=0.0,
        buttons=(),
    ).as_dict()


def _verify_replay(config: E010CompatibilityConfig) -> dict[str, Any]:
    if not config.replay_path.is_file():
        raise FileNotFoundError(f"E000 canary replay is missing: {config.replay_path}")
    byte_length = config.replay_path.stat().st_size
    observed_sha256 = _sha256_file(config.replay_path)
    if observed_sha256 != config.replay_sha256:
        raise RuntimeError(
            f"E000 canary replay SHA-256 mismatch: {observed_sha256} != {config.replay_sha256}"
        )
    return {
        "path": str(config.replay_path),
        "sha256": observed_sha256,
        "byte_length": byte_length,
        "identity_basis": "raw .slp bytes",
    }


def _verify_e000_manifest(config: E010CompatibilityConfig) -> dict[str, Any]:
    manifest_identity = _project_file_record(config.project_root, E000_SAMPLE_MANIFEST_PATH)
    manifest_path = config.project_root / E000_SAMPLE_MANIFEST_PATH
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        matches = [row for row in csv.DictReader(stream) if row.get("replay_sha256") == config.replay_sha256]
    if len(matches) != 1:
        raise RuntimeError(f"expected one E000 manifest row for {config.replay_sha256}, found {len(matches)}")
    row = matches[0]
    checks = {
        "manifest_version": row.get("manifest_version") == "e000.manifest.v1",
        "raw_byte_length": int(row["raw_byte_length"]) == config.replay_path.stat().st_size,
        "libmelee_status": row.get("libmelee_status") == "ok",
        "peppi_status": row.get("peppi_status") == "ok",
        "supported": row.get("supported") == "True",
    }
    if not all(checks.values()):
        failures = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"E000 manifest support gate failed: {failures}")
    return {
        **manifest_identity,
        "checks": checks,
        "dataset_url": row["dataset_url"],
        "dataset_revision": row["dataset_revision"],
        "source_archive": row["source_archive"],
        "source_shard": row["source_shard"],
        "source_member": row["source_member"],
        "slippi_version": row["slippi_version"],
        "stage_id": int(row["stage_id"]),
        "libmelee_status": row["libmelee_status"],
        "peppi_status": row["peppi_status"],
        "supported": True,
    }


def _project_file_record(project_root: Path, relative_path: Path) -> dict[str, Any]:
    path = project_root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"required E010 file is missing: {path}")
    return {
        "path": relative_path.as_posix(),
        "sha256": _sha256_file(path),
        "byte_length": path.stat().st_size,
    }


def _implementation_file_records(config: E010CompatibilityConfig) -> list[dict[str, Any]]:
    return [_project_file_record(config.project_root, path) for path in IMPLEMENTATION_PATHS]


def execute_single_run(
    config: E010CompatibilityConfig,
    *,
    console_factory: ConsoleFactory = _default_console_factory,
    session_factory: SessionFactory = _default_session_factory,
    asset_verifier: AssetVerifier = verify_runtime_assets,
    tensorflow_setup: TensorFlowSetup = configure_cpu_tensorflow,
    manifest_verifier: ManifestVerifier = _verify_e000_manifest,
) -> dict[str, Any]:
    """Execute one policy trace in the current fresh process."""
    started_at = time.perf_counter()
    replay_identity = _verify_replay(config)
    e000_manifest = manifest_verifier(config)
    policy_config = config.policy_config()
    asset_identity = asset_verifier(policy_config)
    tensorflow = tensorflow_setup(config.seed)

    console = console_factory(config.replay_path)
    session = session_factory(policy_config)
    command_trace: list[dict[str, Any]] = []
    vocabulary_results: list[dict[str, bool]] = []
    first_state: Any | None = None
    metadata: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    try:
        if not console.connect():
            raise RuntimeError("libmelee could not connect to the E000 replay")
        session.start()
        for index in range(config.frames):
            gamestate = console.step()
            if gamestate is None:
                raise RuntimeError(f"replay ended before configured frame count at index {index}")
            frame = int(gamestate.frame)
            expected_frame = EXPECTED_RESET_FRAME + index
            if frame != expected_frame:
                raise RuntimeError(
                    f"replay frames are not consecutive: expected {expected_frame}, got {frame}"
                )
            if first_state is None:
                first_state = gamestate

            ports = sorted(int(port) for port in gamestate.players)
            if ports != sorted((config.controlled_port, config.opponent_port)):
                raise RuntimeError(f"replay port mismatch at frame {frame}: {ports}")
            controlled_name = str(gamestate.players[config.controlled_port].character.name)
            opponent_name = str(gamestate.players[config.opponent_port].character.name)
            if controlled_name != config.controlled_character:
                raise RuntimeError(
                    f"controlled character mismatch at frame {frame}: "
                    f"{controlled_name} != {config.controlled_character}"
                )
            if opponent_name != config.opponent_character:
                raise RuntimeError(
                    f"opponent character mismatch at frame {frame}: "
                    f"{opponent_name} != {config.opponent_character}"
                )

            command = session.step(gamestate)
            checks = command_vocabulary_checks(command)
            if not all(checks.values()):
                raise RuntimeError(f"controller vocabulary failure at frame {frame}: {checks}")
            command_trace.append({"frame": frame, "command": command.as_dict()})
            vocabulary_results.append(checks)

        metadata = session.metadata()
        diagnostics = session.diagnostics()
    finally:
        session.close()
        with suppress(AssertionError):
            console.stop()

    assert first_state is not None
    dummy_prefix = command_trace[: config.effective_output_delay_frames]
    expected_dummy = _expected_dummy_command()
    dummy_prefix_exact = all(item["command"] == expected_dummy for item in dummy_prefix)
    decoder_parity = (
        diagnostics.get("capture_decoder_assertions") == config.frames
        and diagnostics.get("capture_decoder_mismatches") == 0
    )
    all_vocabulary_checks = all(all(frame_checks.values()) for frame_checks in vocabulary_results)
    ordered_frames = [item["frame"] for item in command_trace] == list(
        range(EXPECTED_RESET_FRAME, EXPECTED_RESET_FRAME + config.frames)
    )
    upstream_runtime = cast(Mapping[str, Any], metadata.get("upstream_runtime", {}))
    state_assignment = cast(Mapping[str, Any], upstream_runtime.get("state_assignment", {}))
    checkpoint_config = cast(Mapping[str, Any], upstream_runtime.get("checkpoint_config", {}))
    network_config = cast(Mapping[str, Any], checkpoint_config.get("network", {}))
    tx_like_config = cast(Mapping[str, Any], network_config.get("tx_like", {}))
    post_delay_non_dummy = any(
        item["command"] != expected_dummy for item in command_trace[config.effective_output_delay_frames :]
    )
    checks = {
        "replay_raw_sha256": replay_identity["sha256"] == config.replay_sha256,
        "source_revision": (asset_identity.get("source", {}).get("revision") == config.source_revision),
        "checkpoint_sha256": (asset_identity.get("checkpoint", {}).get("sha256") == config.checkpoint_sha256),
        "checkpoint_byte_length": (
            asset_identity.get("checkpoint", {}).get("byte_length") == config.checkpoint_byte_length
        ),
        "e000_manifest_membership": all(e000_manifest["checks"].values()),
        "cpu_only_tensorflow": (
            tensorflow.get("physical_gpu_count") == 0 and tensorflow.get("logical_gpu_count") == 0
        ),
        "first_frame_reset": command_trace[0]["frame"] == EXPECTED_RESET_FRAME,
        "strict_frame_order": ordered_frames,
        "frame_count": len(command_trace) == config.frames,
        "native_dummy_prefix_length": len(dummy_prefix) == config.effective_output_delay_frames,
        "native_dummy_prefix_exact": dummy_prefix_exact,
        "post_delay_output_non_dummy": post_delay_non_dummy,
        "finite_and_in_vocabulary": all_vocabulary_checks,
        "native_controller_decoder_parity": decoder_parity,
        "native_parser": upstream_runtime.get("parser_class") == "slippi_db.parse_libmelee.Parser",
        "sequential_observation_filter": upstream_runtime.get("observation_filter_class")
        == "slippi_ai.observations.ChainObservationFilter",
        "tensorflow_policy_path": (
            upstream_runtime.get("platform") == "tf"
            and ".tf." in str(upstream_runtime.get("controller_head_class", ""))
        ),
        "recurrent_lstm_path": (
            bool(upstream_runtime.get("recurrent_state_class"))
            and network_config.get("name") == "tx_like"
            and tx_like_config.get("recurrent_layer") == "lstm"
        ),
        "policy_delay": upstream_runtime.get("policy_delay_frames") == POLICY_DELAY_FRAMES,
        "effective_policy_delay": upstream_runtime.get("effective_policy_delay_frames")
        == EFFECTIVE_POLICY_DELAY_FRAMES,
        "variable_count": upstream_runtime.get("variable_count") == MEDIUM_V2_VARIABLE_COUNT,
        "parameter_count": upstream_runtime.get("parameter_count") == MEDIUM_V2_PARAMETER_COUNT,
        "state_assignment_shapes": state_assignment.get("shape_mismatches") == [],
        "state_assignment_dtypes": state_assignment.get("dtype_mismatches") == [],
        "state_assignment_values": state_assignment.get("value_mismatches") == [],
        "state_assignment_finite": state_assignment.get("nonfinite_variables") == [],
    }
    if not all(checks.values()):
        failures = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"E010 single-process compatibility gate failed: {failures}")

    trace_sha256 = _sha256_bytes(_canonical_json_bytes(command_trace))
    maximum_resident_set = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "schema_version": SINGLE_RUN_SCHEMA,
        "experiment_id": config.experiment_id,
        "seed": config.seed,
        "process_id": os.getpid(),
        "replay": replay_identity,
        "e000_manifest": e000_manifest,
        "perspective": {
            "controlled_port": config.controlled_port,
            "controlled_character": _enum_fact(first_state.players[config.controlled_port].character),
            "opponent_port": config.opponent_port,
            "opponent_character": _enum_fact(first_state.players[config.opponent_port].character),
            "stage": _enum_fact(first_state.stage),
            "first_frame": int(first_state.frame),
            "last_frame": int(command_trace[-1]["frame"]),
        },
        "tensorflow": tensorflow,
        "asset_identity": asset_identity,
        "policy_metadata": metadata,
        "policy_diagnostics": diagnostics,
        "commands": command_trace,
        "command_trace_sha256": trace_sha256,
        "native_dummy_prefix": {
            "frames": config.effective_output_delay_frames,
            "expected_command": expected_dummy,
            "outputs": dummy_prefix,
            "exact": dummy_prefix_exact,
        },
        "checks": checks,
        "performance": {
            "wall_seconds": time.perf_counter() - started_at,
            "maximum_resident_set_raw": maximum_resident_set,
            "maximum_resident_set_unit": ("bytes" if platform.system() == "Darwin" else "kilobytes"),
            "estimated_flops": None,
            "estimated_flops_reason": "not exposed by the released inference runtime",
        },
        "gate": "pass",
    }


def _source_manifest(config: E010CompatibilityConfig) -> dict[str, Any]:
    source = config.source_directory
    revision = _git(source, "rev-parse", "HEAD")
    if revision != config.source_revision:
        raise RuntimeError(f"source revision changed before artifact writing: {revision}")
    commit_lines = _git(
        source,
        "show",
        "-s",
        "--format=%H%n%T%n%aI%n%s",
        revision,
    ).splitlines()
    if len(commit_lines) != 4:
        raise RuntimeError("could not read the complete pinned source commit record")
    license_path = source / "LICENSE"
    return {
        "repository_url": config.source_repository_url,
        "branch": config.source_branch,
        "revision": revision,
        "git_tree_oid": commit_lines[1],
        "authored_at": commit_lines[2],
        "subject": commit_lines[3],
        "directory": str(source),
        "tracked_tree_clean": _git(source, "status", "--short", "--untracked-files=all") == "",
        "license": {
            "identifier": "MIT",
            "path": str(license_path),
            "sha256": _sha256_file(license_path),
        },
        "native_paths": [
            "slippi_ai/eval_lib.py",
            "slippi_db/parse_libmelee.py",
            "slippi_ai/observations.py",
            "slippi_ai/controller_lib.py",
            "slippi_ai/controller_heads.py",
            "slippi_ai/tf/policies.py",
            "slippi_ai/tf/agents.py",
        ],
    }


def _project_record(config: E010CompatibilityConfig) -> dict[str, Any]:
    tracked_diff = subprocess.run(
        [
            "git",
            "-C",
            str(config.project_root),
            "diff",
            "--quiet",
            "--",
            ".",
            ":(exclude)artifacts/e010/slippi/**",
        ],
        check=False,
        capture_output=True,
    )
    if tracked_diff.returncode not in (0, 1):
        raise RuntimeError("could not inspect the project tracked-tree status")
    implementation_files = _implementation_file_records(config)
    runner = next(
        record
        for record in implementation_files
        if record["path"] == "src/melee_policy/integration/slippi_compatibility.py"
    )
    return {
        "commit": _git(config.project_root, "rev-parse", "HEAD"),
        "tracked_tree_dirty": tracked_diff.returncode == 1,
        "dirty_scope": "tracked files excluding artifacts/e010/slippi",
        "runner_path": runner["path"],
        "runner_sha256": runner["sha256"],
        "implementation_files": implementation_files,
    }


def _environment_record(first: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "precision": "float32",
        "device": "cpu",
        "tensorflow": first["tensorflow"]["tensorflow"],
        "numpy": np.__version__,
        "melee": _distribution_version("melee"),
        "slippi_ai": _distribution_version("slippi-ai", "slippi_ai"),
        "dm_sonnet": _distribution_version("dm-sonnet"),
        "tensorflow_probability": _distribution_version("tfp-nightly", "tensorflow-probability"),
    }


def _requirements_lock_record(config: E010CompatibilityConfig) -> dict[str, Any]:
    path = config.project_root / "requirements-e010.lock"
    if not path.is_file():
        raise FileNotFoundError(f"E010 dependency lock is missing: {path}")
    return {
        "path": str(path.relative_to(config.project_root)),
        "sha256": _sha256_file(path),
        "byte_length": path.stat().st_size,
    }


def _stable_diagnostics(value: object) -> object:
    """Remove only measured process timing from otherwise exact diagnostics."""
    if not isinstance(value, Mapping):
        return value
    result = {str(key): item for key, item in value.items() if key != "step_timing"}
    upstream = result.get("upstream")
    if isinstance(upstream, Mapping):
        result["upstream"] = {
            str(key): item
            for key, item in upstream.items()
            if key not in ("state_queue_profiler", "step_profiler")
        }
    return result


def _stable_single_run(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project a fresh-process result into reproducible tracked evidence."""
    return {
        str(key): (_stable_diagnostics(item) if key == "policy_diagnostics" else item)
        for key, item in value.items()
        if key not in ("process_id", "performance")
    }


def build_artifacts(
    config: E010CompatibilityConfig,
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the two tracked JSON values with stable key and list ordering."""
    first_commands = first.get("commands")
    second_commands = second.get("commands")
    trace_equal = first.get("command_trace_sha256") == second.get("command_trace_sha256")
    commands_equal = first_commands == second_commands
    identity_equal = first.get("asset_identity") == second.get("asset_identity")
    replay_equal = first.get("replay") == second.get("replay")
    both_single_runs_pass = first.get("gate") == second.get("gate") == "pass"
    reload_checks = {
        "trace_sha256_equal": trace_equal,
        "complete_command_trace_equal": commands_equal,
        "asset_identity_equal": identity_equal,
        "replay_identity_equal": replay_equal,
        "both_single_process_gates_pass": both_single_runs_pass,
        "distinct_processes": first.get("process_id") != second.get("process_id"),
    }

    source = _source_manifest(config)
    replay = _verify_replay(config)
    e000_manifest = _verify_e000_manifest(config)
    requirements_lock = _requirements_lock_record(config)
    project = _project_record(config)
    checkpoint = {
        "name": config.checkpoint_name,
        "url": config.checkpoint_url,
        "path": str(config.checkpoint_path),
        "sha256": _sha256_file(config.checkpoint_path),
        "byte_length": config.checkpoint_path.stat().st_size,
        "name_code_label": config.checkpoint_name_label,
    }
    manifest_checks = {
        "source_revision": source["revision"] == config.source_revision,
        "source_tree_clean": source["tracked_tree_clean"] is True,
        "checkpoint_sha256": checkpoint["sha256"] == config.checkpoint_sha256,
        "checkpoint_byte_length": checkpoint["byte_length"] == config.checkpoint_byte_length,
        "replay_sha256": replay["sha256"] == config.replay_sha256,
        "e000_manifest_support": all(e000_manifest["checks"].values()),
        "e000_manifest_file_identity": (
            len(str(e000_manifest["sha256"])) == 64 and int(e000_manifest["byte_length"]) > 0
        ),
        "implementation_files_present": (
            len(project["implementation_files"]) == len(IMPLEMENTATION_PATHS)
            and all(
                int(record["byte_length"]) > 0
                for record in cast(list[dict[str, Any]], project["implementation_files"])
            )
        ),
        "requirements_lock_present": requirements_lock["byte_length"] > 0,
    }
    if not all(manifest_checks.values()):
        failures = sorted(name for name, passed in manifest_checks.items() if not passed)
        raise RuntimeError(f"E010 manifest identity gate failed: {failures}")

    checkpoint_manifest = {
        "schema_version": CHECKPOINT_MANIFEST_SCHEMA,
        "experiment": {
            "id": config.experiment_id,
            "kind": "public-checkpoint-compatibility-canary",
            "config_path": str(config.config_path.relative_to(config.project_root)),
            "config_sha256": config.config_sha256,
            "seed_policy": {
                "seed": config.seed,
                "scope": "Python, NumPy, and TensorFlow in each fresh process",
                "upstream_default_seeded": False,
            },
        },
        "source": source,
        "checkpoint": checkpoint,
        "replay": replay,
        "e000_manifest": e000_manifest,
        "environment": _environment_record(first),
        "requirements_lock": requirements_lock,
        "project": project,
        "manifest_checks": manifest_checks,
    }
    gate_checks = {
        **manifest_checks,
        **reload_checks,
        "first_run_all_checks": all(cast(Mapping[str, bool], first["checks"]).values()),
        "second_run_all_checks": all(cast(Mapping[str, bool], second["checks"]).values()),
    }
    compatibility = {
        "schema_version": COMPATIBILITY_SCHEMA,
        "experiment_id": config.experiment_id,
        "hypothesis": (
            "medium-v2 retains its pinned upstream parser, sequential live observation "
            "filter, recurrent state, delay, categorical sampler, and decoder while only "
            "the decoded controller command crosses the project boundary"
        ),
        "control": "the same pinned upstream policy path on identical ordered E000 frames",
        "changed_variable": "project controller-boundary adaptation only",
        "verified_trace": _stable_single_run(first),
        "fresh_process_reload": {
            "command_trace_sha256": second.get("command_trace_sha256"),
            "checks": second.get("checks"),
            "policy_diagnostics": _stable_diagnostics(second.get("policy_diagnostics")),
        },
        "reload_checks": reload_checks,
        "gate_checks": gate_checks,
        "gate": "pass" if all(gate_checks.values()) else "fail",
    }
    return (
        cast(dict[str, Any], _portable_project_paths(checkpoint_manifest, config.project_root)),
        cast(dict[str, Any], _portable_project_paths(compatibility, config.project_root)),
    )


def write_artifacts(
    config: E010CompatibilityConfig,
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    output_directory: Path | None = None,
) -> dict[str, Any]:
    output = (
        output_directory.resolve()
        if output_directory is not None
        else (config.project_root / ARTIFACT_DIRECTORY).resolve()
    )
    checkpoint_manifest, compatibility = build_artifacts(config, first, second)
    _write_json(output / "checkpoint_manifest.json", checkpoint_manifest)
    _write_json(output / "compatibility.json", compatibility)
    readme = "\n".join(
        [
            "# E010 Slippi-AI compatibility canary",
            "",
            "The released medium-v2 policy runs on one 64-frame E000 replay trace through",
            "the pinned upstream parser, sequential live observation filter, recurrent state,",
            "19-frame effective delay, temperature-1 categorical sampler, and decoder.",
            "Only the decoded complete controller command crosses the project boundary.",
            "",
            "Successful commands:",
            "",
            "```sh",
            "PYTHONPATH=src .e010-env/bin/python -m "
            "melee_policy.integration.slippi_compatibility run "
            "--config configs/e010_slippi.toml",
            "PYTHONPATH=src .e010-env/bin/python -m "
            "melee_policy.integration.slippi_compatibility validate "
            "--config configs/e010_slippi.toml",
            "```",
            "",
            "Rendered smoke command:",
            "",
            "```sh",
            "./scripts/play --p1 slippi-ai --p2 mimic --p1-character FOX "
            "--p2-character FOX --stage BATTLEFIELD --max-game-frames 600 "
            "--artifact-label e010_slippi_vs_mimic_smoke",
            "```",
            "",
            f"Offline compatibility gate: {compatibility['gate']}",
            "Rendered smoke evidence: artifacts/integration/slippi_ai/"
            "e010_slippi_vs_mimic_smoke/summary.json",
            "",
        ]
    )
    (output / "README.md").write_text(readme, encoding="utf-8")
    return {
        "output_directory": str(output),
        "checkpoint_manifest": str(output / "checkpoint_manifest.json"),
        "compatibility": str(output / "compatibility.json"),
        "readme": str(output / "README.md"),
        "gate": compatibility["gate"],
    }


def _run_fresh_process(config: E010CompatibilityConfig, output: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    source_path = str(config.project_root / "src")
    current_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path if not current_pythonpath else source_path + os.pathsep + current_pythonpath
    )
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    environment["PYTHONHASHSEED"] = str(config.seed)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "melee_policy.integration.slippi_compatibility",
            "_single",
            "--config",
            str(config.config_path),
            "--output",
            str(output),
        ],
        cwd=config.project_root,
        env=environment,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"fresh-process E010 canary failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return _read_json(output)


def run_with_fresh_process_reload(
    config: E010CompatibilityConfig,
    *,
    output_directory: Path | None = None,
) -> dict[str, Any]:
    """Run the trace in two new processes and persist the checked artifacts."""
    with tempfile.TemporaryDirectory(prefix="e010-slippi-") as temporary:
        temporary_path = Path(temporary)
        first = _run_fresh_process(config, temporary_path / "first.json")
        second = _run_fresh_process(config, temporary_path / "second.json")
    result = write_artifacts(
        config,
        first,
        second,
        output_directory=output_directory,
    )
    if result["gate"] != "pass":
        raise RuntimeError("E010 fresh-process compatibility gate failed")
    return result


def validate_artifacts(
    config: E010CompatibilityConfig,
    *,
    output_directory: Path | None = None,
) -> dict[str, Any]:
    output = (
        output_directory.resolve()
        if output_directory is not None
        else (config.project_root / ARTIFACT_DIRECTORY).resolve()
    )
    manifest = _read_json(output / "checkpoint_manifest.json")
    compatibility = _read_json(output / "compatibility.json")
    commands = compatibility["verified_trace"]["commands"]
    trace_sha256 = _sha256_bytes(_canonical_json_bytes(commands))
    expected_implementation = _implementation_file_records(config)
    observed_implementation = manifest.get("project", {}).get("implementation_files", [])
    observed_by_path = {
        record.get("path"): record for record in observed_implementation if isinstance(record, Mapping)
    }
    implementation_checks = {
        f"implementation_{Path(record['path']).stem}_identity": (
            observed_by_path.get(record["path"]) == record
        )
        for record in expected_implementation
    }
    expected_e000_manifest = _project_file_record(config.project_root, E000_SAMPLE_MANIFEST_PATH)
    recorded_e000_manifest = manifest.get("e000_manifest", {})
    checks = {
        "manifest_schema": manifest.get("schema_version") == CHECKPOINT_MANIFEST_SCHEMA,
        "compatibility_schema": compatibility.get("schema_version") == COMPATIBILITY_SCHEMA,
        "config_sha256": manifest.get("experiment", {}).get("config_sha256") == config.config_sha256,
        "source_revision": manifest.get("source", {}).get("revision") == config.source_revision,
        "checkpoint_sha256": manifest.get("checkpoint", {}).get("sha256") == config.checkpoint_sha256,
        "replay_sha256": manifest.get("replay", {}).get("sha256") == config.replay_sha256,
        "e000_sample_manifest_identity": all(
            recorded_e000_manifest.get(key) == value for key, value in expected_e000_manifest.items()
        ),
        "requirements_lock_sha256": manifest.get("requirements_lock", {}).get("sha256")
        == _requirements_lock_record(config)["sha256"],
        **implementation_checks,
        "frame_count": len(commands) == config.frames,
        "trace_sha256": compatibility["verified_trace"].get("command_trace_sha256") == trace_sha256,
        "dummy_prefix_count": len(compatibility["verified_trace"]["native_dummy_prefix"]["outputs"])
        == config.effective_output_delay_frames,
        "gate": compatibility.get("gate") == "pass",
    }
    if not all(checks.values()):
        failures = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"E010 artifact validation failed: {failures}")
    return {"schema_version": "melee_policy.e010.slippi.validation.v1", "checks": checks}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "validate"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", type=Path, default=Path("configs/e010_slippi.toml"))
        child.add_argument("--output-directory", type=Path)
    single = subparsers.add_parser("_single", help=argparse.SUPPRESS)
    single.add_argument("--config", type=Path, required=True)
    single.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = E010CompatibilityConfig.load(arguments.config)
    if arguments.command == "_single":
        _write_json(arguments.output, execute_single_run(config))
        return 0
    if arguments.command == "run":
        result = run_with_fresh_process_reload(
            config,
            output_directory=arguments.output_directory,
        )
    else:
        result = validate_artifacts(
            config,
            output_directory=arguments.output_directory,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
