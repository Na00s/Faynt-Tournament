"""Run a fail-closed, mirrored round-robin tournament through ``scripts/play``.

The one-game launcher remains the source of truth for every policy runtime.  This
module only constructs matched port blocks, starts one fresh child process per
game, audits the resulting child summary, and aggregates accepted natural ends.
"""

from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from itertools import combinations
from pathlib import Path
from typing import Any, TextIO, cast
from melee_policy.integration.runtime_equivalence import (
    EXPECTED_DIGESTS as RUNTIME_EQUIVALENCE_EXPECTED_DIGESTS,
)
from melee_policy.integration.runtime_equivalence import (
    FRAME_COUNT as RUNTIME_EQUIVALENCE_FRAME_COUNT,
)
from melee_policy.integration.runtime_equivalence import (
    REPLAY_SHA256 as RUNTIME_EQUIVALENCE_REPLAY_SHA256,
)
from melee_policy.integration.runtime_identity import (
    runtime_environment_identity_sha256,
)

SCHEMA_VERSION = "integration.tournament.v7"
DEFAULT_ENTRANTS = ("mimic", "slippi-ai")
ENTRANT_ORDER = {entrant: index for index, entrant in enumerate(DEFAULT_ENTRANTS)}
DISPLAY_IDENTITIES = {"mimic": "MIMIC", "slippi-ai": "vladfi1/slippi-ai medium-v2"}
LEGAL_STAGES = (
    "BATTLEFIELD",
    "DREAMLAND",
    "FINAL_DESTINATION",
    "FOUNTAIN_OF_DREAMS",
    "POKEMON_STADIUM",
    "YOSHIS_STORY",
)
DEFAULT_MAX_GAME_FRAMES = 36000
DEFAULT_CHILD_WALL_TIMEOUT_SECONDS = 10800.0
CHILD_INTERRUPT_GRACE_SECONDS = 60.0
CHILD_TERMINATE_GRACE_SECONDS = 15.0
E011_MINIMUM_BOOTS = 96
FINAL_SCREENING_MINIMUM_MIRRORED_CONFIGURATIONS = 64
WILSON_Z_95 = 1.959963984540054
POLICY_SEED_DERIVATION_DOMAIN = "melee-policy.tournament.policy-sampling.v1"
IMPLEMENTATION_PATHS = (
    "src/melee_policy/integration/tournament.py",
    "src/melee_policy/integration/replay_result.py",
    "src/melee_policy/e000/enums.py",
    "src/melee_policy/integration/play.py",
    "src/melee_policy/integration/match_runtime.py",
    "src/melee_policy/integration/runtime_identity.py",
    "src/melee_policy/integration/runtime_equivalence.py",
    "src/melee_policy/integration/slippi_match.py",
    "src/melee_policy/integration/slippi_ai_policy.py",
    "src/melee_policy/integration/slippi_compatibility.py",
    "src/melee_policy/integration/state_identity.py",
    "scripts/tournament",
    "scripts/play",
    "requirements-e001.lock",
    "requirements-e010.lock",
)
CHILD_RUNTIME_PATHS = {
    "mimic": (
        "src/melee_policy/integration/match_runtime.py",
        "src/melee_policy/integration/play.py",
        "src/melee_policy/integration/runtime_identity.py",
        "src/melee_policy/integration/state_identity.py",
        "scripts/play",
        "requirements-e001.lock",
    ),
    "slippi-ai-mixed": (
        "src/melee_policy/integration/match_runtime.py",
        "src/melee_policy/integration/slippi_ai_policy.py",
        "src/melee_policy/integration/slippi_compatibility.py",
        "src/melee_policy/integration/slippi_match.py",
        "src/melee_policy/integration/play.py",
        "src/melee_policy/integration/runtime_identity.py",
        "src/melee_policy/integration/state_identity.py",
        "scripts/play",
        "requirements-e010.lock",
    ),
}
CHILD_RUNTIME_INTERPRETERS = {
    "mimic": ".e001-env/bin/python",
    "slippi-ai-mixed": ".e010-env/bin/python",
}
JsonObject = dict[str, Any]
ProcessRunner = Callable[[list[str], Path], subprocess.CompletedProcess[str]]
ReplayAuditor = Callable[[Path, str, Mapping[int, str], Mapping[int, int]], JsonObject]
ControllerBoundaryAuditor = Callable[[Path, Path, Path], JsonObject]


def _default_controller_boundary_auditor(
    trace_path: Path, replay_path: Path, project_root: Path
) -> JsonObject:
    """Recompute sent-command evidence from the trace and physical replay."""
    from melee_policy.integration.slippi_match import _audit_controller_boundary

    return _audit_controller_boundary(trace_path, replay_path, project_root)


def _host_identity() -> JsonObject:
    """Return the report-global host tuple shared by both runtime families."""
    return {
        "host_platform": platform.platform(),
        "host_machine": platform.machine(),
        "host_os_build": platform.version(),
    }


def _runtime_equivalence_record(project_root: Path) -> JsonObject:
    """Run the frozen policy-input canary in both libmelee runtime families."""
    config_path = project_root / "configs" / "integration.toml"
    runtime_specs = {
        "mimic": (".e001-env/bin/python", "0.45.1"),
        "slippi-ai-mixed": (".e010-env/bin/python", "0.47.3"),
    }
    results: JsonObject = {}
    for family, (interpreter_path, melee_version) in runtime_specs.items():
        interpreter = project_root / interpreter_path
        if not interpreter.is_file():
            raise FileNotFoundError(
                f"runtime-equivalence interpreter is missing: {interpreter}"
            )
        environment = os.environ.copy()
        source_path = str(project_root / "src")
        environment["PYTHONPATH"] = (
            source_path + os.pathsep + environment.get("PYTHONPATH", "")
        )
        completed = subprocess.run(
            [
                str(interpreter),
                "-m",
                "melee_policy.integration.runtime_equivalence",
                "--config",
                str(config_path),
                "--expected-melee-version",
                melee_version,
            ],
            cwd=project_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=600.0,
        )
        if completed.returncode != 0:
            raise TournamentRunError(
                f"runtime-equivalence canary failed for {family} with status {completed.returncode}: {completed.stderr[-2000:]}"
            )
        prefix = "RUNTIME_EQUIVALENCE_JSON="
        payloads = [
            line.removeprefix(prefix)
            for line in completed.stdout.splitlines()
            if line.startswith(prefix)
        ]
        if len(payloads) != 1:
            raise TournamentRunError(
                f"runtime-equivalence canary produced {len(payloads)} result records for {family}"
            )
        value = json.loads(payloads[0])
        if not isinstance(value, dict):
            raise TournamentRunError(
                f"runtime-equivalence result is not an object for {family}"
            )
        results[family] = value
    mimic_runtime = cast(Mapping[str, Any], results["mimic"])
    slippi_mixed = cast(Mapping[str, Any], results["slippi-ai-mixed"])
    checks = {
        "both_runtime_canaries_pass": all(
            (
                cast(Mapping[str, Any], value).get("decision") == "pass"
                for value in results.values()
            )
        ),
        "expected_interpreters": mimic_runtime.get("python_executable")
        == ".e001-env/bin/python"
        and slippi_mixed.get("python_executable") == ".e010-env/bin/python",
        "declared_libmelee_versions": mimic_runtime.get("melee_version") == "0.45.1"
        and slippi_mixed.get("melee_version") == "0.47.3",
        "same_pinned_replay": cast(
            Mapping[str, Any], mimic_runtime.get("replay", {})
        ).get("sha256")
        == cast(Mapping[str, Any], slippi_mixed.get("replay", {})).get("sha256")
        == RUNTIME_EQUIVALENCE_REPLAY_SHA256,
        "same_complete_frame_count": mimic_runtime.get("processed_frames")
        == slippi_mixed.get("processed_frames")
        == RUNTIME_EQUIVALENCE_FRAME_COUNT,
        "policy_visible_digests_equal_and_frozen": mimic_runtime.get("digests")
        == slippi_mixed.get("digests")
        == RUNTIME_EQUIVALENCE_EXPECTED_DIGESTS,
        "source_revisions_equal": mimic_runtime.get("source_revisions")
        == slippi_mixed.get("source_revisions"),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": "integration.tournament.runtime_equivalence.v1",
        "classification": "cross-libmelee policy-visible parser and MIMIC input equivalence; model inference and controller transport remain native to each runtime family",
        "decision": "pass" if not failures else "fail",
        "checks": checks,
        "failures": failures,
        "runtimes": results,
    }


class TournamentRunError(RuntimeError):
    """The current attempt failed closed after its report was persisted."""


class ChildProcessTimeout(TimeoutError):
    """A child game exceeded its predeclared wall-clock allowance."""

    def __init__(self, timeout_seconds: float, stdout: str, stderr: str) -> None:
        super().__init__(f"child game exceeded {timeout_seconds:g} wall-clock seconds")
        self.timeout_seconds = timeout_seconds
        self.stdout = stdout
        self.stderr = stderr


@dataclass(frozen=True, slots=True)
class TournamentRequest:
    """Frozen tournament configuration.

    ``games_per_block`` is the total number of games for one pairing, seed, and
    stage.  It must be even.  Half use each port assignment, and each opposite
    assignment pair is retained as one matched port block.
    """

    entrants: tuple[str, ...] = DEFAULT_ENTRANTS
    seeds: tuple[int, ...] = (42,)
    stages: tuple[str, ...] = ("BATTLEFIELD",)
    games_per_block: int = 2
    order_seed: int = 0
    character: str = "FOX"
    max_game_frames: int = DEFAULT_MAX_GAME_FRAMES
    child_wall_timeout_seconds: float = DEFAULT_CHILD_WALL_TIMEOUT_SECONDS
    config_path: Path = Path("configs/integration.toml")
    iso_path: Path | None = None
    report_path: Path = Path("artifacts/integration/tournament/report.json")

    def validate(self) -> None:
        entrants = _canonical_entrants(self.entrants)
        if len(entrants) < 2:
            raise ValueError("a tournament requires at least two distinct entrants")
        if not self.seeds:
            raise ValueError("at least one policy seed is required")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("policy seeds must be unique")
        if any((seed < 0 or seed > 4294967295 for seed in self.seeds)):
            raise ValueError("policy seeds must be in the inclusive uint32 range")
        if not self.stages:
            raise ValueError("at least one stage is required")
        if len(set(self.stages)) != len(self.stages):
            raise ValueError("stages must be unique")
        unsupported_stages = sorted(set(self.stages) - set(LEGAL_STAGES))
        if unsupported_stages:
            raise ValueError(f"unsupported tournament stages: {unsupported_stages}")
        if self.games_per_block < 2 or self.games_per_block % 2 != 0:
            raise ValueError("games_per_block must be an even integer of at least two")
        if self.order_seed < 0:
            raise ValueError("order_seed must be non-negative")
        if not self.character or self.character != self.character.upper():
            raise ValueError("character must be an uppercase libmelee enum name")
        if "mimic" in entrants and self.character != "FOX":
            raise ValueError(
                "the fixed tournament MIMIC entrant is the registered Fox checkpoint and asset bundle; non-Fox tournaments require an explicit alternate MIMIC asset contract"
            )
        if self.max_game_frames < 1:
            raise ValueError("max_game_frames must be positive")
        if (
            isinstance(self.child_wall_timeout_seconds, bool)
            or not isinstance(self.child_wall_timeout_seconds, (int, float))
            or (not math.isfinite(self.child_wall_timeout_seconds))
            or (self.child_wall_timeout_seconds < CHILD_INTERRUPT_GRACE_SECONDS)
        ):
            raise ValueError(
                f"child_wall_timeout_seconds must be finite and at least {CHILD_INTERRUPT_GRACE_SECONDS:g}"
            )


def _canonical_model(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    if normalized == "slippi":
        normalized = "slippi-ai"
    if normalized not in DISPLAY_IDENTITIES:
        raise ValueError(f"unsupported tournament entrant: {value!r}")
    return normalized


def _canonical_entrants(values: Sequence[str]) -> tuple[str, ...]:
    entrants = tuple((_canonical_model(value) for value in values))
    if len(set(entrants)) != len(entrants):
        raise ValueError("tournament entrants must be distinct")
    return tuple(sorted(entrants, key=ENTRANT_ORDER.__getitem__))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(project_root.resolve()))
    except ValueError:
        return str(resolved)


def _resolve_from_project(path: Path, project_root: Path) -> Path:
    return (
        path.expanduser().resolve()
        if path.is_absolute()
        else (project_root / path).resolve()
    )


def _resolved_game_image_path(
    request: TournamentRequest, project_root: Path
) -> tuple[Path, str]:
    if request.iso_path is not None:
        return (_resolve_from_project(request.iso_path, project_root), "command-line")
    config_path = _resolve_from_project(request.config_path, project_root)
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    game_image = config.get("game_image")
    if not isinstance(game_image, Mapping):
        raise ValueError("tournament child configuration has no game_image table")
    environment_variable = game_image.get("environment_variable")
    if not isinstance(environment_variable, str) or not environment_variable:
        raise ValueError("game_image.environment_variable must be a non-empty string")
    environment_path = os.environ.get(environment_variable)
    if environment_path:
        return (
            Path(environment_path).expanduser().resolve(),
            f"environment:{environment_variable}",
        )
    default_path = game_image.get("default_path")
    if not isinstance(default_path, str) or not default_path:
        raise ValueError("game_image.default_path must be a non-empty string")
    return (
        _resolve_from_project(Path(default_path), project_root),
        "configuration-default",
    )


def _expected_emulator_application(
    request: TournamentRequest, project_root: Path
) -> JsonObject:
    config_path = _resolve_from_project(request.config_path, project_root)
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    emulator = config.get("emulator")
    if not isinstance(emulator, Mapping):
        raise ValueError("tournament child configuration has no emulator table")
    directory = emulator.get("directory")
    application = emulator.get("application")
    file_count = emulator.get("application_file_count")
    byte_length = emulator.get("application_byte_length")
    tree_sha256 = emulator.get("application_tree_sha256")
    executable_sha256 = emulator.get("executable_sha256")
    if not isinstance(directory, str) or not isinstance(application, str):
        raise ValueError("emulator directory and application must be strings")
    if (
        not isinstance(file_count, int)
        or isinstance(file_count, bool)
        or file_count < 1
    ):
        raise ValueError("emulator.application_file_count must be positive")
    if (
        not isinstance(byte_length, int)
        or isinstance(byte_length, bool)
        or byte_length < 1
    ):
        raise ValueError("emulator.application_byte_length must be positive")
    if not isinstance(tree_sha256, str) or len(tree_sha256) != 64:
        raise ValueError("emulator.application_tree_sha256 must be a SHA-256")
    if not isinstance(executable_sha256, str) or len(executable_sha256) != 64:
        raise ValueError("emulator.executable_sha256 must be a SHA-256")
    application_path = Path(directory) / application
    return {
        "path": application_path.as_posix(),
        "file_count": file_count,
        "byte_length": byte_length,
        "tree_manifest_sha256": tree_sha256,
        "executable_sha256": executable_sha256,
    }


def _game_image_identity(
    request: TournamentRequest, project_root: Path, *, source: str | None = None
) -> JsonObject:
    path, resolved_source = _resolved_game_image_path(request, project_root)
    if not path.is_file():
        raise FileNotFoundError(f"tournament game image is missing: {path}")
    with path.open("rb") as stream:
        header = stream.read(8)
    disc_game_id = header[:6].decode("ascii", errors="replace")
    disc_revision = header[7] if len(header) > 7 else None
    config_path = _resolve_from_project(request.config_path, project_root)
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    game_image = config.get("game_image")
    if not isinstance(game_image, Mapping):
        raise ValueError("tournament child configuration has no game_image table")
    expected_game_id = game_image.get("disc_game_id")
    expected_revision = game_image.get("disc_revision")
    expected_byte_length = game_image.get("byte_length")
    expected_sha256 = game_image.get("sha256")
    if expected_game_id != "GALE01" or expected_revision != 2:
        raise ValueError(
            "tournament child configuration does not require Melee NTSC 1.02"
        )
    if (
        not isinstance(expected_byte_length, int)
        or isinstance(expected_byte_length, bool)
        or expected_byte_length < 1
    ):
        raise ValueError("game_image.byte_length must be positive")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("game_image.sha256 must be a SHA-256")
    if disc_game_id != expected_game_id or disc_revision != expected_revision:
        raise ValueError(
            f"tournament game image is not Melee NTSC 1.02: disc_game_id={disc_game_id!r}, disc_revision={disc_revision!r}"
        )
    identity = _file_identity(path, project_root)
    if (
        identity["byte_length"] != expected_byte_length
        or identity["sha256"] != expected_sha256
    ):
        raise ValueError(
            f"tournament game image does not match the frozen image identity: byte_length={identity['byte_length']!r}, sha256={identity['sha256']!r}"
        )
    return {
        **identity,
        "resolved_source": resolved_source if source is None else source,
        "disc_game_id": disc_game_id,
        "disc_revision": disc_revision,
    }


def _file_identity(path: Path, project_root: Path) -> JsonObject:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"tournament reproducibility input is missing: {resolved}"
        )
    return {
        "path": _display_path(resolved, project_root),
        "sha256": _sha256_file(resolved),
        "byte_length": resolved.stat().st_size,
    }


def _git_command(project_root: Path, *arguments: str) -> tuple[int, str, str]:
    completed = subprocess.run(
        ["git", "-C", str(project_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    return (completed.returncode, completed.stdout.strip(), completed.stderr.strip())


def _code_identity(project_root: Path) -> JsonObject:
    implementation_files = [
        _file_identity(project_root / relative, project_root)
        for relative in IMPLEMENTATION_PATHS
    ]
    bundle_sha256 = _sha256_bytes(_canonical_json(implementation_files).encode("utf-8"))
    root_code, git_root, root_error = _git_command(
        project_root, "rev-parse", "--show-toplevel"
    )
    commit_code, commit, commit_error = _git_command(project_root, "rev-parse", "HEAD")
    status_code, status, status_error = _git_command(
        project_root, "status", "--porcelain=v1", "--untracked-files=all"
    )
    git_available = root_code == commit_code == status_code == 0
    return {
        "git_available": git_available,
        "git_root": git_root if root_code == 0 else None,
        "commit": commit if commit_code == 0 else None,
        "dirty": bool(status) if status_code == 0 else None,
        "status_porcelain": status.splitlines() if status_code == 0 else None,
        "errors": [
            error
            for code, error in (
                (root_code, root_error),
                (commit_code, commit_error),
                (status_code, status_error),
            )
            if code != 0 and error
        ],
        "implementation_files": implementation_files,
        "implementation_bundle_sha256": bundle_sha256,
    }


def _request_configuration(request: TournamentRequest) -> JsonObject:
    return {
        "entrants": list(_canonical_entrants(request.entrants)),
        "base_policy_sampling_seeds": sorted(request.seeds),
        "policy_sampling_seed_derivation": {
            "domain": POLICY_SEED_DERIVATION_DOMAIN,
            "algorithm": "uint32 big-endian from the first four bytes of SHA-256",
            "inputs": ["base_policy_sampling_seed", "mirror_repetition"],
            "paired_use": "the same effective seed is used for both port orientations",
            "collision_policy": "fail closed across distinct base-seed/repetition inputs",
        },
        "policy_sampling_seed_scope": "Seeds Python, NumPy, Torch, and TensorFlow policy/evaluation sampling where the child runtime supports them. It does not claim to control Dolphin or Melee RNG.",
        "stages": sorted(request.stages, key=LEGAL_STAGES.index),
        "games_per_block": request.games_per_block,
        "matched_port_games_per_block": 2,
        "order_seed": request.order_seed,
        "execution_order_method": "sha256(order_seed, game_id)",
        "character": request.character,
        "expected_replay_costumes": {"p1": 1, "p2": 0},
        "replay_costume_interpretation": "Fail-closed replay expectation for the local duplicate-Fox CSS auto-selection: physical P1 records costume 1 and physical P2 records costume 0. The orchestrator does not claim to control costumes through the one-game CLI. Reversing ports still reverses the replay-observed color assigned to each model.",
        "costume_controlled_by_orchestrator": False,
        "max_game_frames": request.max_game_frames,
        "child_wall_timeout_seconds": request.child_wall_timeout_seconds,
        "child_shutdown_protocol": {
            "initial_signal": "SIGINT",
            "interrupt_grace_seconds": CHILD_INTERRUPT_GRACE_SECONDS,
            "fallback_signal": "SIGTERM",
            "terminate_grace_seconds": CHILD_TERMINATE_GRACE_SECONDS,
            "last_resort": "SIGKILL",
            "scope": "child process group, including Dolphin",
        },
        "inference_mode": "exact",
        "inference_interpretation": "behavior-preserving, frame-complete, and time-dilated; not a real-time compute contest",
        "fresh_process_per_game": True,
        "require_natural_end": True,
        "reject_frame_limit": True,
        "draw_handling": "Accept a complete, conclusive replay draw as a sporting result and score it as one-half point; never rerun it as a technical failure.",
        "sudden_death_transition_handling": "The one-game launcher stops at the first sudden-death transition, selects the unique replay covering the full base-game policy trace, and archives any just-created tiebreak replay as auxiliary evidence. The base-game draw receives one-half point.",
        "port_mirror_randomness": "Port orientations share policy sampling seeds but not Dolphin or Melee RNG. These are port-mirrored blocks, not full common-random-number pairs; stage is retained for stratified analysis.",
        "pause_rule_disclosure": "The local Slippi ruleset records pause enabled. All three policy action spaces omit Start, so entrants cannot invoke pause or LRAS. Replay audit rejects any LRAS Game End as technical rather than scoring it as policy strength.",
        "reject_inconclusive_result": True,
        "retry_policy": "No replacement attempt is permitted inside a report. Any launch, runtime, watchdog, summary, reproducibility, rules, or replay failure invalidates that report so uncontrolled game states cannot be conditioned away.",
    }


def _immutable_manifest_disclosures(
    request: TournamentRequest, schedule: Sequence[Mapping[str, Any]]
) -> JsonObject:
    """Return every derived scientific identity and limitation that resume must recompute."""
    return {
        "classification": "mirrored, behavior-preserving, frame-complete, time-dilated exact-mode local evaluation; not a real-time compute contest",
        "study_label": "local-reference-round-robin",
        "experiment_id": "E011",
        "gate_scope": "The report gate covers schedule, runtime identity, replay rules, and result integrity only. A pass does not support a final strength claim.",
        "evaluation_design": _evaluation_design(schedule),
        "entrants": [
            {"id": entrant, "display_identity": DISPLAY_IDENTITIES[entrant]}
            for entrant in _canonical_entrants(request.entrants)
        ],
    }


def _evaluation_design(schedule: Sequence[Mapping[str, Any]]) -> JsonObject:
    boots = len(schedule)
    mirrored_configurations = len(
        {str(game["matched_port_block_id"]) for game in schedule}
    )
    return {
        "registry_id": "E011",
        "registry_phase": "reference-characterization-archive"
        if boots >= E011_MINIMUM_BOOTS
        else "development-smoke",
        "scheduled_boots": boots,
        "mirrored_configurations": mirrored_configurations,
        "e011_reference_characterization_minimum_boots": E011_MINIMUM_BOOTS,
        "final_screening_minimum_mirrored_configurations": FINAL_SCREENING_MINIMUM_MIRRORED_CONFIGURATIONS,
        "meets_e011_boot_count": boots >= E011_MINIMUM_BOOTS,
        "meets_final_screening_mirrored_configuration_count": mirrored_configurations
        >= FINAL_SCREENING_MINIMUM_MIRRORED_CONFIGURATIONS,
        "paired_stratified_bootstrap_implemented": False,
        "full_performance_profile_implemented": False,
        "result_claim_ready": False,
        "interpretation": "This runner archives trustworthy games and matched port blocks. Its current aggregate is descriptive infrastructure, not the paired-bootstrap, multi-metric statistical report required for a final strength claim.",
    }


def _safe_fragment(value: str) -> str:
    return "".join(
        (character if character.isalnum() else "-" for character in value.lower())
    ).strip("-")


def _derive_policy_sampling_seed(base_seed: int, mirror_repetition: int) -> int:
    material = f"{POLICY_SEED_DERIVATION_DOMAIN}\x00{base_seed}\x00{mirror_repetition}".encode()
    return int.from_bytes(
        hashlib.sha256(material).digest()[:4], byteorder="big", signed=False
    )


def build_schedule(request: TournamentRequest) -> list[JsonObject]:
    """Return the complete schedule in deterministic randomized execution order."""
    request.validate()
    entrants = _canonical_entrants(request.entrants)
    seeds = sorted(request.seeds)
    stages = sorted(request.stages, key=LEGAL_STAGES.index)
    repetitions = request.games_per_block // 2
    derived_seeds = {
        (base_seed, repetition): _derive_policy_sampling_seed(base_seed, repetition)
        for base_seed in seeds
        for repetition in range(1, repetitions + 1)
    }
    reverse_seed_map: dict[int, tuple[int, int]] = {}
    for inputs, effective_seed in derived_seeds.items():
        previous = reverse_seed_map.setdefault(effective_seed, inputs)
        if previous != inputs:
            raise RuntimeError(
                f"derived policy sampling seed collision: {previous} and {inputs} both produced {effective_seed}"
            )
    schedule: list[JsonObject] = []
    for first, second in combinations(entrants, 2):
        pairing_id = f"{first}__{second}"
        for seed in seeds:
            for stage in stages:
                evaluation_block_id = (
                    f"{pairing_id}__base-seed-{seed}__stage-{stage.lower()}"
                )
                for repetition in range(1, repetitions + 1):
                    effective_seed = derived_seeds[seed, repetition]
                    matched_block_id = f"{evaluation_block_id}__mirror-{repetition:03d}"
                    for orientation, (player_1, player_2) in enumerate(
                        ((first, second), (second, first)), start=1
                    ):
                        game_id = f"{matched_block_id}__port-order-{orientation}"
                        schedule.append(
                            {
                                "game_id": game_id,
                                "pairing_id": pairing_id,
                                "evaluation_block_id": evaluation_block_id,
                                "matched_port_block_id": matched_block_id,
                                "mirror_repetition": repetition,
                                "port_order": orientation,
                                "base_policy_sampling_seed": seed,
                                "policy_sampling_seed": effective_seed,
                                "policy_sampling_seed_derivation": {
                                    "domain": POLICY_SEED_DERIVATION_DOMAIN,
                                    "mirror_repetition": repetition,
                                },
                                "policy_sampling_seed_scope": "policy/evaluation sampling only; does not control Dolphin or Melee RNG",
                                "stage": stage,
                                "character": request.character,
                                "expected_replay_costumes": {"p1": 1, "p2": 0},
                                "replay_costume_assignment": "local duplicate-Fox CSS auto-selection",
                                "player_1_model": player_1,
                                "player_1_display_identity": DISPLAY_IDENTITIES[
                                    player_1
                                ],
                                "player_2_model": player_2,
                                "player_2_display_identity": DISPLAY_IDENTITIES[
                                    player_2
                                ],
                            }
                        )
    for game in schedule:
        order_material = f"{request.order_seed}\x00{game['game_id']}".encode()
        game["randomized_order_key_sha256"] = _sha256_bytes(order_material)
    schedule.sort(
        key=lambda game: (game["randomized_order_key_sha256"], game["game_id"])
    )
    for execution_index, game in enumerate(schedule, start=1):
        game["execution_index"] = execution_index
    return schedule


def wilson_interval_95(wins: int, losses: int) -> JsonObject:
    """Wilson 95% interval for decisive Bernoulli games; draws are excluded."""
    if wins < 0 or losses < 0:
        raise ValueError("Wilson counts cannot be negative")
    sample_size = wins + losses
    if sample_size == 0:
        return {
            "method": "Wilson score interval, 95%, decisive games only; draws excluded",
            "wins": wins,
            "losses": losses,
            "sample_size": 0,
            "estimate": None,
            "lower": None,
            "upper": None,
        }
    proportion = wins / sample_size
    z_squared = WILSON_Z_95**2
    denominator = 1.0 + z_squared / sample_size
    center = (proportion + z_squared / (2.0 * sample_size)) / denominator
    half_width = (
        WILSON_Z_95
        * math.sqrt(
            proportion * (1.0 - proportion) / sample_size
            + z_squared / (4.0 * sample_size**2)
        )
        / denominator
    )
    return {
        "method": "Wilson score interval, 95%, decisive games only; draws excluded",
        "wins": wins,
        "losses": losses,
        "sample_size": sample_size,
        "estimate": proportion,
        "lower": max(0.0, center - half_width),
        "upper": min(1.0, center + half_width),
    }


def _empty_record(entrant: str) -> JsonObject:
    return {
        "entrant": entrant,
        "display_identity": DISPLAY_IDENTITIES[entrant],
        "games": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "points": 0.0,
        "by_port": {
            "p1": {"games": 0, "wins": 0, "losses": 0, "draws": 0, "points": 0.0},
            "p2": {"games": 0, "wins": 0, "losses": 0, "draws": 0, "points": 0.0},
        },
    }


def _accepted_outcome(game_record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if game_record.get("status") != "accepted":
        return None
    outcome = game_record.get("outcome")
    return outcome if isinstance(outcome, Mapping) else None


def aggregate_results(
    entrants: Sequence[str],
    schedule: Sequence[Mapping[str, Any]],
    games: Mapping[str, Any],
) -> JsonObject:
    """Aggregate accepted games while retaining each mirrored port block."""
    canonical_entrants = _canonical_entrants(entrants)
    entrant_records = {
        entrant: _empty_record(entrant) for entrant in canonical_entrants
    }
    pairing_records: dict[str, JsonObject] = {}
    matched_blocks: dict[str, list[Mapping[str, Any]]] = {}
    for definition in schedule:
        pairing_id = str(definition["pairing_id"])
        first, second = pairing_id.split("__", maxsplit=1)
        if pairing_id not in pairing_records:
            pairing_records[pairing_id] = {
                "pairing_id": pairing_id,
                "entrants": [
                    {"id": first, "display_identity": DISPLAY_IDENTITIES[first]},
                    {"id": second, "display_identity": DISPLAY_IDENTITIES[second]},
                ],
                "games": 0,
                "wins": {first: 0, second: 0},
                "losses": {first: 0, second: 0},
                "draws": 0,
                "points": {first: 0.0, second: 0.0},
            }
        matched_blocks.setdefault(str(definition["matched_port_block_id"]), []).append(
            definition
        )
        game_record = games.get(str(definition["game_id"]))
        if not isinstance(game_record, Mapping):
            continue
        outcome = _accepted_outcome(game_record)
        if outcome is None:
            continue
        player_1 = str(definition["player_1_model"])
        player_2 = str(definition["player_2_model"])
        winner = outcome.get("winner_model")
        pairing = pairing_records[pairing_id]
        pairing["games"] += 1
        for entrant, port in ((player_1, "p1"), (player_2, "p2")):
            record = entrant_records[entrant]
            record["games"] += 1
            cast(JsonObject, record["by_port"])[port]["games"] += 1
        if winner in (player_1, player_2):
            loser = player_2 if winner == player_1 else player_1
            winner_port = "p1" if winner == player_1 else "p2"
            loser_port = "p2" if winner == player_1 else "p1"
            entrant_records[str(winner)]["wins"] += 1
            entrant_records[loser]["losses"] += 1
            entrant_records[str(winner)]["points"] += 1.0
            cast(JsonObject, entrant_records[str(winner)]["by_port"])[winner_port][
                "wins"
            ] += 1
            cast(JsonObject, entrant_records[loser]["by_port"])[loser_port][
                "losses"
            ] += 1
            cast(JsonObject, entrant_records[str(winner)]["by_port"])[winner_port][
                "points"
            ] += 1.0
            cast(JsonObject, pairing["wins"])[str(winner)] += 1
            cast(JsonObject, pairing["losses"])[loser] += 1
            cast(JsonObject, pairing["points"])[str(winner)] += 1.0
        else:
            entrant_records[player_1]["draws"] += 1
            entrant_records[player_2]["draws"] += 1
            cast(JsonObject, entrant_records[player_1]["by_port"])["p1"]["draws"] += 1
            cast(JsonObject, entrant_records[player_2]["by_port"])["p2"]["draws"] += 1
            entrant_records[player_1]["points"] += 0.5
            entrant_records[player_2]["points"] += 0.5
            cast(JsonObject, entrant_records[player_1]["by_port"])["p1"]["points"] += (
                0.5
            )
            cast(JsonObject, entrant_records[player_2]["by_port"])["p2"]["points"] += (
                0.5
            )
            pairing["draws"] += 1
            cast(JsonObject, pairing["points"])[player_1] += 0.5
            cast(JsonObject, pairing["points"])[player_2] += 0.5
    for record in entrant_records.values():
        record["win_probability_95"] = wilson_interval_95(
            record["wins"], record["losses"]
        )
        record["score_rate"] = (
            record["points"] / record["games"] if record["games"] else None
        )
        for port_record in cast(JsonObject, record["by_port"]).values():
            port_record["score_rate"] = (
                port_record["points"] / port_record["games"]
                if port_record["games"]
                else None
            )
    block_summaries: dict[str, JsonObject] = {}
    for block_id, definitions in sorted(matched_blocks.items()):
        pairing_id = str(definitions[0]["pairing_id"])
        first, second = pairing_id.split("__", maxsplit=1)
        block_wins = {first: 0, second: 0}
        draws = 0
        accepted = 0
        game_ids: list[str] = []
        for definition in definitions:
            game_id = str(definition["game_id"])
            game_ids.append(game_id)
            game_record = games.get(game_id)
            if not isinstance(game_record, Mapping):
                continue
            outcome = _accepted_outcome(game_record)
            if outcome is None:
                continue
            accepted += 1
            winner = outcome.get("winner_model")
            if winner in block_wins:
                block_wins[str(winner)] += 1
            else:
                draws += 1
        complete = accepted == len(definitions) == 2
        if not complete:
            result = "incomplete"
        elif block_wins[first] > block_wins[second]:
            result = first
        elif block_wins[second] > block_wins[first]:
            result = second
        else:
            result = "tie"
        block_summaries[block_id] = {
            "matched_port_block_id": block_id,
            "pairing_id": pairing_id,
            "game_ids": sorted(game_ids),
            "expected_games": len(definitions),
            "accepted_games": accepted,
            "port_balance": {
                first: {
                    "p1": sum((d["player_1_model"] == first for d in definitions)),
                    "p2": sum((d["player_2_model"] == first for d in definitions)),
                },
                second: {
                    "p1": sum((d["player_1_model"] == second for d in definitions)),
                    "p2": sum((d["player_2_model"] == second for d in definitions)),
                },
            },
            "wins": block_wins,
            "draws": draws,
            "complete": complete,
            "result": result,
        }
    for pairing_id, pairing in pairing_records.items():
        first, second = pairing_id.split("__", maxsplit=1)
        wins = cast(JsonObject, pairing["wins"])
        losses = cast(JsonObject, pairing["losses"])
        pairing["win_probability_95"] = {
            first: wilson_interval_95(wins[first], losses[first]),
            second: wilson_interval_95(wins[second], losses[second]),
        }
        pairing_points = cast(JsonObject, pairing["points"])
        pairing["score_rate"] = {
            first: pairing_points[first] / pairing["games"]
            if pairing["games"]
            else None,
            second: pairing_points[second] / pairing["games"]
            if pairing["games"]
            else None,
        }
        relevant_blocks = [
            block
            for block in block_summaries.values()
            if block["pairing_id"] == pairing_id
        ]
        block_results = {first: 0, second: 0, "ties": 0, "incomplete": 0}
        for block in relevant_blocks:
            result = block["result"]
            if result in (first, second):
                block_results[str(result)] += 1
            elif result == "tie":
                block_results["ties"] += 1
            else:
                block_results["incomplete"] += 1
        pairing["matched_port_blocks"] = {
            "count": len(relevant_blocks),
            "complete": sum((bool(block["complete"]) for block in relevant_blocks)),
            "results": block_results,
            "blocks": relevant_blocks,
        }
    return {
        "uncertainty_note": "Wilson 95% intervals treat decisive individual games as Bernoulli observations and exclude draws. Matched port blocks are reported separately and must remain intact for paired analyses.",
        "entrants": entrant_records,
        "pairings": pairing_records,
        "matched_port_blocks": block_summaries,
    }


def _content_sha256_without_key(value: Mapping[str, Any], excluded_key: str) -> str:
    material = {key: item for key, item in value.items() if key != excluded_key}
    return _sha256_bytes(_canonical_json(material).encode("utf-8"))


def _completion_gate(
    request: TournamentRequest,
    *,
    all_accepted: bool,
    all_blocks_complete: bool,
    no_replacement_attempts: bool,
) -> JsonObject:
    complete = all_accepted and all_blocks_complete and no_replacement_attempts
    return {
        "decision": "pass" if complete else "incomplete",
        "scope": "execution and archive integrity, not a final strength claim",
        "result_claim_ready": False,
        "checks": {
            "all_scheduled_games_accepted": all_accepted,
            "all_matched_port_blocks_complete": all_blocks_complete,
            "no_replacement_attempts": no_replacement_attempts,
            "fresh_process_per_game": True,
            "exact_mode_only": True,
            "all_games_require_natural_end": True,
            "all_games_have_conclusive_win_or_draw": all_accepted,
        },
    }


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _record_child_process_group(
    process_group_id: int, attempt: JsonObject, report_path: Path, report: JsonObject
) -> None:
    attempt["child_process_group_id"] = process_group_id
    _atomic_write_json(report_path, report)


@contextmanager
def _exclusive_report_lock(report_path: Path) -> Iterator[None]:
    """Hold one nonblocking advisory lock for the full report lifecycle."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = report_path.with_name(f"{report_path.name}.lock")
    stream: TextIO = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            stream.seek(0)
            owner = stream.read().strip() or "unknown owner"
            raise TournamentRunError(
                f"tournament report is already locked: {report_path}; owner={owner}"
            ) from error
        stream.seek(0)
        stream.truncate()
        json.dump(
            {"pid": os.getpid(), "acquired_at_utc": _utc_now()}, stream, sort_keys=True
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        yield
    finally:
        with suppress(OSError):
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def _load_json_object(path: Path) -> JsonObject:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object in {path}")
    return cast(JsonObject, value)


def _revalidate_recorded_file(
    record_value: Any, project_root: Path, label: str
) -> None:
    record = _identity_record(record_value, label)
    path = Path(str(record["path"]))
    if not path.is_absolute():
        path = project_root / path
    if not path.is_file():
        raise TournamentRunError(f"accepted tournament evidence is missing: {path}")
    actual = _file_identity(path, project_root)
    if actual != record:
        raise TournamentRunError(
            f"accepted tournament evidence identity changed for {label}: {actual} != {record}"
        )


def _revalidate_accepted_evidence(
    report: Mapping[str, Any], project_root: Path
) -> None:
    games = report.get("games")
    if not isinstance(games, Mapping):
        raise TournamentRunError("tournament report has no games object")
    for game_id, game_value in games.items():
        if (
            not isinstance(game_value, Mapping)
            or game_value.get("status") != "accepted"
        ):
            continue
        accepted_attempt = game_value.get("accepted_attempt")
        attempts = game_value.get("attempts")
        if (
            not isinstance(accepted_attempt, int)
            or isinstance(accepted_attempt, bool)
            or (not isinstance(attempts, Sequence))
            or isinstance(attempts, (str, bytes))
            or (not 1 <= accepted_attempt <= len(attempts))
        ):
            raise TournamentRunError(
                f"accepted game {game_id} has an invalid accepted attempt"
            )
        attempt = attempts[accepted_attempt - 1]
        if not isinstance(attempt, Mapping) or attempt.get("status") != "accepted":
            raise TournamentRunError(
                f"accepted game {game_id} does not point to an accepted attempt"
            )
        _revalidate_recorded_file(
            attempt.get("child_summary_file"), project_root, f"{game_id} child summary"
        )
        _revalidate_recorded_file(
            attempt.get("saved_replay_file"), project_root, f"{game_id} saved replay"
        )
        _revalidate_recorded_file(
            attempt.get("controller_trace_file"),
            project_root,
            f"{game_id} controller trace",
        )
        child_replay_files = attempt.get("child_replay_files")
        if not isinstance(child_replay_files, Sequence) or isinstance(
            child_replay_files, (str, bytes)
        ):
            raise TournamentRunError(f"accepted game {game_id} has no replay archive")
        for replay_index, replay_file in enumerate(child_replay_files):
            _revalidate_recorded_file(
                replay_file, project_root, f"{game_id} child replay {replay_index}"
            )


def _validate_report_content(
    report: Mapping[str, Any],
    request: TournamentRequest,
    expected_schedule: list[JsonObject],
) -> None:
    if report.get("schedule") != expected_schedule:
        raise TournamentRunError(
            "tournament report schedule content differs from its frozen request"
        )
    manifest_value = report.get("manifest")
    if not isinstance(manifest_value, Mapping):
        raise TournamentRunError("tournament report has no manifest")
    expected_disclosures = _immutable_manifest_disclosures(request, expected_schedule)
    disclosure_failures = [
        key
        for key, expected in expected_disclosures.items()
        if manifest_value.get(key) != expected
    ]
    if disclosure_failures:
        raise TournamentRunError(
            f"tournament manifest scientific disclosures changed: {disclosure_failures}"
        )
    expected_configuration = _request_configuration(request)
    expected_configuration_sha256 = _sha256_bytes(
        _canonical_json(expected_configuration).encode()
    )
    expected_schedule_sha256 = _sha256_bytes(
        _canonical_json(expected_schedule).encode()
    )
    if (
        manifest_value.get("tournament_configuration") != expected_configuration
        or manifest_value.get("tournament_configuration_sha256")
        != expected_configuration_sha256
        or manifest_value.get("schedule_sha256") != expected_schedule_sha256
        or (manifest_value.get("schedule_game_count") != len(expected_schedule))
    ):
        raise TournamentRunError(
            "tournament manifest configuration or schedule identity changed"
        )
    games_value = report.get("games")
    if not isinstance(games_value, Mapping):
        raise TournamentRunError("tournament report has no games object")
    expected_by_id = {str(game["game_id"]): game for game in expected_schedule}
    if set(games_value) != set(expected_by_id):
        raise TournamentRunError(
            "tournament report game keys differ from the frozen schedule"
        )
    for game_id, definition in expected_by_id.items():
        game_value = games_value[game_id]
        if (
            not isinstance(game_value, Mapping)
            or game_value.get("definition") != definition
        ):
            raise TournamentRunError(
                f"tournament report game definition changed for {game_id}"
            )
        attempts = game_value.get("attempts")
        if not isinstance(attempts, Sequence) or isinstance(attempts, (str, bytes)):
            raise TournamentRunError(
                f"tournament report attempts are invalid for {game_id}"
            )
        if len(attempts) > 1:
            raise TournamentRunError(
                f"tournament report contains a forbidden replacement attempt for {game_id}"
            )
        if game_value.get("status") != "accepted":
            continue
        if game_value.get("accepted_attempt") != 1 or len(attempts) != 1:
            raise TournamentRunError(
                f"accepted game {game_id} has no unique first attempt"
            )
        attempt = attempts[0]
        if not isinstance(attempt, Mapping) or attempt.get("status") != "accepted":
            raise TournamentRunError(
                f"accepted game {game_id} does not point to an accepted attempt"
            )
        attempt_digest = attempt.get("accepted_record_sha256")
        if attempt_digest != _content_sha256_without_key(
            attempt, "accepted_record_sha256"
        ):
            raise TournamentRunError(f"accepted attempt content changed for {game_id}")
        audit = attempt.get("audit")
        audited_outcome = audit.get("outcome") if isinstance(audit, Mapping) else None
        replay_audit = attempt.get("replay_audit")
        controller_boundary_audit = attempt.get("controller_boundary_audit")
        auxiliary_replay_inspections = attempt.get("auxiliary_replay_inspections")
        child_reproducibility = attempt.get("child_reproducibility")
        if not isinstance(audit, Mapping) or audit.get("decision") != "pass":
            raise TournamentRunError(
                f"accepted game {game_id} has a failing child audit"
            )
        if (
            not isinstance(replay_audit, Mapping)
            or replay_audit.get("rules_passed") is not True
            or replay_audit.get("audit_passed") is not True
            or (replay_audit.get("tournament_result_ready") is not True)
        ):
            raise TournamentRunError(
                f"accepted game {game_id} has a failing replay audit"
            )
        boundary_gate = (
            controller_boundary_audit.get("gate")
            if isinstance(controller_boundary_audit, Mapping)
            else None
        )
        boundary_checks = (
            boundary_gate.get("checks") if isinstance(boundary_gate, Mapping) else None
        )
        required_boundary_checks = (
            "both_slots_physical_buttons_exact",
            "both_slots_processed_upstream_buttons_exact",
            "both_slots_intended_raw_main_stick_exact",
            "both_slots_processed_c_stick_within_tolerance",
            "both_slots_physical_analog_shoulders_within_tolerance",
        )
        if (
            not isinstance(boundary_gate, Mapping)
            or boundary_gate.get("decision") != "pass"
            or (not isinstance(boundary_checks, Mapping))
            or any(
                (
                    boundary_checks.get(name) is not True
                    for name in required_boundary_checks
                )
            )
        ):
            raise TournamentRunError(
                f"accepted game {game_id} has a failing controller boundary audit"
            )
        if (
            not isinstance(auxiliary_replay_inspections, Sequence)
            or isinstance(auxiliary_replay_inspections, (str, bytes))
            or any(
                (
                    not isinstance(value, Mapping)
                    or value.get("scoring") is not False
                    or value.get("role") != "sudden-death-transition-auxiliary"
                    or (not isinstance(value.get("inspection"), Mapping))
                    or (value.get("inspection_required_for_acceptance") is not False)
                    for value in auxiliary_replay_inspections
                )
            )
        ):
            raise TournamentRunError(
                f"accepted game {game_id} has an invalid auxiliary replay inspection"
            )
        if (
            not isinstance(child_reproducibility, Mapping)
            or child_reproducibility.get("decision") != "pass"
        ):
            raise TournamentRunError(
                f"accepted game {game_id} has a failing reproducibility audit"
            )
        if (
            not isinstance(audited_outcome, Mapping)
            or audited_outcome.get("status") not in ("win", "draw")
            or audited_outcome.get("natural_game_end") is not True
        ):
            raise TournamentRunError(
                f"accepted game {game_id} has no natural win or draw"
            )
        if game_value.get("outcome") != audited_outcome:
            raise TournamentRunError(
                f"accepted outcome differs from its replay audit for {game_id}"
            )
        game_digest = game_value.get("accepted_record_sha256")
        if game_digest != _content_sha256_without_key(
            game_value, "accepted_record_sha256"
        ):
            raise TournamentRunError(f"accepted game content changed for {game_id}")
    expected_baselines: JsonObject = {}
    for game_id in expected_by_id:
        game_value = cast(Mapping[str, Any], games_value[game_id])
        if game_value.get("status") != "accepted":
            continue
        attempts = cast(Sequence[Mapping[str, Any]], game_value["attempts"])
        attempt = attempts[0]
        child_reproducibility = cast(
            Mapping[str, Any], attempt["child_reproducibility"]
        )
        family = str(child_reproducibility["family"])
        expected_baselines.setdefault(
            family,
            {
                "sha256": child_reproducibility["sha256"],
                "file_identities": child_reproducibility["file_identities"],
                "runtime_environment": child_reproducibility["runtime_environment"],
                "first_accepted_game_id": game_id,
                "first_accepted_attempt": 1,
            },
        )
    if manifest_value.get("accepted_child_reproducibility") != expected_baselines:
        raise TournamentRunError(
            "accepted child reproducibility baselines differ from game evidence"
        )
    recomputed_aggregates = aggregate_results(
        request.entrants, expected_schedule, games_value
    )
    if report.get("aggregates") != recomputed_aggregates:
        raise TournamentRunError(
            "tournament report aggregates differ from accepted game records"
        )
    if report.get("status") != "complete":
        return
    all_accepted = all(
        (
            isinstance(game, Mapping) and game.get("status") == "accepted"
            for game in games_value.values()
        )
    )
    blocks = cast(
        Mapping[str, Mapping[str, Any]], recomputed_aggregates["matched_port_blocks"]
    )
    all_blocks_complete = bool(blocks) and all(
        (block.get("complete") is True for block in blocks.values())
    )
    no_replacement_attempts = all(
        (
            isinstance(game, Mapping)
            and isinstance(game.get("attempts"), Sequence)
            and (len(cast(Sequence[Any], game["attempts"])) == 1)
            for game in games_value.values()
        )
    )
    expected_gate = _completion_gate(
        request,
        all_accepted=all_accepted,
        all_blocks_complete=all_blocks_complete,
        no_replacement_attempts=no_replacement_attempts,
    )
    if report.get("gate") != expected_gate:
        raise TournamentRunError(
            "complete tournament gate differs from recomputed accepted records"
        )
    if not all_accepted or not all_blocks_complete or (not no_replacement_attempts):
        raise TournamentRunError(
            "complete tournament report does not satisfy completion invariants"
        )
    if expected_gate["decision"] != "pass":
        raise TournamentRunError(
            "complete tournament report has a failing completion gate"
        )


def _signal_child_process_group(
    process: subprocess.Popen[str], signal_number: int
) -> None:
    try:
        os.killpg(process.pid, signal_number)
    except ProcessLookupError:
        return
    except OSError:
        if process.poll() is not None:
            return
        with suppress(ProcessLookupError, OSError):
            process.send_signal(signal_number)


def _process_group_commands(process_group_id: int) -> list[str]:
    completed = subprocess.run(
        ["ps", "-axo", "pgid=,command="], check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise TournamentRunError(
            f"could not inspect prior child process group {process_group_id}: {completed.stderr.strip()}"
        )
    commands: list[str] = []
    for raw_line in completed.stdout.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        fields = stripped.split(maxsplit=1)
        if (
            len(fields) == 2
            and fields[0].isdigit()
            and (int(fields[0]) == process_group_id)
        ):
            commands.append(fields[1])
    return commands


def _recorded_process_group_status(process_group_id: int, artifact_label: str) -> str:
    if not artifact_label:
        raise TournamentRunError(
            "recorded child process group has no unique artifact label"
        )
    commands = _process_group_commands(process_group_id)
    if not commands:
        return "absent"
    if any((artifact_label in command for command in commands)):
        return "matching"
    return "unknown"


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_recorded_process_group(process_group_id: int) -> str:
    for signal_number, name, grace in (
        (signal.SIGINT, "sigint", CHILD_INTERRUPT_GRACE_SECONDS),
        (signal.SIGTERM, "sigterm", CHILD_TERMINATE_GRACE_SECONDS),
    ):
        with suppress(ProcessLookupError):
            os.killpg(process_group_id, signal_number)
        deadline = time.monotonic() + grace
        while _process_group_exists(process_group_id) and time.monotonic() < deadline:
            time.sleep(0.1)
        if not _process_group_exists(process_group_id):
            return name
    with suppress(ProcessLookupError):
        os.killpg(process_group_id, signal.SIGKILL)
    deadline = time.monotonic() + CHILD_TERMINATE_GRACE_SECONDS
    while _process_group_exists(process_group_id) and time.monotonic() < deadline:
        time.sleep(0.1)
    if _process_group_exists(process_group_id):
        raise TournamentRunError(
            f"prior child process group {process_group_id} survived SIGINT, SIGTERM, and SIGKILL"
        )
    return "sigkill"


def _stop_child_process_group(process: subprocess.Popen[str]) -> tuple[str, str, str]:
    """Stop the launcher and its Dolphin descendants, draining captured output."""
    _signal_child_process_group(process, signal.SIGINT)
    try:
        stdout, stderr = process.communicate(timeout=CHILD_INTERRUPT_GRACE_SECONDS)
        return (stdout or "", stderr or "", "sigint")
    except subprocess.TimeoutExpired:
        _signal_child_process_group(process, signal.SIGTERM)
    try:
        stdout, stderr = process.communicate(timeout=CHILD_TERMINATE_GRACE_SECONDS)
        return (stdout or "", stderr or "", "sigterm")
    except subprocess.TimeoutExpired:
        _signal_child_process_group(process, signal.SIGKILL)
        try:
            stdout, stderr = process.communicate(timeout=CHILD_TERMINATE_GRACE_SECONDS)
            return (stdout or "", stderr or "", "sigkill")
        except subprocess.TimeoutExpired as error:
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    with suppress(OSError):
                        stream.close()
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=1.0)
            stdout = error.output if isinstance(error.output, str) else ""
            stderr = error.stderr if isinstance(error.stderr, str) else ""
            return (stdout, stderr, "sigkill-output-drain-timeout")


def _default_process_runner(
    command: list[str],
    cwd: Path,
    timeout_seconds: float,
    *,
    on_start: Callable[[int], None] | None = None,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        if on_start is not None:
            on_start(process.pid)
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        stdout, stderr, shutdown = _stop_child_process_group(process)
        raise ChildProcessTimeout(
            timeout_seconds, stdout, f"{stderr}\nwatchdog_shutdown={shutdown}".strip()
        ) from None
    except BaseException:
        _stop_child_process_group(process)
        raise
    return subprocess.CompletedProcess(command, int(process.returncode), stdout, stderr)


def _stream_identity(value: str) -> JsonObject:
    encoded = value.encode("utf-8", errors="replace")
    return {
        "sha256": _sha256_bytes(encoded),
        "byte_length": len(encoded),
        "tail": value[-4000:],
    }


def _summary_slots(
    summary: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    slots = summary.get("slots")
    if isinstance(slots, Mapping):
        player_1 = slots.get("p1")
        player_2 = slots.get("p2")
        if isinstance(player_1, Mapping) and isinstance(player_2, Mapping):
            return (player_1, player_2)
    configuration = summary.get("configuration")
    if isinstance(configuration, Mapping):
        player_1 = configuration.get("player_1")
        player_2 = configuration.get("player_2")
        if isinstance(player_1, Mapping) and isinstance(player_2, Mapping):
            return (player_1, player_2)
    return ({}, {})


def _child_runtime_family(game: Mapping[str, Any]) -> str:
    models = (game.get("player_1_model"), game.get("player_2_model"))
    return "slippi-ai-mixed" if "slippi-ai" in models else "mimic"


def _identity_record(value: Any, label: str) -> JsonObject:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not a file-identity object")
    path = value.get("path")
    digest = value.get("sha256")
    byte_length = value.get("byte_length")
    if not isinstance(path, str) or not path:
        raise ValueError(f"{label} has no path")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{label} has no SHA-256")
    if (
        not isinstance(byte_length, int)
        or isinstance(byte_length, bool)
        or byte_length < 0
    ):
        raise ValueError(f"{label} has an invalid byte length")
    return {"path": path, "sha256": digest, "byte_length": byte_length}


def _parent_file_identities(manifest: Mapping[str, Any]) -> dict[str, JsonObject]:
    code = manifest.get("code")
    if not isinstance(code, Mapping):
        raise ValueError("tournament manifest has no code identity")
    raw_files = code.get("implementation_files")
    if not isinstance(raw_files, Sequence) or isinstance(raw_files, (str, bytes)):
        raise ValueError("tournament manifest implementation identities are invalid")
    records = [
        _identity_record(value, f"tournament implementation identity {index}")
        for index, value in enumerate(raw_files)
    ]
    records.append(
        _identity_record(
            manifest.get("child_configuration"), "child configuration identity"
        )
    )
    result = {str(record["path"]): record for record in records}
    if len(result) != len(records):
        raise ValueError("tournament manifest contains duplicate file-identity paths")
    return result


def _audit_child_reproducibility(
    summary: Mapping[str, Any], game: Mapping[str, Any], manifest: Mapping[str, Any]
) -> JsonObject:
    family = _child_runtime_family(game)
    reproducibility = summary.get("reproducibility")
    if not isinstance(reproducibility, Mapping):
        raise ValueError("child summary has no reproducibility identity")
    raw_implementations = reproducibility.get("implementation_files")
    if not isinstance(raw_implementations, Sequence) or isinstance(
        raw_implementations, (str, bytes)
    ):
        raise ValueError("child implementation identities are invalid")
    records = [
        _identity_record(
            reproducibility.get("configuration"), "child configuration identity"
        ),
        _identity_record(
            reproducibility.get("dependency_lock"), "child dependency-lock identity"
        ),
        *(
            _identity_record(value, f"child implementation identity {index}")
            for index, value in enumerate(raw_implementations)
        ),
    ]
    child_by_path = {str(record["path"]): record for record in records}
    if len(child_by_path) != len(records):
        raise ValueError("child reproducibility identity contains duplicate paths")
    runtime_environment_value = reproducibility.get("runtime_environment")
    if not isinstance(runtime_environment_value, Mapping):
        raise ValueError("child reproducibility identity has no runtime environment")
    runtime_environment = dict(runtime_environment_value)
    recorded_environment_sha256 = runtime_environment.get("identity_sha256")
    calculated_environment_sha256 = runtime_environment_identity_sha256(
        runtime_environment
    )
    lock_validation_value = runtime_environment.get("dependency_lock_validation")
    lock_validation = (
        lock_validation_value if isinstance(lock_validation_value, Mapping) else {}
    )
    child_configuration = _identity_record(
        manifest.get("child_configuration"), "manifest child configuration identity"
    )
    expected_paths = {str(child_configuration["path"]), *CHILD_RUNTIME_PATHS[family]}
    parent_by_path = _parent_file_identities(manifest)
    parent_subset = {path: parent_by_path.get(path) for path in sorted(expected_paths)}
    paths_exact = set(child_by_path) == expected_paths
    parent_exact = paths_exact and all(
        (parent_subset[path] == child_by_path[path] for path in expected_paths)
    )
    digest = _sha256_bytes(
        _canonical_json(
            {
                "file_identities": child_by_path,
                "runtime_environment": runtime_environment,
            }
        ).encode()
    )
    raw_baselines = manifest.get("accepted_child_reproducibility")
    baselines = raw_baselines if isinstance(raw_baselines, Mapping) else {}
    baseline = baselines.get(family)
    baseline_matches = baseline is None or (
        isinstance(baseline, Mapping) and baseline.get("sha256") == digest
    )
    checks = {
        "expected_file_paths_exact": paths_exact,
        "matches_frozen_parent_file_identities": parent_exact,
        "runtime_environment_identity_self_consistent": recorded_environment_sha256
        == calculated_environment_sha256,
        "runtime_environment_lock_gate_passed": lock_validation.get(
            "environment_lock_gate_passed"
        )
        is True,
        "runtime_interpreter_exact": runtime_environment.get("python_executable")
        == CHILD_RUNTIME_INTERPRETERS[family],
        "report_global_host_identity_exact": {
            key: runtime_environment.get(key)
            for key in ("host_platform", "host_machine", "host_os_build")
        }
        == manifest.get("host_identity"),
        "matches_first_accepted_child_for_runtime_family": baseline_matches,
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "decision": "pass" if not failures else "fail",
        "family": family,
        "sha256": digest,
        "file_identities": child_by_path,
        "runtime_environment": runtime_environment,
        "checks": checks,
        "failures": failures,
    }


def _audit_child_summary(
    summary: Mapping[str, Any],
    game: Mapping[str, Any],
    replay_audit: Mapping[str, Any],
    replay_file: Mapping[str, Any],
    child_replay_record: Mapping[str, Any],
    controller_trace_file: Mapping[str, Any],
    child_trace_record: Mapping[str, Any],
    controller_boundary_audit: Mapping[str, Any],
    auxiliary_replay_inspections: Sequence[Mapping[str, Any]],
    child_reproducibility: Mapping[str, Any],
    expected_game_image: Mapping[str, Any],
    expected_emulator_application: Mapping[str, Any],
) -> JsonObject:
    configuration_value = summary.get("configuration")
    configuration = (
        configuration_value if isinstance(configuration_value, Mapping) else {}
    )
    execution_value = summary.get("execution")
    execution = execution_value if isinstance(execution_value, Mapping) else {}
    gate_value = summary.get("gate")
    gate = gate_value if isinstance(gate_value, Mapping) else {}
    child_gate_checks_value = gate.get("checks")
    child_gate_checks = (
        child_gate_checks_value if isinstance(child_gate_checks_value, Mapping) else {}
    )
    player_1, player_2 = _summary_slots(summary)
    natural_end = next(
        (
            execution[key]
            for key in ("natural_game_end", "game_end_observed", "natural_end")
            if key in execution
        ),
        None,
    )
    replay_outcome_value = replay_audit.get("outcome")
    replay_outcome = (
        replay_outcome_value if isinstance(replay_outcome_value, Mapping) else {}
    )
    replay_identity_value = replay_audit.get("replay")
    replay_identity = (
        replay_identity_value if isinstance(replay_identity_value, Mapping) else {}
    )
    child_game_image_value = summary.get("game_image")
    child_game_image = (
        child_game_image_value if isinstance(child_game_image_value, Mapping) else {}
    )
    child_emulator_value = summary.get("emulator_application")
    child_emulator = (
        child_emulator_value if isinstance(child_emulator_value, Mapping) else {}
    )
    child_executable_value = child_emulator.get("executable")
    child_executable = (
        child_executable_value if isinstance(child_executable_value, Mapping) else {}
    )
    boundary_gate_value = controller_boundary_audit.get("gate")
    boundary_gate = (
        boundary_gate_value if isinstance(boundary_gate_value, Mapping) else {}
    )
    boundary_checks_value = boundary_gate.get("checks")
    boundary_checks = (
        boundary_checks_value if isinstance(boundary_checks_value, Mapping) else {}
    )
    boundary_trace_value = controller_boundary_audit.get("trace")
    boundary_trace = (
        boundary_trace_value if isinstance(boundary_trace_value, Mapping) else {}
    )
    boundary_replay_value = controller_boundary_audit.get("replay")
    boundary_replay = (
        boundary_replay_value if isinstance(boundary_replay_value, Mapping) else {}
    )
    winner_port = replay_outcome.get("winner_port")
    outcome_status = replay_outcome.get("status")
    draw = replay_outcome.get("draw") is True
    winner_model = (
        game["player_1_model"]
        if winner_port == 1
        else game["player_2_model"]
        if winner_port == 2
        else None
    )
    mixed_slippi_game = "slippi-ai" in (game["player_1_model"], game["player_2_model"])
    current_frame_barrier_exact = (
        all(
            (
                child_gate_checks.get(name) is True
                for name in (
                    "exact_mode_only",
                    "slippi_policy_called_once_per_processed_frame",
                    "no_recurrent_frame_drop",
                    "slippi_current_frame_inference_barrier",
                    "windowed_exact_inference_count",
                    "windowed_exact_command_age_zero",
                )
            )
        )
        if mixed_slippi_game
        else all(
            (
                child_gate_checks.get(name) is True
                for name in ("mimic_current_frame_inference_barrier",)
            )
        )
    )
    checks = {
        "child_gate_pass": gate.get("decision") == "pass",
        "both_policies_finish_current_step_before_advance": current_frame_barrier_exact,
        "player_1_model_exact": player_1.get("model") == game["player_1_model"],
        "player_2_model_exact": player_2.get("model") == game["player_2_model"],
        "player_1_character_exact": player_1.get(
            "character", player_1.get("requested_character")
        )
        == game["character"],
        "player_2_character_exact": player_2.get(
            "character", player_2.get("requested_character")
        )
        == game["character"],
        "policy_sampling_seed_exact": configuration.get("seed")
        == game["policy_sampling_seed"],
        "stage_exact": configuration.get("stage") == game["stage"],
        "exact_inference": configuration.get("inference_mode")
        in ("exact", "synchronous-concurrent"),
        "natural_end_flag": natural_end is True,
        "natural_end_termination": execution.get("termination") == "natural-game-end",
        "not_frame_limit": "frame-limit" not in str(execution.get("termination", "")),
        "single_child_replay_identity_sha256": child_replay_record.get("sha256")
        == replay_file.get("sha256"),
        "single_child_replay_identity_bytes": child_replay_record.get("byte_length")
        == replay_file.get("byte_length"),
        "replay_audit_identity_sha256": replay_identity.get("sha256")
        == replay_file.get("sha256"),
        "replay_audit_identity_bytes": replay_identity.get("raw_byte_length")
        == replay_file.get("byte_length"),
        "controller_trace_identity_sha256": child_trace_record.get("sha256")
        == controller_trace_file.get("sha256"),
        "controller_trace_identity_bytes": child_trace_record.get("byte_length")
        == controller_trace_file.get("byte_length"),
        "controller_boundary_gate_pass": boundary_gate.get("decision") == "pass",
        "controller_boundary_dispatch_exclusions_exact": boundary_checks.get(
            "both_slots_no_dispatch_rule_exact"
        )
        is True,
        "controller_boundary_trace_identity_exact": (
            boundary_trace.get("sha256"),
            boundary_trace.get("byte_length"),
        )
        == (
            controller_trace_file.get("sha256"),
            controller_trace_file.get("byte_length"),
        ),
        "controller_boundary_replay_identity_exact": (
            boundary_replay.get("sha256"),
            boundary_replay.get("byte_length"),
        )
        == (replay_file.get("sha256"), replay_file.get("byte_length")),
        "controller_boundary_buttons_exact": boundary_checks.get(
            "both_slots_physical_buttons_exact"
        )
        is True
        and boundary_checks.get("both_slots_processed_upstream_buttons_exact") is True,
        "controller_boundary_raw_main_exact": boundary_checks.get(
            "both_slots_intended_raw_main_stick_exact"
        )
        is True,
        "controller_boundary_processed_c_stick_exact": boundary_checks.get(
            "both_slots_processed_c_stick_within_tolerance"
        )
        is True,
        "controller_boundary_physical_analog_shoulders_exact": boundary_checks.get(
            "both_slots_physical_analog_shoulders_within_tolerance"
        )
        is True,
        "auxiliary_replays_retained_and_explicitly_non_scoring": all(
            (
                value.get("scoring") is False
                and value.get("role") == "sudden-death-transition-auxiliary"
                and isinstance(value.get("inspection"), Mapping)
                and (value.get("inspection_required_for_acceptance") is False)
                for value in auxiliary_replay_inspections
            )
        ),
        "replay_parser_exact": replay_identity.get("parser") == "peppi-py"
        and replay_identity.get("parser_version") == "0.9.2",
        "game_image_sha256_frozen": child_game_image.get("sha256")
        == expected_game_image.get("sha256"),
        "game_image_bytes_frozen": child_game_image.get("byte_length")
        == expected_game_image.get("byte_length"),
        "game_image_disc_identity_frozen": (
            child_game_image.get("disc_game_id"),
            child_game_image.get("disc_revision"),
        )
        == (
            expected_game_image.get("disc_game_id"),
            expected_game_image.get("disc_revision"),
        ),
        "emulator_application_tree_frozen": (
            child_emulator.get("path"),
            child_emulator.get("file_count"),
            child_emulator.get("byte_length"),
            child_emulator.get("tree_manifest_sha256"),
        )
        == (
            expected_emulator_application.get("path"),
            expected_emulator_application.get("file_count"),
            expected_emulator_application.get("byte_length"),
            expected_emulator_application.get("tree_manifest_sha256"),
        ),
        "emulator_executable_frozen": child_executable.get("sha256")
        == expected_emulator_application.get("executable_sha256"),
        "emulator_matches_frozen_configuration": child_emulator.get(
            "matches_frozen_configuration"
        )
        is True,
        "child_reproducibility_exact": child_reproducibility.get("decision") == "pass",
        "replay_rules_passed": replay_audit.get("rules_passed") is True,
        "replay_audit_passed": replay_audit.get("audit_passed") is True,
        "replay_result_ready": replay_audit.get("tournament_result_ready") is True,
        "replay_outcome_conclusive": replay_outcome.get("conclusive") is True,
        "replay_game_complete": replay_outcome.get("game_complete") is True,
        "replay_outcome_is_win_or_draw": outcome_status in ("win", "draw"),
        "explicit_win_or_draw": outcome_status == "win"
        and (not draw)
        and (winner_port in (1, 2))
        or (outcome_status == "draw" and draw and (winner_port is None)),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "decision": "pass" if not failures else "fail",
        "checks": checks,
        "failures": failures,
        "outcome": {
            "winner_model": winner_model,
            "winner_display_identity": DISPLAY_IDENTITIES[str(winner_model)]
            if winner_model in DISPLAY_IDENTITIES
            else None,
            "winner_port": winner_port,
            "loser_port": replay_outcome.get("loser_port"),
            "status": outcome_status,
            "draw": draw,
            "requires_tiebreak": replay_outcome.get("requires_tiebreak"),
            "termination": execution.get("termination"),
            "natural_game_end": natural_end,
            "replay_start_random_seed": cast(
                Mapping[str, Any], replay_audit.get("settings", {})
            ).get("game_random_seed"),
            "replay_outcome_reason": replay_outcome.get("reason"),
        },
    }


def _child_replay(
    summary: Mapping[str, Any], summary_path: Path, project_root: Path
) -> tuple[Path, Mapping[str, Any]]:
    """Resolve exactly one current-run replay inside the child's artifact directory."""
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("child summary has no artifacts object")
    replays = artifacts.get("replays")
    if not isinstance(replays, Sequence) or isinstance(replays, (str, bytes)):
        raise ValueError("child summary replay records are missing or invalid")
    selected = [
        record
        for record in replays
        if isinstance(record, Mapping)
        and record.get("tournament_result_replay") is True
    ]
    if len(selected) != 1:
        raise ValueError(
            f"expected exactly one child tournament-result replay record, found {len(selected)} among {len(replays)} records"
        )
    replay_record = cast(Mapping[str, Any], selected[0])
    raw_path = replay_record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("child replay record has no path")
    replay_path = Path(raw_path)
    if not replay_path.is_absolute():
        replay_path = project_root / replay_path
    replay_path = replay_path.resolve()
    artifact_directory = summary_path.parent.resolve()
    if not replay_path.is_relative_to(artifact_directory):
        raise ValueError(
            f"child replay escaped its unique artifact directory: {replay_path} not under {artifact_directory}"
        )
    if not replay_path.is_file():
        raise FileNotFoundError(f"child replay is missing: {replay_path}")
    return (replay_path, replay_record)


def _child_trace(
    summary: Mapping[str, Any], summary_path: Path, project_root: Path
) -> tuple[Path, JsonObject]:
    """Resolve and authenticate the controller trace inside the child's artifact directory."""
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("child summary has no artifacts object")
    trace_record = _identity_record(artifacts.get("trace"), "child controller trace")
    trace_path = Path(str(trace_record["path"]))
    if not trace_path.is_absolute():
        trace_path = project_root / trace_path
    trace_path = trace_path.resolve()
    artifact_directory = summary_path.parent.resolve()
    if not trace_path.is_relative_to(artifact_directory):
        raise ValueError(
            f"child controller trace escaped its unique artifact directory: {trace_path} not under {artifact_directory}"
        )
    if not trace_path.is_file():
        raise FileNotFoundError(f"child controller trace is missing: {trace_path}")
    actual = _file_identity(trace_path, project_root)
    if actual != trace_record:
        raise ValueError(
            f"child controller trace identity differs from its summary: {actual} != {trace_record}"
        )
    return (trace_path, trace_record)


def _child_replay_files(
    summary: Mapping[str, Any], summary_path: Path, project_root: Path
) -> list[JsonObject]:
    """Authenticate every replay produced in the child's unique artifact directory."""
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("child summary has no artifacts object")
    replays = artifacts.get("replays")
    if (
        not isinstance(replays, Sequence)
        or isinstance(replays, (str, bytes))
        or (not replays)
    ):
        raise ValueError("child summary replay records are missing or invalid")
    identities: list[JsonObject] = []
    seen_paths: set[str] = set()
    artifact_directory = summary_path.parent.resolve()
    for index, value in enumerate(replays):
        record = _identity_record(value, f"child replay {index}")
        replay_path = Path(str(record["path"]))
        if not replay_path.is_absolute():
            replay_path = project_root / replay_path
        replay_path = replay_path.resolve()
        if not replay_path.is_relative_to(artifact_directory):
            raise ValueError(
                f"child replay escaped its unique artifact directory: {replay_path} not under {artifact_directory}"
            )
        actual = _file_identity(replay_path, project_root)
        if actual != record:
            raise ValueError(
                f"child replay identity differs from its summary: {actual} != {record}"
            )
        path_key = str(actual["path"])
        if path_key in seen_paths:
            raise ValueError(
                f"child summary contains duplicate replay path: {path_key}"
            )
        seen_paths.add(path_key)
        identities.append(actual)
    return identities


def _inspect_auxiliary_replays(
    summary: Mapping[str, Any],
    replay_files: Sequence[Mapping[str, Any]],
    project_root: Path,
    *,
    expected_stage: str,
    expected_characters: Mapping[int, str],
    expected_costumes: Mapping[int, int],
) -> list[JsonObject]:
    """Retain unscored sudden-death files and inspect them without gating a valid base draw."""
    artifacts = summary.get("artifacts")
    raw_replays = artifacts.get("replays") if isinstance(artifacts, Mapping) else None
    if not isinstance(raw_replays, Sequence) or isinstance(raw_replays, (str, bytes)):
        raise ValueError("child summary replay records are missing or invalid")
    if len(raw_replays) != len(replay_files):
        raise ValueError("child replay archive length differs from its summary")
    from melee_policy.integration.replay_result import audit_tiebreaker_replay

    results: list[JsonObject] = []
    for index, (raw_record, replay_file) in enumerate(
        zip(raw_replays, replay_files, strict=True)
    ):
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"child replay {index} is not an object")
        if raw_record.get("tournament_result_replay") is True:
            continue
        if raw_record.get("tournament_result_replay") is not False:
            raise ValueError(
                f"child replay {index} has no explicit scoring classification"
            )
        if raw_record.get("role") != "sudden-death-transition-auxiliary":
            raise ValueError(
                f"child replay {index} is not a declared sudden-death auxiliary"
            )
        replay_path = Path(str(replay_file["path"]))
        if not replay_path.is_absolute():
            replay_path = project_root / replay_path
        try:
            inspection = audit_tiebreaker_replay(
                replay_path,
                expected_stage=expected_stage,
                expected_characters=expected_characters,
                expected_costumes=expected_costumes,
            )
        except Exception as error:
            inspection = {
                "schema_version": "integration.replay_tiebreaker.v1",
                "classification": "unverified or incomplete non-scoring sudden-death transition replay",
                "decision": "unavailable",
                "checks": {},
                "failures": ["inspection_exception"],
                "error": f"{type(error).__name__}: {error}",
            }
        results.append(
            {
                "replay": dict(replay_file),
                "role": "sudden-death-transition-auxiliary",
                "scoring": False,
                "inspection": inspection,
                "inspection_required_for_acceptance": False,
                "disclosure": "Slippi may leave this automatically opened sudden-death replay incomplete because the launcher stops at the transition. It is archived but never scored.",
            }
        )
    execution = summary.get("execution")
    transition_observed = (
        isinstance(execution, Mapping)
        and execution.get("sudden_death_transition_observed") is True
    )
    if results and (not transition_observed):
        raise ValueError(
            "child produced a sudden-death auxiliary without an observed transition"
        )
    if len(results) > 1:
        raise ValueError("child produced more than one sudden-death auxiliary replay")
    return results


def _default_replay_auditor(
    replay_path: Path,
    stage: str,
    expected_characters: Mapping[int, str],
    expected_costumes: Mapping[int, int],
) -> JsonObject:
    from melee_policy.integration.replay_result import audit_replay

    return audit_replay(
        replay_path,
        expected_stage=stage,
        expected_characters=expected_characters,
        expected_costumes=expected_costumes,
    )


def _expected_child_summary_path(
    project_root: Path, game: Mapping[str, Any], label: str
) -> Path:
    models = (game["player_1_model"], game["player_2_model"])
    if "slippi-ai" in models:
        return (
            project_root
            / "artifacts"
            / "integration"
            / "slippi_ai"
            / label
            / "summary.json"
        )
    return project_root / "artifacts" / "integration" / label / "summary.json"


def _artifact_label(run_id: str, game: Mapping[str, Any], attempt_number: int) -> str:
    fragment = _safe_fragment(str(game["game_id"]))
    return f"tournament-{run_id}-{fragment}-attempt-{attempt_number:03d}"


def _child_command(
    project_root: Path,
    request: TournamentRequest,
    game: Mapping[str, Any],
    artifact_label: str,
) -> list[str]:
    config_path = _resolve_from_project(request.config_path, project_root)
    command = [
        str(project_root / "scripts" / "play"),
        "--config",
        str(config_path),
        "--p1",
        str(game["player_1_model"]),
        "--p2",
        str(game["player_2_model"]),
        "--p1-character",
        str(game["character"]),
        "--p2-character",
        str(game["character"]),
        "--stage",
        str(game["stage"]),
        "--seed",
        str(game["policy_sampling_seed"]),
        "--max-game-frames",
        str(request.max_game_frames),
        "--inference-mode",
        "exact",
        "--require-natural-end",
        "--artifact-label",
        artifact_label,
    ]
    if request.iso_path is not None:
        command.extend(
            ["--iso-path", str(_resolve_from_project(request.iso_path, project_root))]
        )
    return command


def _new_report(
    request: TournamentRequest,
    project_root: Path,
    schedule: list[JsonObject],
    code_identity: JsonObject,
) -> JsonObject:
    run_id = uuid.uuid4().hex[:16]
    configuration = _request_configuration(request)
    configuration_sha256 = _sha256_bytes(_canonical_json(configuration).encode())
    schedule_sha256 = _sha256_bytes(_canonical_json(schedule).encode())
    child_config_path = _resolve_from_project(request.config_path, project_root)
    game_image = _game_image_identity(request, project_root)
    emulator_application = _expected_emulator_application(request, project_root)
    runtime_equivalence = _runtime_equivalence_record(project_root)
    if runtime_equivalence.get("decision") != "pass":
        raise TournamentRunError(
            f"cross-runtime policy-input equivalence failed: {runtime_equivalence.get('failures')}"
        )
    games: JsonObject = {
        str(game["game_id"]): {
            "definition": game,
            "status": "pending",
            "attempts": [],
            "accepted_attempt": None,
            "outcome": None,
        }
        for game in schedule
    }
    report: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "status": "pending",
        "gate": {"decision": "incomplete", "checks": {}},
        "manifest": {
            "run_id": run_id,
            "created_at_utc": _utc_now(),
            **_immutable_manifest_disclosures(request, schedule),
            "tournament_configuration": configuration,
            "tournament_configuration_sha256": configuration_sha256,
            "child_configuration": _file_identity(child_config_path, project_root),
            "host_identity": _host_identity(),
            "cross_runtime_policy_input_equivalence": runtime_equivalence,
            "game_image": game_image,
            "emulator_application": emulator_application,
            "orchestrator_command": [sys.executable, *sys.argv],
            "code": code_identity,
            "accepted_child_reproducibility": {},
            "schedule_sha256": schedule_sha256,
            "schedule_game_count": len(schedule),
        },
        "schedule": schedule,
        "games": games,
        "aggregates": aggregate_results(request.entrants, schedule, games),
        "last_error": None,
    }
    return report


def _load_or_create_report(
    request: TournamentRequest, project_root: Path, report_path: Path
) -> JsonObject:
    request.validate()
    schedule = build_schedule(request)
    code_identity = _code_identity(project_root)
    expected_configuration_sha256 = _sha256_bytes(
        _canonical_json(_request_configuration(request)).encode()
    )
    expected_schedule_sha256 = _sha256_bytes(_canonical_json(schedule).encode())
    if not report_path.exists():
        report = _new_report(request, project_root, schedule, code_identity)
        _atomic_write_json(report_path, report)
        return report
    report = _load_json_object(report_path)
    if report.get("schema_version") != SCHEMA_VERSION:
        raise TournamentRunError(
            f"cannot resume report with schema {report.get('schema_version')!r}"
        )
    manifest_value = report.get("manifest")
    if not isinstance(manifest_value, Mapping):
        raise TournamentRunError("cannot resume report without a manifest")
    manifest = manifest_value
    current_runtime_equivalence = _runtime_equivalence_record(project_root)
    stored_code_value = manifest.get("code")
    stored_code = stored_code_value if isinstance(stored_code_value, Mapping) else {}
    child_configuration = _file_identity(
        _resolve_from_project(request.config_path, project_root), project_root
    )
    resume_checks = {
        "configuration": manifest.get("tournament_configuration_sha256")
        == expected_configuration_sha256,
        "schedule": manifest.get("schedule_sha256") == expected_schedule_sha256,
        "implementation": stored_code.get("implementation_bundle_sha256")
        == code_identity["implementation_bundle_sha256"]
        and stored_code.get("implementation_files")
        == code_identity["implementation_files"]
        and (stored_code.get("git_available") == code_identity["git_available"])
        and (stored_code.get("git_root") == code_identity["git_root"])
        and (stored_code.get("commit") == code_identity["commit"]),
        "child_configuration": manifest.get("child_configuration")
        == child_configuration,
        "game_image": manifest.get("game_image")
        == _game_image_identity(request, project_root),
        "emulator_application": manifest.get("emulator_application")
        == _expected_emulator_application(request, project_root),
        "host_identity": manifest.get("host_identity") == _host_identity(),
        "cross_runtime_policy_input_equivalence": current_runtime_equivalence.get(
            "decision"
        )
        == "pass"
        and manifest.get("cross_runtime_policy_input_equivalence")
        == current_runtime_equivalence,
    }
    failures = [name for name, passed in resume_checks.items() if not passed]
    if failures:
        raise TournamentRunError(
            f"refusing to mix incompatible runs in one tournament report: {failures}"
        )
    _validate_report_content(report, request, schedule)
    _revalidate_accepted_evidence(report, project_root)
    return report


def _mark_interrupted(
    report: JsonObject, report_path: Path, request: TournamentRequest, message: str
) -> None:
    games = cast(JsonObject, report["games"])
    report["status"] = "invalid"
    report["last_error"] = message
    report["aggregates"] = aggregate_results(
        request.entrants, report["schedule"], games
    )
    report["gate"] = {
        "decision": "fail",
        "scope": "run invalidated; no replacement attempts are permitted inside this report",
        "checks": {
            "all_scheduled_games_accepted": False,
            "every_accepted_game_natural": all(
                (
                    cast(Mapping[str, Any], game.get("outcome", {})).get(
                        "natural_game_end"
                    )
                    is True
                    for game in games.values()
                    if isinstance(game, Mapping) and game.get("status") == "accepted"
                )
            ),
            "every_accepted_game_has_conclusive_result": all(
                (
                    cast(Mapping[str, Any], game.get("outcome", {})).get("status")
                    in ("win", "draw")
                    for game in games.values()
                    if isinstance(game, Mapping) and game.get("status") == "accepted"
                )
            ),
        },
    }
    _atomic_write_json(report_path, report)


def _run_tournament_unlocked(
    request: TournamentRequest,
    *,
    project_root: Path | None = None,
    process_runner: ProcessRunner | None = None,
    replay_auditor: ReplayAuditor | None = None,
    controller_boundary_auditor: ControllerBoundaryAuditor | None = None,
) -> JsonObject:
    """Run a tournament while the caller holds its exclusive report lock."""
    request.validate()
    root = (
        Path(__file__).resolve().parents[3]
        if project_root is None
        else project_root.expanduser().resolve()
    )
    runner = process_runner
    audit_replay = _default_replay_auditor if replay_auditor is None else replay_auditor
    audit_controller_boundary = (
        _default_controller_boundary_auditor
        if controller_boundary_auditor is None
        else controller_boundary_auditor
    )
    play_script = root / "scripts" / "play"
    if not play_script.is_file():
        raise FileNotFoundError(f"one-game launcher is missing: {play_script}")
    report_path = _resolve_from_project(request.report_path, root)
    report = _load_or_create_report(request, root, report_path)
    if report.get("status") == "complete":
        return report
    if report.get("status") == "invalid":
        raise TournamentRunError(
            "this tournament report is invalid and cannot be retried; preserve it and start a new report path"
        )
    games = cast(JsonObject, report["games"])
    manifest = cast(JsonObject, report["manifest"])
    run_id = str(manifest["run_id"])
    report["status"] = "running"
    report["last_error"] = None
    _atomic_write_json(report_path, report)
    for game_value in cast(list[JsonObject], report["schedule"]):
        game_id = str(game_value["game_id"])
        game_record = cast(JsonObject, games[game_id])
        if game_record.get("status") == "accepted":
            continue
        attempts = cast(list[JsonObject], game_record["attempts"])
        for prior_attempt in attempts:
            if prior_attempt.get("status") == "running":
                process_group_id = prior_attempt.get("child_process_group_id")
                if not isinstance(process_group_id, int) or isinstance(
                    process_group_id, bool
                ):
                    message = f"cannot safely resume {game_id}: prior running attempt has no recorded child process group"
                    game_record["status"] = "invalid"
                    _mark_interrupted(report, report_path, request, message)
                    raise TournamentRunError(message)
                prior_label = str(prior_attempt.get("artifact_label", ""))
                if not prior_label:
                    message = f"cannot safely resume {game_id}: prior running attempt has no unique artifact label"
                    game_record["status"] = "invalid"
                    _mark_interrupted(report, report_path, request, message)
                    raise TournamentRunError(message)
                group_status = _recorded_process_group_status(
                    process_group_id, prior_label
                )
                if group_status == "unknown":
                    message = f"cannot safely resume {game_id}: process group {process_group_id} is live but does not match the recorded child command"
                    game_record["status"] = "invalid"
                    _mark_interrupted(report, report_path, request, message)
                    raise TournamentRunError(message)
                cleanup = (
                    _terminate_recorded_process_group(process_group_id)
                    if group_status == "matching"
                    else "already-absent"
                )
                prior_attempt["status"] = "abandoned-on-resume"
                prior_attempt["ended_at_utc"] = _utc_now()
                prior_attempt["error"] = (
                    "orchestrator resumed before this attempt was accepted"
                )
                prior_attempt["resume_process_group_cleanup"] = cleanup
                game_record["status"] = "invalid"
                message = f"cannot resume {game_id}: the prior child was interrupted before a result was accepted, and replacement attempts are forbidden"
                _mark_interrupted(report, report_path, request, message)
                raise TournamentRunError(message)
        if attempts:
            message = f"cannot resume {game_id}: a prior nonaccepted attempt exists and replacement attempts are forbidden"
            game_record["status"] = "invalid"
            _mark_interrupted(report, report_path, request, message)
            raise TournamentRunError(message)
        attempt_number = len(attempts) + 1
        label = _artifact_label(run_id, game_value, attempt_number)
        summary_path = _expected_child_summary_path(root, game_value, label)
        if summary_path.parent.exists() and any(summary_path.parent.iterdir()):
            message = (
                f"new artifact label unexpectedly already exists: {summary_path.parent}"
            )
            _mark_interrupted(report, report_path, request, message)
            raise TournamentRunError(message)
        command = _child_command(root, request, game_value, label)
        attempt: JsonObject = {
            "attempt": attempt_number,
            "artifact_label": label,
            "status": "running",
            "started_at_utc": _utc_now(),
            "orchestrator_pid": os.getpid(),
            "command": command,
            "expected_summary_path": _display_path(summary_path, root),
        }
        attempts.append(attempt)
        game_record["status"] = "running"
        _atomic_write_json(report_path, report)
        try:
            if runner is None:
                completed = _default_process_runner(
                    command,
                    root,
                    request.child_wall_timeout_seconds,
                    on_start=partial(
                        _record_child_process_group,
                        attempt=attempt,
                        report_path=report_path,
                        report=report,
                    ),
                )
            else:
                completed = runner(command, root)
        except BaseException as error:
            attempt["status"] = "launch-failed"
            attempt["ended_at_utc"] = _utc_now()
            attempt["error"] = f"{type(error).__name__}: {error}"
            if isinstance(error, ChildProcessTimeout):
                attempt["status"] = "child-wall-timeout"
                attempt["stdout"] = _stream_identity(error.stdout)
                attempt["stderr"] = _stream_identity(error.stderr)
            game_record["status"] = "invalid"
            message = f"child launch failed for {game_id}: {attempt['error']}"
            _mark_interrupted(report, report_path, request, message)
            raise
        attempt["returncode"] = completed.returncode
        attempt["stdout"] = _stream_identity(completed.stdout or "")
        attempt["stderr"] = _stream_identity(completed.stderr or "")
        attempt["ended_at_utc"] = _utc_now()
        summary: JsonObject | None = None
        load_error: str | None = None
        try:
            summary = _load_json_object(summary_path)
            attempt["child_summary_file"] = _file_identity(summary_path, root)
            attempt["child_summary"] = summary
            replay_path, child_replay_record = _child_replay(
                summary, summary_path, root
            )
            replay_file = _file_identity(replay_path, root)
            attempt["saved_replay_file"] = replay_file
            child_replay_files = _child_replay_files(summary, summary_path, root)
            attempt["child_replay_files"] = child_replay_files
            trace_path, child_trace_record = _child_trace(summary, summary_path, root)
            controller_trace_file = dict(child_trace_record)
            attempt["controller_trace_file"] = controller_trace_file
            child_reproducibility = _audit_child_reproducibility(
                summary, game_value, manifest
            )
            attempt["child_reproducibility"] = child_reproducibility
            expected_replay_costumes = cast(
                Mapping[str, Any], game_value["expected_replay_costumes"]
            )
            replay_audit = audit_replay(
                replay_path,
                str(game_value["stage"]),
                {1: str(game_value["character"]), 2: str(game_value["character"])},
                {
                    1: int(expected_replay_costumes["p1"]),
                    2: int(expected_replay_costumes["p2"]),
                },
            )
            if not isinstance(replay_audit, dict):
                raise TypeError("replay auditor did not return a JSON object")
            attempt["replay_audit"] = replay_audit
            controller_boundary_audit = audit_controller_boundary(
                trace_path, replay_path, root
            )
            if not isinstance(controller_boundary_audit, dict):
                raise TypeError(
                    "controller boundary auditor did not return a JSON object"
                )
            attempt["controller_boundary_audit"] = controller_boundary_audit
            auxiliary_replay_inspections = _inspect_auxiliary_replays(
                summary,
                child_replay_files,
                root,
                expected_stage=str(game_value["stage"]),
                expected_characters={
                    1: str(game_value["character"]),
                    2: str(game_value["character"]),
                },
                expected_costumes={
                    1: int(expected_replay_costumes["p1"]),
                    2: int(expected_replay_costumes["p2"]),
                },
            )
            attempt["auxiliary_replay_inspections"] = auxiliary_replay_inspections
            attempt["audit"] = _audit_child_summary(
                summary,
                game_value,
                replay_audit,
                replay_file,
                child_replay_record,
                controller_trace_file,
                child_trace_record,
                controller_boundary_audit,
                auxiliary_replay_inspections,
                child_reproducibility,
                cast(Mapping[str, Any], manifest["game_image"]),
                cast(Mapping[str, Any], manifest["emulator_application"]),
            )
        except Exception as error:
            load_error = f"{type(error).__name__}: {error}"
            attempt["summary_load_error"] = load_error
        audit = attempt.get("audit")
        audit_passed = isinstance(audit, Mapping) and audit.get("decision") == "pass"
        if completed.returncode != 0 or summary is None or (not audit_passed):
            if completed.returncode != 0:
                reason = f"child process exited with status {completed.returncode}"
                attempt["status"] = "child-process-failed"
            elif summary is None or not isinstance(audit, Mapping):
                reason = f"child summary or replay audit could not be completed: {load_error}"
                attempt["status"] = "summary-or-replay-audit-invalid"
            else:
                failures = audit.get("failures")
                reason = f"child summary failed tournament audit: {failures}"
                attempt["status"] = "summary-rejected"
            attempt["error"] = reason
            game_record["status"] = "invalid"
            _mark_interrupted(report, report_path, request, f"{game_id}: {reason}")
            raise TournamentRunError(f"{game_id}: {reason}")
        audited_outcome = cast(
            Mapping[str, Any], cast(Mapping[str, Any], audit)["outcome"]
        )
        child_reproducibility = cast(JsonObject, attempt["child_reproducibility"])
        baselines = cast(JsonObject, manifest["accepted_child_reproducibility"])
        family = str(child_reproducibility["family"])
        baselines.setdefault(
            family,
            {
                "sha256": child_reproducibility["sha256"],
                "file_identities": child_reproducibility["file_identities"],
                "runtime_environment": child_reproducibility["runtime_environment"],
                "first_accepted_game_id": game_id,
                "first_accepted_attempt": attempt_number,
            },
        )
        attempt["status"] = "accepted"
        game_record["status"] = "accepted"
        game_record["accepted_attempt"] = attempt_number
        game_record["outcome"] = dict(audited_outcome)
        attempt["accepted_record_sha256"] = _content_sha256_without_key(
            attempt, "accepted_record_sha256"
        )
        game_record["accepted_record_sha256"] = _content_sha256_without_key(
            game_record, "accepted_record_sha256"
        )
        report["aggregates"] = aggregate_results(
            request.entrants, report["schedule"], games
        )
        _atomic_write_json(report_path, report)
    all_accepted = all(
        (
            isinstance(game, Mapping) and game.get("status") == "accepted"
            for game in games.values()
        )
    )
    aggregates = aggregate_results(request.entrants, report["schedule"], games)
    matched_blocks = cast(
        Mapping[str, Mapping[str, Any]], aggregates["matched_port_blocks"]
    )
    all_blocks_complete = bool(matched_blocks) and all(
        (block.get("complete") is True for block in matched_blocks.values())
    )
    no_replacement_attempts = all(
        (
            isinstance(game, Mapping)
            and isinstance(game.get("attempts"), Sequence)
            and (len(cast(Sequence[Any], game["attempts"])) == 1)
            for game in games.values()
        )
    )
    report["aggregates"] = aggregates
    report["status"] = (
        "complete"
        if all_accepted and all_blocks_complete and no_replacement_attempts
        else "invalid"
    )
    report["completed_at_utc"] = _utc_now() if report["status"] == "complete" else None
    report["last_error"] = (
        None if report["status"] == "complete" else "schedule is incomplete"
    )
    report["gate"] = _completion_gate(
        request,
        all_accepted=all_accepted,
        all_blocks_complete=all_blocks_complete,
        no_replacement_attempts=no_replacement_attempts,
    )
    _atomic_write_json(report_path, report)
    if report["status"] != "complete":
        raise TournamentRunError("tournament schedule did not complete")
    return report


def run_tournament(
    request: TournamentRequest,
    *,
    project_root: Path | None = None,
    process_runner: ProcessRunner | None = None,
    replay_auditor: ReplayAuditor | None = None,
    controller_boundary_auditor: ControllerBoundaryAuditor | None = None,
) -> JsonObject:
    """Run or inspect one report while excluding concurrent orchestrators."""
    request.validate()
    root = (
        Path(__file__).resolve().parents[3]
        if project_root is None
        else project_root.expanduser().resolve()
    )
    report_path = _resolve_from_project(request.report_path, root)
    with _exclusive_report_lock(report_path):
        return _run_tournament_unlocked(
            request,
            project_root=root,
            process_runner=process_runner,
            replay_auditor=replay_auditor,
            controller_boundary_auditor=controller_boundary_auditor,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/integration.toml"),
        help="one-game integration configuration",
    )
    parser.add_argument("--iso-path", type=Path)
    parser.add_argument(
        "--entrant",
        action="append",
        choices=("mimic", "slippi-ai", "slippi"),
        help="repeat to override the default MIMIC and Slippi-AI field",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        help="repeat for base policy/evaluation sampling seeds; paired effective seeds are derived from these values and do not control Dolphin or Melee RNG",
    )
    parser.add_argument(
        "--stage",
        action="append",
        choices=LEGAL_STAGES,
        help="repeat for multiple legal stages",
    )
    parser.add_argument("--games-per-block", type=int, default=2)
    parser.add_argument("--order-seed", type=int, default=0)
    parser.add_argument("--character", default="FOX")
    parser.add_argument("--max-game-frames", type=int, default=DEFAULT_MAX_GAME_FRAMES)
    parser.add_argument(
        "--child-timeout-seconds",
        type=float,
        default=DEFAULT_CHILD_WALL_TIMEOUT_SECONDS,
        help="wall-clock watchdog per game, including policy inference and Dolphin shutdown",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/integration/tournament/report.json"),
        help="atomic resumable JSON report",
    )
    return parser


def main() -> None:
    arguments = _build_parser().parse_args()
    request = TournamentRequest(
        entrants=DEFAULT_ENTRANTS
        if arguments.entrant is None
        else tuple(arguments.entrant),
        seeds=(42,) if arguments.seed is None else tuple(arguments.seed),
        stages=("BATTLEFIELD",) if arguments.stage is None else tuple(arguments.stage),
        games_per_block=arguments.games_per_block,
        order_seed=arguments.order_seed,
        character=arguments.character,
        max_game_frames=arguments.max_game_frames,
        child_wall_timeout_seconds=arguments.child_timeout_seconds,
        config_path=arguments.config,
        iso_path=arguments.iso_path,
        report_path=arguments.report,
    )
    report = run_tournament(request)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
