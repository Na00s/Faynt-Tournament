"""Faithful runtime boundary for supported Frisson Transformer policies.

This policy shares the pinned Slippi-AI game and ``custom_v1`` controller
representations, but it is a distinct PyTorch Transformer trained by Frisson.
The boundary deliberately uses the checkpoint's own model configuration and
actor contract.  State at frame ``t`` is sampled into the command for frame
``t + 1`` with no Slippi-AI 21-frame policy FIFO.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, Self, cast

from melee_policy.integration.faynt_d0_checkpoints import (
    CHECKPOINTS as FAYNT_D0_CHECKPOINTS,
)
from melee_policy.integration.faynt_d0_checkpoints import (
    FORMAT as FAYNT_D0_CHECKPOINT_FORMAT,
)
from melee_policy.integration.post_rl_checkpoints import (
    CHECKPOINTS as POST_RL_CHECKPOINTS,
)
from melee_policy.integration.post_rl_checkpoints import (
    FORMAT as FRISSON_POST_RL_CHECKPOINT_FORMAT,
)
from melee_policy.integration.slippi_ai_policy import (
    DIGITAL_BUTTON_ORDER,
    CanonicalControllerCommand,
)

FRISSON_POLICY_IDENTITY = "frisson-ai/melee-rl-transformer-custom-v1"
FRISSON_CHECKPOINT_FORMAT = "melee_rl.rl_checkpoint.v1"
FRISSON_BC_CHECKPOINT_FORMAT = "melee_policy.bc_checkpoint.v1"
FRISSON_FAMILY_CHECKPOINT_FORMAT = "melee_policy.final_pretraining_checkpoint.v1"
FRISSON_POSTTRAINING_CHECKPOINT_FORMAT = (
    "melee_policy.posttraining_curriculum_checkpoint.v1"
)
FRISSON_CHECKPOINT_FORMATS = frozenset(
    (
        FRISSON_CHECKPOINT_FORMAT,
        FRISSON_BC_CHECKPOINT_FORMAT,
        FRISSON_FAMILY_CHECKPOINT_FORMAT,
        FRISSON_POSTTRAINING_CHECKPOINT_FORMAT,
        FRISSON_POST_RL_CHECKPOINT_FORMAT,
        FAYNT_D0_CHECKPOINT_FORMAT,
    )
)
FRISSON_CONFIG_FORMAT = "melee_rl.rl_config.v1"
FRISSON_FAMILY_RUNTIME_SCHEMA = "melee-policy.final-pretraining-runtime.v1"
FRISSON_FAMILY_RECOVERY_IDENTITY_SCHEMA = (
    "melee-policy.final-pretraining-recovery-operational-identity.v1"
)
FRISSON_SOURCE_REVISION = "268031e7bddebb4e8c7a40026cd0b95f824d9d0d"
FRISSON_RUNTIME_SOURCE_REVISION = "2c535c04693dc9a232fc81f244cbd24cb2359771"
SLIPPI_AI_SOURCE_REVISION = "577965a7731dc53e3472ea63d9e9853a4e9d65fa"

MODEL_SOURCE_SHA256 = "2e61e7b27b1412c4becbd2dd93ee8d8bebad15c14cb7a2babaf954993205854d"
CONTROLLER_CODEC_SOURCE_SHA256 = (
    "4a51a6eb2bd1da48a1a7c852194d914ab6e5072b7d353096f351e89b24252286"
)
TENSOR_BATCH_SOURCE_SHA256 = (
    "07c39e9e5d5152b9d8d73aa74edf77c690db53f45b6ec03c6d4e64c80f4814ed"
)
PARSER_SOURCE_SHA256 = (
    "37024389c321fa9bcfa1f1c3e274ec88164190eaaa89c53437b43860a8c3ff5d"
)

EXPECTED_STATE_TENSOR_COUNT = 125
EXPECTED_PARAMETER_COUNT = 20_040_877
EXPECTED_CODEC: dict[str, Any] = {
    "name": "custom_v1",
    "vocab_sizes": {"buttons": 728, "main_stick": 85},
}
EXPECTED_ACTION_OFFSET_FRAMES = 1
EXPECTED_ACTOR_DELAY_FRAMES = 0
EXPECTED_ACTOR_CONTEXT_MODE = "ring"
EXPECTED_ACTOR_CONTEXT_FRAMES = 128
EXPECTED_MODEL_CONTEXT_LENGTH = 256
EXPECTED_BATCH_STEPS = 1
EXPECTED_SAMPLE_TEMPERATURE = 1.0
EXPECTED_ACTOR_SEED = 0

_FAMILY_PROFILE_CONTRACTS: dict[str, dict[str, Any]] = {
    "10m": {
        "study_id": "family-pretrain-10m-hp-v2",
        "state_tensor_count": 109,
        "model": {
            "profile": "10m",
            "d_model": 384,
            "n_layers": 5,
            "n_heads": 6,
            "n_kv_heads": 2,
            "head_dim": 64,
            "d_ff": 768,
            "encoder_parameters": 1_789_824,
            "backbone_parameters": 7_945_216,
            "controller_head_parameters": 428_589,
            "total_trainable_parameters": 10_163_629,
            "muon_matrix_count": 35,
            "muon_parameters": 6_389_760,
            "auxiliary_parameters": 3_773_869,
            "gradient_checkpointing": False,
            "prevalidated_inputs": True,
        },
    },
    "75m": {
        "study_id": "family-pretrain-75m-hp-v2",
        "state_tensor_count": 205,
        "model": {
            "profile": "75m",
            "d_model": 768,
            "n_layers": 11,
            "n_heads": 12,
            "n_kv_heads": 3,
            "head_dim": 64,
            "d_ff": 1_920,
            "encoder_parameters": 1_789_824,
            "backbone_parameters": 73_038_144,
            "controller_head_parameters": 477_741,
            "total_trainable_parameters": 75_305_709,
            "muon_matrix_count": 77,
            "muon_parameters": 64_880_640,
            "auxiliary_parameters": 10_425_069,
            "gradient_checkpointing": False,
            "prevalidated_inputs": True,
        },
    },
}

_POSTTRAINING_DATASET_CONTRACT: dict[str, Any] = {
    "train_mds_path": "full/mds/train",
    "validation_mds_path": "full/mds/validation",
    "dataset_manifest_path": "full/manifests/dataset_manifest.json",
    "split_manifest_hash": "9c4ae826ae9789c1a1b294dac00db9847a174273b085c03355b205c9953de98c",
    "parse_manifest_id": "9340436a32b8a093dbaa3b0243f120d299a0c0e61a5013a8b421414076d68268",
    "dataset_manifest_id": "e018460bb8b2d5398b9005217a8cc0ba1ff0f93fb3fe8fb26b7e5f18e03d4ee3",
    "repository": "erickfm/melee-ranked-replays",
    "dataset_revision": "ef54ddc230e01373ff2a179b49a7e559af788081",
    "metadata_revision": "694d40f6b13dc79c477d754ef6097ce9f72eac00",
    "slippi_ai_commit": SLIPPI_AI_SOURCE_REVISION,
    "replay_schema_version": f"slippi-ai-{SLIPPI_AI_SOURCE_REVISION[:7]}.replay-mds.v1",
}

_TRUSTED_POSTTRAINING_WINNERS: dict[str, dict[str, Any]] = {
    "10m": {
        "checkpoint_sha256": "63b5ff05ef30476c4f41590c478f4a7218f3b72a8b5ef24a0e336eeb2b7c287b",
        "checkpoint_bytes": 96_452_075,
        "launch_id": "cur-6-kd",
        "trial_id": "10m-muon-low",
        "track_identity_sha256": "a288fc4794cd25a42623825cba9ad7300c9258855d4ee16505a3272e0ae499e4",
        "trial_identity_sha256": "cd4c4a9034ece6c445c13e666728fb2e89297b8202df62a6443cdf409b97344d",
        "wandb_run_id": "mp-01fa6731541b0fd42623",
        "optimizer_steps": 195_248,
        "processed_target_frames": 12_795_772_928,
        "microbatches": 195_248,
        "start_frames": 11_895_832_576,
        "hold_start_frames": 11_913_789_440,
        "decay_start_frames": 12_615_811_072,
        "start_multiplier": 0.05,
        "start_muon_learning_rate": 0.000562886624941572,
        "end_muon_learning_rate": 0.00000562886624941572,
        "source_checkpoint_path": (
            "/mnt/melee-policy-checkpoints/curriculum-v1/cur-4-kd/10m/B-mix10/steps/step-181516.pt"
        ),
        "source_checkpoint_sha256": (
            "f5c35e51beadadb41a297d77a882b5a8957a34e4c8cd4ddba17cd51170442f31"
        ),
        "latest_path": "/mnt/melee-policy-checkpoints/curriculum-v1/cur-6-kd/10m/B-mix10/latest.pt",
        "immutable_step_path": (
            "/mnt/melee-policy-checkpoints/curriculum-v1/cur-6-kd/10m/B-mix10/steps/step-195248.pt"
        ),
        "distillation": {
            "teacher_checkpoint": "curriculum-v1/cur-3/75m/B-mix10/steps/step-127214.pt",
            "teacher_profile": "75m",
            "teacher_optimizer_steps": 127_214,
            "alpha": 0.5,
            "temperature": 1.0,
            "compiled_teacher": True,
        },
    },
    "75m": {
        "checkpoint_sha256": "8211f1832198646e9f4e3bacde26f181614f93320e68bd326dd5942c4e0f4077",
        "checkpoint_bytes": 644_291_987,
        "launch_id": "cur-3",
        "trial_id": "75m-muon-low",
        "track_identity_sha256": "9415792c37482e18c2d83028e78bfedabdac5c71c349bef8447be700b166f951",
        "trial_identity_sha256": "b759246874c284982c1d5cf2134bd06c7f608d9af1d0b29b6604a44d8a6fa5e9",
        "wandb_run_id": "mp-ce9d4e1e59ea7f66ee1a",
        "optimizer_steps": 127_214,
        "processed_target_frames": 8_337_096_704,
        "microbatches": 508_856,
        "start_frames": 6_837_108_736,
        "hold_start_frames": 6_867_058_688,
        "decay_start_frames": 8_037_138_432,
        "start_multiplier": 0.1,
        "start_muon_learning_rate": 0.0005,
        "end_muon_learning_rate": 0.000005,
        "source_checkpoint_path": (
            "/mnt/melee-policy-checkpoints/curriculum-v1/cur-1/75m/A-outcome-first/steps/step-104326.pt"
        ),
        "source_checkpoint_sha256": (
            "855deb6416f3d0333c55cde921aa5476e5b9a02873a6dea4e513d89bdf09c8eb"
        ),
        "latest_path": "/mnt/melee-policy-checkpoints/curriculum-v1/cur-3/75m/B-mix10/latest.pt",
        "immutable_step_path": (
            "/mnt/melee-policy-checkpoints/curriculum-v1/cur-3/75m/B-mix10/steps/step-127214.pt"
        ),
        "distillation": None,
    },
}

_TRUSTED_POSTTRAINING_SHA256 = frozenset(
    contract["checkpoint_sha256"] for contract in _TRUSTED_POSTTRAINING_WINNERS.values()
)

_IMPORT_LOCK = threading.RLock()
_MISSING = object()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except TypeError:
            pass
        except ValueError:
            pass
    return repr(value)


def _require_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key, _MISSING)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"checkpoint field {key!r} must be a mapping")
    return cast(Mapping[str, Any], value)


def _require_integer(
    parent: Mapping[str, Any],
    key: str,
    *,
    field_name: str | None = None,
) -> int:
    value = parent.get(key, _MISSING)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"checkpoint field {field_name or key!r} must be an integer")
    return cast(int, value)


def _require_nonempty_string(
    parent: Mapping[str, Any],
    key: str,
    *,
    field_name: str | None = None,
) -> str:
    value = parent.get(key, _MISSING)
    if not isinstance(value, str) or not value:
        raise RuntimeError(
            f"checkpoint field {field_name or key!r} must be a nonempty string"
        )
    return value


def _require_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError(f"checkpoint field {field_name!r} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise RuntimeError(
            f"checkpoint field {field_name!r} must be hexadecimal"
        ) from error
    return value


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_relative_checkpoint_path(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(
            f"checkpoint field {field_name!r} must be a nonempty relative path"
        )
    path = Path(value)
    if path.is_absolute() or any(
        component in ("", ".", "..") for component in path.parts
    ):
        raise RuntimeError(
            f"checkpoint field {field_name!r} must be a traversal-free relative path"
        )
    return value


def _require_identifier(
    value: Any, *, prefix: str, hexadecimal_length: int, field_name: str
) -> str:
    if not isinstance(value, str) or not value.startswith(prefix):
        raise RuntimeError(f"checkpoint field {field_name!r} has an invalid identifier")
    suffix = value.removeprefix(prefix)
    if len(suffix) != hexadecimal_length or any(
        character not in "0123456789abcdef" for character in suffix
    ):
        raise RuntimeError(f"checkpoint field {field_name!r} has an invalid identifier")
    return value


@dataclass(frozen=True, slots=True)
class FrissonPolicyConfig:
    """Frozen launch contract for one compatible Frisson checkpoint."""

    model_source_directory: Path
    slippi_ai_source_directory: Path
    checkpoint_path: Path
    port: int
    opponent_port: int
    sample_temperature: float = EXPECTED_SAMPLE_TEMPERATURE
    evaluation_seed: int = EXPECTED_ACTOR_SEED
    device: str = "cpu"
    deterministic_algorithms: bool = True
    fast_step: bool = True
    context_mode: str = EXPECTED_ACTOR_CONTEXT_MODE
    actor_context_frames: int = EXPECTED_ACTOR_CONTEXT_FRAMES
    model_context_length: int = EXPECTED_MODEL_CONTEXT_LENGTH
    delay_frames: int = EXPECTED_ACTOR_DELAY_FRAMES
    batch_steps: int = EXPECTED_BATCH_STEPS
    model_source_repository: Path | None = None
    model_source_revision: str | None = None
    model_source_subdirectory: str | None = None

    @classmethod
    def from_project_root(
        cls, project_root: Path, *, port: int, opponent_port: int
    ) -> Self:
        root = project_root.expanduser().resolve()
        return cls(
            model_source_directory=root / "sources" / "faynt" / FRISSON_SOURCE_REVISION,
            slippi_ai_source_directory=root / ".e001-cache" / "slippi-ai-source",
            checkpoint_path=root / ".e010-cache" / "selfplay-20m-bc" / "best.pt",
            port=port,
            opponent_port=opponent_port,
        )

    def validate(self) -> None:
        if self.port not in (1, 2, 3, 4):
            raise ValueError(f"policy port must be in 1..4, got {self.port}")
        if self.opponent_port not in (1, 2, 3, 4):
            raise ValueError(f"opponent port must be in 1..4, got {self.opponent_port}")
        if self.port == self.opponent_port:
            raise ValueError("policy port and opponent port must differ")
        required = {
            "sample_temperature": (
                self.sample_temperature,
                EXPECTED_SAMPLE_TEMPERATURE,
            ),
            "device": (self.device, "cpu"),
            "deterministic_algorithms": (self.deterministic_algorithms, True),
            "fast_step": (self.fast_step, True),
            "context_mode": (self.context_mode, EXPECTED_ACTOR_CONTEXT_MODE),
            "actor_context_frames": (
                self.actor_context_frames,
                EXPECTED_ACTOR_CONTEXT_FRAMES,
            ),
            "model_context_length": (
                self.model_context_length,
                EXPECTED_MODEL_CONTEXT_LENGTH,
            ),
            "delay_frames": (self.delay_frames, EXPECTED_ACTOR_DELAY_FRAMES),
            "batch_steps": (self.batch_steps, EXPECTED_BATCH_STEPS),
        }
        mismatches = {
            name: {"observed": observed, "required": expected}
            for name, (observed, expected) in required.items()
            if observed != expected
        }
        if mismatches:
            raise ValueError(f"Frisson actor contract mismatch: {mismatches}")
        if isinstance(self.evaluation_seed, bool) or not isinstance(
            self.evaluation_seed, int
        ):
            raise ValueError("evaluation_seed must be an integer")
        if not 0 <= self.evaluation_seed < 2**32:
            raise ValueError(
                "evaluation_seed must be between 0 and 4294967295 inclusive"
            )
        if not math.isfinite(self.sample_temperature):
            raise ValueError("sample_temperature must be finite")
        provenance = (
            self.model_source_repository,
            self.model_source_revision,
            self.model_source_subdirectory,
        )
        if any(value is not None for value in provenance) and not all(
            value is not None for value in provenance
        ):
            raise ValueError(
                "model source repository, revision, and subdirectory must be provided together"
            )
        if self.model_source_revision is not None:
            revision = self.model_source_revision
            if len(revision) != 40 or any(
                character not in "0123456789abcdef" for character in revision
            ):
                raise ValueError(
                    "model source revision must be a lowercase 40-character Git commit"
                )
        if self.model_source_subdirectory is not None:
            components = Path(self.model_source_subdirectory).parts
            if (
                Path(self.model_source_subdirectory).is_absolute()
                or not components
                or any(component in ("", ".", "..") for component in components)
            ):
                raise ValueError(
                    "model source subdirectory must be a relative path without traversal"
                )


def _pinned_model_source_files() -> dict[str, tuple[str, str]]:
    return {
        "model": ("model.py", MODEL_SOURCE_SHA256),
        "controller_codec": ("controller_codec.py", CONTROLLER_CODEC_SOURCE_SHA256),
        "tensor_batch": ("tensor_batch.py", TENSOR_BATCH_SOURCE_SHA256),
    }


def _git_blob(repository: Path, revision: str, repository_path: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repository), "show", f"{revision}:{repository_path}"],
        check=True,
        capture_output=True,
    ).stdout


def _bundled_source_manifest(
    repository: Path, revision: str, source_subdirectory: str
) -> dict[str, Any] | None:
    """Authenticate a release bundle against the original pinned source hashes."""
    directory = repository.expanduser().resolve() / source_subdirectory
    path = directory / "source-manifest.json"
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    expected_files = {
        filename: digest for filename, digest in _pinned_model_source_files().values()
    }
    if (
        not isinstance(record, dict)
        or record.get("schema_version") != "melee_policy.bundled_model_source.v1"
        or record.get("source_revision") != revision
        or revision != FRISSON_SOURCE_REVISION
        or record.get("runtime_source_revision") != FRISSON_RUNTIME_SOURCE_REVISION
        or record.get("files") != expected_files
    ):
        raise RuntimeError(
            "bundled Faynt source provenance differs from the pinned contract"
        )
    for filename, digest in expected_files.items():
        candidate = directory / filename
        if not candidate.is_file() or _sha256_file(candidate) != digest:
            raise RuntimeError(f"bundled Faynt source hash mismatch: {filename}")
    return record


def _verified_pinned_model_blobs(
    repository: Path,
    revision: str,
    source_subdirectory: str,
) -> dict[str, bytes]:
    source_repository = repository.expanduser().resolve()
    if not source_repository.is_dir():
        raise FileNotFoundError(
            f"Frisson source repository is missing: {source_repository}"
        )
    bundle = _bundled_source_manifest(source_repository, revision, source_subdirectory)
    if bundle is not None:
        return {
            filename: (source_repository / source_subdirectory / filename).read_bytes()
            for filename, _ in _pinned_model_source_files().values()
        }
    resolved_revision = _git_output(
        source_repository,
        "rev-parse",
        "--verify",
        f"{revision}^{{commit}}",
    )
    if resolved_revision != revision:
        raise RuntimeError(
            f"Frisson source revision mismatch: {resolved_revision} != {revision}"
        )

    blobs: dict[str, bytes] = {}
    for name, (filename, expected_sha256) in _pinned_model_source_files().items():
        repository_path = f"{source_subdirectory.rstrip('/')}/{filename}"
        content = _git_blob(source_repository, revision, repository_path)
        observed_sha256 = hashlib.sha256(content).hexdigest()
        if observed_sha256 != expected_sha256:
            raise RuntimeError(
                f"Frisson {name} Git blob hash mismatch: {observed_sha256} != {expected_sha256}"
            )
        blobs[filename] = content
    return blobs


def _verify_materialized_model_source(
    directory: Path, blobs: Mapping[str, bytes]
) -> None:
    if not directory.is_dir():
        raise FileNotFoundError(
            f"pinned Frisson source snapshot is missing: {directory}"
        )
    mismatches: dict[str, dict[str, str | None]] = {}
    for filename, expected_content in blobs.items():
        path = directory / filename
        expected_sha256 = hashlib.sha256(expected_content).hexdigest()
        observed_sha256 = _sha256_file(path) if path.is_file() else None
        if observed_sha256 != expected_sha256:
            mismatches[filename] = {
                "observed_sha256": observed_sha256,
                "expected_sha256": expected_sha256,
            }
    if mismatches:
        raise RuntimeError(
            f"pinned Frisson source snapshot identity mismatch: {mismatches}"
        )


def materialize_pinned_model_source(
    *,
    repository: Path,
    revision: str,
    source_subdirectory: str,
    destination: Path,
) -> Path:
    """Materialize immutable files from a verified Git commit or release bundle."""

    target = destination.expanduser().resolve()
    blobs = _verified_pinned_model_blobs(repository, revision, source_subdirectory)
    if target.exists():
        _verify_materialized_model_source(target, blobs)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        for filename, content in blobs.items():
            path = temporary / filename
            path.write_bytes(content)
            path.chmod(0o444)
        try:
            os.rename(temporary, target)
        except FileExistsError:
            _verify_materialized_model_source(target, blobs)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    _verify_materialized_model_source(target, blobs)
    return target


def verify_source_assets(config: FrissonPolicyConfig) -> dict[str, Any]:
    """Verify the exact local model files and pinned parser before importing."""
    config.validate()
    model_source = config.model_source_directory.expanduser().resolve()
    slippi_source = config.slippi_ai_source_directory.expanduser().resolve()
    if not model_source.is_dir():
        raise FileNotFoundError(f"Frisson model source is missing: {model_source}")
    if not slippi_source.is_dir():
        raise FileNotFoundError(f"pinned Slippi-AI source is missing: {slippi_source}")

    model_files = {
        name: (model_source / filename, expected_sha256)
        for name, (filename, expected_sha256) in _pinned_model_source_files().items()
    }
    file_identity: dict[str, Any] = {}
    for name, (path, expected_sha256) in model_files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Frisson {name} source is missing: {path}")
        observed_sha256 = _sha256_file(path)
        if observed_sha256 != expected_sha256:
            raise RuntimeError(
                f"Frisson {name} source hash mismatch: {observed_sha256} != {expected_sha256}"
            )
        file_identity[name] = {"path": str(path), "sha256": observed_sha256}

    direct_bundle = _bundled_source_manifest(model_source, FRISSON_SOURCE_REVISION, ".")
    if config.model_source_repository is None and direct_bundle is not None:
        source_repository = model_source
        source_subdirectory = "."
        selected_revision = FRISSON_SOURCE_REVISION
        selection_method = "verified-bundled-source"
    elif config.model_source_repository is None:
        source_repository = model_source
        source_subdirectory = ""
        selected_revision = _git_output(model_source, "rev-parse", "HEAD")
        selected_changes = _git_output(
            model_source,
            "status",
            "--short",
            "--",
            "model.py",
            "controller_codec.py",
            "tensor_batch.py",
        )
        if selected_changes:
            raise RuntimeError(
                f"Frisson runtime source files are modified: {selected_changes}"
            )
        selection_method = "verified-working-tree"
    else:
        source_repository = config.model_source_repository.expanduser().resolve()
        source_subdirectory = cast(str, config.model_source_subdirectory)
        selected_revision = cast(str, config.model_source_revision)
        blobs = _verified_pinned_model_blobs(
            source_repository,
            selected_revision,
            source_subdirectory,
        )
        _verify_materialized_model_source(model_source, blobs)
        selection_method = "verified-git-object-snapshot"

    bundled_identity = (
        _bundled_source_manifest(
            source_repository, selected_revision, source_subdirectory
        )
        if config.model_source_repository is not None or direct_bundle is not None
        else None
    )
    if bundled_identity is not None:
        repository_head = None
        runtime_source_revision = bundled_identity["runtime_source_revision"]
        selection_method = "verified-bundled-source"
    else:
        repository_head = _git_output(source_repository, "rev-parse", "HEAD")
        source_paths = [
            f"{source_subdirectory.rstrip('/')}/{filename}"
            if source_subdirectory
            else filename
            for filename, _ in _pinned_model_source_files().values()
        ]
        runtime_source_revision = _git_output(
            source_repository,
            "log",
            "-1",
            "--format=%H",
            selected_revision,
            "--",
            *source_paths,
        )
    if runtime_source_revision != FRISSON_RUNTIME_SOURCE_REVISION:
        raise RuntimeError(
            "Frisson runtime source revision mismatch: "
            f"{runtime_source_revision} != {FRISSON_RUNTIME_SOURCE_REVISION}"
        )
    parser_path = slippi_source / "slippi_db" / "parse_libmelee.py"
    if not parser_path.is_file():
        raise FileNotFoundError(f"pinned Slippi-AI parser is missing: {parser_path}")
    parser_sha256 = _sha256_file(parser_path)
    if parser_sha256 != PARSER_SOURCE_SHA256:
        raise RuntimeError(
            f"Slippi-AI parser hash mismatch: {parser_sha256} != {PARSER_SOURCE_SHA256}"
        )
    slippi_revision = _git_output(slippi_source, "rev-parse", "HEAD")
    if slippi_revision != SLIPPI_AI_SOURCE_REVISION:
        raise RuntimeError(
            f"Slippi-AI revision mismatch: {slippi_revision} != {SLIPPI_AI_SOURCE_REVISION}"
        )
    slippi_changes = _git_output(
        slippi_source, "status", "--short", "--untracked-files=all"
    )
    if slippi_changes:
        raise RuntimeError(f"pinned Slippi-AI source is modified: {slippi_changes}")

    return {
        "frisson_source": {
            "directory": str(model_source),
            "producing_revision": FRISSON_SOURCE_REVISION,
            "selected_revision": selected_revision,
            "runtime_source_revision": runtime_source_revision,
            "repository_head": repository_head,
            "repository": str(source_repository),
            "repository_subdirectory": source_subdirectory,
            "selection_method": selection_method,
            "working_tree_independent": config.model_source_repository is not None or bundled_identity is not None,
            "runtime_files_clean": True,
            "whole_worktree_clean": (
                not bool(
                    _git_output(
                        model_source, "status", "--short", "--untracked-files=all"
                    )
                )
                if config.model_source_repository is None and bundled_identity is None
                else None
            ),
            "files": file_identity,
        },
        "slippi_ai_source": {
            "directory": str(slippi_source),
            "revision": slippi_revision,
            "tree_clean": True,
            "parser_path": str(parser_path),
            "parser_sha256": parser_sha256,
        },
    }


def _module_alias(prefix: str, path: Path) -> str:
    suffix = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
    return f"_melee_policy_{prefix}_{suffix}"


def _load_module_from_path(alias: str, path: Path) -> ModuleType:
    existing = sys.modules.get(alias)
    if existing is not None:
        existing_file = Path(cast(str, existing.__file__)).resolve()
        if existing_file != path.resolve():
            raise RuntimeError(
                f"module alias collision for {alias}: {existing_file} != {path}"
            )
        return existing
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot create an import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(alias, None)
        raise
    return module


def _activate_runtime_sources(config: FrissonPolicyConfig) -> dict[str, ModuleType]:
    model_source = config.model_source_directory.expanduser().resolve()
    slippi_source = config.slippi_ai_source_directory.expanduser().resolve()
    with _IMPORT_LOCK:
        codec_path = model_source / "controller_codec.py"
        codec = _load_module_from_path(
            _module_alias("frisson_codec", codec_path), codec_path
        )

        model_path = model_source / "model.py"
        model_alias = _module_alias("frisson_model", model_path)
        existing_model = sys.modules.get(model_alias)
        if existing_model is None:
            previous_codec = sys.modules.get("controller_codec", _MISSING)
            sys.modules["controller_codec"] = codec
            try:
                model = _load_module_from_path(model_alias, model_path)
            finally:
                if previous_codec is _MISSING:
                    sys.modules.pop("controller_codec", None)
                else:
                    sys.modules["controller_codec"] = cast(ModuleType, previous_codec)
        else:
            model = existing_model

        tensor_path = model_source / "tensor_batch.py"
        tensor_batch = _load_module_from_path(
            _module_alias("frisson_tensor_batch", tensor_path),
            tensor_path,
        )

        source_text = str(slippi_source)
        inserted = source_text not in sys.path
        if inserted:
            sys.path.insert(0, source_text)
        try:
            parser = importlib.import_module("slippi_db.parse_libmelee")
            types = importlib.import_module("slippi_ai.types")
        finally:
            if inserted and sys.path and sys.path[0] == source_text:
                sys.path.pop(0)

    for name, module, required_root in (
        ("model", model, model_source),
        ("controller codec", codec, model_source),
        ("tensor batch", tensor_batch, model_source),
        ("parser", parser, slippi_source),
        ("Slippi-AI types", types, slippi_source),
    ):
        module_file_value = getattr(module, "__file__", None)
        if not isinstance(module_file_value, str):
            raise RuntimeError(f"loaded {name} module has no concrete source path")
        module_file = Path(module_file_value).resolve()
        if not module_file.is_relative_to(required_root):
            raise RuntimeError(
                f"loaded {name} outside its verified source: {module_file}"
            )
    if getattr(codec, "SLIPPI_AI_COMMIT", None) != SLIPPI_AI_SOURCE_REVISION:
        raise RuntimeError(
            "Frisson controller codec is not pinned to the required Slippi-AI commit"
        )
    return {
        "model": model,
        "controller_codec": codec,
        "tensor_batch": tensor_batch,
        "parser": parser,
    }


def _validate_checkpoint_actor(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("delay_frames") != EXPECTED_ACTOR_DELAY_FRAMES:
        raise RuntimeError(
            "checkpoint delay_frames is not the deployed zero-delay contract"
        )
    if payload.get("context_mode") != EXPECTED_ACTOR_CONTEXT_MODE:
        raise RuntimeError("checkpoint context_mode is not the deployed ring contract")
    rl_config = _require_mapping(payload, "rl_config")
    if rl_config.get("format") != FRISSON_CONFIG_FORMAT:
        raise RuntimeError(
            f"checkpoint rl_config format is not {FRISSON_CONFIG_FORMAT!r}"
        )
    runtime = _require_mapping(rl_config, "runtime")
    policy = _require_mapping(rl_config, "policy")
    actor = _require_mapping(rl_config, "actor")
    required = {
        "runtime.seed": (runtime.get("seed"), EXPECTED_ACTOR_SEED),
        "runtime.deterministic": (runtime.get("deterministic"), True),
        "policy.profile": (policy.get("profile"), "20m"),
        "policy.seed": (policy.get("seed"), EXPECTED_ACTOR_SEED),
        "policy.compute_dtype": (policy.get("compute_dtype"), "float32"),
        "policy.cache_dtype": (policy.get("cache_dtype"), "float32"),
        "policy.fast_step": (policy.get("fast_step"), True),
        "actor.context_frames": (
            actor.get("context_frames"),
            EXPECTED_ACTOR_CONTEXT_FRAMES,
        ),
        "actor.delay_frames": (actor.get("delay_frames"), EXPECTED_ACTOR_DELAY_FRAMES),
        "actor.context_mode": (actor.get("context_mode"), EXPECTED_ACTOR_CONTEXT_MODE),
        "actor.batch_steps": (actor.get("batch_steps"), EXPECTED_BATCH_STEPS),
        "actor.temperature": (actor.get("temperature"), EXPECTED_SAMPLE_TEMPERATURE),
        "actor.seed": (actor.get("seed"), EXPECTED_ACTOR_SEED),
    }
    mismatches = {
        name: {"observed": observed, "required": expected}
        for name, (observed, expected) in required.items()
        if observed != expected
    }
    if mismatches:
        raise RuntimeError(f"checkpoint deployed actor contract mismatch: {mismatches}")
    return {
        "runtime": {
            "seed": runtime["seed"],
            "deterministic": runtime["deterministic"],
        },
        "policy": {
            "profile": policy["profile"],
            "compute_dtype": policy["compute_dtype"],
            "cache_dtype": policy["cache_dtype"],
            "fast_step": policy["fast_step"],
        },
        "actor": {
            "context_frames": actor["context_frames"],
            "delay_frames": actor["delay_frames"],
            "context_mode": actor["context_mode"],
            "batch_steps": actor["batch_steps"],
            "temperature": actor["temperature"],
            "seed": actor["seed"],
        },
    }


def _pinned_evaluation_actor_contract(profile: str = "20m") -> dict[str, Any]:
    """Return the explicit actor contract for checkpoints without one."""

    return {
        "source": "launcher-pinned-frisson-evaluation.v1",
        "checkpoint_embedded": False,
        "runtime": {
            "deterministic": True,
        },
        "policy": {
            "profile": profile,
            "compute_dtype": "float32",
            "cache_dtype": "float32",
            "fast_step": True,
        },
        "actor": {
            "context_frames": EXPECTED_ACTOR_CONTEXT_FRAMES,
            "delay_frames": EXPECTED_ACTOR_DELAY_FRAMES,
            "context_mode": EXPECTED_ACTOR_CONTEXT_MODE,
            "batch_steps": EXPECTED_BATCH_STEPS,
            "temperature": EXPECTED_SAMPLE_TEMPERATURE,
            "seed": None,
        },
    }


def _posttraining_checkpoint_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one of the two content-addressed curriculum-winner envelopes."""

    expected_top_level = {
        "format_version",
        "model",
        "optimizer_bundle",
        "training_state",
        "frame_cursor",
        "rng_state",
        "resolved_config",
        "metadata",
    }
    if set(payload) != expected_top_level:
        raise RuntimeError(
            "post-training checkpoint top-level fields changed: "
            f"observed={sorted(payload)!r}, required={sorted(expected_top_level)!r}"
        )
    if payload.get("format_version") != 1:
        raise RuntimeError("post-training checkpoint format_version must be 1")

    resolved = _require_mapping(payload, "resolved_config")
    metadata = _require_mapping(payload, "metadata")
    if resolved.get("schema_version") != FRISSON_FAMILY_RUNTIME_SCHEMA:
        raise RuntimeError("post-training checkpoint resolved_config schema mismatch")
    if set(metadata) != {"curriculum", "checkpoint_storage"}:
        raise RuntimeError(
            f"post-training checkpoint metadata fields changed: observed={sorted(metadata)!r}"
        )

    resolved_model = dict(_require_mapping(resolved, "model"))
    profile = resolved_model.get("profile")
    if not isinstance(profile, str) or profile not in _TRUSTED_POSTTRAINING_WINNERS:
        raise RuntimeError(f"unsupported post-training checkpoint profile: {profile!r}")
    winner = _TRUSTED_POSTTRAINING_WINNERS[profile]
    profile_contract = _FAMILY_PROFILE_CONTRACTS[profile]
    expected_model = cast(dict[str, Any], profile_contract["model"])
    if resolved_model != expected_model:
        mismatches = {
            name: {
                "observed": resolved_model.get(name, _MISSING),
                "required": expected_model.get(name, _MISSING),
            }
            for name in sorted(set(resolved_model) | set(expected_model))
            if resolved_model.get(name, _MISSING) != expected_model.get(name, _MISSING)
        }
        raise RuntimeError(
            f"post-training checkpoint model contract mismatch: {mismatches}"
        )

    study_id = _require_nonempty_string(
        resolved, "study_id", field_name="resolved_config.study_id"
    )
    trial_id = _require_nonempty_string(
        resolved, "trial_id", field_name="resolved_config.trial_id"
    )
    launch = _require_mapping(resolved, "launch")
    launch_id = _require_nonempty_string(
        launch, "launch_id", field_name="resolved_config.launch.launch_id"
    )
    header_required = {
        "study_id": profile_contract["study_id"],
        "trial_id": winner["trial_id"],
        "launch_id": winner["launch_id"],
    }
    header_observed = {
        "study_id": study_id,
        "trial_id": trial_id,
        "launch_id": launch_id,
    }
    if header_observed != header_required:
        raise RuntimeError(
            "post-training checkpoint profile and launch identity mismatch: "
            f"observed={header_observed!r}, required={header_required!r}"
        )

    dataset = dict(_require_mapping(resolved, "dataset"))
    if dataset != _POSTTRAINING_DATASET_CONTRACT:
        mismatches = {
            name: {
                "observed": dataset.get(name, _MISSING),
                "required": _POSTTRAINING_DATASET_CONTRACT.get(name, _MISSING),
            }
            for name in sorted(set(dataset) | set(_POSTTRAINING_DATASET_CONTRACT))
            if dataset.get(name, _MISSING)
            != _POSTTRAINING_DATASET_CONTRACT.get(name, _MISSING)
        }
        raise RuntimeError(
            f"post-training checkpoint dataset identity mismatch: {mismatches}"
        )

    identity = dict(_require_mapping(resolved, "identity"))
    trial_identity = cast(str, winner["trial_identity_sha256"])
    checkpoint_relative_path = (
        f"family-pretraining-v1/{profile}/{study_id}/"
        f"{_POSTTRAINING_DATASET_CONTRACT['dataset_manifest_id'][:12]}/"
        f"{trial_id}-{trial_identity[:12]}/latest.pt"
    )
    identity_required = {
        "track_identity_sha256": winner["track_identity_sha256"],
        "trial_identity_sha256": trial_identity,
        "checkpoint_relative_path": checkpoint_relative_path,
        "wandb_run_id": winner["wandb_run_id"],
    }
    if identity != identity_required:
        raise RuntimeError(
            "post-training checkpoint base-pretraining identity mismatch: "
            f"observed={identity!r}, required={identity_required!r}"
        )
    _require_sha256(
        identity["track_identity_sha256"],
        "resolved_config.identity.track_identity_sha256",
    )
    _require_sha256(
        identity["trial_identity_sha256"],
        "resolved_config.identity.trial_identity_sha256",
    )
    _require_relative_checkpoint_path(
        identity["checkpoint_relative_path"],
        "resolved_config.identity.checkpoint_relative_path",
    )

    training = _require_mapping(resolved, "training")
    training_required = {
        "global_target_frame_batch": 65_536,
        "context_length": EXPECTED_MODEL_CONTEXT_LENGTH,
        "action_offset_frames": EXPECTED_ACTION_OFFSET_FRAMES,
        "precision": "bfloat16",
    }
    training_mismatches = {
        name: {"observed": training.get(name, _MISSING), "required": required}
        for name, required in training_required.items()
        if training.get(name, _MISSING) != required
    }
    if training_mismatches:
        raise RuntimeError(
            f"post-training checkpoint training contract mismatch: {training_mismatches}"
        )

    validation = _require_mapping(resolved, "validation")
    validation_required = {
        "deterministic": True,
        "seed": 15_031,
        "windows_per_replay_per_perspective": 1,
        "window_selection": "first-gap-free-window-in-stable-replay-order",
        "window_selection_version": "replay-validation.v1",
        "shuffle": False,
        "promotion_metric": "validation/total_policy_nll",
        "lower_is_better": True,
        "split_scope": "entire-fixed-validation-window-manifest",
        "maximum_target_frames": None,
    }
    if dict(validation) != validation_required:
        raise RuntimeError(
            "post-training checkpoint validation dataset contract mismatch"
        )

    wandb = _require_mapping(resolved, "wandb")
    wandb_required = {
        "entity": "frisson-labs",
        "project": "Frisson-AI-Melee",
        "group": f"final-pretrain-{profile}-hp-v2",
        "resume_policy": "must",
        "persistent_run_id_per_trial": True,
    }
    if dict(wandb) != wandb_required:
        raise RuntimeError("post-training checkpoint base W&B identity mismatch")

    curriculum = dict(_require_mapping(resolved, "curriculum"))
    metadata_curriculum = dict(_require_mapping(metadata, "curriculum"))
    if curriculum != metadata_curriculum:
        raise RuntimeError(
            "post-training checkpoint curriculum metadata is not self-consistent"
        )
    curriculum_required = {
        "schema_version": "melee-policy.curriculum.v1",
        "arm": "B-mix10",
        "start_frames": winner["start_frames"],
        "horizon_frames": winner["processed_target_frames"],
        "lr_schedule": "wsd",
        "weight_decay_follows_lr": True,
        "start_multiplier": winner["start_multiplier"],
        "start_muon_learning_rate": winner["start_muon_learning_rate"],
        "end_muon_learning_rate": winner["end_muon_learning_rate"],
        "hold_fraction": 0.5,
        "end_fraction": 0.01,
        "hold_start_frames": winner["hold_start_frames"],
        "decay_start_frames": winner["decay_start_frames"],
        "source_checkpoint_path": winner["source_checkpoint_path"],
        "source_checkpoint_sha256": winner["source_checkpoint_sha256"],
        "chunk_order_size": 128,
    }
    curriculum_mismatches = {
        name: {"observed": curriculum.get(name, _MISSING), "required": required}
        for name, required in curriculum_required.items()
        if curriculum.get(name, _MISSING) != required
    }
    if curriculum_mismatches:
        raise RuntimeError(
            f"post-training checkpoint curriculum contract mismatch: {curriculum_mismatches}"
        )
    _require_sha256(
        curriculum["source_checkpoint_sha256"],
        "resolved_config.curriculum.source_checkpoint_sha256",
    )

    stages = curriculum.get("stages")
    if (
        not isinstance(stages, list)
        or len(stages) != 1
        or not isinstance(stages[0], Mapping)
    ):
        raise RuntimeError(
            "post-training checkpoint curriculum must contain one mapping stage"
        )
    stage = cast(Mapping[str, Any], stages[0])
    secondary = _require_mapping(stage, "secondary")
    secondary_ranks = secondary.get("ranks")
    if type(secondary_ranks) is not frozenset or secondary_ranks != frozenset(
        ("diamond",)
    ):
        raise RuntimeError(
            "post-training checkpoint secondary rank set must be frozen diamond"
        )
    secondary_required = {
        "name": "diamond-winners",
        "fraction": 1.0,
        "ranks": frozenset(("diamond",)),
        "winners_only": True,
        "loser_weight": 0.0,
        "natural_floor": 0.0,
        "secondary": None,
        "secondary_share": 0.0,
    }
    stage_required = {
        "name": "master-winners+10pct-diamond-winners",
        "fraction": 1.0,
        "ranks": ["master"],
        "winners_only": True,
        "loser_weight": 0.0,
        "natural_floor": 0.0,
        "secondary": secondary_required,
        "secondary_share": 0.1,
        "end_frames": winner["processed_target_frames"],
    }
    if dict(stage) != stage_required:
        raise RuntimeError(
            "post-training checkpoint weighted curriculum stage mismatch"
        )

    observed_distillation = curriculum.get("distillation", _MISSING)
    expected_distillation = winner["distillation"]
    if expected_distillation is None:
        if observed_distillation is not _MISSING:
            raise RuntimeError(
                "75M post-training winner unexpectedly contains distillation metadata"
            )
    elif observed_distillation != expected_distillation:
        raise RuntimeError("10M post-training winner distillation lineage mismatch")

    training_state = _require_mapping(payload, "training_state")
    processed_target_frames = _require_integer(
        training_state,
        "processed_target_frames",
        field_name="training_state.processed_target_frames",
    )
    frame_cursor = _require_integer(
        training_state,
        "frame_cursor",
        field_name="training_state.frame_cursor",
    )
    optimizer_steps = _require_integer(
        training_state,
        "optimizer_steps",
        field_name="training_state.optimizer_steps",
    )
    microbatches = _require_integer(
        training_state,
        "microbatches",
        field_name="training_state.microbatches",
    )
    counters_observed = {
        "processed_target_frames": processed_target_frames,
        "frame_cursor": frame_cursor,
        "optimizer_steps": optimizer_steps,
        "microbatches": microbatches,
    }
    counters_required = {
        "processed_target_frames": winner["processed_target_frames"],
        "frame_cursor": winner["processed_target_frames"],
        "optimizer_steps": winner["optimizer_steps"],
        "microbatches": winner["microbatches"],
    }
    if (
        counters_observed != counters_required
        or payload.get("frame_cursor") != frame_cursor
    ):
        raise RuntimeError(
            "post-training checkpoint training counters mismatch: "
            f"observed={counters_observed!r}, required={counters_required!r}"
        )

    storage = dict(_require_mapping(metadata, "checkpoint_storage"))
    storage_required = {
        "schema_version": "melee-policy.immutable-checkpoint.v1",
        "latest_path": winner["latest_path"],
        "immutable_step_path": winner["immutable_step_path"],
        "optimizer_steps": winner["optimizer_steps"],
        "processed_target_frames": winner["processed_target_frames"],
        "path_semantics": "immutable step with atomic latest symlink",
        "publication_mode": "copy-atomic-rename",
        "single_writer_required": True,
    }
    if storage != storage_required:
        raise RuntimeError(
            "post-training checkpoint immutable storage identity mismatch: "
            f"observed={storage!r}, required={storage_required!r}"
        )

    return {
        "format": FRISSON_POSTTRAINING_CHECKPOINT_FORMAT,
        "profile": profile,
        "step": optimizer_steps,
        "processed_target_frames": processed_target_frames,
        "study_id": study_id,
        "trial_id": trial_id,
        "launch_id": launch_id,
        "track_identity_sha256": identity["track_identity_sha256"],
        "trial_identity_sha256": identity["trial_identity_sha256"],
        "wandb_run_id": identity["wandb_run_id"],
        "checkpoint_relative_path": checkpoint_relative_path,
        "source_checkpoint_path": curriculum["source_checkpoint_path"],
        "source_checkpoint_sha256": curriculum["source_checkpoint_sha256"],
        "immutable_step_path": storage["immutable_step_path"],
        "dataset_identity": dataset,
        "curriculum": _json_safe(curriculum),
        "resolved_config_sha256": _canonical_sha256(resolved),
        "resolved_model": resolved_model,
        "resolved_training": dict(training),
        "metadata": _json_safe(metadata),
        "expected_checkpoint_sha256": winner["checkpoint_sha256"],
        "expected_checkpoint_bytes": winner["checkpoint_bytes"],
        "expected_state_tensor_count": profile_contract["state_tensor_count"],
        "expected_parameter_count": expected_model["total_trainable_parameters"],
        "expected_component_parameter_counts": {
            "encoder": expected_model["encoder_parameters"],
            "backbone": expected_model["backbone_parameters"],
            "controller_head": expected_model["controller_head_parameters"],
            "total": expected_model["total_trainable_parameters"],
        },
        "deployed_actor": _pinned_evaluation_actor_contract(profile),
    }


def _family_checkpoint_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the self-identifying final-pretraining checkpoint envelope."""

    expected_top_level = {
        "format_version",
        "model",
        "optimizer_bundle",
        "training_state",
        "frame_cursor",
        "rng_state",
        "resolved_config",
        "metadata",
    }
    if set(payload) != expected_top_level:
        raise RuntimeError(
            "family checkpoint top-level fields changed: "
            f"observed={sorted(payload)!r}, required={sorted(expected_top_level)!r}"
        )
    if payload.get("format_version") != 1:
        raise RuntimeError("family checkpoint format_version must be 1")

    resolved = _require_mapping(payload, "resolved_config")
    metadata = _require_mapping(payload, "metadata")
    if resolved.get("schema_version") != FRISSON_FAMILY_RUNTIME_SCHEMA:
        raise RuntimeError("family checkpoint resolved_config schema mismatch")
    if metadata.get("schema_version") != FRISSON_FAMILY_RUNTIME_SCHEMA:
        raise RuntimeError("family checkpoint metadata schema mismatch")

    resolved_model = dict(_require_mapping(resolved, "model"))
    profile = resolved_model.get("profile")
    if not isinstance(profile, str) or profile not in _FAMILY_PROFILE_CONTRACTS:
        raise RuntimeError(f"unsupported family checkpoint profile: {profile!r}")
    profile_contract = _FAMILY_PROFILE_CONTRACTS[profile]
    expected_model = cast(dict[str, Any], profile_contract["model"])
    if resolved_model != expected_model:
        mismatches = {
            name: {
                "observed": resolved_model.get(name, _MISSING),
                "required": expected_model.get(name, _MISSING),
            }
            for name in sorted(set(resolved_model) | set(expected_model))
            if resolved_model.get(name, _MISSING) != expected_model.get(name, _MISSING)
        }
        raise RuntimeError(f"family checkpoint model contract mismatch: {mismatches}")

    study_id = _require_nonempty_string(
        resolved, "study_id", field_name="resolved_config.study_id"
    )
    if study_id != profile_contract["study_id"]:
        raise RuntimeError(
            "family checkpoint study_id disagrees with its model profile"
        )
    trial_id = _require_nonempty_string(
        resolved, "trial_id", field_name="resolved_config.trial_id"
    )
    if not trial_id.startswith(f"{profile}-"):
        raise RuntimeError(
            "family checkpoint trial_id disagrees with its model profile"
        )

    training = _require_mapping(resolved, "training")
    required_training = {
        "training.context_length": (
            training.get("context_length"),
            EXPECTED_MODEL_CONTEXT_LENGTH,
        ),
        "training.action_offset_frames": (
            training.get("action_offset_frames"),
            EXPECTED_ACTION_OFFSET_FRAMES,
        ),
        "training.precision": (training.get("precision"), "bfloat16"),
    }
    training_mismatches = {
        name: {"observed": observed, "required": required}
        for name, (observed, required) in required_training.items()
        if observed != required
    }
    if training_mismatches:
        raise RuntimeError(
            f"family checkpoint training contract mismatch: {training_mismatches}"
        )

    dataset = _require_mapping(resolved, "dataset")
    representation = {
        "dataset.slippi_ai_commit": (
            dataset.get("slippi_ai_commit"),
            SLIPPI_AI_SOURCE_REVISION,
        ),
        "dataset.replay_schema_version": (
            dataset.get("replay_schema_version"),
            f"slippi-ai-{SLIPPI_AI_SOURCE_REVISION[:7]}.replay-mds.v1",
        ),
    }
    representation_mismatches = {
        name: {"observed": observed, "required": required}
        for name, (observed, required) in representation.items()
        if observed != required
    }
    if representation_mismatches:
        raise RuntimeError(
            f"family checkpoint Slippi representation mismatch: {representation_mismatches}"
        )

    identity = _require_mapping(resolved, "identity")
    track_identity = _require_sha256(
        identity.get("track_identity_sha256"),
        "resolved_config.identity.track_identity_sha256",
    )
    trial_identity = _require_sha256(
        identity.get("trial_identity_sha256"),
        "resolved_config.identity.trial_identity_sha256",
    )
    wandb_run_id = _require_identifier(
        identity.get("wandb_run_id"),
        prefix="mpr-",
        hexadecimal_length=20,
        field_name="resolved_config.identity.wandb_run_id",
    )
    launch = _require_mapping(resolved, "launch")
    launch_id = _require_identifier(
        launch.get("launch_id"),
        prefix="fpr-",
        hexadecimal_length=20,
        field_name="resolved_config.launch.launch_id",
    )
    checkpoint_relative_path = _require_relative_checkpoint_path(
        identity.get("checkpoint_relative_path"),
        "resolved_config.identity.checkpoint_relative_path",
    )
    expected_checkpoint_relative_path = (
        f"family-pretraining-recovery-v1/launches/{launch_id}/{profile}/"
        f"{trial_id}-{trial_identity[:12]}/latest.pt"
    )
    if checkpoint_relative_path != expected_checkpoint_relative_path:
        raise RuntimeError(
            "family checkpoint relative path disagrees with its embedded identity"
        )

    source_wandb_run_id = _require_identifier(
        identity.get("source_wandb_run_id"),
        prefix="mp-",
        hexadecimal_length=20,
        field_name="resolved_config.identity.source_wandb_run_id",
    )
    if source_wandb_run_id != f"mp-{trial_identity[:20]}":
        raise RuntimeError(
            "family checkpoint source W&B run ID disagrees with trial identity"
        )
    source_checkpoint_relative_path = _require_relative_checkpoint_path(
        identity.get("source_checkpoint_relative_path"),
        "resolved_config.identity.source_checkpoint_relative_path",
    )
    dataset_manifest_id = _require_sha256(
        dataset.get("dataset_manifest_id"),
        "resolved_config.dataset.dataset_manifest_id",
    )
    expected_source_path = (
        f"family-pretraining-v1/{profile}/{study_id}/{dataset_manifest_id[:12]}/"
        f"{trial_id}-{trial_identity[:12]}/latest.pt"
    )
    if source_checkpoint_relative_path != expected_source_path:
        raise RuntimeError(
            "family checkpoint source path disagrees with its embedded identity"
        )

    recovery = _require_mapping(identity, "recovery_operational_identity")
    if recovery.get("schema_version") != FRISSON_FAMILY_RECOVERY_IDENTITY_SCHEMA:
        raise RuntimeError("family checkpoint recovery identity schema mismatch")
    _require_sha256(
        recovery.get("manifest_payload_sha256"),
        "resolved_config.identity.recovery_operational_identity.manifest_payload_sha256",
    )
    recovery_required = {
        "launch_id": launch_id,
        "wandb_run_id": wandb_run_id,
        "checkpoint_relative_path": checkpoint_relative_path,
        "validation_checkpoint_relative_path_template": checkpoint_relative_path.replace(
            "latest.pt",
            "validated/frames-<processed_target_frames>.pt",
        ),
        "wandb_group": f"final-pretraining-recovery-v1-{launch_id}",
        "initial_wandb_resume_policy": "never",
        "subsequent_wandb_resume_policy": "must",
        "automatic_continuation": False,
    }
    recovery_mismatches = {
        name: {"observed": recovery.get(name, _MISSING), "required": required}
        for name, required in recovery_required.items()
        if recovery.get(name, _MISSING) != required
    }
    if recovery_mismatches:
        raise RuntimeError(
            f"family checkpoint recovery identity mismatch: {recovery_mismatches}"
        )

    wandb = _require_mapping(resolved, "wandb")
    wandb_required = {
        "entity": "frisson-labs",
        "project": "Frisson-AI-Melee",
        "run_id": wandb_run_id,
        "group": recovery_required["wandb_group"],
        "initial_resume_policy": "never",
        "subsequent_resume_policy": "must",
        "resume_policy": "must",
        "persistent_run_id_per_trial": True,
    }
    wandb_mismatches = {
        name: {"observed": wandb.get(name, _MISSING), "required": required}
        for name, required in wandb_required.items()
        if wandb.get(name, _MISSING) != required
    }
    if wandb_mismatches:
        raise RuntimeError(
            f"family checkpoint W&B identity mismatch: {wandb_mismatches}"
        )

    metadata_required = {
        "study_id": study_id,
        "trial_id": trial_id,
        "track_identity_sha256": track_identity,
        "trial_identity_sha256": trial_identity,
        "checkpoint_relative_path": checkpoint_relative_path,
        "wandb_run_id": wandb_run_id,
        "launch_id": launch_id,
        "dataset": dict(dataset),
        "promotion_metric": "validation/total_policy_nll",
        "planning_projection_used_for_live_hpo_ranking": False,
    }
    metadata_mismatches = {
        name: {"observed": metadata.get(name, _MISSING), "required": required}
        for name, required in metadata_required.items()
        if metadata.get(name, _MISSING) != required
    }
    if metadata_mismatches:
        raise RuntimeError(
            f"family checkpoint metadata identity mismatch: {metadata_mismatches}"
        )
    validation = _require_mapping(resolved, "validation")
    validation_manifest = _require_sha256(
        validation.get("window_manifest_sha256"),
        "resolved_config.validation.window_manifest_sha256",
    )
    if metadata.get("validation_window_manifest_sha256") != validation_manifest:
        raise RuntimeError("family checkpoint validation manifest identity mismatch")
    runtime_observability = _require_mapping(resolved, "runtime_observability")
    if metadata.get("remote_call_id") != runtime_observability.get("remote_call_id"):
        raise RuntimeError("family checkpoint remote call identity mismatch")
    _require_sha256(
        metadata.get("resume_contract_sha256"), "metadata.resume_contract_sha256"
    )

    training_state = _require_mapping(payload, "training_state")
    processed_target_frames = _require_integer(
        training_state,
        "processed_target_frames",
        field_name="training_state.processed_target_frames",
    )
    frame_cursor = _require_integer(
        training_state,
        "frame_cursor",
        field_name="training_state.frame_cursor",
    )
    optimizer_steps = _require_integer(
        training_state,
        "optimizer_steps",
        field_name="training_state.optimizer_steps",
    )
    microbatches = _require_integer(
        training_state,
        "microbatches",
        field_name="training_state.microbatches",
    )
    if min(processed_target_frames, frame_cursor, optimizer_steps, microbatches) < 0:
        raise RuntimeError("family checkpoint training counters must be nonnegative")
    if (
        frame_cursor != processed_target_frames
        or payload.get("frame_cursor") != frame_cursor
    ):
        raise RuntimeError("family checkpoint frame cursor identity mismatch")

    storage = _require_mapping(metadata, "checkpoint_storage")
    storage_required = {
        "schema_version": "melee-policy.immutable-checkpoint.v1",
        "latest_path": f"/mnt/melee-policy-checkpoints/{checkpoint_relative_path}",
        "immutable_step_path": (
            f"/mnt/melee-policy-checkpoints/{Path(checkpoint_relative_path).parent}/"
            f"steps/step-{optimizer_steps}.pt"
        ),
        "optimizer_steps": optimizer_steps,
        "processed_target_frames": processed_target_frames,
        "path_semantics": "immutable step with atomic latest symlink",
    }
    storage_mismatches = {
        name: {"observed": storage.get(name, _MISSING), "required": required}
        for name, required in storage_required.items()
        if storage.get(name, _MISSING) != required
    }
    if storage_mismatches:
        raise RuntimeError(
            f"family checkpoint storage identity mismatch: {storage_mismatches}"
        )

    return {
        "format": FRISSON_FAMILY_CHECKPOINT_FORMAT,
        "profile": profile,
        "step": optimizer_steps,
        "processed_target_frames": processed_target_frames,
        "wandb_run_id": wandb_run_id,
        "study_id": study_id,
        "trial_id": trial_id,
        "launch_id": launch_id,
        "track_identity_sha256": track_identity,
        "trial_identity_sha256": trial_identity,
        "checkpoint_relative_path": checkpoint_relative_path,
        "source_checkpoint_relative_path": source_checkpoint_relative_path,
        "source_wandb_run_id": source_wandb_run_id,
        "resolved_config_sha256": _canonical_sha256(resolved),
        "resolved_model": resolved_model,
        "resolved_training": dict(training),
        "metadata": _json_safe(metadata),
        "expected_state_tensor_count": profile_contract["state_tensor_count"],
        "expected_parameter_count": expected_model["total_trainable_parameters"],
        "expected_component_parameter_counts": {
            "encoder": expected_model["encoder_parameters"],
            "backbone": expected_model["backbone_parameters"],
            "controller_head": expected_model["controller_head_parameters"],
            "total": expected_model["total_trainable_parameters"],
        },
        "deployed_actor": _pinned_evaluation_actor_contract(profile),
    }


def _post_rl_checkpoint_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(_require_mapping(payload, "metadata"))
    rl = dict(_require_mapping(metadata, "rl_post_training"))
    base = dict(payload)
    base["metadata"] = {
        key: value for key, value in metadata.items() if key != "rl_post_training"
    }
    inherited = _posttraining_checkpoint_contract(base)
    expected = POST_RL_CHECKPOINTS[inherited["profile"]]
    for key, wanted in {
        "step": expected["step"],
        "wandb_run": expected["wandb_run_id"],
        "source_checkpoint": expected["source_checkpoint"],
        "init_checkpoint": f"melee-rl-runs:/bc/frisson-melee-{inherited['profile']}-posttrained-best-val.pt",
    }.items():
        if rl.get(key) != wanted:
            raise RuntimeError(f"P21 RL provenance mismatch: {key}")
    return {
        **inherited,
        "format": FRISSON_POST_RL_CHECKPOINT_FORMAT,
        "step": expected["step"],
        "processed_target_frames": None,
        "wandb_run_id": expected["wandb_run_id"],
        "trial_id": f"p21-{inherited['profile']}-rl",
        "expected_checkpoint_sha256": expected["sha256"],
        "expected_checkpoint_bytes": expected["byte_length"],
        "metadata": _json_safe(metadata),
        "rl_provenance": {
            **rl,
            "initialization_training_state": _json_safe(payload["training_state"]),
            "initialization_processed_target_frames": inherited[
                "processed_target_frames"
            ],
            "rl_frames_seen": expected["rl_frames_seen"],
            "native_checkpoint_created": expected["created"],
            "native_context_mode": "prefix",
            "evaluation_context_mode": "ring",
            "evaluation_contract": "unchanged existing benchmark runtime",
        },
    }


def _faynt_d0_checkpoint_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate new RL weights separately from their inherited BC envelope."""
    metadata = dict(_require_mapping(payload, "metadata"))
    rl = dict(_require_mapping(metadata, "rl_post_training"))
    base = dict(payload)
    base["metadata"] = {
        key: value
        for key, value in metadata.items()
        if key not in {"rl_post_training", "trained_delay"}
    }
    inherited = _posttraining_checkpoint_contract(base)
    expected = FAYNT_D0_CHECKPOINTS[inherited["profile"]]
    for key, wanted in {
        "step": expected["step"],
        "wandb_run": expected["wandb_run_id"],
        "source_checkpoint": expected["source_checkpoint"],
        "init_checkpoint": f"melee-rl-runs:/bc/frisson-melee-{inherited['profile']}-posttrained-best-val.pt",
        "exported": expected["exported"],
    }.items():
        if rl.get(key) != wanted or (
            isinstance(wanted, int) and isinstance(rl.get(key), bool)
        ):
            raise RuntimeError(f"Faynt D0 RL provenance mismatch: {key}")
    for mapping, key in ((metadata, "trained_delay"), (rl, "delay_frames")):
        if key in mapping or expected["explicit_ali_delay_required"]:
            value = _require_integer(mapping, key)
            if value != 0:
                raise RuntimeError(f"Faynt D0 checkpoint requires zero {key}")
    return {
        **inherited,
        "format": FAYNT_D0_CHECKPOINT_FORMAT,
        "step": expected["step"],
        "processed_target_frames": None,
        "wandb_run_id": expected["wandb_run_id"],
        "trial_id": f"faynt-d0-{inherited['profile']}-rl-step-{expected['step']}",
        "expected_checkpoint_sha256": expected["sha256"],
        "expected_checkpoint_bytes": expected["byte_length"],
        "trained_delay": 0,
        "metadata": _json_safe(metadata),
        "initialization_provenance": inherited,
        "rl_provenance": {
            **rl,
            "trained_delay": 0,
            "delay_evidence": expected["delay_evidence"],
            "native_checkpoint_sha256": expected["native_sha256"],
            "native_checkpoint_byte_length": expected["native_byte_length"],
            "native_checkpoint_created": expected["created"],
            "policy_export_sha256": expected["policy_export_sha256"],
            "policy_export_byte_length": expected["policy_export_byte_length"],
            "initialization_training_state": _json_safe(payload["training_state"]),
            "initialization_processed_target_frames": inherited[
                "processed_target_frames"
            ],
            "inherited_optimizer_state_used_for_inference": False,
            "training_compute_dtype": "float32",
            "initialization_training_compute_dtype": inherited["resolved_training"][
                "precision"
            ],
            "rl_frames_seen": expected["rl_frames_seen"],
            "native_context_mode": expected["native_context_mode"],
            "evaluation_context_mode": "ring",
            "evaluation_contract": "unchanged existing benchmark runtime; zero added delay",
        },
    }


def _checkpoint_envelope(
    payload: Mapping[str, Any],
) -> tuple[str, int, str | None, dict[str, Any], dict[str, Any] | None]:
    """Validate one supported safe checkpoint envelope and normalize its metadata."""

    checkpoint_format = payload.get("format")
    if checkpoint_format is None and payload.get("format_version") == 1:
        resolved = payload.get("resolved_config")
        if (
            isinstance(resolved, Mapping)
            and resolved.get("schema_version") == FRISSON_FAMILY_RUNTIME_SCHEMA
        ):
            metadata = payload.get("metadata")
            if isinstance(metadata, Mapping) and "rl_post_training" in metadata:
                rl = _require_mapping(metadata, "rl_post_training")
                if rl.get("source_checkpoint") in {
                    record["source_checkpoint"]
                    for record in FAYNT_D0_CHECKPOINTS.values()
                }:
                    family = _faynt_d0_checkpoint_contract(payload)
                    checkpoint_format = FAYNT_D0_CHECKPOINT_FORMAT
                else:
                    family = _post_rl_checkpoint_contract(payload)
                    checkpoint_format = FRISSON_POST_RL_CHECKPOINT_FORMAT
            elif (
                isinstance(metadata, Mapping)
                and "curriculum" in resolved
                and "curriculum" in metadata
            ):
                family = _posttraining_checkpoint_contract(payload)
                checkpoint_format = FRISSON_POSTTRAINING_CHECKPOINT_FORMAT
            else:
                family = _family_checkpoint_contract(payload)
                checkpoint_format = FRISSON_FAMILY_CHECKPOINT_FORMAT
            return (
                checkpoint_format,
                cast(int, family["step"]),
                None,
                cast(dict[str, Any], family["deployed_actor"]),
                {
                    "processed_target_frames": family["processed_target_frames"],
                    "profile": family["profile"],
                    "wandb_run_id": family["wandb_run_id"],
                    "trial_id": family["trial_id"],
                },
            )
    if checkpoint_format not in FRISSON_CHECKPOINT_FORMATS:
        raise RuntimeError(
            f"Frisson checkpoint format mismatch: {checkpoint_format!r} not in "
            f"{sorted(FRISSON_CHECKPOINT_FORMATS)!r}"
        )
    common_keys = {
        "format",
        "state_dict",
        "model_config",
        "codec",
        "slippi_ai_commit",
        "config_yaml_sha256",
    }
    required_keys = set(common_keys)
    if checkpoint_format == FRISSON_CHECKPOINT_FORMAT:
        required_keys.update(("step", "rl_config", "delay_frames", "context_mode"))
    elif checkpoint_format == FRISSON_BC_CHECKPOINT_FORMAT:
        required_keys.add("training")
    elif checkpoint_format == FRISSON_POSTTRAINING_CHECKPOINT_FORMAT:
        raise RuntimeError(
            "post-training checkpoint must use its versioned curriculum envelope"
        )
    else:
        raise RuntimeError(
            "family checkpoint must use its versioned final-pretraining envelope"
        )
    missing_keys = sorted(required_keys - set(payload))
    if missing_keys:
        raise RuntimeError(
            f"Frisson checkpoint is missing required fields: {missing_keys!r}"
        )

    training: dict[str, Any] | None = None
    if checkpoint_format == FRISSON_CHECKPOINT_FORMAT:
        checkpoint_step = payload["step"]
        actor_contract = _validate_checkpoint_actor(payload)
    else:
        training = dict(_require_mapping(payload, "training"))
        checkpoint_step = training.get("step")
        source_checkpoint = training.get("source_checkpoint")
        source_sha256 = training.get("source_sha256")
        if not isinstance(source_checkpoint, str) or not source_checkpoint:
            raise RuntimeError(
                "BC checkpoint training source_checkpoint must be nonempty"
            )
        if not isinstance(source_sha256, str) or len(source_sha256) != 64:
            raise RuntimeError("BC checkpoint training source_sha256 must be a digest")
        try:
            int(source_sha256, 16)
        except ValueError as error:
            raise RuntimeError(
                "BC checkpoint training source_sha256 is not hexadecimal"
            ) from error
        actor_contract = _pinned_evaluation_actor_contract()

    if isinstance(checkpoint_step, bool) or not isinstance(checkpoint_step, int):
        raise RuntimeError("checkpoint training step must be an integer")
    if checkpoint_step < 0:
        raise RuntimeError("checkpoint training step must be nonnegative")

    config_sha = payload["config_yaml_sha256"]
    if config_sha is not None:
        if not isinstance(config_sha, str) or len(config_sha) != 64:
            raise RuntimeError(
                "checkpoint config_yaml_sha256 is not a 64-character digest"
            )
        try:
            int(config_sha, 16)
        except ValueError as error:
            raise RuntimeError(
                "checkpoint config_yaml_sha256 is not hexadecimal"
            ) from error
    elif checkpoint_format == FRISSON_CHECKPOINT_FORMAT:
        raise RuntimeError("RL checkpoint config_yaml_sha256 cannot be null")

    return checkpoint_format, checkpoint_step, config_sha, actor_contract, training


def _safe_load_checkpoint_payload(
    checkpoint: Path, *, torch: ModuleType
) -> Mapping[str, Any]:
    unsafe_globals = torch.serialization.get_unsafe_globals_in_checkpoint(
        str(checkpoint)
    )
    unsupported_globals = sorted(set(unsafe_globals) - {"builtins.frozenset"})
    if unsupported_globals:
        raise RuntimeError(
            f"checkpoint requires unsafe globals: {unsupported_globals!r}"
        )
    safe_globals: list[type[Any]] = []
    if unsafe_globals:
        checkpoint_sha256 = _sha256_file(checkpoint)
        if set(unsafe_globals) != {"builtins.frozenset"}:
            raise RuntimeError(
                f"checkpoint safe-global declaration changed: {unsafe_globals!r}"
            )
        if checkpoint_sha256 not in _TRUSTED_POSTTRAINING_SHA256 | {
            row["sha256"] for row in POST_RL_CHECKPOINTS.values()
        } | {row["sha256"] for row in FAYNT_D0_CHECKPOINTS.values()}:
            raise RuntimeError(
                "builtins.frozenset is allowed only for a content-addressed trusted post-training checkpoint"
            )
        safe_globals.append(frozenset)
    with torch.serialization.safe_globals(safe_globals):
        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    if not isinstance(payload, Mapping):
        raise RuntimeError("Frisson checkpoint payload must be a mapping")
    return cast(Mapping[str, Any], payload)


def _validate_checkpoint_state_dict(
    state_dict: Any,
    *,
    torch: ModuleType,
    expected_tensor_count: int,
    expected_parameter_count: int,
) -> tuple[Mapping[str, Any], int]:
    if not isinstance(state_dict, Mapping):
        raise RuntimeError("checkpoint state_dict must be a mapping")
    if len(state_dict) != expected_tensor_count:
        raise RuntimeError(
            f"checkpoint tensor count mismatch: {len(state_dict)} != {expected_tensor_count}"
        )
    state_elements = 0
    for name, value in state_dict.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise RuntimeError(
                "checkpoint state_dict must map string names to tensors only"
            )
        state_elements += int(value.numel())
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(
                f"checkpoint state tensor {name!r} contains nonfinite values"
            )
    if state_elements != expected_parameter_count:
        raise RuntimeError(
            f"checkpoint state element count mismatch: {state_elements} != {expected_parameter_count}"
        )
    return cast(Mapping[str, Any], state_dict), state_elements


def _inspect_loaded_checkpoint(
    checkpoint: Path,
    payload: Mapping[str, Any],
    *,
    torch: ModuleType,
) -> tuple[Mapping[str, Any], dict[str, Any], dict[str, Any] | None]:
    checkpoint_format, checkpoint_step, config_sha, actor_contract, training = (
        _checkpoint_envelope(payload)
    )
    family: dict[str, Any] | None = None
    if checkpoint_format in {
        FRISSON_FAMILY_CHECKPOINT_FORMAT,
        FRISSON_POSTTRAINING_CHECKPOINT_FORMAT,
        FRISSON_POST_RL_CHECKPOINT_FORMAT,
        FAYNT_D0_CHECKPOINT_FORMAT,
    }:
        if checkpoint_format == FAYNT_D0_CHECKPOINT_FORMAT:
            family = _faynt_d0_checkpoint_contract(payload)
        elif checkpoint_format == FRISSON_POST_RL_CHECKPOINT_FORMAT:
            family = _post_rl_checkpoint_contract(payload)
        elif checkpoint_format == FRISSON_POSTTRAINING_CHECKPOINT_FORMAT:
            family = _posttraining_checkpoint_contract(payload)
        else:
            family = _family_checkpoint_contract(payload)
        expected_tensor_count = cast(int, family["expected_state_tensor_count"])
        expected_parameter_count = cast(int, family["expected_parameter_count"])
        state_dict_value = payload.get("model")
        profile = cast(str, family["profile"])
        slippi_ai_commit = SLIPPI_AI_SOURCE_REVISION
        codec = EXPECTED_CODEC
        embedded_model_config: Any = {
            "source": "resolved_config.model + resolved_config.training",
            "model": family["resolved_model"],
            "training": {
                "context_length": family["resolved_training"]["context_length"],
                "action_offset_frames": family["resolved_training"][
                    "action_offset_frames"
                ],
                "precision": family["resolved_training"]["precision"],
            },
        }
    else:
        if payload["slippi_ai_commit"] != SLIPPI_AI_SOURCE_REVISION:
            raise RuntimeError("checkpoint Slippi-AI representation commit mismatch")
        if payload["codec"] != EXPECTED_CODEC:
            raise RuntimeError(
                f"checkpoint custom_v1 vocabulary mismatch: {payload['codec']!r}"
            )
        expected_tensor_count = EXPECTED_STATE_TENSOR_COUNT
        expected_parameter_count = EXPECTED_PARAMETER_COUNT
        state_dict_value = payload.get("state_dict")
        profile = "20m"
        slippi_ai_commit = payload["slippi_ai_commit"]
        codec = payload["codec"]
        embedded_model_config = dict(_require_mapping(payload, "model_config"))
        if embedded_model_config.get("prevalidated_inputs", False) is True:
            raise RuntimeError(
                "checkpoint requires the unpinned prevalidated-input runtime path"
            )

    state_dict, _ = _validate_checkpoint_state_dict(
        state_dict_value,
        torch=torch,
        expected_tensor_count=expected_tensor_count,
        expected_parameter_count=expected_parameter_count,
    )
    checkpoint_sha256 = _sha256_file(checkpoint)
    checkpoint_bytes = checkpoint.stat().st_size
    curriculum_envelope = checkpoint_format in {
        FRISSON_POSTTRAINING_CHECKPOINT_FORMAT,
        FRISSON_POST_RL_CHECKPOINT_FORMAT,
        FAYNT_D0_CHECKPOINT_FORMAT,
    }
    if curriculum_envelope and family is not None:
        expected_sha256 = cast(str, family["expected_checkpoint_sha256"])
        expected_bytes = cast(int, family["expected_checkpoint_bytes"])
        if checkpoint_sha256 != expected_sha256 or checkpoint_bytes != expected_bytes:
            raise RuntimeError(
                "post-training checkpoint content identity mismatch: "
                f"sha256={checkpoint_sha256!r}, bytes={checkpoint_bytes!r}, "
                f"required_sha256={expected_sha256!r}, required_bytes={expected_bytes!r}"
            )
    identity = {
        "format": checkpoint_format,
        "profile": profile,
        "path": str(checkpoint),
        "sha256": checkpoint_sha256,
        "byte_length": checkpoint_bytes,
        "step": checkpoint_step,
        "processed_target_frames": (
            None if family is None else family["processed_target_frames"]
        ),
        "config_yaml_sha256": config_sha,
        "training": _json_safe(training),
        "slippi_ai_commit": slippi_ai_commit,
        "codec": _json_safe(codec),
        "model_config": _json_safe(embedded_model_config),
        "state_tensor_count": len(state_dict),
        "parameter_count": expected_parameter_count,
        "all_state_tensors_finite": True,
        "safe_load": {
            "weights_only": True,
            "mmap": True,
            "map_location": "cpu",
            "unsafe_globals": (["builtins.frozenset"] if curriculum_envelope else []),
            "allowlisted_safe_builtins": (
                ["builtins.frozenset"] if curriculum_envelope else []
            ),
        },
        "deployed_actor": actor_contract,
    }
    if family is not None:
        provenance_key = "posttraining" if curriculum_envelope else "family_pretraining"
        provenance_names = [
            "wandb_run_id",
            "study_id",
            "trial_id",
            "launch_id",
            "track_identity_sha256",
            "trial_identity_sha256",
            "checkpoint_relative_path",
            "resolved_config_sha256",
            "metadata",
        ]
        if curriculum_envelope:
            provenance_names.extend(
                (
                    "source_checkpoint_path",
                    "source_checkpoint_sha256",
                    "immutable_step_path",
                    "dataset_identity",
                    "curriculum",
                )
            )
        else:
            provenance_names.extend(
                ("source_checkpoint_relative_path", "source_wandb_run_id")
            )
        identity[provenance_key] = {
            name: _json_safe(family[name]) for name in provenance_names
        }
        if checkpoint_format in {
            FRISSON_POST_RL_CHECKPOINT_FORMAT,
            FAYNT_D0_CHECKPOINT_FORMAT,
        }:
            identity["initialization"] = identity.pop(provenance_key)
            identity["post_rl"] = family["rl_provenance"]
        if checkpoint_format == FAYNT_D0_CHECKPOINT_FORMAT:
            identity["trained_delay"] = family["trained_delay"]
            identity["initialization"] = {
                name: _json_safe(family["initialization_provenance"][name])
                for name in provenance_names
            }
        identity["producing_source_attestation"] = {
            "checkpoint_embedded_source_snapshot_sha256": None,
            "cryptographic_producing_source_verification": False,
            "inference_source_boundary": "launcher-verified pinned Git source",
        }
    return state_dict, identity, family


def inspect_frisson_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    """Safely identify a checkpoint before match-contract validation."""

    checkpoint = checkpoint_path.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Frisson checkpoint is missing: {checkpoint}")
    torch = importlib.import_module("torch")
    payload = _safe_load_checkpoint_payload(checkpoint, torch=torch)
    _, identity, _ = _inspect_loaded_checkpoint(checkpoint, payload, torch=torch)
    return identity


def _load_checkpoint(
    checkpoint_path: Path,
    *,
    torch: ModuleType,
    model_module: ModuleType,
) -> tuple[Any, dict[str, Any]]:
    checkpoint = checkpoint_path.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Frisson checkpoint is missing: {checkpoint}")

    payload = _safe_load_checkpoint_payload(checkpoint, torch=torch)
    state_dict, identity, family = _inspect_loaded_checkpoint(
        checkpoint, payload, torch=torch
    )
    model_config_type = model_module.ModelConfig
    runtime_field_names = {
        field.name for field in dataclasses.fields(model_config_type)
    }
    if family is None:
        embedded_model_config = dict(_require_mapping(payload, "model_config"))
        prevalidated_inputs = embedded_model_config.get("prevalidated_inputs", False)
        if not isinstance(prevalidated_inputs, bool):
            raise RuntimeError(
                "checkpoint prevalidated_inputs must be boolean when present"
            )
        if prevalidated_inputs:
            raise RuntimeError(
                "checkpoint requires the unpinned prevalidated-input runtime path"
            )
        normalized_embedded_config = dict(embedded_model_config)
        normalized_embedded_config.pop("prevalidated_inputs", None)
        runtime_model_kwargs = dict(normalized_embedded_config)
        if "prevalidated_inputs" in runtime_field_names:
            runtime_model_kwargs["prevalidated_inputs"] = False
        try:
            model_config = model_config_type(**runtime_model_kwargs)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "checkpoint model_config cannot construct the pinned ModelConfig"
            ) from error
        reconstructed = dataclasses.asdict(model_config)
        normalized_reconstructed = dict(reconstructed)
        normalized_reconstructed.pop("prevalidated_inputs", None)
        if normalized_embedded_config != normalized_reconstructed:
            raise RuntimeError(
                "checkpoint model_config is incomplete or contains noncanonical values"
            )
        expected_model_config = dataclasses.asdict(
            model_config_type(compute_dtype="float32", cache_dtype="float32")
        )
        normalized_expected_model_config = dict(expected_model_config)
        normalized_expected_model_config.pop("prevalidated_inputs", None)
        if normalized_reconstructed != normalized_expected_model_config:
            mismatches = {
                name: {
                    "observed": normalized_reconstructed.get(name),
                    "required": normalized_expected_model_config.get(name),
                }
                for name in sorted(
                    set(normalized_reconstructed)
                    | set(normalized_expected_model_config)
                )
                if normalized_reconstructed.get(name)
                != normalized_expected_model_config.get(name)
            }
            raise RuntimeError(
                f"checkpoint embedded model contract mismatch: {mismatches}"
            )
    else:
        prevalidated_inputs = True
        baseline = dataclasses.asdict(model_config_type())
        resolved_model = cast(Mapping[str, Any], family["resolved_model"])
        resolved_training = cast(Mapping[str, Any], family["resolved_training"])
        runtime_model_kwargs = dict(baseline)
        runtime_model_kwargs.update(
            {
                name: resolved_model[name]
                for name in (
                    "profile",
                    "d_model",
                    "n_layers",
                    "n_heads",
                    "n_kv_heads",
                    "head_dim",
                    "d_ff",
                    "gradient_checkpointing",
                )
            }
        )
        runtime_model_kwargs.update(
            {
                "context_length": resolved_training["context_length"],
                "action_offset_frames": resolved_training["action_offset_frames"],
                "parameter_dtype": "float32",
                "compute_dtype": "float32",
                "cache_dtype": "float32",
                "softmax_dtype": "float32",
            }
        )
        if "prevalidated_inputs" in runtime_field_names:
            runtime_model_kwargs["prevalidated_inputs"] = False
        try:
            model_config = model_config_type(**runtime_model_kwargs)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "family checkpoint cannot construct the pinned ModelConfig"
            ) from error
        reconstructed = dataclasses.asdict(model_config)
        if reconstructed != runtime_model_kwargs:
            raise RuntimeError(
                "family checkpoint runtime model reconstruction is noncanonical"
            )

    if model_config.action_offset_frames != EXPECTED_ACTION_OFFSET_FRAMES:
        raise RuntimeError("checkpoint action offset is not exactly one frame")
    if model_config.context_length != EXPECTED_MODEL_CONTEXT_LENGTH:
        raise RuntimeError("checkpoint Transformer context capacity is not exactly 256")

    policy = model_module.MeleePolicy(model_config)
    observed_vocab_sizes = tuple(int(value) for value in policy.codec.vocabulary_sizes)
    expected_vocab_sizes = tuple(
        EXPECTED_CODEC["vocab_sizes"][name] for name in ("buttons", "main_stick")
    )
    if observed_vocab_sizes != expected_vocab_sizes:
        raise RuntimeError(
            f"constructed custom_v1 vocabulary mismatch: {observed_vocab_sizes} != {expected_vocab_sizes}"
        )
    incompatible = policy.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "strict state load returned incompatibilities: "
            f"missing={incompatible.missing_keys!r}, unexpected={incompatible.unexpected_keys!r}"
        )
    expected_parameter_count = cast(int, identity["parameter_count"])
    parameter_count = sum(int(parameter.numel()) for parameter in policy.parameters())
    if parameter_count != expected_parameter_count:
        raise RuntimeError(
            f"constructed parameter count mismatch: {parameter_count} != {expected_parameter_count}"
        )
    component_counts = policy.parameter_counts()
    if (
        family is not None
        and component_counts != family["expected_component_parameter_counts"]
    ):
        raise RuntimeError(
            "constructed family component parameter counts mismatch: "
            f"{component_counts} != {family['expected_component_parameter_counts']}"
        )
    for name, parameter in policy.named_parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise RuntimeError(
                f"loaded model parameter {name!r} contains nonfinite values"
            )
        if parameter.dtype != torch.float32 or parameter.device.type != "cpu":
            raise RuntimeError(
                "Frisson inference parameters must load as float32 CPU tensors"
            )
    policy.eval()

    identity.update(
        {
            "runtime_model_config": reconstructed,
            "model_config_compatibility": {
                "checkpoint_prevalidated_inputs": prevalidated_inputs,
                "runtime_prevalidated_inputs": reconstructed.get(
                    "prevalidated_inputs", False
                ),
                "prevalidated_inputs_changes_parameterization": False,
                "checked_runtime_inputs": True,
                "training_compute_dtype": (
                    "float32"
                    if identity["format"] == FAYNT_D0_CHECKPOINT_FORMAT
                    else None
                    if family is None
                    else family["resolved_training"]["precision"]
                ),
                "inference_compute_dtype": "float32",
                "inference_cache_dtype": "float32",
                "inference_parameter_dtype": "float32",
            },
            "constructed_component_parameter_counts": _json_safe(component_counts),
            "strict_state_dict": True,
        },
    )
    return policy, identity


def _scalar_tensor(
    value: Any, *, torch: ModuleType, device: Any, field_name: str
) -> Any:
    tensor = torch.as_tensor(value, device=device)
    if tensor.numel() != 1:
        raise ValueError(
            f"parsed {field_name} must be scalar, got shape {tuple(tensor.shape)}"
        )
    return tensor.reshape(1, 1)


def _tensorize_parsed_game(
    game: Any,
    *,
    tensor_batch: ModuleType,
    torch: ModuleType,
    device: Any,
) -> Any:
    """Convert one pinned parser scalar Game into sibling ``[1, 1]`` batches."""

    def scalar(value: Any, name: str) -> Any:
        return _scalar_tensor(value, torch=torch, device=device, field_name=name)

    def buttons(value: Any, prefix: str) -> Any:
        return tensor_batch.ButtonsBatch(
            **{
                name: scalar(getattr(value, name), f"{prefix}.{name}")
                for name in DIGITAL_BUTTON_ORDER
            }
        )

    def stick(value: Any, prefix: str) -> Any:
        return tensor_batch.StickBatch(
            x=scalar(value.x, f"{prefix}.x"),
            y=scalar(value.y, f"{prefix}.y"),
        )

    def controller(value: Any, prefix: str) -> Any:
        return tensor_batch.ControllerBatch(
            main_stick=stick(value.main_stick, f"{prefix}.main_stick"),
            c_stick=stick(value.c_stick, f"{prefix}.c_stick"),
            shoulder=scalar(value.shoulder, f"{prefix}.shoulder"),
            buttons=buttons(value.buttons, f"{prefix}.buttons"),
        )

    def nana(value: Any, prefix: str) -> Any:
        return tensor_batch.NanaBatch(
            **{
                name: scalar(getattr(value, name), f"{prefix}.{name}")
                for name in (
                    "exists",
                    "percent",
                    "facing",
                    "x",
                    "y",
                    "action",
                    "invulnerable",
                    "character",
                    "jumps_left",
                    "shield_strength",
                    "on_ground",
                )
            }
        )

    def player(value: Any, prefix: str) -> Any:
        core_names = (
            "percent",
            "facing",
            "x",
            "y",
            "action",
            "invulnerable",
            "character",
            "jumps_left",
            "shield_strength",
            "on_ground",
        )
        return tensor_batch.PlayerBatch(
            **{
                name: scalar(getattr(value, name), f"{prefix}.{name}")
                for name in core_names
            },
            controller=controller(value.controller, f"{prefix}.controller"),
            nana=nana(value.nana, f"{prefix}.nana"),
        )

    item_slots = [getattr(game.items, f"item_{index}") for index in range(15)]
    item_tensors = {
        name: torch.stack(
            [
                scalar(getattr(item, name), f"items.item_{index}.{name}")
                for index, item in enumerate(item_slots)
            ],
            dim=-1,
        ).reshape(1, 1, 15)
        for name in ("exists", "type", "state", "x", "y")
    }
    return tensor_batch.GameStateBatch(
        p0=player(game.p0, "p0"),
        p1=player(game.p1, "p1"),
        stage=scalar(game.stage, "stage"),
        randall=tensor_batch.RandallBatch(
            x=scalar(game.randall.x, "randall.x"),
            y=scalar(game.randall.y, "randall.y"),
        ),
        fod_platforms=tensor_batch.FoDPlatformsBatch(
            left=scalar(game.fod_platforms.left, "fod_platforms.left"),
            right=scalar(game.fod_platforms.right, "fod_platforms.right"),
        ),
        items=tensor_batch.ItemsBatch(**item_tensors),
    )


def _slice_last_controller(controller: Any, *, torch: ModuleType) -> Any:
    def sliced(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            if value.ndim < 2 or tuple(value.shape[:2]) != (1, 1):
                raise ValueError(
                    "controller batch leaves must start with singleton [batch, time] dimensions"
                )
            return value[:, -1]
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return type(value)(
                **{
                    field.name: sliced(getattr(value, field.name))
                    for field in dataclasses.fields(value)
                }
            )
        raise TypeError(f"unexpected controller batch node: {type(value).__name__}")

    return sliced(controller)


def _single_tensor_value(value: Any, field_name: str) -> Any:
    if int(value.numel()) != 1:
        raise ValueError(f"decoded {field_name} must contain exactly one value")
    return value.detach().cpu().reshape(()).item()


def capture_frisson_controller_command(
    native_controller: Any,
) -> CanonicalControllerCommand:
    """Capture the sibling custom_v1 decoder without changing its semantics."""
    pressed = tuple(
        name
        for name in DIGITAL_BUTTON_ORDER
        if bool(
            _single_tensor_value(
                getattr(native_controller.buttons, name), f"buttons.{name}"
            )
        )
    )
    return CanonicalControllerCommand(
        main_stick=(
            _single_tensor_value(native_controller.main_stick.x, "main_stick.x"),
            _single_tensor_value(native_controller.main_stick.y, "main_stick.y"),
        ),
        c_stick=(
            _single_tensor_value(native_controller.c_stick.x, "c_stick.x"),
            _single_tensor_value(native_controller.c_stick.y, "c_stick.y"),
        ),
        analog_l=_single_tensor_value(native_controller.shoulder, "shoulder"),
        analog_r=0.0,
        buttons=pressed,
    )


def _sample_policy_frame(
    *,
    policy: Any,
    cache: Any,
    game_batch: Any,
    reset: bool,
    generator: Any,
    torch: ModuleType,
) -> tuple[CanonicalControllerCommand, Any, dict[str, float]]:
    """Run the deployed frame operation order synchronously."""
    current_controller = _slice_last_controller(game_batch.p0.controller, torch=torch)
    timings: dict[str, float] = {}
    started = time.perf_counter()
    encoded = policy.encoder(game_batch, game_batch.p0.controller)
    timings["encoder_seconds"] = time.perf_counter() - started
    if encoded.ndim != 3 or tuple(encoded.shape[:2]) != (1, 1):
        raise RuntimeError(
            f"Frisson encoder returned invalid shape {tuple(encoded.shape)}"
        )
    if not bool(torch.isfinite(encoded).all()):
        raise RuntimeError("Frisson encoder returned nonfinite values")

    started = time.perf_counter()
    hidden, cache = policy.backbone.step(
        encoded[:, -1],
        cache,
        reset_mask=torch.tensor([reset], dtype=torch.bool, device=encoded.device),
    )
    timings["backbone_seconds"] = time.perf_counter() - started
    if not bool(torch.isfinite(hidden).all()):
        raise RuntimeError("Frisson Transformer returned nonfinite values")

    started = time.perf_counter()
    outputs = policy.controller_head.sample(
        hidden,
        current_controller,
        temperature=EXPECTED_SAMPLE_TEMPERATURE,
        generator=generator,
    )
    timings["sample_seconds"] = time.perf_counter() - started
    for name, logits in outputs.logits.items():
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError(
                f"Frisson controller logits {name!r} contain nonfinite values"
            )
    return capture_frisson_controller_command(outputs.controller_state), cache, timings


@dataclass(slots=True)
class _TimingAccumulator:
    count: int = 0
    total_seconds: float = 0.0
    minimum_seconds: float | None = None
    maximum_seconds: float | None = None
    last_seconds: float | None = None

    def add(self, seconds: float) -> None:
        self.count += 1
        self.total_seconds += seconds
        self.last_seconds = seconds
        self.minimum_seconds = (
            seconds
            if self.minimum_seconds is None
            else min(self.minimum_seconds, seconds)
        )
        self.maximum_seconds = (
            seconds
            if self.maximum_seconds is None
            else max(self.maximum_seconds, seconds)
        )

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "count": self.count,
            "total_seconds": self.total_seconds,
            "mean_seconds": self.total_seconds / self.count if self.count else None,
            "minimum_seconds": self.minimum_seconds,
            "maximum_seconds": self.maximum_seconds,
            "last_seconds": self.last_seconds,
        }


class _Runtime(Protocol):
    @property
    def metadata(self) -> Mapping[str, Any]: ...

    def start(self) -> None: ...

    def step(self, gamestate: Any) -> CanonicalControllerCommand: ...

    def diagnostics(self) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


RuntimeFactory = Callable[[FrissonPolicyConfig], _Runtime]


class _NativeFrissonRuntime:
    def __init__(self, config: FrissonPolicyConfig) -> None:
        self.config = config
        self._started = False
        self._closed = False
        self._torch: ModuleType | None = None
        self._modules: dict[str, ModuleType] = {}
        self._policy: Any = None
        self._parser: Any = None
        self._cache: Any = None
        self._generator: Any = None
        self._metadata: dict[str, Any] = {}
        self._last_frame: int | None = None
        self._parser_resets = 0
        self._cache_resets = 0
        self._frames = 0
        self._finite_output_checks = 0
        self._timing = {
            name: _TimingAccumulator()
            for name in ("parse", "tensorize", "encoder", "backbone", "sample", "total")
        }
        self._previous_deterministic: bool | None = None
        self._previous_deterministic_warn_only: bool | None = None

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._metadata

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("Frisson runtime is closed")
        if self._started:
            return
        self.config.validate()
        source_identity = verify_source_assets(self.config)
        modules = _activate_runtime_sources(self.config)
        torch = importlib.import_module("torch")
        policy, checkpoint_identity = _load_checkpoint(
            self.config.checkpoint_path,
            torch=torch,
            model_module=modules["model"],
        )
        device = torch.device(self.config.device)
        policy.to(device=device)
        policy.eval()
        generator = torch.Generator(device=device).manual_seed(
            self.config.evaluation_seed
        )

        self._previous_deterministic = bool(
            torch.are_deterministic_algorithms_enabled()
        )
        warn_only_getter = getattr(
            torch, "is_deterministic_algorithms_warn_only_enabled", None
        )
        self._previous_deterministic_warn_only = (
            bool(warn_only_getter()) if warn_only_getter else False
        )
        torch.use_deterministic_algorithms(True)

        self._torch = torch
        self._modules = modules
        self._policy = policy
        self._generator = generator
        self._metadata = {
            **source_identity,
            "checkpoint": checkpoint_identity,
            "execution": {
                "device": str(device),
                "deterministic_algorithms": True,
                "private_torch_generator": True,
                "evaluation_seed": self.config.evaluation_seed,
                "generator_seed": self.config.evaluation_seed,
                "sample_temperature": self.config.sample_temperature,
                "checkpoint_fast_step": True,
                "execution_path": "encoder + backbone.step + controller_head.sample",
                "fast_step_implementation": "sibling model primitives, not the producing actor helper",
            },
        }
        self._started = True

    @staticmethod
    def _frame(gamestate: Any) -> int:
        value = getattr(gamestate, "frame", None)
        if value is None or isinstance(value, bool):
            raise TypeError("gamestate.frame must be an integer")
        try:
            frame = int(value)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"gamestate.frame must be an integer, got {value!r}"
            ) from error
        if value != frame:
            raise TypeError(f"gamestate.frame must be integral, got {value!r}")
        return frame

    def step(self, gamestate: Any) -> CanonicalControllerCommand:
        self.start()
        torch = cast(ModuleType, self._torch)
        frame = self._frame(gamestate)
        if frame == -123:
            self._parser = self._modules["parser"].Parser(
                ports=(self.config.port, self.config.opponent_port)
            )
            self._cache = self._policy.backbone.init_cache(1, device=self.config.device)
            self._last_frame = None
            self._parser_resets += 1
            self._cache_resets += 1
        elif self._last_frame is None:
            raise ValueError(f"first Frisson runtime frame must be -123, got {frame}")
        else:
            expected = self._last_frame + 1
            if frame != expected:
                raise ValueError(
                    f"Frisson runtime expected frame {expected}, got {frame}"
                )

        total_started = time.perf_counter()
        parse_started = time.perf_counter()
        parsed = self._parser.get_game(gamestate)
        self._timing["parse"].add(time.perf_counter() - parse_started)

        tensor_started = time.perf_counter()
        game_batch = _tensorize_parsed_game(
            parsed,
            tensor_batch=self._modules["tensor_batch"],
            torch=torch,
            device=torch.device(self.config.device),
        )
        self._timing["tensorize"].add(time.perf_counter() - tensor_started)
        with torch.inference_mode():
            command, self._cache, component_timing = _sample_policy_frame(
                policy=self._policy,
                cache=self._cache,
                game_batch=game_batch,
                reset=frame == -123,
                generator=self._generator,
                torch=torch,
            )
        self._timing["encoder"].add(component_timing["encoder_seconds"])
        self._timing["backbone"].add(component_timing["backbone_seconds"])
        self._timing["sample"].add(component_timing["sample_seconds"])
        self._timing["total"].add(time.perf_counter() - total_started)
        self._frames += 1
        self._finite_output_checks += 1
        self._last_frame = frame
        return command

    def diagnostics(self) -> Mapping[str, Any]:
        cache = self._cache
        cache_status: dict[str, Any] | None = None
        if cache is not None:
            cache_status = {
                "capacity": int(cache.capacity),
                "valid_length": [
                    int(value) for value in cache.valid_length.detach().cpu().tolist()
                ],
                "write_position": [
                    int(value) for value in cache.write_position.detach().cpu().tolist()
                ],
                "next_position": [
                    int(value) for value in cache.next_position.detach().cpu().tolist()
                ],
            }
        return {
            "started": self._started,
            "closed": self._closed,
            "frames": self._frames,
            "last_frame": self._last_frame,
            "parser_resets": self._parser_resets,
            "cache_resets": self._cache_resets,
            "finite_output_checks": self._finite_output_checks,
            "cache": cache_status,
            "timing": {
                name: accumulator.as_dict()
                for name, accumulator in self._timing.items()
            },
        }

    def close(self) -> None:
        if self._closed:
            return
        torch = self._torch
        if torch is not None and self._previous_deterministic is not None:
            torch.use_deterministic_algorithms(
                self._previous_deterministic,
                warn_only=bool(self._previous_deterministic_warn_only),
            )
        self._parser = None
        self._cache = None
        self._policy = None
        self._generator = None
        self._started = False
        self._closed = True


def _default_runtime_factory(config: FrissonPolicyConfig) -> _Runtime:
    return _NativeFrissonRuntime(config)


@dataclass(slots=True)
class _SessionState:
    started: bool = False
    closed: bool = False
    fault: str | None = None
    generation: int = 0
    resets: int = 0
    frames_total: int = 0
    frames_in_generation: int = 0
    first_frame: int | None = None
    last_frame: int | None = None
    inference_barriers: int = 0
    awaiting_reset_frame: bool = False
    reset_history: list[dict[str, Any]] = field(default_factory=list)
    timing: _TimingAccumulator = field(default_factory=_TimingAccumulator)


class FrissonPolicySession:
    """One synchronous, frame-exact Frisson policy session."""

    def __init__(
        self,
        config: FrissonPolicyConfig,
        *,
        runtime_factory: RuntimeFactory | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self._factory = runtime_factory or _default_runtime_factory
        self._runtime: _Runtime | None = None
        self._runtime_metadata: Mapping[str, Any] = {}
        self._state = _SessionState()
        self._lock = threading.RLock()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._state.closed:
            raise RuntimeError("Frisson policy session is closed")
        if self._state.fault is not None:
            raise RuntimeError(
                f"Frisson policy session is faulted: {self._state.fault}"
            )

    def start(self) -> None:
        with self._lock:
            self._require_open()
            if self._state.started:
                return
            runtime = self._factory(self.config)
            try:
                runtime.start()
            except BaseException:
                runtime.close()
                raise
            self._runtime = runtime
            self._runtime_metadata = dict(runtime.metadata)
            self._state.started = True

    @staticmethod
    def _frame(gamestate: Any) -> int:
        value = getattr(gamestate, "frame", None)
        if value is None or isinstance(value, bool):
            raise TypeError("gamestate.frame must be an integer")
        try:
            frame = int(value)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"gamestate.frame must be an integer, got {value!r}"
            ) from error
        if value != frame:
            raise TypeError(f"gamestate.frame must be integral, got {value!r}")
        return frame

    def _record_reset(self, reason: str) -> None:
        self._state.reset_history.append(
            {
                "generation": self._state.generation,
                "reason": reason,
                "frames": self._state.frames_in_generation,
                "first_frame": self._state.first_frame,
                "last_frame": self._state.last_frame,
            }
        )
        self._state.generation += 1
        self._state.resets += 1
        self._state.frames_in_generation = 0
        self._state.first_frame = None
        self._state.last_frame = None

    def _validate_next_frame(self, frame: int) -> None:
        if self._state.frames_in_generation == 0:
            if frame != -123:
                raise ValueError(
                    f"first Frisson policy frame must be -123, got {frame}"
                )
            self._state.awaiting_reset_frame = False
            return
        if frame == -123:
            self._record_reset("observed-frame--123")
            self._state.awaiting_reset_frame = False
            return
        assert self._state.last_frame is not None
        expected = self._state.last_frame + 1
        if frame != expected:
            raise ValueError(
                f"Frisson policy frames must be consecutive and unique: expected {expected}, got {frame}"
            )

    def step(self, gamestate: Any) -> CanonicalControllerCommand:
        frame = self._frame(gamestate)
        with self._lock:
            self._require_open()
            self._validate_next_frame(frame)
            self.start()
            runtime = cast(_Runtime, self._runtime)
            started = time.perf_counter()
            try:
                command = runtime.step(gamestate)
                command.validate()
            except BaseException as error:
                self._state.fault = f"{type(error).__name__}: {error}"
                raise
            self._state.timing.add(time.perf_counter() - started)
            self._state.inference_barriers += 1
            self._state.frames_total += 1
            self._state.frames_in_generation += 1
            if self._state.first_frame is None:
                self._state.first_frame = frame
            self._state.last_frame = frame
            return command

    def reset(self, reason: str = "explicit") -> None:
        """Require a fresh frame -123, which recreates both parser and KV cache."""
        if not reason:
            raise ValueError("reset reason must be non-empty")
        with self._lock:
            self._require_open()
            if self._state.frames_in_generation == 0:
                raise RuntimeError(
                    "cannot reset before the current game has received a frame"
                )
            self._record_reset(reason)
            self._state.awaiting_reset_frame = True

    def metadata(self) -> dict[str, Any]:
        with self._lock:
            checkpoint = self._runtime_metadata.get("checkpoint")
            checkpoint_identity = (
                dict(checkpoint) if isinstance(checkpoint, Mapping) else {}
            )
            checkpoint_step = checkpoint_identity.get("step")
            checkpoint_sha256 = checkpoint_identity.get("sha256")
            policy_instance = FRISSON_POLICY_IDENTITY
            if isinstance(checkpoint_step, int) and isinstance(checkpoint_sha256, str):
                policy_instance = f"{FRISSON_POLICY_IDENTITY}/step-{checkpoint_step}/sha256-{checkpoint_sha256}"
            return {
                "schema_version": "melee_policy.frisson_policy.metadata.v1",
                "identity": {
                    "policy": FRISSON_POLICY_IDENTITY,
                    "policy_instance": policy_instance,
                    "checkpoint_step": checkpoint_step,
                    "checkpoint_sha256": checkpoint_sha256,
                    "family": "Frisson-AI",
                    "distinct_from": ["vladfi1/slippi-ai medium-v2", "MIMIC"],
                },
                "runtime_contract": {
                    "port": self.config.port,
                    "opponent_port": self.config.opponent_port,
                    "perspective": {
                        "p0": self.config.port,
                        "p1": self.config.opponent_port,
                    },
                    "parser": "pinned slippi_db.parse_libmelee.Parser",
                    "observation_filter": None,
                    "sample_temperature": self.config.sample_temperature,
                    "evaluation_seed": self.config.evaluation_seed,
                    "private_torch_generator_seed": self.config.evaluation_seed,
                    "deterministic_algorithms": self.config.deterministic_algorithms,
                    "deployed_fast_step": self.config.fast_step,
                    "context_mode": self.config.context_mode,
                    "actor_trajectory_context_frames": self.config.actor_context_frames,
                    "kv_cache_capacity_frames": self.config.model_context_length,
                    "ring_cache_behavior": "continuous across frames and rolls at capacity 256",
                    "periodic_128_frame_reset": False,
                    "delay_frames": self.config.delay_frames,
                    "batch_steps": self.config.batch_steps,
                    "first_required_frame": -123,
                    "frame_order": "strictly consecutive, exactly once",
                    "reset_boundary": "frame -123 recreates parser and KV cache",
                    "render_advance_barrier": (
                        "step returns only after encoder, Transformer, and categorical sample complete"
                    ),
                },
                "action_contract": {
                    "codec": EXPECTED_CODEC,
                    "state_frame": "t",
                    "command_frame": "t+1",
                    "action_offset_frames": EXPECTED_ACTION_OFFSET_FRAMES,
                    "slippi_ai_21_frame_fifo": False,
                    "expected_replay_audit_lag_frames": 1,
                    "digital_buttons": list(DIGITAL_BUTTON_ORDER),
                    "shared_analog_shoulder_output": "L",
                    "analog_r": 0.0,
                },
                "upstream_runtime": _json_safe(self._runtime_metadata),
            }

    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            runtime = (
                dict(self._runtime.diagnostics()) if self._runtime is not None else {}
            )
            expected = (
                None if self._state.last_frame is None else self._state.last_frame + 1
            )
            return {
                "schema_version": "melee_policy.frisson_policy.diagnostics.v1",
                "started": self._state.started,
                "closed": self._state.closed,
                "fault": self._state.fault,
                "generation": self._state.generation,
                "resets": self._state.resets,
                "frames_total": self._state.frames_total,
                "frames_in_generation": self._state.frames_in_generation,
                "first_frame": self._state.first_frame,
                "last_frame": self._state.last_frame,
                "expected_next_frame": expected,
                "awaiting_reset_frame": self._state.awaiting_reset_frame,
                "current_frame_inference_barriers": self._state.inference_barriers,
                "current_frame_inference_barrier_every_frame": (
                    self._state.inference_barriers == self._state.frames_total
                ),
                "step_timing": self._state.timing.as_dict(),
                "reset_history": list(self._state.reset_history),
                "upstream": _json_safe(runtime),
            }

    def close(self) -> None:
        with self._lock:
            if self._state.closed:
                return
            runtime = self._runtime
            close_error: BaseException | None = None
            if runtime is not None:
                try:
                    runtime.close()
                except BaseException as error:
                    close_error = error
            self._runtime = None
            self._state.started = False
            self._state.closed = True
            if close_error is not None:
                raise RuntimeError(
                    "failed to close the Frisson policy runtime"
                ) from close_error


__all__ = [
    "EXPECTED_ACTOR_CONTEXT_FRAMES",
    "EXPECTED_MODEL_CONTEXT_LENGTH",
    "EXPECTED_PARAMETER_COUNT",
    "FRISSON_BC_CHECKPOINT_FORMAT",
    "FRISSON_CHECKPOINT_FORMAT",
    "FRISSON_CHECKPOINT_FORMATS",
    "FRISSON_FAMILY_CHECKPOINT_FORMAT",
    "FRISSON_POLICY_IDENTITY",
    "FRISSON_RUNTIME_SOURCE_REVISION",
    "FrissonPolicyConfig",
    "FrissonPolicySession",
    "capture_frisson_controller_command",
    "inspect_frisson_checkpoint",
    "materialize_pinned_model_source",
    "verify_source_assets",
]
