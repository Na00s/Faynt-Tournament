"""Pinned Slippi-AI policy boundary for rendered, frame-exact evaluation.

The upstream runtime remains responsible for replay-state parsing, observation
filtering, recurrent state, categorical sampling, policy delay, and native
controller decoding.  This module verifies that runtime and adapts its native
single-shoulder controller value to the project's complete controller command.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import importlib
import math
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Protocol, Self, cast

import numpy as np

SLIPPI_AI_REPOSITORY_URL = "https://github.com/vladfi1/slippi-ai"
SLIPPI_AI_SOURCE_REVISION = "577965a7731dc53e3472ea63d9e9853a4e9d65fa"
SLIPPI_AI_SOURCE_BRANCH = "main"
SLIPPI_AI_LICENSE_SHA256 = "a632127717250e9d7d32a7451f8fecb668da1003e333850a4808bcce38348112"
MEDIUM_V2_CHECKPOINT_SHA256 = "48dcfd87c52fde9899fb37b0293ce2a597fb2384b7b40baf7fb49c760152a96c"
MEDIUM_V2_CHECKPOINT_BYTES = 95_559_707
MEDIUM_V2_VARIABLE_COUNT = 141
MEDIUM_V2_PARAMETER_COUNT = 23_887_032
DK_D18_IMITATION_V2_CHECKPOINT_SHA256 = "e4e3f07d32812154b4401e40d6d83b741cbf153887dcd14417fa50c35eeeae4d"
DK_D18_IMITATION_V2_CHECKPOINT_BYTES = 42_078_949
DOC_D18_IMITATION_V3_CHECKPOINT_SHA256 = "ade88cc5c04974df5123da1a0539ab2324d9b76dc41c5561f8e94a54c6d2ac14"
DOC_D18_IMITATION_V3_CHECKPOINT_BYTES = 42_078_998
SPECIALIST_VARIABLE_COUNT = 137
SPECIALIST_PARAMETER_COUNT = 10_517_970
MEDIUM_V2_SUPPORTED_CHARACTERS = (
    "FOX",
    "FALCO",
    "MARTH",
    "SHEIK",
    "JIGGLYPUFF",
    "CPTFALCON",
    "PEACH",
    "YOSHI",
    "POPO",
    "LUIGI",
    "PIKACHU",
    "SAMUS",
)
MEDIUM_V2_SUPPORTED_STAGES = (
    "BATTLEFIELD",
    "DREAMLAND",
    "FINAL_DESTINATION",
    "FOUNTAIN_OF_DREAMS",
    "POKEMON_STADIUM",
    "YOSHIS_STORY",
)
MEDIUM_V2_ALLOWED_OPPONENTS = "all"
MEDIUM_V2_CHARACTER_EMBEDDING_SIZE = 33
MEDIUM_V2_STAGE_EMBEDDING_SIZE = 64
DEFAULT_PLAYER_NAME = "Master Player"
MEDIUM_V2_RL_TRAINED_NAMES = (DEFAULT_PLAYER_NAME,) * len(MEDIUM_V2_SUPPORTED_CHARACTERS)
POLICY_DELAY_FRAMES = 21
UPSTREAM_EVAL_TWO_CONSOLE_DELAY_FRAMES = 2
# The pinned eval_two.py recipe uses console delay 2. The checkpoint training
# configuration and generic DolphinConfig use delay 0 instead.
EFFECTIVE_POLICY_DELAY_FRAMES = POLICY_DELAY_FRAMES - UPSTREAM_EVAL_TWO_CONSOLE_DELAY_FRAMES
SAMPLE_TEMPERATURE = 1.0
LIBMELEE_DISPATCH_VERSION = "0.47.3"
SLIPPI_AI_RELEASE_CONTRACT_SCHEMA_VERSION = "melee_policy.slippi_ai.release.v1"


@dataclass(frozen=True, slots=True)
class SlippiAIReleaseContract:
    """Immutable identity and runtime contract for one official release file."""

    key: str
    display_name: str
    checkpoint_sha256: str
    checkpoint_bytes: int
    variable_count: int
    parameter_count: int
    policy_delay_frames: int
    supported_characters: tuple[str, ...]
    allowed_opponents: str
    checkpoint_kind: str
    checkpoint_tag: str
    checkpoint_config_version: int
    checkpoint_step: int | None
    rl_trained_names: tuple[str, ...]
    official_url: str

    def checkpoint_path(self, project_root: Path) -> Path:
        return project_root.expanduser().resolve() / ".e001-cache" / "slippi-ai" / "models" / self.key

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SLIPPI_AI_RELEASE_CONTRACT_SCHEMA_VERSION,
            "key": self.key,
            "display_name": self.display_name,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_bytes": self.checkpoint_bytes,
            "variable_count": self.variable_count,
            "parameter_count": self.parameter_count,
            "policy_delay_frames": self.policy_delay_frames,
            "supported_characters": list(self.supported_characters),
            "allowed_opponents": self.allowed_opponents,
            "checkpoint_kind": self.checkpoint_kind,
            "checkpoint_tag": self.checkpoint_tag,
            "checkpoint_config_version": self.checkpoint_config_version,
            "checkpoint_step": self.checkpoint_step,
            "rl_trained_names": list(self.rl_trained_names),
            "official_url": self.official_url,
        }


SLIPPI_AI_RELEASE_CONTRACTS: dict[str, SlippiAIReleaseContract] = {
    "medium-v2": SlippiAIReleaseContract(
        key="medium-v2",
        display_name="vladfi1 Slippi-AI medium-v2 Master Player",
        checkpoint_sha256=MEDIUM_V2_CHECKPOINT_SHA256,
        checkpoint_bytes=MEDIUM_V2_CHECKPOINT_BYTES,
        variable_count=MEDIUM_V2_VARIABLE_COUNT,
        parameter_count=MEDIUM_V2_PARAMETER_COUNT,
        policy_delay_frames=POLICY_DELAY_FRAMES,
        supported_characters=MEDIUM_V2_SUPPORTED_CHARACTERS,
        allowed_opponents=MEDIUM_V2_ALLOWED_OPPONENTS,
        checkpoint_kind="self-play-rl",
        checkpoint_tag="top12_d21_imitation_v5",
        checkpoint_config_version=5,
        checkpoint_step=6_829,
        rl_trained_names=MEDIUM_V2_RL_TRAINED_NAMES,
        official_url=(
            "https://www.dropbox.com/scl/fi/lpi9krfei1knfvfw7up7v/medium-v2"
            "?rlkey=qmah3qfz5anwva93x48zcx01k&st=sxo8hbeb&dl=0"
        ),
    ),
    "dk_d18_imitation_v2": SlippiAIReleaseContract(
        key="dk_d18_imitation_v2",
        display_name="vladfi1 Slippi-AI dk_d18_imitation_v2 native DK imitation",
        checkpoint_sha256=DK_D18_IMITATION_V2_CHECKPOINT_SHA256,
        checkpoint_bytes=DK_D18_IMITATION_V2_CHECKPOINT_BYTES,
        variable_count=SPECIALIST_VARIABLE_COUNT,
        parameter_count=SPECIALIST_PARAMETER_COUNT,
        policy_delay_frames=18,
        supported_characters=("DK",),
        allowed_opponents="all",
        checkpoint_kind="imitation",
        checkpoint_tag="dk_delay_18_v2",
        checkpoint_config_version=3,
        checkpoint_step=None,
        rl_trained_names=(),
        official_url=(
            "https://www.dropbox.com/scl/fo/mg916t9exid4stqmx2bjf/"
            "AGBmOvmlKjo7iGjm4CEvyf4/dk_d18_imitation_v2"
            "?rlkey=baqxnfxg2uytvcz62w9o8mwzt&dl=1"
        ),
    ),
    "doc_d18_imitation_v3": SlippiAIReleaseContract(
        key="doc_d18_imitation_v3",
        display_name="vladfi1 Slippi-AI doc_d18_imitation_v3 native Dr. Mario imitation",
        checkpoint_sha256=DOC_D18_IMITATION_V3_CHECKPOINT_SHA256,
        checkpoint_bytes=DOC_D18_IMITATION_V3_CHECKPOINT_BYTES,
        variable_count=SPECIALIST_VARIABLE_COUNT,
        parameter_count=SPECIALIST_PARAMETER_COUNT,
        policy_delay_frames=18,
        supported_characters=("DOC",),
        allowed_opponents="all",
        checkpoint_kind="imitation",
        checkpoint_tag="doc_d18_as16_imitation_v3",
        checkpoint_config_version=3,
        checkpoint_step=None,
        rl_trained_names=(),
        official_url=(
            "https://www.dropbox.com/scl/fo/mg916t9exid4stqmx2bjf/"
            "AOigWSZfGvUkNT5MB8VOI7o/doc_d18_imitation_v3"
            "?rlkey=baqxnfxg2uytvcz62w9o8mwzt&dl=1"
        ),
    ),
}


def slippi_ai_release_contract(release: str) -> SlippiAIReleaseContract:
    """Resolve an exact supported release key and reject aliases or unknown files."""
    try:
        return SLIPPI_AI_RELEASE_CONTRACTS[release]
    except KeyError as error:
        raise ValueError(
            f"unsupported Slippi-AI release {release!r}; allowed={tuple(SLIPPI_AI_RELEASE_CONTRACTS)}"
        ) from error


def slippi_ai_release_capabilities(release: str) -> dict[str, Any]:
    """Return the declared controller and rendered-launcher capabilities."""
    contract = slippi_ai_release_contract(release)
    return {
        "controlled_characters": list(contract.supported_characters),
        "opponents": contract.allowed_opponents,
        "rendered_stages": list(MEDIUM_V2_SUPPORTED_STAGES),
        "character_embedding_size": MEDIUM_V2_CHARACTER_EMBEDDING_SIZE,
        "stage_embedding_size": MEDIUM_V2_STAGE_EMBEDDING_SIZE,
        "policy_ports": [1, 2, 3, 4],
        "rendered_two_player_ports": [1, 2],
        "perspective": {
            "p0": "configured controlled port",
            "p1": "configured opponent port",
        },
    }


DIGITAL_BUTTON_ORDER = ("A", "B", "X", "Y", "Z", "L", "R", "D_UP")
_DIGITAL_BUTTON_SET = frozenset(DIGITAL_BUTTON_ORDER)
_ANALOG_CONTROL_NAMES = frozenset(("MAIN", "C", "L", "R"))


def medium_v2_capabilities() -> dict[str, Any]:
    """Return the frozen public-checkpoint and rendered-launcher capabilities."""
    return slippi_ai_release_capabilities("medium-v2")


def _finite_unit_interval(value: Any, field_name: str) -> float:
    try:
        scalar = value.item() if hasattr(value, "item") else value
        result = float(scalar)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a scalar number, got {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite, got {result!r}")
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1], got {result!r}")
    return result


def _button_name(button: Any) -> str:
    value = getattr(button, "value", button)
    name = str(value)
    if name.startswith("BUTTON_"):
        name = name.removeprefix("BUTTON_")
    return name


@dataclass(frozen=True, slots=True)
class CanonicalControllerCommand:
    """One validated, complete GameCube controller command for one frame."""

    main_stick: tuple[float, float]
    c_stick: tuple[float, float]
    analog_l: float
    analog_r: float
    buttons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        main_stick = self._validate_stick(self.main_stick, "main_stick")
        c_stick = self._validate_stick(self.c_stick, "c_stick")
        analog_l = _finite_unit_interval(self.analog_l, "analog_l")
        analog_r = _finite_unit_interval(self.analog_r, "analog_r")

        supplied_buttons = tuple(str(button) for button in self.buttons)
        if "START" in supplied_buttons:
            raise ValueError("START is forbidden for policy controller commands")
        unknown = set(supplied_buttons) - _DIGITAL_BUTTON_SET
        if unknown:
            raise ValueError(f"unsupported digital controller buttons: {sorted(unknown)!r}")
        if len(supplied_buttons) != len(set(supplied_buttons)):
            raise ValueError("digital controller buttons must not contain duplicates")
        ordered_buttons = tuple(button for button in DIGITAL_BUTTON_ORDER if button in supplied_buttons)

        object.__setattr__(self, "main_stick", main_stick)
        object.__setattr__(self, "c_stick", c_stick)
        object.__setattr__(self, "analog_l", analog_l)
        object.__setattr__(self, "analog_r", analog_r)
        object.__setattr__(self, "buttons", ordered_buttons)

    @staticmethod
    def _validate_stick(value: Any, field_name: str) -> tuple[float, float]:
        try:
            x, y = value
        except (TypeError, ValueError) as error:
            raise ValueError(f"{field_name} must contain exactly two axes") from error
        return (
            _finite_unit_interval(x, f"{field_name}.x"),
            _finite_unit_interval(y, f"{field_name}.y"),
        )

    @classmethod
    def neutral(cls) -> Self:
        return cls(
            main_stick=(0.5, 0.5),
            c_stick=(0.5, 0.5),
            analog_l=0.0,
            analog_r=0.0,
            buttons=(),
        )

    def validate(self) -> None:
        """Re-run validation before applying a potentially external command."""
        type(self)(
            main_stick=self.main_stick,
            c_stick=self.c_stick,
            analog_l=self.analog_l,
            analog_r=self.analog_r,
            buttons=self.buttons,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "main_stick": list(self.main_stick),
            "c_stick": list(self.c_stick),
            "analog_l": self.analog_l,
            "analog_r": self.analog_r,
            "buttons": list(self.buttons),
        }


def capture_native_controller_command(native_controller: Any) -> CanonicalControllerCommand:
    """Convert the pinned native decoder value without changing its semantics."""
    buttons = tuple(name for name in DIGITAL_BUTTON_ORDER if bool(getattr(native_controller.buttons, name)))
    return CanonicalControllerCommand(
        main_stick=(native_controller.main_stick.x, native_controller.main_stick.y),
        c_stick=(native_controller.c_stick.x, native_controller.c_stick.y),
        analog_l=native_controller.shoulder,
        # The pinned action type has one analog shoulder and its sender assigns
        # that value to L.  R is therefore independently represented as neutral.
        analog_r=0.0,
        buttons=buttons,
    )


@dataclass(frozen=True, slots=True)
class ControllerDispatchRecord:
    """Immutable description of one canonical-to-libmelee dispatch.

    ``canonical_command`` contains the policy-facing values.  The ``pipe_*``
    fields contain the values that libmelee 0.47.3 writes after applying its
    default input correction exactly once.  Pipe values are Dolphin inputs,
    not processed replay values.
    """

    canonical_command: CanonicalControllerCommand
    pipe_main_stick: tuple[float, float]
    pipe_c_stick: tuple[float, float]
    pipe_analog_l: float
    pipe_analog_r: float
    button_states: tuple[tuple[str, bool], ...]
    pipe_commands: tuple[str, ...]
    libmelee_version: str = LIBMELEE_DISPATCH_VERSION
    fix_analog_inputs: bool = True
    upstream_native_sender_flush_count: int = 0
    project_adapter_flush_count: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "melee_policy.controller_dispatch.libmelee_0_47_3.v1",
            "canonical": self.canonical_command.as_dict(),
            "pipe": {
                "main_stick": list(self.pipe_main_stick),
                "c_stick": list(self.pipe_c_stick),
                "analog_l": self.pipe_analog_l,
                "analog_r": self.pipe_analog_r,
                "button_states": [
                    {"button": name, "pressed": pressed} for name, pressed in self.button_states
                ],
                "commands": list(self.pipe_commands),
            },
            "libmelee": {
                "version": self.libmelee_version,
                "fix_analog_inputs": self.fix_analog_inputs,
                "analog_correction_applications": 1,
            },
            "flush": {
                "upstream_native_sender_count": self.upstream_native_sender_flush_count,
                "project_adapter_count": self.project_adapter_flush_count,
                "project_boundary_deliberately_flushes_once": self.project_adapter_flush_count == 1,
            },
            "interpretation": (
                "Pipe analog values are corrected Dolphin pipe inputs and must not be "
                "interpreted as processed replay values."
            ),
        }


@functools.cache
def _verified_libmelee_dispatch_version() -> str:
    observed_version = importlib_metadata.version("melee")
    if observed_version != LIBMELEE_DISPATCH_VERSION:
        raise RuntimeError(
            f"controller dispatch requires melee {LIBMELEE_DISPATCH_VERSION}, got {observed_version}"
        )
    return observed_version


def describe_controller_dispatch(
    command: CanonicalControllerCommand, *, flush: bool = True
) -> ControllerDispatchRecord:
    """Describe the exact default libmelee 0.47.3 writes without sending them."""
    from melee.controller import fix_analog_stick, fix_analog_trigger

    command.validate()
    _verified_libmelee_dispatch_version()

    pipe_main_stick = tuple(fix_analog_stick(value) for value in command.main_stick)
    pipe_c_stick = tuple(fix_analog_stick(value) for value in command.c_stick)
    pipe_analog_l = fix_analog_trigger(command.analog_l)
    pipe_analog_r = fix_analog_trigger(command.analog_r)
    button_states = tuple((name, name in command.buttons) for name in DIGITAL_BUTTON_ORDER)
    pipe_commands = (
        *(f"{'PRESS' if pressed else 'RELEASE'} {name}\n" for name, pressed in button_states),
        f"SET MAIN {pipe_main_stick[0]} {pipe_main_stick[1]}\n",
        f"SET C {pipe_c_stick[0]} {pipe_c_stick[1]}\n",
        f"SET L {pipe_analog_l}\n",
        f"SET R {pipe_analog_r}\n",
    )
    if flush:
        pipe_commands = (*pipe_commands, "FLUSH\n")
    return ControllerDispatchRecord(
        canonical_command=command,
        pipe_main_stick=cast(tuple[float, float], pipe_main_stick),
        pipe_c_stick=cast(tuple[float, float], pipe_c_stick),
        pipe_analog_l=pipe_analog_l,
        pipe_analog_r=pipe_analog_r,
        button_states=button_states,
        pipe_commands=pipe_commands,
        project_adapter_flush_count=int(flush),
    )


def send_canonical_controller(
    controller: Any,
    command: CanonicalControllerCommand,
    *,
    flush: bool = True,
) -> ControllerDispatchRecord:
    """Apply one complete command, optionally retaining upstream deferred flush timing."""
    import melee

    dispatch = describe_controller_dispatch(command, flush=flush)
    pressed = frozenset(command.buttons)
    for name in DIGITAL_BUTTON_ORDER:
        button = getattr(melee.Button, f"BUTTON_{name}")
        if name in pressed:
            controller.press_button(button)
        else:
            controller.release_button(button)
    controller.tilt_analog(melee.Button.BUTTON_MAIN, *command.main_stick)
    controller.tilt_analog(melee.Button.BUTTON_C, *command.c_stick)
    controller.press_shoulder(melee.Button.BUTTON_L, command.analog_l)
    controller.press_shoulder(melee.Button.BUTTON_R, command.analog_r)
    if flush:
        controller.flush()
    return dispatch


@dataclass(frozen=True, slots=True)
class SlippiAIPolicyConfig:
    """Frozen compatibility contract for one supported official policy release."""

    source_directory: Path
    checkpoint_path: Path
    port: int
    opponent_port: int
    release: str = "medium-v2"
    name: str = DEFAULT_PLAYER_NAME
    sample_temperature: float = SAMPLE_TEMPERATURE
    policy_delay_frames: int = POLICY_DELAY_FRAMES
    console_delay_frames: int = UPSTREAM_EVAL_TWO_CONSOLE_DELAY_FRAMES
    async_inference: bool = True
    compile: bool = True
    tf_jit_compile: bool = False
    batch_steps: int = 0
    mirror: bool = False
    requested_character: str | None = None
    allow_ood_character: bool = False

    @classmethod
    def from_project_root(
        cls,
        project_root: Path,
        *,
        port: int,
        opponent_port: int,
        release: str = "medium-v2",
    ) -> Self:
        root = project_root.expanduser().resolve()
        contract = slippi_ai_release_contract(release)
        return cls(
            source_directory=root / ".e001-cache" / "slippi-ai-source",
            checkpoint_path=contract.checkpoint_path(root),
            port=port,
            opponent_port=opponent_port,
            release=release,
            policy_delay_frames=contract.policy_delay_frames,
        )

    @property
    def release_contract(self) -> SlippiAIReleaseContract:
        return slippi_ai_release_contract(self.release)

    def validate(self) -> None:
        contract = self.release_contract
        if self.port not in (1, 2, 3, 4):
            raise ValueError(f"policy port must be in 1..4, got {self.port}")
        if self.opponent_port not in (1, 2, 3, 4):
            raise ValueError(f"opponent port must be in 1..4, got {self.opponent_port}")
        if self.port == self.opponent_port:
            raise ValueError("policy port and opponent port must differ")
        if not self.name.strip():
            raise ValueError("checkpoint player name must be non-empty")
        if not isinstance(self.allow_ood_character, bool):
            raise TypeError("allow_ood_character must be a boolean")
        if self.requested_character is not None:
            if not self.requested_character or self.requested_character != self.requested_character.upper():
                raise ValueError("requested_character must be an uppercase libmelee enum name")
            if self.requested_character not in contract.supported_characters and not self.allow_ood_character:
                raise ValueError(
                    f"{contract.key} does not cover requested character {self.requested_character!r}"
                )
        if not math.isfinite(self.sample_temperature) or self.sample_temperature <= 0.0:
            raise ValueError(
                "categorical sample_temperature must be finite and greater than zero, "
                f"got {self.sample_temperature!r}"
            )
        if isinstance(self.console_delay_frames, bool) or not isinstance(self.console_delay_frames, int):
            raise ValueError("console_delay_frames must be an integer")
        if not 0 <= self.console_delay_frames <= self.policy_delay_frames:
            raise ValueError(
                "console_delay_frames must be between zero and policy_delay_frames inclusive, "
                f"got {self.console_delay_frames}"
            )
        required = {
            "policy_delay_frames": (self.policy_delay_frames, contract.policy_delay_frames),
            "async_inference": (self.async_inference, True),
            "compile": (self.compile, True),
            "tf_jit_compile": (self.tf_jit_compile, False),
            "batch_steps": (self.batch_steps, 0),
            "mirror": (self.mirror, False),
        }
        mismatches = {
            field_name: {"observed": observed, "required": expected}
            for field_name, (observed, expected) in required.items()
            if observed != expected
        }
        if mismatches:
            raise ValueError(f"{contract.key} runtime contract mismatch: {mismatches}")


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


def verify_runtime_assets(config: SlippiAIPolicyConfig) -> dict[str, Any]:
    """Verify exact source and checkpoint identities before importing either."""
    config.validate()
    contract = config.release_contract
    source = config.source_directory.expanduser().resolve()
    checkpoint = config.checkpoint_path.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"pinned Slippi-AI source is missing: {source}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"released {contract.key} checkpoint is missing: {checkpoint}")

    revision = _git_output(source, "rev-parse", "HEAD")
    if revision != SLIPPI_AI_SOURCE_REVISION:
        raise RuntimeError(f"Slippi-AI source revision mismatch: {revision} != {SLIPPI_AI_SOURCE_REVISION}")
    tracked_changes = _git_output(source, "status", "--short", "--untracked-files=all")
    if tracked_changes:
        raise RuntimeError(f"pinned Slippi-AI source has tracked modifications: {tracked_changes}")
    remote_url = _git_output(source, "remote", "get-url", "origin")
    if remote_url.removesuffix(".git") != SLIPPI_AI_REPOSITORY_URL:
        raise RuntimeError(
            f"Slippi-AI source remote mismatch: {remote_url!r} != {SLIPPI_AI_REPOSITORY_URL!r}"
        )
    branch = _git_output(source, "branch", "--show-current")
    if branch != SLIPPI_AI_SOURCE_BRANCH:
        raise RuntimeError(f"Slippi-AI source branch mismatch: {branch!r} != {SLIPPI_AI_SOURCE_BRANCH!r}")
    license_path = source / "LICENSE"
    license_sha256 = _sha256_file(license_path)
    if license_sha256 != SLIPPI_AI_LICENSE_SHA256:
        raise RuntimeError(f"Slippi-AI license hash mismatch: {license_sha256} != {SLIPPI_AI_LICENSE_SHA256}")

    checkpoint_bytes = checkpoint.stat().st_size
    checkpoint_sha256 = _sha256_file(checkpoint)
    if checkpoint_bytes != contract.checkpoint_bytes:
        raise RuntimeError(
            f"{contract.key} byte length mismatch: {checkpoint_bytes} != {contract.checkpoint_bytes}"
        )
    if checkpoint_sha256 != contract.checkpoint_sha256:
        raise RuntimeError(
            f"{contract.key} SHA-256 mismatch: {checkpoint_sha256} != {contract.checkpoint_sha256}"
        )
    return {
        "release_contract": contract.as_dict(),
        "source": {
            "repository_url": SLIPPI_AI_REPOSITORY_URL,
            "remote_url": remote_url,
            "branch": branch,
            "revision": revision,
            "directory": str(source),
            "tracked_tree_clean": True,
            "license": "MIT",
            "license_path": str(license_path),
            "license_sha256": license_sha256,
        },
        "checkpoint": {
            "release": contract.key,
            "path": str(checkpoint),
            "sha256": checkpoint_sha256,
            "byte_length": checkpoint_bytes,
        },
    }


class _ControllerCapture:
    """Capture exactly what the pinned native sender applies for one frame."""

    def __init__(self, port: int) -> None:
        self.port = port
        self._active = False
        self._buttons: dict[str, bool] = {}
        self._button_touches: dict[str, int] = {}
        self._analogs: dict[str, tuple[float, ...]] = {}
        self._analog_touches: dict[str, int] = {}
        self._flushes = 0
        self.last_trace: tuple[tuple[str, tuple[Any, ...]], ...] = ()
        self._trace: list[tuple[str, tuple[Any, ...]]] = []

    def begin_frame(self) -> None:
        if self._active:
            raise RuntimeError("controller capture frame is already active")
        self._active = True
        self._buttons = {}
        self._button_touches = {}
        self._analogs = {}
        self._analog_touches = {}
        self._flushes = 0
        self._trace = []

    def abort_frame(self) -> None:
        self._active = False
        self._trace = []

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("controller operation occurred outside a capture frame")

    def _record_button(self, button: Any, pressed: bool) -> None:
        self._require_active()
        name = _button_name(button)
        if name == "START":
            raise ValueError("START is forbidden for policy controller commands")
        if name not in _DIGITAL_BUTTON_SET:
            raise ValueError(f"native sender used unsupported digital button {name!r}")
        self._button_touches[name] = self._button_touches.get(name, 0) + 1
        self._buttons[name] = pressed
        operation = "press_button" if pressed else "release_button"
        self._trace.append((operation, (name,)))

    def press_button(self, button: Any) -> None:
        self._record_button(button, True)

    def release_button(self, button: Any) -> None:
        self._record_button(button, False)

    def tilt_analog(self, button: Any, x: Any, y: Any) -> None:
        self._require_active()
        name = _button_name(button)
        if name not in ("MAIN", "C"):
            raise ValueError(f"native sender used unsupported stick {name!r}")
        values = (
            _finite_unit_interval(x, f"native {name} stick x"),
            _finite_unit_interval(y, f"native {name} stick y"),
        )
        self._analog_touches[name] = self._analog_touches.get(name, 0) + 1
        self._analogs[name] = values
        self._trace.append(("tilt_analog", (name, *values)))

    def press_shoulder(self, button: Any, value: Any) -> None:
        self._require_active()
        name = _button_name(button)
        if name not in ("L", "R"):
            raise ValueError(f"native sender used unsupported analog shoulder {name!r}")
        shoulder = _finite_unit_interval(value, f"native analog {name}")
        self._analog_touches[name] = self._analog_touches.get(name, 0) + 1
        self._analogs[name] = (shoulder,)
        self._trace.append(("press_shoulder", (name, shoulder)))

    def flush(self) -> None:
        self._require_active()
        self._flushes += 1
        self._trace.append(("flush", ()))

    def finish_native_frame(self) -> CanonicalControllerCommand:
        self._require_active()
        self._active = False
        expected_button_touches = {name: 1 for name in DIGITAL_BUTTON_ORDER}
        if self._button_touches != expected_button_touches:
            raise AssertionError(
                f"native sender did not emit one complete digital-button frame: {self._button_touches!r}"
            )
        expected_analog_touches = {"MAIN": 1, "C": 1, "L": 1}
        if self._analog_touches != expected_analog_touches:
            raise AssertionError(
                f"native sender did not emit its complete analog frame: {self._analog_touches!r}"
            )
        if self._flushes != 0:
            raise AssertionError(f"pinned native sender unexpectedly flushed {self._flushes} times")
        if set(self._analogs) - _ANALOG_CONTROL_NAMES:
            raise AssertionError(f"unexpected captured analog controls: {self._analogs!r}")
        self.last_trace = tuple(self._trace)
        return CanonicalControllerCommand(
            main_stick=cast(tuple[float, float], self._analogs["MAIN"]),
            c_stick=cast(tuple[float, float], self._analogs["C"]),
            analog_l=self._analogs["L"][0],
            analog_r=0.0,
            buttons=tuple(name for name in DIGITAL_BUTTON_ORDER if self._buttons[name]),
        )


class _PolicyRuntime(Protocol):
    @property
    def metadata(self) -> Mapping[str, Any]: ...

    def start(self) -> None: ...

    def step(self, gamestate: Any) -> Any: ...

    def wait_current_frame(self, timeout: float = 30.0) -> float: ...

    def decode_sample_outputs(self, sample_outputs: Any) -> CanonicalControllerCommand: ...

    def diagnostics(self) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


RuntimeFactory = Callable[[SlippiAIPolicyConfig, _ControllerCapture], _PolicyRuntime]


def _activate_pinned_source(source_directory: Path) -> dict[str, Any]:
    source = source_directory.expanduser().resolve()
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

    importlib.import_module("slippi_ai")
    eval_lib = importlib.import_module("slippi_ai.eval_lib")
    saving = importlib.import_module("slippi_ai.saving")
    utils = importlib.import_module("slippi_ai.utils")
    policies = importlib.import_module("slippi_ai.policies")
    mirror_lib = importlib.import_module("slippi_ai.mirror")
    parse_libmelee = importlib.import_module("slippi_db.parse_libmelee")

    # slippi_ai is a namespace package at the pinned revision, so its
    # ``__file__`` is None.  Verify a concrete core module instead.
    module_file_value = getattr(eval_lib, "__file__", None)
    if not isinstance(module_file_value, str):
        raise RuntimeError("loaded slippi_ai.eval_lib has no concrete source path")
    module_file = Path(module_file_value).resolve()
    parser_file = Path(cast(str, parse_libmelee.__file__)).resolve()
    if not module_file.is_relative_to(source):
        raise RuntimeError(f"loaded slippi_ai outside pinned source: {module_file}")
    if not parser_file.is_relative_to(source):
        raise RuntimeError(f"loaded Slippi-AI Parser outside pinned source: {parser_file}")
    return {
        "eval_lib": eval_lib,
        "saving": saving,
        "utils": utils,
        "policies": policies,
        "mirror_lib": mirror_lib,
        "module_file": module_file,
        "parser_file": parser_file,
    }


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
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        return _json_safe(enum_value)
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except TypeError:
            pass
        except ValueError:
            pass
    return repr(value)


@dataclass(frozen=True, slots=True)
class _PolicyNameIdentity:
    requested_name: str
    effective_name: str
    requested_name_code: int
    effective_name_code: int
    rl_trained_names: tuple[str, ...]


def _validate_policy_name_identity(
    requested_name: str,
    name_map: Mapping[str, Any],
    rl_names: object,
    *,
    release: str = "medium-v2",
    expected_rl_names: tuple[str, ...] = MEDIUM_V2_RL_TRAINED_NAMES,
) -> _PolicyNameIdentity:
    """Reject a label that the pinned upstream builder would silently replace."""
    if expected_rl_names:
        if not isinstance(rl_names, (list, tuple)) or not rl_names:
            raise RuntimeError(f"{release} checkpoint has no RL-trained player-name list")
        trained_names = tuple(str(name) for name in rl_names)
    else:
        if rl_names is not None:
            raise RuntimeError(f"{release} imitation checkpoint unexpectedly declares RL player names")
        trained_names = ()
    if trained_names != expected_rl_names:
        raise RuntimeError(
            f"{release} RL-trained player-name list mismatch: {trained_names!r} != {expected_rl_names!r}"
        )
    effective_name = (
        requested_name if not trained_names or requested_name in trained_names else trained_names[0]
    )
    if effective_name != requested_name:
        raise ValueError(
            f"{release} was RL-trained only with {sorted(set(trained_names))!r}; "
            f"pinned upstream would coerce {requested_name!r} to {effective_name!r}"
        )
    if requested_name not in name_map:
        raise RuntimeError(f"checkpoint has no {requested_name!r} category in name_map")
    requested_code = int(name_map[requested_name])
    return _PolicyNameIdentity(
        requested_name=requested_name,
        effective_name=effective_name,
        requested_name_code=requested_code,
        effective_name_code=requested_code,
        rl_trained_names=trained_names,
    )


class _NativePolicyRuntime:
    def __init__(self, config: SlippiAIPolicyConfig, capture: _ControllerCapture) -> None:
        release = config.release_contract
        modules = _activate_pinned_source(config.source_directory)
        self._utils = modules["utils"]
        self._policies = modules["policies"]
        self._mirror_lib = modules["mirror_lib"]
        saving = modules["saving"]
        eval_lib = modules["eval_lib"]

        state = saving.load_state_from_disk(str(config.checkpoint_path.expanduser().resolve()))
        if not isinstance(state, dict):
            raise RuntimeError(f"upstream saving returned {type(state).__name__}, expected dict")
        checkpoint_config = state.get("config")
        if not isinstance(checkpoint_config, dict):
            raise RuntimeError(f"{release.key} checkpoint has no configuration dictionary")
        checkpoint_delay = checkpoint_config.get("policy", {}).get("delay")
        if checkpoint_delay != config.policy_delay_frames:
            raise RuntimeError(
                f"{release.key} policy delay mismatch: {checkpoint_delay!r} != {config.policy_delay_frames}"
            )
        platform = saving.get_platform(checkpoint_config)
        if platform is not self._policies.Platform.TF:
            raise RuntimeError(f"{release.key} requires the TensorFlow runtime, got {platform!r}")
        name_map = state.get("name_map")
        if not isinstance(name_map, dict):
            raise RuntimeError(f"{release.key} checkpoint has no player-name category map")
        name_identity = _validate_policy_name_identity(
            config.name,
            name_map,
            eval_lib.get_name_from_rl_state(state),
            release=release.key,
            expected_rl_names=release.rl_trained_names,
        )
        dataset_config = checkpoint_config.get("dataset", {})
        dataset_allowed_characters = dataset_config.get("allowed_characters")
        if not isinstance(dataset_allowed_characters, str):
            raise RuntimeError(f"{release.key} checkpoint has no dataset controlled-character declaration")
        dataset_characters = tuple(
            character.strip().upper()
            for character in dataset_allowed_characters.split(",")
            if character.strip()
        )
        if dataset_characters != release.supported_characters:
            raise RuntimeError(
                f"{release.key} dataset controlled-character list mismatch: "
                f"{dataset_characters!r} != {release.supported_characters!r}"
            )
        allowed_opponents = dataset_config.get("allowed_opponents")
        if allowed_opponents != release.allowed_opponents:
            raise RuntimeError(
                f"{release.key} opponent declaration mismatch: "
                f"{allowed_opponents!r} != {release.allowed_opponents!r}"
            )
        if release.checkpoint_kind == "self-play-rl":
            rl_agent_config = state.get("rl_config", {}).get("agent", {})
            checkpoint_characters = tuple(
                str(getattr(character, "name", character)) for character in rl_agent_config.get("char", ())
            )
        elif release.checkpoint_kind == "imitation":
            if "rl_config" in state or "agent_config" in state:
                raise RuntimeError(f"{release.key} release contract requires an imitation checkpoint")
            checkpoint_characters = dataset_characters
        else:
            raise AssertionError(f"unknown checkpoint kind: {release.checkpoint_kind!r}")
        if checkpoint_characters != release.supported_characters:
            raise RuntimeError(
                f"{release.key} supported-character list mismatch: "
                f"{checkpoint_characters!r} != {release.supported_characters!r}"
            )
        checkpoint_tag = checkpoint_config.get("tag")
        if checkpoint_tag != release.checkpoint_tag:
            raise RuntimeError(
                f"{release.key} checkpoint tag mismatch: {checkpoint_tag!r} != {release.checkpoint_tag!r}"
            )
        checkpoint_version = checkpoint_config.get("version")
        if checkpoint_version != release.checkpoint_config_version:
            raise RuntimeError(
                f"{release.key} checkpoint config version mismatch: "
                f"{checkpoint_version!r} != {release.checkpoint_config_version!r}"
            )
        checkpoint_step = state.get("step")
        if checkpoint_step != release.checkpoint_step:
            raise RuntimeError(
                f"{release.key} checkpoint step mismatch: {checkpoint_step!r} != {release.checkpoint_step!r}"
            )

        policy = eval_lib.build_agent(
            state=state,
            controller=capture,
            port=config.port,
            opponent_port=config.opponent_port,
            name=config.name,
            mirror=config.mirror,
            console_delay=config.console_delay_frames,
            async_inference=config.async_inference,
            sample_temperature=config.sample_temperature,
            compile=config.compile,
            batch_steps=config.batch_steps,
            tf={"jit_compile": config.tf_jit_compile},
        )
        if not isinstance(policy, eval_lib.Agent):
            raise RuntimeError(f"upstream builder returned unexpected type {type(policy).__name__}")
        if type(policy._agent).__name__ != "AsyncDelayedAgent":
            raise RuntimeError("upstream runtime did not preserve asynchronous inference")
        if policy._agent.policy.delay != config.policy_delay_frames:
            raise RuntimeError(f"instantiated policy delay mismatch: {policy._agent.policy.delay}")
        effective_delay = policy._agent.delay
        expected_effective_delay = config.policy_delay_frames - config.console_delay_frames
        if effective_delay != expected_effective_delay:
            raise RuntimeError(
                f"effective policy delay mismatch: {effective_delay} != {expected_effective_delay}"
            )
        actual_name_codes = np.asarray(policy._agent.name_code)
        if actual_name_codes.shape != (1,):
            raise RuntimeError(
                f"{release.key} effective player-name code must have shape (1,), "
                f"got {actual_name_codes.shape}"
            )
        actual_name_code = int(actual_name_codes[0])
        if actual_name_code != name_identity.effective_name_code:
            raise RuntimeError(
                "pinned upstream changed the effective player-name code: "
                f"{actual_name_code} != {name_identity.effective_name_code}"
            )

        tree = importlib.import_module("tree")
        loaded_variables = list(tree.flatten(policy._agent.policy.variables))
        saved_variables = list(tree.flatten(state["state"]["policy"]))
        if len(loaded_variables) != release.variable_count:
            raise RuntimeError(
                f"{release.key} variable count mismatch: {len(loaded_variables)} != {release.variable_count}"
            )
        if len(saved_variables) != len(loaded_variables):
            raise RuntimeError(
                f"{release.key} saved and instantiated variable trees differ in leaf count: "
                f"{len(saved_variables)} != {len(loaded_variables)}"
            )
        parameter_count = 0
        nonfinite_variables: list[int] = []
        assignment_mismatches: list[int] = []
        shape_mismatches: list[int] = []
        dtype_mismatches: list[int] = []
        variable_shapes: list[list[int]] = []
        variable_dtypes: list[str] = []
        for index, (variable, saved_value) in enumerate(zip(loaded_variables, saved_variables, strict=True)):
            loaded_array = np.asarray(variable.numpy())
            saved_array = np.asarray(saved_value)
            variable_shapes.append(list(loaded_array.shape))
            variable_dtypes.append(str(loaded_array.dtype))
            parameter_count += int(loaded_array.size)
            if loaded_array.shape != saved_array.shape:
                shape_mismatches.append(index)
                continue
            if loaded_array.dtype != saved_array.dtype:
                dtype_mismatches.append(index)
            if not np.isfinite(loaded_array).all():
                nonfinite_variables.append(index)
            if not np.array_equal(loaded_array, saved_array):
                assignment_mismatches.append(index)
        if parameter_count != release.parameter_count:
            raise RuntimeError(
                f"{release.key} parameter count mismatch: {parameter_count} != {release.parameter_count}"
            )
        if shape_mismatches or dtype_mismatches or assignment_mismatches or nonfinite_variables:
            raise RuntimeError(
                f"{release.key} parameter restoration mismatch: "
                f"shape={shape_mismatches}, dtype={dtype_mismatches}, "
                f"assignment={assignment_mismatches}, nonfinite={nonfinite_variables}"
            )

        self._policy = policy
        self._mirror = config.mirror
        self._started = False
        self.metadata = {
            "release_contract": release.as_dict(),
            "saving_loader": "slippi_ai.saving.load_state_from_disk",
            "runtime_class": f"{type(policy).__module__}.{type(policy).__name__}",
            "delayed_runtime_class": (f"{type(policy._agent).__module__}.{type(policy._agent).__name__}"),
            "parser_class": "slippi_db.parse_libmelee.Parser",
            "parser_source_path": str(modules["parser_file"]),
            "observation_filter_class": (
                f"{type(policy._observation_filter).__module__}.{type(policy._observation_filter).__name__}"
            ),
            "controller_head_class": (
                f"{type(policy._agent.policy.controller_head).__module__}."
                f"{type(policy._agent.policy.controller_head).__name__}"
            ),
            "platform": platform.value,
            "checkpoint_config": _json_safe(checkpoint_config),
            "checkpoint_step": _json_safe(checkpoint_step),
            "checkpoint_name_map_size": len(name_map),
            "requested_name": name_identity.requested_name,
            "requested_name_code": name_identity.requested_name_code,
            "effective_name": name_identity.effective_name,
            "effective_name_code": actual_name_code,
            "rl_trained_names": list(name_identity.rl_trained_names),
            # Backwards-compatible fields now describe the effective identity.
            "selected_name": name_identity.effective_name,
            "selected_name_code": actual_name_code,
            "supported_characters": list(checkpoint_characters),
            "dataset_supported_characters": list(dataset_characters),
            "allowed_opponents": allowed_opponents,
            "supported_rendered_stages": list(MEDIUM_V2_SUPPORTED_STAGES),
            "character_embedding_size": MEDIUM_V2_CHARACTER_EMBEDDING_SIZE,
            "stage_embedding_size": MEDIUM_V2_STAGE_EMBEDDING_SIZE,
            "variable_count": len(loaded_variables),
            "parameter_count": parameter_count,
            "variable_shapes": variable_shapes,
            "variable_dtypes": variable_dtypes,
            "state_assignment": {
                "shape_mismatches": shape_mismatches,
                "dtype_mismatches": dtype_mismatches,
                "value_mismatches": assignment_mismatches,
                "nonfinite_variables": nonfinite_variables,
            },
            "name_code": actual_name_code,
            "recurrent_state_class": type(policy._agent.hidden_state).__name__,
            "policy_delay_frames": policy._agent.policy.delay,
            "console_delay_frames": config.console_delay_frames,
            "effective_policy_delay_frames": effective_delay,
            "effective_delay_equation": "policy_delay_frames - console_delay_frames",
            "native_analog_shoulder": "L",
            "native_analog_r_emitted": False,
        }

    def start(self) -> None:
        if self._started:
            raise RuntimeError("native Slippi-AI runtime is already started")
        self._policy.start()
        self._started = True

    def step(self, gamestate: Any) -> Any:
        if not self._started:
            raise RuntimeError("native Slippi-AI runtime has not been started")
        return self._policy.step(gamestate)

    def wait_current_frame(self, timeout: float = 30.0) -> float:
        """Wait for the recurrent update submitted by the immediately preceding step."""

        if timeout <= 0:
            raise ValueError("current-frame inference timeout must be positive")
        delayed = self._policy._agent
        state_queue = getattr(delayed, "_state_queue", None)
        worker = getattr(delayed, "_worker_thread", None)
        if state_queue is None or worker is None:
            raise RuntimeError("pinned Slippi-AI async inference queue or worker is unavailable")
        started_at = time.perf_counter()
        deadline = time.monotonic() + timeout
        # Queue.join() has no timeout and can deadlock if the upstream worker
        # faults. Wait on the same standard-library condition while checking
        # the deadline and worker liveness.
        with state_queue.all_tasks_done:
            while state_queue.unfinished_tasks:
                if not worker.is_alive():
                    raise RuntimeError(
                        "pinned Slippi-AI inference worker stopped before completing the current frame"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("pinned Slippi-AI current-frame inference timed out")
                state_queue.all_tasks_done.wait(timeout=min(0.05, remaining))
        return time.perf_counter() - started_at

    def decode_sample_outputs(self, sample_outputs: Any) -> CanonicalControllerCommand:
        action = sample_outputs.controller_state
        if self._policy._agent.policy.platform is self._policies.Platform.JAX:
            jax = importlib.import_module("jax")
            action = jax.device_get(action)
        action = self._utils.map_single_structure(lambda value: value[0], action)
        native = self._policy._agent.decode_controller(action)
        if self._mirror:
            native = self._mirror_lib.mirror_controller(native)
        return capture_native_controller_command(native)

    def diagnostics(self) -> Mapping[str, Any]:
        delayed = self._policy._agent
        result: dict[str, Any] = {
            "worker_running": bool(getattr(delayed, "_worker_thread", None)),
            "effective_policy_delay_frames": int(delayed.delay),
        }
        for name in ("state_queue_profiler", "step_profiler"):
            profiler = getattr(delayed, name, None)
            if profiler is None:
                continue
            calls = int(getattr(profiler, "num_calls", 0))
            result[name] = {
                "calls": calls,
                "cumulative_seconds": float(getattr(profiler, "cumtime", 0.0)),
                "last_seconds": (float(profiler.last_time) if hasattr(profiler, "last_time") else None),
            }
        return result

    def close(self) -> None:
        if self._started:
            self._policy.stop()
            self._started = False


def _default_runtime_factory(config: SlippiAIPolicyConfig, capture: _ControllerCapture) -> _PolicyRuntime:
    return _NativePolicyRuntime(config, capture)


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
        self.minimum_seconds = seconds if self.minimum_seconds is None else min(self.minimum_seconds, seconds)
        self.maximum_seconds = seconds if self.maximum_seconds is None else max(self.maximum_seconds, seconds)

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "count": self.count,
            "total_seconds": self.total_seconds,
            "mean_seconds": self.total_seconds / self.count if self.count else None,
            "minimum_seconds": self.minimum_seconds,
            "maximum_seconds": self.maximum_seconds,
            "last_seconds": self.last_seconds,
        }


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
    capture_decoder_assertions: int = 0
    capture_decoder_mismatches: int = 0
    current_frame_inference_barriers: int = 0
    current_frame_inference_barrier_timing: _TimingAccumulator = field(default_factory=_TimingAccumulator)
    awaiting_reset_frame: bool = False
    reset_history: list[dict[str, Any]] = field(default_factory=list)
    timing: _TimingAccumulator = field(default_factory=_TimingAccumulator)


class SlippiAIPolicySession:
    """One ordered game session backed by a pinned official Slippi-AI release."""

    def __init__(
        self,
        config: SlippiAIPolicyConfig,
        *,
        agent_factory: RuntimeFactory | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self._asset_identity = verify_runtime_assets(config)
        self._factory = agent_factory or _default_runtime_factory
        self._capture = _ControllerCapture(config.port)
        self._runtime: _PolicyRuntime | None = None
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
            raise RuntimeError("Slippi-AI policy session is closed")
        if self._state.fault is not None:
            raise RuntimeError(f"Slippi-AI policy session is faulted: {self._state.fault}")

    def start(self) -> None:
        with self._lock:
            self._require_open()
            if self._state.started:
                return
            runtime = self._factory(self.config, self._capture)
            try:
                runtime.start()
            except BaseException:
                runtime.close()
                raise
            self._runtime = runtime
            self._runtime_metadata = dict(runtime.metadata)
            self._state.started = True

    def _validate_next_frame(self, frame: int) -> None:
        if self._state.frames_in_generation == 0:
            if frame != -123:
                raise ValueError(f"first policy frame must be -123, got {frame}")
            self._state.awaiting_reset_frame = False
            return
        assert self._state.last_frame is not None
        if frame == -123:
            self._record_reset("observed-frame--123")
            self._state.awaiting_reset_frame = False
            return
        expected = self._state.last_frame + 1
        if frame != expected:
            raise ValueError(
                f"policy frames must be submitted exactly once in order: expected {expected}, got {frame}"
            )

    def step(self, gamestate: Any) -> CanonicalControllerCommand:
        frame_value = getattr(gamestate, "frame", None)
        if frame_value is None:
            raise TypeError("gamestate.frame must be an integer, got None")
        if isinstance(frame_value, bool):
            raise TypeError("gamestate.frame must be an integer, not bool")
        try:
            frame = int(frame_value)
        except (TypeError, ValueError) as error:
            raise TypeError(f"gamestate.frame must be an integer, got {frame_value!r}") from error
        if frame_value != frame:
            raise TypeError(f"gamestate.frame must be integral, got {frame_value!r}")

        with self._lock:
            self._require_open()
            self._validate_next_frame(frame)
            self.start()
            runtime = cast(_PolicyRuntime, self._runtime)
            self._capture.begin_frame()
            started_at = time.perf_counter()
            try:
                sample_outputs = runtime.step(gamestate)
                barrier_seconds = runtime.wait_current_frame()
                self._state.current_frame_inference_barrier_timing.add(barrier_seconds)
                self._state.current_frame_inference_barriers += 1
                captured = self._capture.finish_native_frame()
                decoded = runtime.decode_sample_outputs(sample_outputs)
                self._state.capture_decoder_assertions += 1
                if captured != decoded:
                    self._state.capture_decoder_mismatches += 1
                    raise AssertionError(
                        "native controller capture does not exactly match the checkpoint decoder: "
                        f"captured={captured!r}, decoded={decoded!r}"
                    )
            except BaseException as error:
                self._capture.abort_frame()
                self._state.fault = f"{type(error).__name__}: {error}"
                raise
            elapsed = time.perf_counter() - started_at
            self._state.timing.add(elapsed)
            self._state.frames_total += 1
            self._state.frames_in_generation += 1
            if self._state.first_frame is None:
                self._state.first_frame = frame
            self._state.last_frame = frame
            return captured

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

    def reset(self, reason: str = "explicit") -> None:
        """Require a new frame -123 while preserving upstream multi-game state."""
        if not reason:
            raise ValueError("reset reason must be non-empty")
        with self._lock:
            self._require_open()
            if self._state.frames_in_generation == 0:
                raise RuntimeError("cannot reset before the current game has received a frame")
            self._record_reset(reason)
            self._state.awaiting_reset_frame = True

    def metadata(self) -> dict[str, Any]:
        with self._lock:
            release = self.config.release_contract
            effective_name = str(self._runtime_metadata.get("effective_name", self.config.name))
            return {
                "schema_version": "melee_policy.slippi_ai_policy.metadata.v1",
                **self._asset_identity,
                "capabilities": slippi_ai_release_capabilities(release.key),
                "runtime_contract": {
                    "release": release.key,
                    "player_name": effective_name,
                    "requested_player_name": self.config.name,
                    "effective_player_name": effective_name,
                    "port": self.config.port,
                    "opponent_port": self.config.opponent_port,
                    "sample_temperature": self.config.sample_temperature,
                    "categorical_sampling_framework": "tensorflow",
                    "policy_delay_frames": self.config.policy_delay_frames,
                    "console_delay_frames": self.config.console_delay_frames,
                    "rendered_console_delay_frames": self.config.console_delay_frames,
                    "effective_policy_delay_frames": (
                        self.config.policy_delay_frames - self.config.console_delay_frames
                    ),
                    "effective_delay_equation": "policy_delay_frames - console_delay_frames",
                    "async_inference": self.config.async_inference,
                    "compile": self.config.compile,
                    "tf_jit_compile": self.config.tf_jit_compile,
                    "batch_steps": self.config.batch_steps,
                    "mirror": self.config.mirror,
                    "first_required_frame": -123,
                    "frame_order": "strictly consecutive, exactly once",
                    "render_advance_barrier": (
                        "current recurrent inference completes before the next rendered frame"
                    ),
                },
                "controller_contract": {
                    "digital_buttons": list(DIGITAL_BUTTON_ORDER),
                    "forbidden_buttons": ["START"],
                    "independent_analog_shoulders": ["L", "R"],
                    "native_checkpoint_analog_shoulders": ["L"],
                    "native_checkpoint_analog_r_value": 0.0,
                    "native_sender_flushes": False,
                    "adapter_flush_mode": "call-site-configurable",
                    "adapter_flushes_by_default": True,
                    "source_exact_deferred_flush_supported": True,
                    "capture_vs_native_decoder": "exact",
                },
                "upstream_runtime": _json_safe(self._runtime_metadata),
                "character_transfer": {
                    "requested_character": self.config.requested_character,
                    "checkpoint_declared_characters": list(release.supported_characters),
                    "requested_character_covered_by_training": (
                        self.config.requested_character in release.supported_characters
                        if self.config.requested_character is not None
                        else None
                    ),
                    "forced_ood_character_transfer": (
                        self.config.requested_character is not None
                        and self.config.requested_character not in release.supported_characters
                        and self.config.allow_ood_character
                    ),
                    "checkpoint_bytes_unchanged": True,
                    "native_inference_path_unchanged": True,
                },
            }

    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            runtime_diagnostics = dict(self._runtime.diagnostics()) if self._runtime is not None else {}
            expected_next_frame = None if self._state.last_frame is None else self._state.last_frame + 1
            return {
                "schema_version": "melee_policy.slippi_ai_policy.diagnostics.v1",
                "started": self._state.started,
                "closed": self._state.closed,
                "fault": self._state.fault,
                "generation": self._state.generation,
                "resets": self._state.resets,
                "frames_total": self._state.frames_total,
                "frames_in_generation": self._state.frames_in_generation,
                "first_frame": self._state.first_frame,
                "last_frame": self._state.last_frame,
                "expected_next_frame": expected_next_frame,
                "awaiting_reset_frame": self._state.awaiting_reset_frame,
                "capture_decoder_assertions": self._state.capture_decoder_assertions,
                "capture_decoder_mismatches": self._state.capture_decoder_mismatches,
                "current_frame_inference_barriers": self._state.current_frame_inference_barriers,
                "current_frame_inference_barrier_every_frame": (
                    self._state.current_frame_inference_barriers == self._state.frames_total
                ),
                "current_frame_inference_barrier_timing": (
                    self._state.current_frame_inference_barrier_timing.as_dict()
                ),
                "step_timing": self._state.timing.as_dict(),
                "reset_history": list(self._state.reset_history),
                "last_native_capture": [
                    [operation, list(arguments)] for operation, arguments in self._capture.last_trace
                ],
                "upstream": _json_safe(runtime_diagnostics),
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
                raise RuntimeError("failed to close the Slippi-AI policy runtime") from close_error


def ordered_buttons(buttons: Iterable[str]) -> tuple[str, ...]:
    """Return a canonical digital-button tuple, validating every member."""
    return CanonicalControllerCommand(
        main_stick=(0.5, 0.5),
        c_stick=(0.5, 0.5),
        analog_l=0.0,
        analog_r=0.0,
        buttons=tuple(buttons),
    ).buttons
