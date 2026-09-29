from __future__ import annotations

import dataclasses
import io
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import melee_policy.integration.slippi_ai_policy as policy_module
from melee_policy.integration.slippi_ai_policy import (
    DEFAULT_PLAYER_NAME,
    DIGITAL_BUTTON_ORDER,
    DK_D18_IMITATION_V2_CHECKPOINT_BYTES,
    DK_D18_IMITATION_V2_CHECKPOINT_SHA256,
    DOC_D18_IMITATION_V3_CHECKPOINT_BYTES,
    DOC_D18_IMITATION_V3_CHECKPOINT_SHA256,
    EFFECTIVE_POLICY_DELAY_FRAMES,
    MEDIUM_V2_ALLOWED_OPPONENTS,
    MEDIUM_V2_CHARACTER_EMBEDDING_SIZE,
    MEDIUM_V2_CHECKPOINT_BYTES,
    MEDIUM_V2_CHECKPOINT_SHA256,
    MEDIUM_V2_RL_TRAINED_NAMES,
    MEDIUM_V2_STAGE_EMBEDDING_SIZE,
    MEDIUM_V2_SUPPORTED_CHARACTERS,
    MEDIUM_V2_SUPPORTED_STAGES,
    POLICY_DELAY_FRAMES,
    SLIPPI_AI_SOURCE_REVISION,
    CanonicalControllerCommand,
    ControllerDispatchRecord,
    SlippiAIPolicyConfig,
    SlippiAIPolicySession,
    capture_native_controller_command,
    describe_controller_dispatch,
    medium_v2_capabilities,
    send_canonical_controller,
    slippi_ai_release_capabilities,
    slippi_ai_release_contract,
    verify_runtime_assets,
)


def _native_controller(*, main_x: float = 0.25) -> Any:
    from slippi_ai import types  # type: ignore[import-not-found]

    return types.Controller(
        main_stick=types.Stick(np.float32(main_x), np.float32(0.75)),
        c_stick=types.Stick(np.float32(0.5), np.float32(0.125)),
        shoulder=np.float32(0.6),
        buttons=types.Buttons(
            A=np.bool_(True),
            B=np.bool_(False),
            X=np.bool_(False),
            Y=np.bool_(True),
            Z=np.bool_(False),
            L=np.bool_(True),
            R=np.bool_(False),
            D_UP=np.bool_(True),
        ),
    )


def _send_native_controller(controller: Any, native: Any) -> None:
    from slippi_ai.controller_lib import send_controller  # type: ignore[import-not-found]

    send_controller(controller, native)


def test_native_capture_is_complete_and_exactly_matches_direct_decode() -> None:
    native = _native_controller()
    capture = policy_module._ControllerCapture(port=2)
    capture.begin_frame()
    _send_native_controller(capture, native)
    captured = capture.finish_native_frame()

    assert captured == capture_native_controller_command(native)
    assert captured == CanonicalControllerCommand(
        main_stick=(0.25, 0.75),
        c_stick=(0.5, 0.125),
        analog_l=float(np.float32(0.6)),
        analog_r=0.0,
        buttons=("A", "Y", "L", "D_UP"),
    )
    assert len(capture.last_trace) == 11
    assert [entry[1][0] for entry in capture.last_trace[:8]] == list(DIGITAL_BUTTON_ORDER)


def test_native_capture_rejects_start_and_incomplete_frames() -> None:
    import melee

    capture = policy_module._ControllerCapture(port=1)
    capture.begin_frame()
    with pytest.raises(ValueError, match="START is forbidden"):
        capture.press_button(melee.Button.BUTTON_START)
    capture.abort_frame()

    capture.begin_frame()
    capture.press_button(melee.Button.BUTTON_A)
    with pytest.raises(AssertionError, match="complete digital-button frame"):
        capture.finish_native_frame()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.01, 1.01])
def test_canonical_command_rejects_nonfinite_or_out_of_range_values(value: float) -> None:
    with pytest.raises(ValueError):
        CanonicalControllerCommand(
            main_stick=(value, 0.5),
            c_stick=(0.5, 0.5),
            analog_l=0.0,
            analog_r=0.0,
        )


def test_canonical_command_forbids_start_and_canonicalizes_button_order() -> None:
    with pytest.raises(ValueError, match="START is forbidden"):
        CanonicalControllerCommand(
            main_stick=(0.5, 0.5),
            c_stick=(0.5, 0.5),
            analog_l=0.0,
            analog_r=0.0,
            buttons=("START",),
        )

    command = CanonicalControllerCommand(
        main_stick=(0.5, 0.5),
        c_stick=(0.5, 0.5),
        analog_l=0.0,
        analog_r=0.0,
        buttons=("D_UP", "B", "A"),
    )
    assert command.buttons == ("A", "B", "D_UP")


class _RecordingController:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def press_button(self, *arguments: Any) -> None:
        self.calls.append(("press_button", arguments))

    def release_button(self, *arguments: Any) -> None:
        self.calls.append(("release_button", arguments))

    def tilt_analog(self, *arguments: Any) -> None:
        self.calls.append(("tilt_analog", arguments))

    def press_shoulder(self, *arguments: Any) -> None:
        self.calls.append(("press_shoulder", arguments))

    def flush(self) -> None:
        self.calls.append(("flush", ()))


def test_send_canonical_controller_applies_full_frame_and_flushes_once() -> None:
    import melee

    command = CanonicalControllerCommand(
        main_stick=(0.2, 0.8),
        c_stick=(0.1, 0.9),
        analog_l=0.3,
        analog_r=0.7,
        buttons=("A", "R", "D_UP"),
    )
    controller = _RecordingController()
    dispatch = send_canonical_controller(controller, command)

    digital_calls = controller.calls[:8]
    assert [call[1][0] for call in digital_calls] == [
        getattr(melee.Button, f"BUTTON_{name}") for name in DIGITAL_BUTTON_ORDER
    ]
    assert [call[0] for call in digital_calls] == [
        "press_button",
        "release_button",
        "release_button",
        "release_button",
        "release_button",
        "release_button",
        "press_button",
        "press_button",
    ]
    assert controller.calls[8] == (
        "tilt_analog",
        (melee.Button.BUTTON_MAIN, 0.2, 0.8),
    )
    assert controller.calls[9] == (
        "tilt_analog",
        (melee.Button.BUTTON_C, 0.1, 0.9),
    )
    assert controller.calls[10] == (
        "press_shoulder",
        (melee.Button.BUTTON_L, 0.3),
    )
    assert controller.calls[11] == (
        "press_shoulder",
        (melee.Button.BUTTON_R, 0.7),
    )
    assert controller.calls[12:] == [("flush", ())]
    assert isinstance(dispatch, ControllerDispatchRecord)
    assert dispatch.canonical_command == command
    assert dispatch.upstream_native_sender_flush_count == 0
    assert dispatch.project_adapter_flush_count == 1


def test_dispatch_description_separates_canonical_and_single_corrected_pipe_values() -> None:
    command = CanonicalControllerCommand(
        main_stick=(0.0, 0.5),
        c_stick=(1.0, 0.25),
        analog_l=0.0,
        analog_r=1.0,
        buttons=("A", "L", "D_UP"),
    )
    dispatch = describe_controller_dispatch(command)

    assert dispatch.canonical_command == command
    assert dispatch.pipe_main_stick == ((-80 + 0.1) / 254 + 0.5, 0.1 / 254 + 0.5)
    assert dispatch.pipe_c_stick == ((80 + 0.1) / 254 + 0.5, (-40 + 0.1) / 254 + 0.5)
    assert dispatch.pipe_analog_l == 0.1 / 255
    assert dispatch.pipe_analog_r == (140 + 0.1) / 255
    assert dispatch.button_states == (
        ("A", True),
        ("B", False),
        ("X", False),
        ("Y", False),
        ("Z", False),
        ("L", True),
        ("R", False),
        ("D_UP", True),
    )

    serialized = dispatch.as_dict()
    json.dumps(serialized, sort_keys=True, allow_nan=False)
    assert serialized["canonical"] == command.as_dict()
    assert serialized["pipe"]["main_stick"] == list(dispatch.pipe_main_stick)
    assert serialized["libmelee"] == {
        "version": "0.47.3",
        "fix_analog_inputs": True,
        "analog_correction_applications": 1,
    }
    assert serialized["flush"] == {
        "upstream_native_sender_count": 0,
        "project_adapter_count": 1,
        "project_boundary_deliberately_flushes_once": True,
    }
    assert "must not be interpreted as processed replay values" in serialized["interpretation"]


def test_libmelee_default_correction_round_trips_every_declared_raw_grid_value() -> None:
    from melee.controller import fix_analog_stick, fix_analog_trigger

    for raw_axis in range(-80, 81):
        canonical = raw_axis / 160 + 0.5
        pipe_value = fix_analog_stick(canonical)
        assert math.floor((pipe_value - 0.5) * 254) == raw_axis

    for raw_trigger in range(141):
        canonical = raw_trigger / 140
        pipe_value = fix_analog_trigger(canonical)
        assert math.floor(pipe_value * 255) == raw_trigger


def test_dispatch_applies_analog_correction_once_not_twice() -> None:
    from melee.controller import fix_analog_stick, fix_analog_trigger

    command = CanonicalControllerCommand(
        main_stick=(0.2, 0.8),
        c_stick=(0.3, 0.7),
        analog_l=0.3,
        analog_r=0.7,
    )
    dispatch = describe_controller_dispatch(command)

    expected_main_x = fix_analog_stick(command.main_stick[0])
    double_main_x = fix_analog_stick(expected_main_x)
    expected_analog_l = fix_analog_trigger(command.analog_l)
    double_analog_l = fix_analog_trigger(expected_analog_l)
    assert dispatch.pipe_main_stick[0] == expected_main_x
    assert dispatch.pipe_main_stick[0] != double_main_x
    assert dispatch.pipe_analog_l == expected_analog_l
    assert dispatch.pipe_analog_l != double_analog_l


def test_actual_libmelee_pipe_order_matches_dispatch_record_and_flushes_once() -> None:
    import melee
    from melee.controller import ControllerState

    pipe = io.StringIO()
    controller = object.__new__(melee.Controller)
    controller.pipe = pipe
    controller.port = 1
    controller.current = ControllerState()
    controller.prev = ControllerState()
    controller.logger = None
    controller._fix_analog_inputs = True
    command = CanonicalControllerCommand(
        main_stick=(0.2, 0.8),
        c_stick=(0.1, 0.9),
        analog_l=0.3,
        analog_r=0.7,
        buttons=("A", "R", "D_UP"),
    )

    dispatch = send_canonical_controller(controller, command)
    actual_commands = tuple(pipe.getvalue().splitlines(keepends=True))

    assert controller._fix_analog_inputs is True
    assert actual_commands == dispatch.pipe_commands
    assert actual_commands[:8] == (
        "PRESS A\n",
        "RELEASE B\n",
        "RELEASE X\n",
        "RELEASE Y\n",
        "RELEASE Z\n",
        "RELEASE L\n",
        "PRESS R\n",
        "PRESS D_UP\n",
    )
    assert actual_commands[-1] == "FLUSH\n"
    assert actual_commands.count("FLUSH\n") == 1
    controller.disconnect()


def test_source_exact_dispatch_defers_flush_to_the_next_console_step() -> None:
    import melee
    from melee.controller import ControllerState

    pipe = io.StringIO()
    controller = object.__new__(melee.Controller)
    controller.pipe = pipe
    controller.port = 1
    controller.current = ControllerState()
    controller.prev = ControllerState()
    controller.logger = None
    controller._fix_analog_inputs = True
    command = CanonicalControllerCommand(
        main_stick=(0.25, 0.75),
        c_stick=(0.5, 0.125),
        analog_l=0.6,
        analog_r=0.0,
        buttons=("A", "Y", "L", "D_UP"),
    )

    dispatch = send_canonical_controller(controller, command, flush=False)
    actual_commands = tuple(pipe.getvalue().splitlines(keepends=True))

    assert actual_commands == dispatch.pipe_commands
    assert actual_commands[:8] == (
        "PRESS A\n",
        "RELEASE B\n",
        "RELEASE X\n",
        "PRESS Y\n",
        "RELEASE Z\n",
        "PRESS L\n",
        "RELEASE R\n",
        "PRESS D_UP\n",
    )
    assert actual_commands[-1].startswith("SET R ")
    assert "FLUSH\n" not in actual_commands
    assert dispatch.upstream_native_sender_flush_count == 0
    assert dispatch.project_adapter_flush_count == 0
    assert dispatch.as_dict()["flush"] == {
        "upstream_native_sender_count": 0,
        "project_adapter_count": 0,
        "project_boundary_deliberately_flushes_once": False,
    }
    controller.disconnect()


def _fake_asset_identity() -> dict[str, Any]:
    return {
        "source": {
            "repository_url": "https://github.com/vladfi1/slippi-ai",
            "revision": SLIPPI_AI_SOURCE_REVISION,
            "tracked_tree_clean": True,
        },
        "checkpoint": {
            "release": "medium-v2",
            "sha256": MEDIUM_V2_CHECKPOINT_SHA256,
            "byte_length": MEDIUM_V2_CHECKPOINT_BYTES,
        },
    }


class _FakeRuntime:
    def __init__(self, capture: Any, *, mismatch: bool = False) -> None:
        self.capture = capture
        self.mismatch = mismatch
        self.metadata = {
            "parser_class": "slippi_db.parse_libmelee.Parser",
            "policy_delay_frames": POLICY_DELAY_FRAMES,
            "effective_policy_delay_frames": EFFECTIVE_POLICY_DELAY_FRAMES,
        }
        self.started = False
        self.closed = False
        self.frames: list[int] = []
        self.barrier_frames: list[int] = []

    def start(self) -> None:
        assert not self.started
        self.started = True

    def step(self, gamestate: Any) -> Any:
        assert self.started
        self.frames.append(int(gamestate.frame))
        native = _native_controller()
        _send_native_controller(self.capture, native)
        return native

    def wait_current_frame(self, timeout: float = 30.0) -> float:
        assert timeout > 0
        self.barrier_frames.append(self.frames[-1])
        return 0.0

    def decode_sample_outputs(self, sample_outputs: Any) -> CanonicalControllerCommand:
        command = capture_native_controller_command(sample_outputs)
        if not self.mismatch:
            return command
        return CanonicalControllerCommand(
            main_stick=(0.5, command.main_stick[1]),
            c_stick=command.c_stick,
            analog_l=command.analog_l,
            analog_r=command.analog_r,
            buttons=command.buttons,
        )

    def diagnostics(self) -> dict[str, Any]:
        return {"worker_running": self.started, "submitted_frames": list(self.frames)}

    def close(self) -> None:
        self.started = False
        self.closed = True


def _fake_config(tmp_path: Path) -> SlippiAIPolicyConfig:
    return SlippiAIPolicyConfig(
        source_directory=tmp_path / "source",
        checkpoint_path=tmp_path / "medium-v2",
        port=1,
        opponent_port=2,
    )


def test_session_requires_minus_123_and_preserves_every_frame_and_native_resets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(policy_module, "verify_runtime_assets", lambda _config: _fake_asset_identity())
    runtimes: list[_FakeRuntime] = []

    def factory(_config: SlippiAIPolicyConfig, capture: Any) -> _FakeRuntime:
        runtime = _FakeRuntime(capture)
        runtimes.append(runtime)
        return runtime

    session = SlippiAIPolicySession(_fake_config(tmp_path), agent_factory=factory)
    with pytest.raises(ValueError, match="first policy frame must be -123"):
        session.step(SimpleNamespace(frame=-122))
    assert not runtimes

    first = session.step(SimpleNamespace(frame=-123))
    second = session.step(SimpleNamespace(frame=-122))
    assert first == second
    with pytest.raises(ValueError, match="expected -121, got -120"):
        session.step(SimpleNamespace(frame=-120))
    assert runtimes[0].frames == [-123, -122]

    # A later -123 is passed to the same upstream runtime.  That is the
    # published Parser/filter/recurrent reset behavior and retains its delay queue.
    session.step(SimpleNamespace(frame=-123))
    session.step(SimpleNamespace(frame=-122))
    diagnostics = session.diagnostics()
    assert diagnostics["generation"] == 1
    assert diagnostics["resets"] == 1
    assert diagnostics["frames_total"] == 4
    assert diagnostics["frames_in_generation"] == 2
    assert diagnostics["first_frame"] == -123
    assert diagnostics["last_frame"] == -122
    assert diagnostics["expected_next_frame"] == -121
    assert diagnostics["capture_decoder_assertions"] == 4
    assert diagnostics["capture_decoder_mismatches"] == 0
    assert diagnostics["current_frame_inference_barriers"] == 4
    assert diagnostics["current_frame_inference_barrier_every_frame"] is True
    assert diagnostics["current_frame_inference_barrier_timing"]["count"] == 4
    assert diagnostics["step_timing"]["count"] == 4
    assert diagnostics["upstream"]["submitted_frames"] == [-123, -122, -123, -122]
    assert runtimes[0].barrier_frames == [-123, -122, -123, -122]
    assert len(runtimes) == 1

    metadata = session.metadata()
    assert metadata["source"]["revision"] == SLIPPI_AI_SOURCE_REVISION
    assert metadata["checkpoint"]["sha256"] == MEDIUM_V2_CHECKPOINT_SHA256
    assert metadata["runtime_contract"]["sample_temperature"] == 1.0
    assert metadata["runtime_contract"]["effective_policy_delay_frames"] == 19
    assert metadata["runtime_contract"]["console_delay_frames"] == 2
    assert metadata["runtime_contract"]["requested_player_name"] == "Master Player"
    assert metadata["runtime_contract"]["effective_player_name"] == "Master Player"
    assert metadata["capabilities"] == medium_v2_capabilities()
    assert metadata["upstream_runtime"]["parser_class"] == "slippi_db.parse_libmelee.Parser"

    session.reset("test-next-game")
    assert session.diagnostics()["awaiting_reset_frame"]
    with pytest.raises(ValueError, match="first policy frame must be -123"):
        session.step(SimpleNamespace(frame=-122))
    session.step(SimpleNamespace(frame=-123))
    assert len(runtimes) == 1

    session.close()
    session.close()
    assert runtimes[0].closed
    assert session.diagnostics()["closed"]
    with pytest.raises(RuntimeError, match="session is closed"):
        session.step(SimpleNamespace(frame=-122))


def test_session_faults_on_capture_decoder_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(policy_module, "verify_runtime_assets", lambda _config: _fake_asset_identity())

    def factory(_config: SlippiAIPolicyConfig, capture: Any) -> _FakeRuntime:
        return _FakeRuntime(capture, mismatch=True)

    session = SlippiAIPolicySession(_fake_config(tmp_path), agent_factory=factory)
    try:
        with pytest.raises(AssertionError, match="does not exactly match"):
            session.step(SimpleNamespace(frame=-123))
        diagnostics = session.diagnostics()
        assert diagnostics["capture_decoder_assertions"] == 1
        assert diagnostics["capture_decoder_mismatches"] == 1
        assert diagnostics["fault"].startswith("AssertionError:")
    finally:
        session.close()


def test_config_preserves_defaults_and_validates_runtime_options(tmp_path: Path) -> None:
    config = _fake_config(tmp_path)
    config.validate()
    assert config.name == "Master Player"
    assert config.sample_temperature == 1.0
    SlippiAIPolicyConfig(
        source_directory=config.source_directory,
        checkpoint_path=config.checkpoint_path,
        port=1,
        opponent_port=2,
        name="Diamond Player",
        sample_temperature=0.5,
    ).validate()
    no_console_delay = SlippiAIPolicyConfig(
        source_directory=config.source_directory,
        checkpoint_path=config.checkpoint_path,
        port=1,
        opponent_port=2,
        console_delay_frames=0,
    )
    no_console_delay.validate()
    assert no_console_delay.policy_delay_frames - no_console_delay.console_delay_frames == 21
    with pytest.raises(ValueError, match="greater than zero"):
        SlippiAIPolicyConfig(
            source_directory=config.source_directory,
            checkpoint_path=config.checkpoint_path,
            port=1,
            opponent_port=2,
            sample_temperature=0.0,
        ).validate()
    for invalid_delay in (-1, 22):
        with pytest.raises(ValueError, match="between zero and policy_delay_frames"):
            SlippiAIPolicyConfig(
                source_directory=config.source_directory,
                checkpoint_path=config.checkpoint_path,
                port=1,
                opponent_port=2,
                console_delay_frames=invalid_delay,
            ).validate()


def test_policy_name_identity_rejects_labels_upstream_would_coerce() -> None:
    name_map = {"Master Player": 1, "Diamond Player": 2}
    accepted = policy_module._validate_policy_name_identity(
        "Master Player",
        name_map,
        list(MEDIUM_V2_RL_TRAINED_NAMES),
    )
    assert accepted.requested_name == "Master Player"
    assert accepted.effective_name == "Master Player"
    assert accepted.requested_name_code == 1
    assert accepted.effective_name_code == 1

    with pytest.raises(
        ValueError,
        match="pinned upstream would coerce 'Diamond Player' to 'Master Player'",
    ):
        policy_module._validate_policy_name_identity(
            "Diamond Player",
            name_map,
            list(MEDIUM_V2_RL_TRAINED_NAMES),
        )


def test_declared_medium_v2_capabilities_are_complete_and_deterministic() -> None:
    import melee

    expected = {
        "controlled_characters": list(MEDIUM_V2_SUPPORTED_CHARACTERS),
        "opponents": MEDIUM_V2_ALLOWED_OPPONENTS,
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
    assert medium_v2_capabilities() == expected
    assert medium_v2_capabilities() == expected
    assert len(set(MEDIUM_V2_SUPPORTED_CHARACTERS)) == 12
    assert len(set(MEDIUM_V2_SUPPORTED_STAGES)) == 6
    assert all(character in melee.Character.__members__ for character in MEDIUM_V2_SUPPORTED_CHARACTERS)
    assert all(stage in melee.Stage.__members__ for stage in MEDIUM_V2_SUPPORTED_STAGES)


@pytest.mark.parametrize(
    ("release", "character", "sha256", "byte_length"),
    (
        (
            "dk_d18_imitation_v2",
            "DK",
            DK_D18_IMITATION_V2_CHECKPOINT_SHA256,
            DK_D18_IMITATION_V2_CHECKPOINT_BYTES,
        ),
        (
            "doc_d18_imitation_v3",
            "DOC",
            DOC_D18_IMITATION_V3_CHECKPOINT_SHA256,
            DOC_D18_IMITATION_V3_CHECKPOINT_BYTES,
        ),
    ),
)
def test_native_specialist_release_contracts_are_strict_and_delay_18(
    tmp_path: Path,
    release: str,
    character: str,
    sha256: str,
    byte_length: int,
) -> None:
    contract = slippi_ai_release_contract(release)
    assert contract.checkpoint_kind == "imitation"
    assert contract.supported_characters == (character,)
    assert contract.allowed_opponents == "all"
    assert contract.policy_delay_frames == 18
    assert contract.checkpoint_sha256 == sha256
    assert contract.checkpoint_bytes == byte_length
    assert contract.variable_count == 137
    assert contract.parameter_count == 10_517_970
    assert contract.rl_trained_names == ()
    assert slippi_ai_release_capabilities(release)["controlled_characters"] == [character]

    config = SlippiAIPolicyConfig(
        source_directory=tmp_path / "source",
        checkpoint_path=tmp_path / release,
        port=2,
        opponent_port=1,
        release=release,
        policy_delay_frames=18,
        console_delay_frames=0,
        requested_character=character,
    )
    config.validate()
    with pytest.raises(ValueError, match="does not cover requested character"):
        SlippiAIPolicyConfig(
            source_directory=config.source_directory,
            checkpoint_path=config.checkpoint_path,
            port=2,
            opponent_port=1,
            release=release,
            policy_delay_frames=18,
            requested_character="FOX",
        ).validate()
    with pytest.raises(ValueError, match="runtime contract mismatch"):
        SlippiAIPolicyConfig(
            source_directory=config.source_directory,
            checkpoint_path=config.checkpoint_path,
            port=2,
            opponent_port=1,
            release=release,
            policy_delay_frames=21,
            requested_character=character,
        ).validate()


def test_imitation_policy_name_identity_has_no_rl_coercion() -> None:
    accepted = policy_module._validate_policy_name_identity(
        "Master Player",
        {"Master Player": 7},
        None,
        release="dk_d18_imitation_v2",
        expected_rl_names=(),
    )
    assert accepted.effective_name == "Master Player"
    assert accepted.effective_name_code == 7
    assert accepted.rl_trained_names == ()


@pytest.mark.local_replays
def test_fixed_seed_real_replays_cover_all_declared_characters_and_stages() -> None:
    import melee
    import tensorflow as tf  # type: ignore[import-untyped]

    project_root = Path(__file__).resolve().parents[2]
    cases = (
        ("0230f29d67e81b7bb61bc748a7a30d8c0cfa78b26ee2bcdd6c0be40efa6c6296", "FOX", "BATTLEFIELD"),
        ("2762abfa2c245c8715d30a41f4bd36eee785b736e960dac6d9d18f547395f347", "FALCO", "BATTLEFIELD"),
        ("0a229f0da774ee5929e11d6291829bfc1d3c6c16ade288b215d4050171d5f6e5", "MARTH", "BATTLEFIELD"),
        ("0944d1bbd98047cf9a007dea3c4ac06730e73cfa1fb167acb4c31fb5fd963ce5", "SHEIK", "BATTLEFIELD"),
        ("3b5001ec386c01497c8f0ae1a86d754916887a3f63fbe4c960316d17faf2f6c6", "JIGGLYPUFF", "BATTLEFIELD"),
        ("33fd1b4d43ddabc34247ce382ece4c2d65bed2a9550d47fc978efd3ebd6882a6", "CPTFALCON", "BATTLEFIELD"),
        ("2c78a4f626ba71560a964379145752dafac5b93d87cf4e8f46744f83353906fc", "PEACH", "BATTLEFIELD"),
        ("62020862c6e22d1a2a4267d173d3e4b9fc4a28d8a6caa07d58c7890ac0f4fe9d", "YOSHI", "FINAL_DESTINATION"),
        ("4a1f14a04dfe1e8cc9fb3b85b6f1b31bfbf821abb6ec2dd2b6b7fcb7459608af", "POPO", "DREAMLAND"),
        ("943dc64c28ba786702c72da41f20e1645aa125c59501a97317b870ae47e9c7d1", "LUIGI", "POKEMON_STADIUM"),
        ("470108cdeada1632ce9e0212f2d8d6649e9d2649e9d1159b0b83a33721a0bafa", "PIKACHU", "FOUNTAIN_OF_DREAMS"),
        ("5af8c38cdcf0981352728e725641dd0d4b15c52a9fec9bb8a23da1dda393eba0", "SAMUS", "YOSHIS_STORY"),
    )
    replay_paths = [project_root / ".e000-cache" / "raw" / f"{sha256}.slp" for sha256, _, _ in cases]
    default_config = SlippiAIPolicyConfig.from_project_root(project_root, port=1, opponent_port=2)
    if (
        not default_config.source_directory.is_dir()
        or not default_config.checkpoint_path.is_file()
        or not all(path.is_file() for path in replay_paths)
    ):
        pytest.skip("ignored pinned source, checkpoint, or E000 capability replay is unavailable")

    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    config = SlippiAIPolicyConfig(
        source_directory=default_config.source_directory,
        checkpoint_path=default_config.checkpoint_path,
        port=1,
        opponent_port=2,
        console_delay_frames=0,
    )
    observed_characters: set[str] = set()
    observed_stages: set[str] = set()
    with SlippiAIPolicySession(config) as session:
        for path, (_, expected_character, expected_stage) in zip(replay_paths, cases, strict=True):
            console = melee.Console(path=str(path), is_dolphin=False, allow_old_version=True)
            assert console.connect()
            try:
                for frame_offset in range(2):
                    gamestate = console.step()
                    assert gamestate is not None
                    assert int(gamestate.frame) == -123 + frame_offset
                    assert gamestate.players[1].character.name == expected_character
                    assert gamestate.stage.name == expected_stage
                    command = session.step(gamestate)
                    command.validate()
                observed_characters.add(expected_character)
                observed_stages.add(expected_stage)
            finally:
                console.stop()

        diagnostics = session.diagnostics()
        assert diagnostics["frames_total"] == 24
        assert diagnostics["resets"] == 11
        assert diagnostics["capture_decoder_assertions"] == 24
        assert diagnostics["capture_decoder_mismatches"] == 0
        assert diagnostics["fault"] is None

    assert observed_characters == set(MEDIUM_V2_SUPPORTED_CHARACTERS)
    assert observed_stages == set(MEDIUM_V2_SUPPORTED_STAGES)


def test_pinned_source_and_medium_v2_checkpoint_identity_when_cached() -> None:
    project_root = Path(__file__).resolve().parents[2]
    config = SlippiAIPolicyConfig.from_project_root(project_root, port=1, opponent_port=2)
    if not config.source_directory.is_dir() or not config.checkpoint_path.is_file():
        pytest.skip("ignored pinned Slippi-AI source or medium-v2 checkpoint is unavailable")
    identity = verify_runtime_assets(config)
    assert identity["source"]["revision"] == SLIPPI_AI_SOURCE_REVISION
    assert identity["source"]["branch"] == "main"
    assert identity["source"]["remote_url"] == "https://github.com/vladfi1/slippi-ai.git"
    assert identity["source"]["license"] == "MIT"
    assert identity["source"]["tracked_tree_clean"] is True
    assert identity["checkpoint"] == {
        "release": "medium-v2",
        "path": str(config.checkpoint_path.resolve()),
        "sha256": MEDIUM_V2_CHECKPOINT_SHA256,
        "byte_length": MEDIUM_V2_CHECKPOINT_BYTES,
    }


@pytest.mark.parametrize(
    ("release", "character"),
    (("dk_d18_imitation_v2", "DK"), ("doc_d18_imitation_v3", "DOC")),
)
def test_pinned_native_specialist_checkpoint_identity_when_cached(
    release: str,
    character: str,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    config = SlippiAIPolicyConfig.from_project_root(
        project_root,
        port=2,
        opponent_port=1,
        release=release,
    )
    config = dataclasses.replace(
        config,
        console_delay_frames=0,
        requested_character=character,
    )
    if not config.source_directory.is_dir() or not config.checkpoint_path.is_file():
        pytest.skip(f"ignored pinned Slippi-AI source or {release} checkpoint is unavailable")
    identity = verify_runtime_assets(config)
    contract = slippi_ai_release_contract(release)
    assert identity["release_contract"] == contract.as_dict()
    assert identity["checkpoint"] == {
        "release": release,
        "path": str(config.checkpoint_path.resolve()),
        "sha256": contract.checkpoint_sha256,
        "byte_length": contract.checkpoint_bytes,
    }


@pytest.mark.parametrize(
    ("release", "character"),
    (("dk_d18_imitation_v2", "DK"), ("doc_d18_imitation_v3", "DOC")),
)
def test_native_specialist_builds_through_pinned_upstream_runtime_when_cached(
    release: str,
    character: str,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    config = dataclasses.replace(
        SlippiAIPolicyConfig.from_project_root(
            project_root,
            port=2,
            opponent_port=1,
            release=release,
        ),
        console_delay_frames=0,
        requested_character=character,
    )
    if not config.source_directory.is_dir() or not config.checkpoint_path.is_file():
        pytest.skip(f"ignored pinned Slippi-AI source or {release} checkpoint is unavailable")
    session = SlippiAIPolicySession(config)
    try:
        session.start()
        metadata = session.metadata()
        upstream = metadata["upstream_runtime"]
        assert upstream["release_contract"]["key"] == release
        assert upstream["supported_characters"] == [character]
        assert upstream["dataset_supported_characters"] == [character]
        assert upstream["rl_trained_names"] == []
        assert upstream["effective_name"] == DEFAULT_PLAYER_NAME
        assert upstream["policy_delay_frames"] == 18
        assert upstream["effective_policy_delay_frames"] == 18
        assert upstream["variable_count"] == 137
        assert upstream["parameter_count"] == 10_517_970
        assert upstream["state_assignment"] == {
            "shape_mismatches": [],
            "dtype_mismatches": [],
            "value_mismatches": [],
            "nonfinite_variables": [],
        }
    finally:
        session.close()


@pytest.mark.local_replays
@pytest.mark.parametrize("console_delay_frames,expected_effective_delay", [(0, 21), (2, 19)])
def test_real_replay_frame_runs_exact_pinned_runtime_when_cached(
    console_delay_frames: int,
    expected_effective_delay: int,
) -> None:
    import melee

    project_root = Path(__file__).resolve().parents[2]
    replay = (
        project_root
        / ".e000-cache"
        / "raw"
        / "09a0d7d468df588b3ca0d1a6e763c4c31385e3a0564dba41998cb24a7a6ed498.slp"
    )
    source = project_root / ".e001-cache" / "slippi-ai-source"
    checkpoint = project_root / ".e001-cache" / "slippi-ai" / "models" / "medium-v2"
    if not replay.is_file() or not source.is_dir() or not checkpoint.is_file():
        pytest.skip("ignored E000 replay or pinned Slippi-AI cache is unavailable")

    console = melee.Console(path=str(replay), is_dolphin=False, allow_old_version=True)
    assert console.connect()
    try:
        gamestate = console.step()
        assert gamestate is not None
        assert int(gamestate.frame) == -123
        ports = sorted(gamestate.players)
        assert len(ports) == 2
        default_config = SlippiAIPolicyConfig.from_project_root(
            project_root, port=ports[0], opponent_port=ports[1]
        )
        config = SlippiAIPolicyConfig(
            source_directory=default_config.source_directory,
            checkpoint_path=default_config.checkpoint_path,
            port=ports[0],
            opponent_port=ports[1],
            console_delay_frames=console_delay_frames,
        )
        with SlippiAIPolicySession(config) as session:
            command = session.step(gamestate)
            command.validate()
            assert command.main_stick == (0.0, 0.0)
            assert command.c_stick == (0.0, 0.0)
            metadata = session.metadata()
            diagnostics = session.diagnostics()
            assert metadata["upstream_runtime"]["policy_delay_frames"] == 21
            assert metadata["upstream_runtime"]["console_delay_frames"] == console_delay_frames
            assert metadata["upstream_runtime"]["effective_policy_delay_frames"] == expected_effective_delay
            assert metadata["runtime_contract"]["console_delay_frames"] == console_delay_frames
            assert metadata["runtime_contract"]["effective_policy_delay_frames"] == expected_effective_delay
            assert metadata["upstream_runtime"]["requested_name"] == "Master Player"
            assert metadata["upstream_runtime"]["effective_name"] == "Master Player"
            assert metadata["upstream_runtime"]["requested_name_code"] == 1
            assert metadata["upstream_runtime"]["effective_name_code"] == 1
            assert metadata["upstream_runtime"]["rl_trained_names"] == list(MEDIUM_V2_RL_TRAINED_NAMES)
            assert metadata["upstream_runtime"]["variable_count"] == 141
            assert metadata["upstream_runtime"]["parameter_count"] == 23_887_032
            assert metadata["upstream_runtime"]["supported_characters"] == [
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
            ]
            assert metadata["upstream_runtime"]["dataset_supported_characters"] == list(
                MEDIUM_V2_SUPPORTED_CHARACTERS
            )
            assert metadata["upstream_runtime"]["allowed_opponents"] == "all"
            assert metadata["upstream_runtime"]["supported_rendered_stages"] == list(
                MEDIUM_V2_SUPPORTED_STAGES
            )
            assert metadata["capabilities"] == medium_v2_capabilities()
            assert metadata["upstream_runtime"]["state_assignment"] == {
                "shape_mismatches": [],
                "dtype_mismatches": [],
                "value_mismatches": [],
                "nonfinite_variables": [],
            }
            assert diagnostics["frames_total"] == 1
            assert diagnostics["capture_decoder_assertions"] == 1
    finally:
        console.stop()
