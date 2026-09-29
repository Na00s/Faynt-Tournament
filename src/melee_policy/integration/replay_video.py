"""Render a completed Slippi replay to an isolated-window MP4 on macOS."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import pty
import re
import shutil
import signal
import stat
import statistics
import struct
import subprocess
import tempfile
import wave
from collections.abc import Mapping
from dataclasses import replace
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

from melee_policy.integration.game_bundle import VideoRenderContext

DEFAULT_PLAYBACK_APPLICATION = ".e003-cache/slippi-launcher/user-data/playback/Slippi Dolphin.app"
DEFAULT_FRAME_RATE = 60.0
RENDER_CAPTURE_EMULATION_SPEED = 0.5
RENDER_CAPTURE_FPS = 120
NORMAL_OUTPUT_PLAYBACK_SPEED = 1.0
DEFAULT_AUDIO_PRESENTATION_DELAY_SECONDS = 0.050
MINIMUM_TERMINAL_TAIL_SECONDS = 2.0
CONTENT_AUDIT_METHOD = "raw-source-central-gameplay-region-frame-scan-v1"
CONTENT_AUDIT_MAXIMUM_SUSTAINED_BLANK_SECONDS = 0.500
OUTPUT_CONTENT_AUDIT_METHOD = "final-output-central-gameplay-region-frame-scan-v1"
OUTPUT_CONTENT_AUDIT_MAXIMUM_JOIN_SAMPLE_GAP_SECONDS = 0.100
SOURCE_BLANK_INTERVAL_SCHEMA = "canonical-coalesced-half-open-v1"
SOURCE_BLANK_INTERVAL_TIME_AXIS = "source-video-seconds"
SOURCE_FAITHFUL_BLANK_FRAME_POLICY = (
    "source-faithful-near-uniform-blank-frames-over-gameplay"
)
SOURCE_BLANK_INTERVAL_MAPPING_METHOD = "clock-landmark-piecewise-linear-half-open-v1"
SOURCE_BLANK_INTERVAL_MAPPING_QUANTIZATION = "per-segment-composition-cmtime-60000-v1"
SOURCE_BLANK_INTERVAL_MAPPING_TIMESCALE = 60000
SOURCE_BLANK_INTERVAL_MAPPING_NUMERICAL_SLACK_SECONDS = 2.0 / 60000.0
SOURCE_BLANK_CLASSES = ("near-white", "near-black", "near-neutral-blank")
CONTENT_AUDIT_MAXIMUM_FRAME_GAP_SECONDS = 0.040
OUTPUT_CONTENT_AUDIT_FRAME_INTERVAL_TOLERANCE_SECONDS = 0.001
OUTPUT_CONTENT_AUDIT_MINIMUM_FRAME_GAP_SECONDS = (
    (1.0 / DEFAULT_FRAME_RATE) - OUTPUT_CONTENT_AUDIT_FRAME_INTERVAL_TOLERANCE_SECONDS
)
OUTPUT_CONTENT_AUDIT_MAXIMUM_FRAME_GAP_SECONDS = (
    (1.0 / DEFAULT_FRAME_RATE) + OUTPUT_CONTENT_AUDIT_FRAME_INTERVAL_TOLERANCE_SECONDS
)
CAPTURE_DELIVERY_SCHEMA = "sck-stream-output-avassetwriter-v3"
CAPTURE_DELIVERY_MAXIMUM_GAP_SECONDS = 0.040
CAPTURE_DELIVERY_TERMINAL_RECOVERY_CEILING_SECONDS = 0.050
CAPTURE_DELIVERY_MAXIMUM_CALLBACK_LAG_SECONDS = 0.050
CAPTURE_DELIVERY_MAXIMUM_CALLBACK_SERVICE_SECONDS = 1.0 / RENDER_CAPTURE_FPS
CAPTURE_DELIVERY_MAXIMUM_SUSTAINED_BLANK_SECONDS = 0.500
STARTUP_SYNC_SCHEMA = "idle-command-zero-audio-two-stop-inclusive-terminal-v7"
STARTUP_SYNC_COMMAND_BOUNDARY_SEMANTICS = "endFrame-is-exclusive-control-boundary"
STARTUP_SYNC_GENERATION_BOUNDARY_SEMANTICS = (
    "CURRENT_FRAME-ends-at-requested-inclusive-last-content-frame"
)
CAPTURE_ENCODED_PTS_SEQUENCE_ERROR_SECONDS = 1.0 / 60000.0 + 0.000001
CAPTURE_ENCODED_PTS_TRACE_ROUNDING_SECONDS = 0.000001
CAPTURE_ENCODED_PTS_NUMERICAL_SLACK_SECONDS = 0.000000000001
CAPTURE_ENCODED_PTS_GAP_RECONCILIATION_ERROR_SECONDS = (
    2.0 * CAPTURE_ENCODED_PTS_SEQUENCE_ERROR_SECONDS
    + CAPTURE_ENCODED_PTS_TRACE_ROUNDING_SECONDS
)
CAPTURE_ENCODED_PTS_MAXIMUM_GAP_SECONDS = (
    CAPTURE_DELIVERY_MAXIMUM_GAP_SECONDS
    + CAPTURE_ENCODED_PTS_GAP_RECONCILIATION_ERROR_SECONDS
)
CAPTURE_ENCODED_PTS_ENDPOINT_RECONCILIATION_ERROR_SECONDS = (
    CAPTURE_ENCODED_PTS_SEQUENCE_ERROR_SECONDS + CAPTURE_ENCODED_PTS_TRACE_ROUNDING_SECONDS
)
MINIMUM_CAPTURE_DIMENSION_FRACTION = 0.750
NATIVE_RECORDER_STARTUP_TIMEOUT_PHASE_COUNT = 10
NATIVE_RECORDER_ACTIVATION_TIMEOUT_MAX_SECONDS = 5.0
NATIVE_RECORDER_TERMINAL_APPLICATION_PROOF_MAX_SECONDS = 1.0
NATIVE_RECORDER_TEARDOWN_MARGIN_SECONDS = 30.1
AUDIO_PHYSICAL_MAXIMUM_OVERSHOOT_SECONDS = 1.0 / DEFAULT_FRAME_RATE + 0.002000
AUDIO_PHYSICAL_MAXIMUM_FINALIZED_EXTENSION_SECONDS = 0.100
AUDIO_PHYSICAL_MAXIMUM_UNDERSHOOT_SECONDS = (
    AUDIO_PHYSICAL_MAXIMUM_FINALIZED_EXTENSION_SECONDS
)
AUDIO_CONTENT_SEAL_SCHEMA = "dolphin-audio-content-seal-v1"
AUDIO_CONTENT_SEAL_ROUNDING_RULE = (
    "shared-start=ceiling;output-payload=nearest-half-up;"
    "source-nominal=nearest-half-up;source-support="
    "ceiling((output-frames-1)*source-rate/output-rate)+1;"
    "source-payload=max(nominal,support)"
)
AUDIO_PHYSICAL_SEAL_SCHEMA = "dolphin-audio-physical-seal-v1"

_MODEL_DISPLAY_NAMES = {
    "cpu": "Melee CPU 9",
    "frisson-ai": "Frisson-AI",
    "frisson_ai": "Frisson-AI",
    "mimic": "MIMIC",
    "slippi-ai": "vladfi1 Slippi-AI",
    "slippi_ai": "vladfi1 Slippi-AI",
    "vladfi1-slippi-ai": "vladfi1 Slippi-AI",
    "vladfi1_slippi_ai": "vladfi1 Slippi-AI",
}


def _render_speed_metadata() -> dict[str, Any]:
    return {
        "normal_speed": True,
        "capture_emulation_speed": RENDER_CAPTURE_EMULATION_SPEED,
        "requested_capture_fps": RENDER_CAPTURE_FPS,
        "output_playback_speed": NORMAL_OUTPUT_PLAYBACK_SPEED,
        "output_frame_rate": DEFAULT_FRAME_RATE,
        "output_speed_semantics": "normal-speed Dolphin audio clock",
    }


def _video_config(context: VideoRenderContext) -> dict[str, Any]:
    raw = context.config.get("video")
    values = dict(raw) if isinstance(raw, Mapping) else {}
    expected_capture_width = values.get("expected_capture_width")
    expected_capture_height = values.get("expected_capture_height")
    return {
        "playback_application": values.get("playback_application", DEFAULT_PLAYBACK_APPLICATION),
        "startup_timeout_seconds": float(values.get("startup_timeout_seconds", 30.0)),
        "tail_padding_seconds": float(values.get("tail_padding_seconds", MINIMUM_TERMINAL_TAIL_SECONDS)),
        "resume_delay_seconds": float(values.get("resume_delay_seconds", 0.25)),
        "audio_presentation_delay_seconds": float(
            values.get(
                "audio_presentation_delay_seconds",
                DEFAULT_AUDIO_PRESENTATION_DELAY_SECONDS,
            )
        ),
        "frame_rate": float(values.get("frame_rate", DEFAULT_FRAME_RATE)),
        "width": int(values.get("width", 1280)),
        "height": int(values.get("height", 720)),
        "expected_capture_width": (
            int(expected_capture_width) if expected_capture_width is not None else None
        ),
        "expected_capture_height": (
            int(expected_capture_height) if expected_capture_height is not None else None
        ),
    }


def _resolve_project_path(value: object, project_root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise TypeError(f"video path setting must be a nonempty string, got {value!r}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _native_source(name: str) -> Path:
    return Path(__file__).resolve().parent / "native" / name


def _compile_native_tool(
    project_root: Path,
    *,
    source_name: str,
    executable_name: str,
) -> tuple[Path, str]:
    source = _native_source(source_name)
    if not source.is_file():
        raise FileNotFoundError(f"macOS replay recorder source is missing: {source}")
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    cache_directory = project_root / ".e010-cache" / "video-tools"
    cache_directory.mkdir(parents=True, exist_ok=True)
    executable = cache_directory / f"{executable_name}-{source_sha256[:16]}"
    if executable.is_file() and os.access(executable, os.X_OK):
        return executable, source_sha256
    clang = shutil.which("clang")
    if clang is None:
        raise FileNotFoundError("clang is required to build the macOS replay recorder")
    temporary = executable.with_name(f".{executable.name}.{os.getpid()}.tmp")
    module_cache = cache_directory / "clang-module-cache"
    module_cache.mkdir(parents=True, exist_ok=True)
    command = [
        clang,
        "-fobjc-arc",
        "-fmodules",
        "-framework",
        "AppKit",
        "-framework",
        "ApplicationServices",
        "-framework",
        "AVFoundation",
        "-framework",
        "CoreMedia",
        "-framework",
        "CoreVideo",
        "-framework",
        "Foundation",
        "-framework",
        "QuartzCore",
        "-framework",
        "ScreenCaptureKit",
        str(source),
        "-o",
        str(temporary),
    ]
    try:
        compile_environment = os.environ.copy()
        compile_environment["CLANG_MODULE_CACHE_PATH"] = str(module_cache)
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=compile_environment,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "could not compile the macOS replay recorder: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
        temporary.chmod(0o755)
        temporary.replace(executable)
    finally:
        temporary.unlink(missing_ok=True)
    return executable, source_sha256


def _compile_native_recorder(project_root: Path) -> tuple[Path, str]:
    return _compile_native_tool(
        project_root,
        source_name="macos_replay_recorder.m",
        executable_name="macos-replay-recorder",
    )


def _compile_native_muxer(project_root: Path) -> tuple[Path, str]:
    return _compile_native_tool(
        project_root,
        source_name="macos_mux_replay_audio.m",
        executable_name="macos-replay-audio-muxer",
    )


def _write_playback_user(
    user_directory: Path,
    *,
    width: int,
    height: int,
    dump_directory: Path | None = None,
) -> None:
    config_directory = user_directory / "Config"
    config_directory.mkdir(parents=True, exist_ok=True)
    (user_directory / "ishiiruka").touch()
    dump_directory = user_directory / "Dump" if dump_directory is None else dump_directory
    dolphin_ini = f"""[General]
DumpPath = {dump_directory.resolve()}
[Interface]
ConfirmStop = False
UsePanicHandlers = False
OnScreenDisplayMessages = False
HideCursor = True
PauseOnFocusLost = False
AutoHideCursor = False
MainWindowWidth = 628
MainWindowHeight = 422
ShowToolbar = True
ShowStatusbar = True
ShowSeekbar = True
[Display]
Fullscreen = False
RenderToMain = True
RenderWindowWidth = {width}
RenderWindowHeight = {height}
RenderWindowAutoSize = False
[Core]
EmulationSpeed = {RENDER_CAPTURE_EMULATION_SPEED:.8f}
FrameSkip = 0x00000000
SlippiPlaybackDisplayFrameIndex = False
[Movie]
PauseMovie = False
[DSP]
Backend = CoreAudio
Volume = 100
DumpAudio = True
DumpAudioSilent = True
[Input]
BackgroundInput = True
[Analytics]
Enabled = False
"""
    gfx_ini = """[Settings]
AspectRatio = 0
Crop = False
wideScreenHack = False
ShowFPS = False
ShowFrameTimes = False
DumpFramesAsImages = False
EFBScale = 2
VSync = True
[Hacks]
ForceProgressive = True
"""
    (config_directory / "Dolphin.ini").write_text(dolphin_ini, encoding="utf-8")
    (config_directory / "GFX.ini").write_text(gfx_ini, encoding="utf-8")


def _write_playback_command(
    path: Path,
    replay_path: Path,
    context: VideoRenderContext,
) -> None:
    command = {
        "mode": "normal",
        "replay": str(replay_path.resolve()),
        "startFrame": context.first_frame,
        "endFrame": context.last_frame,
        "isRealTimeMode": False,
        "shouldResync": False,
        "rollbackDisplayMethod": "off",
        "commandId": f"game-video-{hashlib.sha256(replay_path.read_bytes()).hexdigest()[:16]}",
    }
    path.write_text(json.dumps(command, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_idle_playback_command(
    path: Path,
    replay_path: Path,
    context: VideoRenderContext,
) -> None:
    replay_identity = hashlib.sha256(replay_path.read_bytes()).hexdigest()[:16]
    command = {
        "mode": "normal",
        "replay": "",
        "startFrame": context.first_frame,
        "endFrame": context.last_frame,
        "isRealTimeMode": False,
        "shouldResync": False,
        "rollbackDisplayMethod": "off",
        "commandId": f"game-video-idle-{replay_identity}",
    }
    path.write_text(json.dumps(command, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _playback_launch_command(
    executable: Path,
    user_directory: Path,
    active_playback_command: Path,
    iso_path: Path,
) -> list[str]:
    return [
        str(executable),
        "-b",
        "--cout",
        "-u",
        str(user_directory),
        "-i",
        str(active_playback_command),
        "-e",
        str(iso_path),
    ]


def _playback_sync_recorder_arguments(
    *,
    pty_master_fd: int,
    active_playback_command: Path,
    capture_playback_command: Path,
    dolphin_log: Path,
    requested_start_frame: int,
    requested_inclusive_end_frame: int,
) -> list[str]:
    if pty_master_fd < 0:
        raise ValueError("Dolphin PTY master descriptor must be nonnegative")
    if requested_inclusive_end_frame < requested_start_frame:
        raise ValueError("requested replay frame interval must be nonempty")
    return [
        str(pty_master_fd),
        str(active_playback_command),
        str(capture_playback_command),
        str(dolphin_log),
        str(requested_start_frame),
        str(requested_inclusive_end_frame),
    ]


def _playback_recorder_command(
    recorder: Path,
    *,
    startup_timeout_seconds: float,
    game_seconds: float,
    tail_padding_seconds: float,
    resume_delay_seconds: float,
    dsp_audio_path: Path,
    dtk_audio_path: Path,
    minimum_capture_width: int,
    minimum_capture_height: int,
    expected_capture_width: int | None,
    expected_capture_height: int | None,
    audio_presentation_delay_seconds: float,
    output_path: Path,
) -> list[str]:
    return [
        str(recorder),
        f"{startup_timeout_seconds:.6f}",
        f"{game_seconds:.9f}",
        f"{tail_padding_seconds:.6f}",
        f"{resume_delay_seconds:.6f}",
        str(dsp_audio_path),
        str(dtk_audio_path),
        str(minimum_capture_width),
        str(minimum_capture_height),
        str(expected_capture_width or 0),
        str(expected_capture_height or 0),
        str(RENDER_CAPTURE_FPS),
        f"{RENDER_CAPTURE_EMULATION_SPEED:.9f}",
        f"{audio_presentation_delay_seconds:.9f}",
        str(output_path),
    ]


def _run_playback_recorder(
    launch: list[str],
    environment: Mapping[str, str],
    recorder_command: list[str],
    *,
    active_playback_command: Path,
    capture_playback_command: Path,
    dolphin_log: Path,
    requested_start_frame: int,
    requested_inclusive_end_frame: int,
    timeout_seconds: float,
) -> tuple[subprocess.CompletedProcess[str], str]:
    if not recorder_command:
        raise ValueError("replay recorder command must contain an executable")
    pty_master_fd, pty_slave_fd = pty.openpty()
    process: subprocess.Popen[str] | None = None
    shutdown_method = "unattempted"
    try:
        try:
            process = subprocess.Popen(
                launch,
                env=environment,
                stdout=pty_slave_fd,
                stderr=pty_slave_fd,
                text=True,
                start_new_session=True,
            )
        finally:
            os.close(pty_slave_fd)
        completed = subprocess.run(
            [
                recorder_command[0],
                str(process.pid),
                *recorder_command[1:],
                *_playback_sync_recorder_arguments(
                    pty_master_fd=pty_master_fd,
                    active_playback_command=active_playback_command,
                    capture_playback_command=capture_playback_command,
                    dolphin_log=dolphin_log,
                    requested_start_frame=requested_start_frame,
                    requested_inclusive_end_frame=requested_inclusive_end_frame,
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            pass_fds=(pty_master_fd,),
        )
    finally:
        try:
            if process is not None:
                shutdown_method = _stop_process_group(process)
        finally:
            os.close(pty_master_fd)
    return completed, shutdown_method


def _stop_process_group(process: subprocess.Popen[str]) -> str:
    if process.poll() is not None:
        return "already-exited"
    try:
        # The recorder leaves Dolphin suspended at its sealed audio boundary.
        # Queue shutdown first so resuming the process only flushes and exits.
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        process.wait(timeout=5.0)
        return "already-exited"
    try:
        os.killpg(process.pid, signal.SIGCONT)
    except ProcessLookupError:
        process.wait(timeout=5.0)
        return "sigint"
    try:
        process.wait(timeout=5.0)
        return "sigint"
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=5.0)
        return "sigint"
    try:
        process.wait(timeout=5.0)
        return "sigterm"
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        process.wait(timeout=5.0)
        return "sigterm"
    process.wait(timeout=5.0)
    return "sigkill"


def _playback_recorder_timeout_seconds(
    *,
    startup_timeout_seconds: float,
    game_seconds: float,
    tail_padding_seconds: float,
    resume_delay_seconds: float,
) -> float:
    expected_capture_wall_seconds = game_seconds / RENDER_CAPTURE_EMULATION_SPEED
    maximum_playback_seconds = expected_capture_wall_seconds + startup_timeout_seconds
    # Native startup has ten sequential waits that each use startupTimeout:
    # window, first SCK frame, three PTY quiet barriers, two exact frame stops,
    # opening-frame proof, terminal trace, and terminal PTY drain. The terminal
    # application proof has its own one-second ceiling. Application activation
    # has its own five-second ceiling.
    bounded_startup_seconds = (
        NATIVE_RECORDER_STARTUP_TIMEOUT_PHASE_COUNT * startup_timeout_seconds
        + min(startup_timeout_seconds, NATIVE_RECORDER_ACTIVATION_TIMEOUT_MAX_SECONDS)
    )
    required_stable_seconds = max(tail_padding_seconds + 0.100, 0.350)
    return (
        bounded_startup_seconds
        + maximum_playback_seconds
        + NATIVE_RECORDER_TERMINAL_APPLICATION_PROOF_MAX_SECONDS
        + required_stable_seconds
        + resume_delay_seconds
        + NATIVE_RECORDER_TEARDOWN_MARGIN_SECONDS
    )


def _capture_frame_interval(first_frame: int, last_frame: int) -> tuple[int, int, int]:
    capture_start_frame = max(first_frame, 0)
    frame_count = last_frame - capture_start_frame + 1
    if frame_count < 1:
        raise ValueError("video replay has no audio-backed capture frame range")
    return capture_start_frame, last_frame, frame_count


def _last_log_lines(path: Path, count: int = 30) -> str:
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-count:])


def _label_fragment(value: object, *, fallback: str) -> str:
    if not isinstance(value, str):
        return fallback
    normalized = " ".join(value.split())
    if not normalized:
        return fallback
    return normalized[:80]


def _player_video_label(player: Mapping[str, Any], expected_port: int) -> str:
    port = player.get("port")
    if isinstance(port, bool) or not isinstance(port, int) or port != expected_port:
        raise ValueError(f"video player {expected_port} must declare physical port {expected_port}")
    raw_model = _label_fragment(player.get("model"), fallback="Unknown model")
    model = _MODEL_DISPLAY_NAMES.get(raw_model.casefold(), raw_model)
    character = _label_fragment(player.get("character"), fallback="Unknown character")
    return f"P{port}  {model}  {character.upper()}"


def _validate_finalized_wave(path: Path) -> int:
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
            raise RuntimeError(f"Dolphin audio dump has an invalid RIFF/WAVE header: {path}")
        declared_riff_size = struct.unpack_from("<I", header, 4)[0] + 8
        if declared_riff_size != file_size:
            raise RuntimeError(
                f"Dolphin audio dump was not finalized: declared={declared_riff_size} actual={file_size}"
            )
        data_length: int | None = None
        position = 12
        while position + 8 <= file_size:
            stream.seek(position)
            chunk_header = stream.read(8)
            chunk_name, chunk_length = struct.unpack("<4sI", chunk_header)
            payload_end = position + 8 + chunk_length
            if payload_end > file_size:
                raise RuntimeError(f"Dolphin audio dump contains an incomplete chunk: {path}")
            if chunk_name == b"data":
                data_length = chunk_length
            position = payload_end + (chunk_length & 1)
        if position != file_size or data_length is None:
            raise RuntimeError(f"Dolphin audio dump has an invalid finalized chunk layout: {path}")
    return data_length


def _validate_dolphin_audio_inventory(audio_directory: Path) -> tuple[Path, Path]:
    expected_names = ("dspdump.wav", "dtkdump.wav")
    if not audio_directory.is_dir():
        raise RuntimeError(f"Dolphin audio dump directory is missing: {audio_directory}")
    entries = sorted(audio_directory.iterdir(), key=lambda path: path.name)
    actual_names = [path.name for path in entries]
    if actual_names != list(expected_names):
        raise RuntimeError(
            "Dolphin audio dump inventory must contain exactly "
            f"{list(expected_names)}; found {actual_names}"
        )
    for path in entries:
        if not stat.S_ISREG(path.lstat().st_mode):
            raise RuntimeError(f"Dolphin audio dump must be a regular file: {path}")
    return audio_directory / expected_names[0], audio_directory / expected_names[1]


def _round_nonnegative_fraction_half_up(value: Fraction) -> int:
    if value < 0:
        raise ValueError("audio boundary rounding requires a nonnegative value")
    return (2 * value.numerator + value.denominator) // (2 * value.denominator)


def _ceil_nonnegative_fraction(value: Fraction) -> int:
    if value < 0:
        raise ValueError("audio boundary rounding requires a nonnegative value")
    return (value.numerator + value.denominator - 1) // value.denominator


def _derive_audio_content_seal(
    *,
    expected_game_frames: int,
    audio_presentation_delay_seconds: float,
    dsp_source_rate: int,
    dtk_source_rate: int,
    dsp_barrier_frames: int,
    dtk_barrier_frames: int,
    output_rate: int = 48_000,
) -> dict[str, Any]:
    """Derive exact source and output boundaries for normalized sound-sync audio."""
    integer_values = {
        "expected_game_frames": expected_game_frames,
        "dsp_source_rate": dsp_source_rate,
        "dtk_source_rate": dtk_source_rate,
        "dsp_barrier_frames": dsp_barrier_frames,
        "dtk_barrier_frames": dtk_barrier_frames,
        "output_rate": output_rate,
    }
    if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_values.values()):
        raise TypeError("audio content-seal frame counts and rates must be integers")
    if (
        expected_game_frames <= 0
        or dsp_source_rate <= 0
        or dtk_source_rate <= 0
        or output_rate <= 0
        or dsp_barrier_frames < 0
        or dtk_barrier_frames < 0
    ):
        raise ValueError("audio content-seal frame counts or rates are outside their domains")
    if (
        isinstance(audio_presentation_delay_seconds, bool)
        or not isinstance(audio_presentation_delay_seconds, (int, float))
        or not math.isfinite(float(audio_presentation_delay_seconds))
        or audio_presentation_delay_seconds < 0.0
    ):
        raise ValueError("audio content-seal presentation delay must be finite and nonnegative")

    game_duration = Fraction(expected_game_frames, int(DEFAULT_FRAME_RATE))
    presentation_delay = Fraction(str(float(audio_presentation_delay_seconds)))
    target_duration = game_duration - presentation_delay
    if target_duration <= 0:
        raise ValueError("audio content-seal target duration must be positive")
    shared_barrier = max(
        Fraction(dsp_barrier_frames, dsp_source_rate),
        Fraction(dtk_barrier_frames, dtk_source_rate),
    )

    output_frames = _round_nonnegative_fraction_half_up(target_duration * output_rate)
    output_duration = Fraction(output_frames, output_rate)
    streams: dict[str, dict[str, Any]] = {}
    for stream_name, source_rate, barrier_frames in (
        ("dsp", dsp_source_rate, dsp_barrier_frames),
        ("dtk", dtk_source_rate, dtk_barrier_frames),
    ):
        content_start = _ceil_nonnegative_fraction(shared_barrier * source_rate)
        nominal_payload = _round_nonnegative_fraction_half_up(
            target_duration * source_rate
        )
        # Linear interpolation at output index N - 1 reads source position
        # (N - 1) * source_rate / output_rate. Include that position without
        # relying on endpoint clamping, including mixed-rate 32 kHz streams.
        resampling_support = (
            _ceil_nonnegative_fraction(
                Fraction((output_frames - 1) * source_rate, output_rate)
            )
            + 1
        )
        content_payload = max(nominal_payload, resampling_support)
        content_end = content_start + content_payload
        content_duration = Fraction(content_payload, source_rate)
        streams[stream_name] = {
            "source_rate": source_rate,
            "logical_barrier_frames": barrier_frames,
            "content_start_frames": content_start,
            "nominal_content_payload_frames": nominal_payload,
            "resampling_support_payload_frames": resampling_support,
            "content_payload_frames": content_payload,
            "content_end_frames": content_end,
            "source_coverage_duration_seconds": float(content_duration),
            "source_coverage_rounding_offset_seconds": float(
                content_duration - target_duration
            ),
        }

    return {
        "schema": AUDIO_CONTENT_SEAL_SCHEMA,
        "method": AUDIO_CONTENT_SEAL_SCHEMA,
        "rounding_rule": AUDIO_CONTENT_SEAL_ROUNDING_RULE,
        "expected_game_frames": expected_game_frames,
        "game_frame_rate": int(DEFAULT_FRAME_RATE),
        "expected_game_duration_seconds": float(game_duration),
        "audio_presentation_delay_seconds": float(presentation_delay),
        "target_raw_duration_numerator": target_duration.numerator,
        "target_raw_duration_denominator": target_duration.denominator,
        "target_raw_duration_seconds": float(target_duration),
        "shared_barrier_numerator": shared_barrier.numerator,
        "shared_barrier_denominator": shared_barrier.denominator,
        "shared_barrier_seconds": float(shared_barrier),
        "output_rate": output_rate,
        "output_frames": output_frames,
        "output_duration_seconds": float(output_duration),
        "output_rounding_offset_seconds": float(output_duration - target_duration),
        "streams": streams,
        "passed": True,
    }


def _build_audio_physical_seal(
    *,
    content_seal: Mapping[str, Any],
    dsp_source_frames: int,
    dtk_source_frames: int,
    recorder_sealed_dsp_frames: int,
    recorder_sealed_dtk_frames: int,
) -> dict[str, Any]:
    """Bind finalized WAV layouts to the recorder seal and exact content target."""
    raw_streams = content_seal.get("streams")
    if not isinstance(raw_streams, Mapping):
        raise TypeError("audio content seal streams must be an object")
    try:
        shared_barrier = Fraction(
            int(content_seal["shared_barrier_numerator"]),
            int(content_seal["shared_barrier_denominator"]),
        )
        target_duration = Fraction(
            int(content_seal["target_raw_duration_numerator"]),
            int(content_seal["target_raw_duration_denominator"]),
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise TypeError("audio content seal has invalid exact rational boundaries") from error
    streams: dict[str, dict[str, Any]] = {}
    for stream_name, source_frames, recorder_sealed in (
        ("dsp", dsp_source_frames, recorder_sealed_dsp_frames),
        ("dtk", dtk_source_frames, recorder_sealed_dtk_frames),
    ):
        if (
            isinstance(source_frames, bool)
            or not isinstance(source_frames, int)
            or source_frames < 0
            or isinstance(recorder_sealed, bool)
            or not isinstance(recorder_sealed, int)
            or recorder_sealed < 0
        ):
            raise TypeError("audio physical-seal frame counts must be nonnegative integers")
        raw_stream = raw_streams.get(stream_name)
        if not isinstance(raw_stream, Mapping):
            raise TypeError(f"audio content seal lacks {stream_name} stream accounting")
        source_rate = raw_stream.get("source_rate")
        barrier = raw_stream.get("logical_barrier_frames")
        content_start = raw_stream.get("content_start_frames")
        content_end = raw_stream.get("content_end_frames")
        content_payload = raw_stream.get("content_payload_frames")
        if isinstance(source_rate, bool) or not isinstance(source_rate, int):
            raise TypeError("audio content-seal source rate must be an integer")
        if isinstance(barrier, bool) or not isinstance(barrier, int):
            raise TypeError("audio content-seal logical barrier must be an integer")
        if isinstance(content_start, bool) or not isinstance(content_start, int):
            raise TypeError("audio content-seal start must be an integer")
        if isinstance(content_end, bool) or not isinstance(content_end, int):
            raise TypeError("audio content-seal end must be an integer")
        if isinstance(content_payload, bool) or not isinstance(content_payload, int):
            raise TypeError("audio content-seal stream boundaries must be integers")
        if source_rate <= 0:
            raise ValueError("audio content-seal source rate must be positive")
        recorder_minus_content_end_frames = recorder_sealed - content_end
        recorder_endpoint = Fraction(recorder_sealed, source_rate) - shared_barrier
        recorder_offset = recorder_endpoint - target_duration
        finalized_extension_frames = source_frames - recorder_sealed
        finalized_content_extension_frames = source_frames - content_end
        recorder_offset_seconds = float(recorder_offset)
        finalized_extension_seconds = finalized_extension_frames / source_rate
        if not (
            -AUDIO_PHYSICAL_MAXIMUM_UNDERSHOOT_SECONDS - 1e-12
            <= recorder_offset_seconds
            <= AUDIO_PHYSICAL_MAXIMUM_OVERSHOOT_SECONDS + 1e-12
        ):
            raise RuntimeError(
                f"recorder {stream_name.upper()} physical seal is outside its bounded window"
            )
        if finalized_extension_frames < 0:
            raise RuntimeError(
                f"finalized {stream_name.upper()} WAV is shorter than the recorder seal"
            )
        if (
            finalized_extension_seconds
            > AUDIO_PHYSICAL_MAXIMUM_FINALIZED_EXTENSION_SECONDS + 1e-12
        ):
            raise RuntimeError(
                f"finalized {stream_name.upper()} WAV extension exceeds 100 ms"
            )
        if source_frames < content_end:
            raise RuntimeError(
                f"finalized {stream_name.upper()} WAV does not contain the content seal"
            )
        streams[stream_name] = {
            "source_rate": source_rate,
            "source_frames": source_frames,
            "logical_barrier_frames": barrier,
            "content_start_frames": content_start,
            "content_end_frames": content_end,
            "content_payload_frames": content_payload,
            "recorder_sealed_frames": recorder_sealed,
            "recorder_minus_content_end_frames": recorder_minus_content_end_frames,
            "recorder_physical_endpoint_seconds": float(recorder_endpoint),
            "recorder_physical_offset_seconds": recorder_offset_seconds,
            "finalized_extension_frames": finalized_extension_frames,
            "finalized_extension_seconds": finalized_extension_seconds,
            "finalized_content_extension_frames": finalized_content_extension_frames,
            "finalized_content_extension_seconds": (
                finalized_content_extension_frames / source_rate
            ),
            "target_available": True,
            "passed": True,
        }
    return {
        "schema": AUDIO_PHYSICAL_SEAL_SCHEMA,
        "method": AUDIO_PHYSICAL_SEAL_SCHEMA,
        "maximum_recorder_overshoot_seconds": AUDIO_PHYSICAL_MAXIMUM_OVERSHOOT_SECONDS,
        "maximum_recorder_undershoot_seconds": AUDIO_PHYSICAL_MAXIMUM_UNDERSHOOT_SECONDS,
        "maximum_finalized_extension_seconds": (
            AUDIO_PHYSICAL_MAXIMUM_FINALIZED_EXTENSION_SECONDS
        ),
        "streams": streams,
        "passed": True,
    }


def _validate_finalized_audio_durations(
    metadata: dict[str, Any],
    *,
    expected_game_frames: int,
    audio_presentation_delay_seconds: float,
    recorder_clock_landmarks: list[dict[str, float]],
    capture_emulation_speed: float = RENDER_CAPTURE_EMULATION_SPEED,
) -> None:
    content_seal = metadata.get("content_seal")
    physical_seal = metadata.get("physical_seal")
    if not isinstance(content_seal, Mapping):
        raise TypeError("finalized audio lacks a concrete content seal")
    if not isinstance(physical_seal, Mapping):
        raise TypeError("finalized audio lacks a concrete physical seal")
    stream_seals = content_seal.get("streams")
    physical_streams = physical_seal.get("streams")
    if not isinstance(stream_seals, Mapping) or not isinstance(physical_streams, Mapping):
        raise TypeError("finalized audio seal streams must be objects")

    source_rates: dict[str, int] = {}
    barriers: dict[str, int] = {}
    source_frame_counts: dict[str, int] = {}
    for stream_name in ("dsp", "dtk"):
        source_rate = metadata.get(f"{stream_name}_source_rate")
        barrier = metadata.get(f"{stream_name}_barrier_frames")
        if (
            isinstance(source_rate, bool)
            or not isinstance(source_rate, int)
            or source_rate <= 0
            or isinstance(barrier, bool)
            or not isinstance(barrier, int)
            or barrier < 0
        ):
            raise TypeError("finalized audio source rates and barriers must be integers")
        source_rates[stream_name] = source_rate
        barriers[stream_name] = barrier
    expected_seal = _derive_audio_content_seal(
        expected_game_frames=expected_game_frames,
        audio_presentation_delay_seconds=audio_presentation_delay_seconds,
        dsp_source_rate=source_rates["dsp"],
        dtk_source_rate=source_rates["dtk"],
        dsp_barrier_frames=barriers["dsp"],
        dtk_barrier_frames=barriers["dtk"],
    )
    if dict(content_seal) != expected_seal:
        raise RuntimeError("finalized audio content seal differs from exact frame accounting")
    if (
        physical_seal.get("schema") != AUDIO_PHYSICAL_SEAL_SCHEMA
        or physical_seal.get("method") != AUDIO_PHYSICAL_SEAL_SCHEMA
        or physical_seal.get("passed") is not True
    ):
        raise RuntimeError("finalized audio physical seal used an unknown contract")
    for key, expected in (
        ("maximum_recorder_overshoot_seconds", AUDIO_PHYSICAL_MAXIMUM_OVERSHOOT_SECONDS),
        ("maximum_recorder_undershoot_seconds", AUDIO_PHYSICAL_MAXIMUM_UNDERSHOOT_SECONDS),
        (
            "maximum_finalized_extension_seconds",
            AUDIO_PHYSICAL_MAXIMUM_FINALIZED_EXTENSION_SECONDS,
        ),
    ):
        value = physical_seal.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isclose(float(value), expected, abs_tol=1e-12)
        ):
            raise RuntimeError(f"finalized audio physical seal {key} is invalid")

    output_rate = expected_seal["output_rate"]
    output_frames = expected_seal["output_frames"]
    expected_game_seconds = expected_game_frames / DEFAULT_FRAME_RATE
    if (
        metadata.get("schema") != "dolphin-audio-mix-content-boundaries-v2"
        or metadata.get("method") != "dolphin-audio-mix-content-boundaries-v2"
        or metadata.get("sample_rate") != output_rate
        or metadata.get("channels") != 2
        or metadata.get("frames") != output_frames
        or metadata.get("target_output_frames") != output_frames
        or metadata.get("explicit_content_boundaries") is not True
    ):
        raise RuntimeError("finalized mixed audio does not match its exact content boundary")
    output_duration = output_frames / output_rate
    duration = metadata.get("duration_seconds")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isclose(float(duration), output_duration, abs_tol=1e-12)
    ):
        raise RuntimeError("finalized mixed audio duration disagrees with its frame count")

    endpoint_error = abs(
        output_duration + audio_presentation_delay_seconds - expected_game_seconds
    )
    maximum_output_error = 0.5 / output_rate
    if endpoint_error > maximum_output_error + 1e-12:
        raise RuntimeError("finalized mixed audio exceeded half-sample endpoint rounding")
    metadata["audio_presentation_delay_seconds"] = audio_presentation_delay_seconds
    metadata["endpoint_error_seconds"] = endpoint_error
    metadata["maximum_allowed_endpoint_error_seconds"] = maximum_output_error
    metadata["endpoint_method"] = "deterministic-content-sealed-normalized-endpoint-v1"

    for stream_name in ("dsp", "dtk"):
        expected_stream = expected_seal["streams"][stream_name]
        raw_physical_stream = physical_streams.get(stream_name)
        if not isinstance(raw_physical_stream, Mapping):
            raise TypeError(f"finalized {stream_name} physical seal must be an object")
        source_rate = source_rates[stream_name]
        source_frames = metadata.get(f"{stream_name}_source_frames")
        if (
            isinstance(source_frames, bool)
            or not isinstance(source_frames, int)
            or source_frames < 0
        ):
            raise TypeError(f"finalized {stream_name} source frames must be an integer")
        source_frame_counts[stream_name] = source_frames
        recorder_sealed = raw_physical_stream.get("recorder_sealed_frames")
        if (
            isinstance(recorder_sealed, bool)
            or not isinstance(recorder_sealed, int)
            or recorder_sealed < 0
        ):
            raise TypeError(f"finalized {stream_name} recorder seal must be an integer")
        content_start = expected_stream["content_start_frames"]
        content_end = expected_stream["content_end_frames"]
        content_payload = expected_stream["content_payload_frames"]
        shared_barrier = Fraction(
            expected_seal["shared_barrier_numerator"],
            expected_seal["shared_barrier_denominator"],
        )
        target_duration = Fraction(
            expected_seal["target_raw_duration_numerator"],
            expected_seal["target_raw_duration_denominator"],
        )
        recorder_minus_content_end_frames = recorder_sealed - content_end
        recorder_endpoint = Fraction(recorder_sealed, source_rate) - shared_barrier
        recorder_offset = recorder_endpoint - target_duration
        finalized_extension_frames = source_frames - recorder_sealed
        finalized_content_extension_frames = source_frames - content_end
        expected_physical_stream = {
            "source_rate": source_rate,
            "source_frames": source_frames,
            "logical_barrier_frames": barriers[stream_name],
            "content_start_frames": content_start,
            "content_end_frames": content_end,
            "content_payload_frames": content_payload,
            "recorder_sealed_frames": recorder_sealed,
            "recorder_minus_content_end_frames": recorder_minus_content_end_frames,
            "recorder_physical_endpoint_seconds": float(recorder_endpoint),
            "recorder_physical_offset_seconds": float(recorder_offset),
            "finalized_extension_frames": finalized_extension_frames,
            "finalized_extension_seconds": finalized_extension_frames / source_rate,
            "finalized_content_extension_frames": finalized_content_extension_frames,
            "finalized_content_extension_seconds": (
                finalized_content_extension_frames / source_rate
            ),
            "target_available": source_frames >= content_end,
            "passed": True,
        }
        if dict(raw_physical_stream) != expected_physical_stream:
            raise RuntimeError(
                f"finalized {stream_name} physical seal differs from frame accounting"
            )
        physical_offset = float(recorder_offset)
        if not (
            -AUDIO_PHYSICAL_MAXIMUM_UNDERSHOOT_SECONDS - 1e-12
            <= physical_offset
            <= AUDIO_PHYSICAL_MAXIMUM_OVERSHOOT_SECONDS + 1e-12
        ):
            raise RuntimeError(f"finalized {stream_name} recorder physical seal is out of bounds")
        if (
            finalized_extension_frames < 0
            or finalized_extension_frames / source_rate
            > AUDIO_PHYSICAL_MAXIMUM_FINALIZED_EXTENSION_SECONDS + 1e-12
            or source_frames < content_end
        ):
            raise RuntimeError(f"finalized {stream_name} source does not contain its content seal")
        for key, expected in (
            (f"{stream_name}_content_start_frames", content_start),
            (f"{stream_name}_content_end_frames", content_end),
            (f"{stream_name}_content_payload_frames", content_payload),
            (f"{stream_name}_trimmed_frames", content_start),
            (f"sealed_{stream_name}_frames", content_end),
            (f"recorder_sealed_{stream_name}_frames", recorder_sealed),
        ):
            if metadata.get(key) != expected:
                raise RuntimeError(f"finalized audio {key} differs from its seal")
        content_duration = output_duration
        reported_duration = metadata.get(f"{stream_name}_duration_seconds")
        if (
            isinstance(reported_duration, bool)
            or not isinstance(reported_duration, (int, float))
            or not math.isclose(float(reported_duration), content_duration, abs_tol=1e-12)
        ):
            raise RuntimeError(f"finalized {stream_name} duration differs from its seal")
        coverage_duration = content_payload / source_rate
        reported_coverage = metadata.get(
            f"{stream_name}_source_coverage_duration_seconds"
        )
        if (
            isinstance(reported_coverage, bool)
            or not isinstance(reported_coverage, (int, float))
            or not math.isclose(float(reported_coverage), coverage_duration, abs_tol=1e-12)
        ):
            raise RuntimeError(
                f"finalized {stream_name} source coverage differs from its seal"
            )
        metadata[f"{stream_name}_endpoint_error_seconds"] = endpoint_error
        metadata[f"finalized_{stream_name}_buffer_seconds"] = (
            finalized_extension_frames / source_rate
        )

    physical_duration = max(
        Fraction(source_frame_counts["dsp"], source_rates["dsp"]) - shared_barrier,
        Fraction(source_frame_counts["dtk"], source_rates["dtk"]) - shared_barrier,
    )
    expected_physical_landmarks, expected_extension = _extend_landmarks_to_finalized_audio(
        recorder_clock_landmarks,
        finalized_audio_duration_seconds=float(physical_duration),
        capture_emulation_speed=capture_emulation_speed,
    )
    _, expected_clock_seal = _seal_landmarks_to_content_audio(
        expected_physical_landmarks,
        content_audio_duration_seconds=output_duration,
    )
    raw_clock_seal = metadata.get("content_clock_seal")
    if not isinstance(raw_clock_seal, Mapping) or dict(raw_clock_seal) != expected_clock_seal:
        raise RuntimeError("finalized audio content-clock seal differs from physical landmarks")
    physical_clock_duration = metadata.get("physical_finalized_clock_duration_seconds")
    reported_extension = metadata.get("finalized_clock_extension_seconds")
    if (
        isinstance(physical_clock_duration, bool)
        or not isinstance(physical_clock_duration, (int, float))
        or not math.isclose(
            float(physical_clock_duration), float(physical_duration), abs_tol=1e-12
        )
        or isinstance(reported_extension, bool)
        or not isinstance(reported_extension, (int, float))
        or not math.isclose(float(reported_extension), expected_extension, abs_tol=1e-12)
        or metadata.get("finalized_buffer_extension_warning")
        is not (expected_extension > 0.050)
    ):
        raise RuntimeError("finalized audio physical clock extension metadata is invalid")


def _validated_pcm16_stereo_layout(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Dolphin audio dump is missing: {path}")
    data_length = _validate_finalized_wave(path)
    with wave.open(str(path), "rb") as stream:
        if stream.getnchannels() != 2 or stream.getsampwidth() != 2:
            raise RuntimeError(f"Dolphin audio dump must be 16-bit stereo PCM: {path}")
        sample_rate = stream.getframerate()
        declared_frames = stream.getnframes()
    if data_length != declared_frames * 4:
        raise RuntimeError(f"Dolphin audio dump data length does not match its finalized header: {path}")
    if sample_rate <= 0:
        raise RuntimeError(f"Dolphin audio dump is malformed: {path}")
    return declared_frames, sample_rate


def _read_pcm16_stereo(path: Path) -> tuple[Any, int]:
    import numpy as np

    declared_frames, sample_rate = _validated_pcm16_stereo_layout(path)
    with wave.open(str(path), "rb") as stream:
        values = np.frombuffer(stream.readframes(declared_frames), dtype="<i2").copy()
    if values.size != declared_frames * 2 or values.size % 2:
        raise RuntimeError(f"Dolphin audio dump is malformed: {path}")
    return values.reshape((-1, 2)), sample_rate


def _resample_pcm(
    values: Any,
    source_rate: int,
    target_rate: int,
    *,
    target_frames: int | None = None,
) -> Any:
    import numpy as np

    explicit_target_frames = target_frames is not None
    if target_frames is not None and (
        isinstance(target_frames, bool) or not isinstance(target_frames, int) or target_frames < 0
    ):
        raise ValueError("resampled target frame count must be a nonnegative integer")
    if len(values) == 0:
        return np.empty((0, 2), dtype=np.float64)
    if target_frames is None:
        target_frames = max(1, round(len(values) * target_rate / source_rate))
    if source_rate == target_rate and target_frames == len(values):
        return values.astype(np.float64)
    if target_frames == 0:
        return np.empty((0, 2), dtype=np.float64)
    if explicit_target_frames and (target_frames - 1) * source_rate > (
        len(values) - 1
    ) * target_rate:
        raise RuntimeError(
            "content-sealed source audio does not cover the final resampling position"
        )
    source_positions: Any = np.arange(len(values), dtype=np.float64)
    target_positions: Any = (
        np.arange(target_frames, dtype=np.float64) * source_rate / target_rate
    )
    target_positions = np.minimum(target_positions, len(values) - 1)
    return np.column_stack(
        [np.interp(target_positions, source_positions, values[:, channel]) for channel in range(2)]
    )


def _mix_dolphin_audio(
    dsp_path: Path,
    dtk_path: Path,
    output_path: Path,
    *,
    minimum_duration_seconds: float = 0.0,
    dsp_barrier_frames: int = 0,
    dtk_barrier_frames: int = 0,
    sealed_dsp_frames: int | None = None,
    sealed_dtk_frames: int | None = None,
    dsp_content_start_frames: int | None = None,
    dtk_content_start_frames: int | None = None,
    dsp_content_end_frames: int | None = None,
    dtk_content_end_frames: int | None = None,
    target_output_frames: int | None = None,
) -> dict[str, Any]:
    import numpy as np

    target_rate = 48_000
    dsp, dsp_rate = _read_pcm16_stereo(dsp_path)
    dtk, dtk_rate = _read_pcm16_stereo(dtk_path)
    for label, barrier_frames, sealed_frames, content_start, content_end, frame_count in (
        (
            "DSP",
            dsp_barrier_frames,
            sealed_dsp_frames,
            dsp_content_start_frames,
            dsp_content_end_frames,
            len(dsp),
        ),
        (
            "DTK",
            dtk_barrier_frames,
            sealed_dtk_frames,
            dtk_content_start_frames,
            dtk_content_end_frames,
            len(dtk),
        ),
    ):
        if isinstance(barrier_frames, bool) or not isinstance(barrier_frames, int):
            raise TypeError(f"{label} audio barrier must be an integer frame count")
        if barrier_frames < 0 or barrier_frames > frame_count:
            raise ValueError(
                f"{label} audio barrier {barrier_frames} exceeds source frame count {frame_count}"
            )
        if sealed_frames is not None:
            if isinstance(sealed_frames, bool) or not isinstance(sealed_frames, int):
                raise TypeError(f"{label} sealed audio boundary must be an integer frame count")
            if sealed_frames < barrier_frames or sealed_frames > frame_count:
                raise ValueError(
                    f"{label} sealed audio boundary {sealed_frames} is outside "
                    f"[{barrier_frames}, {frame_count}]"
                )
        for boundary_name, boundary in (("start", content_start), ("end", content_end)):
            if boundary is not None and (
                isinstance(boundary, bool) or not isinstance(boundary, int)
            ):
                raise TypeError(f"{label} content {boundary_name} must be an integer frame count")
    if target_output_frames is not None and (
        isinstance(target_output_frames, bool)
        or not isinstance(target_output_frames, int)
        or target_output_frames <= 0
    ):
        raise ValueError("mixed-audio target output boundary must be a positive integer")
    dsp_source_frames = len(dsp)
    dtk_source_frames = len(dtk)
    barrier_seconds = max(
        dsp_barrier_frames / dsp_rate,
        dtk_barrier_frames / dtk_rate,
    )
    default_dsp_start = math.ceil(barrier_seconds * dsp_rate - 1e-12)
    default_dtk_start = math.ceil(barrier_seconds * dtk_rate - 1e-12)
    dsp_content_start_frames = (
        default_dsp_start if dsp_content_start_frames is None else dsp_content_start_frames
    )
    dtk_content_start_frames = (
        default_dtk_start if dtk_content_start_frames is None else dtk_content_start_frames
    )
    dsp_content_end_frames = (
        sealed_dsp_frames if dsp_content_end_frames is None and sealed_dsp_frames is not None
        else dsp_source_frames if dsp_content_end_frames is None
        else dsp_content_end_frames
    )
    dtk_content_end_frames = (
        sealed_dtk_frames if dtk_content_end_frames is None and sealed_dtk_frames is not None
        else dtk_source_frames if dtk_content_end_frames is None
        else dtk_content_end_frames
    )
    if (
        dsp_content_start_frames < 0
        or dtk_content_start_frames < 0
        or dsp_content_end_frames < dsp_content_start_frames
        or dtk_content_end_frames < dtk_content_start_frames
        or dsp_content_end_frames > dsp_source_frames
        or dtk_content_end_frames > dtk_source_frames
    ):
        raise RuntimeError("Dolphin audio ended before the shared sample barrier could be normalized")
    dsp = dsp[dsp_content_start_frames:dsp_content_end_frames]
    dtk = dtk[dtk_content_start_frames:dtk_content_end_frames]
    dsp_source_coverage_duration_seconds = len(dsp) / dsp_rate
    dtk_source_coverage_duration_seconds = len(dtk) / dtk_rate
    dsp_resampled = _resample_pcm(
        dsp, dsp_rate, target_rate, target_frames=target_output_frames
    )
    dtk_resampled = _resample_pcm(
        dtk, dtk_rate, target_rate, target_frames=target_output_frames
    )
    minimum_frames = max(0, round(minimum_duration_seconds * target_rate))
    if target_output_frames is not None and minimum_frames > target_output_frames:
        raise RuntimeError("mixed-audio target is shorter than its requested minimum duration")
    output_frames = (
        target_output_frames
        if target_output_frames is not None
        else max(len(dsp_resampled), len(dtk_resampled), minimum_frames)
    )
    dsp_duration_seconds = (
        output_frames / target_rate
        if target_output_frames is not None
        else dsp_source_coverage_duration_seconds
    )
    dtk_duration_seconds = (
        output_frames / target_rate
        if target_output_frames is not None
        else dtk_source_coverage_duration_seconds
    )
    if len(dsp_resampled) == 0 and len(dtk_resampled) == 0 and minimum_duration_seconds > 0.25:
        raise RuntimeError("both Dolphin audio dumps are empty for a nontrivial replay")
    mixed: Any = np.zeros((output_frames, 2), dtype=np.float64)
    mixed[: len(dsp_resampled)] += dsp_resampled
    mixed[: len(dtk_resampled)] += dtk_resampled
    pcm = np.clip(np.rint(mixed), -32768, 32767).astype("<i2")
    with wave.open(str(output_path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(target_rate)
        stream.writeframes(pcm.tobytes())
    metadata: dict[str, Any] = {
        "schema": "dolphin-audio-mix-content-boundaries-v2",
        "method": "dolphin-audio-mix-content-boundaries-v2",
        "sample_rate": target_rate,
        "channels": 2,
        "frames": output_frames,
        "duration_seconds": output_frames / target_rate,
        "dsp_source_rate": dsp_rate,
        "dtk_source_rate": dtk_rate,
        "dsp_source_frames": dsp_source_frames,
        "dtk_source_frames": dtk_source_frames,
        "sealed_dsp_frames": dsp_content_end_frames,
        "sealed_dtk_frames": dtk_content_end_frames,
        "dsp_barrier_frames": dsp_barrier_frames,
        "dtk_barrier_frames": dtk_barrier_frames,
        "barrier_seconds": barrier_seconds,
        "dsp_trimmed_frames": dsp_content_start_frames,
        "dtk_trimmed_frames": dtk_content_start_frames,
        "dsp_content_start_frames": dsp_content_start_frames,
        "dtk_content_start_frames": dtk_content_start_frames,
        "dsp_content_end_frames": dsp_content_end_frames,
        "dtk_content_end_frames": dtk_content_end_frames,
        "dsp_content_payload_frames": len(dsp),
        "dtk_content_payload_frames": len(dtk),
        "target_output_frames": output_frames,
        "explicit_content_boundaries": target_output_frames is not None,
        "dsp_duration_seconds": dsp_duration_seconds,
        "dtk_duration_seconds": dtk_duration_seconds,
        "dsp_source_coverage_duration_seconds": dsp_source_coverage_duration_seconds,
        "dtk_source_coverage_duration_seconds": dtk_source_coverage_duration_seconds,
        "alignment": "Dolphin audio sample barrier",
    }
    absolute_pcm = np.max(np.abs(pcm.astype(np.int32)), axis=1)
    for threshold in (64, 256):
        matches = np.flatnonzero(absolute_pcm >= threshold)
        metadata[f"first_amplitude_{threshold}_observed"] = bool(matches.size)
        if matches.size:
            metadata[f"first_amplitude_{threshold}_seconds"] = int(matches[0]) / target_rate
    return metadata


def _finite_number(metadata: Mapping[str, Any], key: str) -> float:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"recorder timing metadata {key} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"recorder timing metadata {key} must be finite")
    return result


def _validated_source_blank_intervals(
    audit: Mapping[str, Any],
    *,
    source_start_seconds: float,
    source_end_seconds: float,
) -> list[dict[str, float | int | str]]:
    """Validate and normalize canonical half-open raw-source blank runs."""
    if audit.get("blank_interval_schema") != SOURCE_BLANK_INTERVAL_SCHEMA:
        raise RuntimeError("Slippi replay muxer content audit used an unknown blank interval schema")
    if audit.get("blank_interval_time_axis") != SOURCE_BLANK_INTERVAL_TIME_AXIS:
        raise RuntimeError("Slippi replay muxer content audit used the wrong blank interval time axis")
    raw_intervals = audit.get("blank_intervals")
    if not isinstance(raw_intervals, list):
        raise TypeError("Slippi replay muxer content audit blank_intervals must be an array")
    if audit.get("blank_interval_count") != len(raw_intervals):
        raise RuntimeError("Slippi replay muxer content audit reported the wrong blank interval count")

    class_frame_count_keys = {
        "near-white": "near_white_frame_count",
        "near-black": "near_black_frame_count",
        "near-neutral-blank": "near_neutral_blank_frame_count",
    }
    class_interval_count_keys = {
        "near-white": "near_white_blank_interval_count",
        "near-black": "near_black_blank_interval_count",
        "near-neutral-blank": "near_neutral_blank_interval_count",
    }
    frame_counts = dict.fromkeys(SOURCE_BLANK_CLASSES, 0)
    interval_counts = dict.fromkeys(SOURCE_BLANK_CLASSES, 0)
    normalized: list[dict[str, float | int | str]] = []
    previous_end = -math.inf
    previous_class: str | None = None
    for index, raw_interval in enumerate(raw_intervals):
        if not isinstance(raw_interval, Mapping):
            raise TypeError(f"Slippi replay muxer source blank interval {index} must be an object")
        frame_class = raw_interval.get("class")
        if frame_class not in SOURCE_BLANK_CLASSES:
            raise RuntimeError(f"Slippi replay muxer source blank interval {index} has an unknown class")

        def interval_number(
            key: str,
            *,
            interval: Mapping[str, Any] = raw_interval,
            interval_index: int = index,
        ) -> float:
            value = interval.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"Slippi replay muxer source blank interval {interval_index} {key} must be a number"
                )
            result = float(value)
            if not math.isfinite(result):
                raise ValueError(
                    f"Slippi replay muxer source blank interval {interval_index} {key} must be finite"
                )
            return result

        start = interval_number("source_start_seconds")
        end = interval_number("source_end_seconds")
        duration = interval_number("source_duration_seconds")
        source_frame_count = raw_interval.get("source_frame_count")
        if (
            isinstance(source_frame_count, bool)
            or not isinstance(source_frame_count, int)
            or source_frame_count <= 0
        ):
            raise RuntimeError(
                f"Slippi replay muxer source blank interval {index} has an invalid frame count"
            )
        if (
            start < source_start_seconds - 1e-9
            or end > source_end_seconds + 1e-9
            or end <= start
            or not math.isclose(duration, end - start, abs_tol=1e-9)
            or start < previous_end - 1e-9
        ):
            raise RuntimeError("Slippi replay muxer source blank intervals are invalid or unordered")
        if (
            previous_class == frame_class
            and math.isclose(start, previous_end, abs_tol=1e-9)
        ):
            raise RuntimeError("Slippi replay muxer source blank intervals are not coalesced")
        normalized.append(
            {
                "class": cast(str, frame_class),
                "source_start_seconds": start,
                "source_end_seconds": end,
                "source_duration_seconds": duration,
                "source_frame_count": source_frame_count,
            }
        )
        frame_counts[cast(str, frame_class)] += source_frame_count
        interval_counts[cast(str, frame_class)] += 1
        previous_end = end
        previous_class = cast(str, frame_class)

    for frame_class in SOURCE_BLANK_CLASSES:
        if audit.get(class_frame_count_keys[frame_class]) != frame_counts[frame_class]:
            raise RuntimeError("Slippi replay muxer source blank interval frame counts disagree")
        if audit.get(class_interval_count_keys[frame_class]) != interval_counts[frame_class]:
            raise RuntimeError("Slippi replay muxer source blank interval class counts disagree")
    return normalized


def _validate_mux_content_audit(
    metadata: Mapping[str, Any],
    *,
    expected_source_start_seconds: float,
    expected_source_end_seconds: float,
    expected_width: int,
    expected_height: int,
) -> None:
    """Require the native muxer's complete raw-gameplay pixel audit contract."""
    raw_audit = metadata.get("content_audit")
    if not isinstance(raw_audit, Mapping):
        raise TypeError("Slippi replay muxer content audit must be an object")
    if raw_audit.get("method") != CONTENT_AUDIT_METHOD:
        raise RuntimeError("Slippi replay muxer used an unknown content audit method")
    if raw_audit.get("passed") is not True:
        raise RuntimeError("Slippi replay muxer content audit did not pass")

    def number(key: str) -> float:
        value = raw_audit.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Slippi replay muxer content audit {key} must be a number")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"Slippi replay muxer content audit {key} must be finite")
        return result

    start = number("source_interval_start_seconds")
    end = number("source_interval_end_seconds")
    duration = number("source_interval_duration_seconds")
    if not math.isclose(start, expected_source_start_seconds, abs_tol=1e-3):
        raise RuntimeError("Slippi replay muxer content audit started outside the landmark interval")
    if not math.isclose(end, expected_source_end_seconds, abs_tol=1e-3):
        raise RuntimeError("Slippi replay muxer content audit ended outside the landmark interval")
    if end <= start or not math.isclose(duration, end - start, abs_tol=1e-6):
        raise RuntimeError("Slippi replay muxer content audit reported an invalid source interval")

    blank_intervals = _validated_source_blank_intervals(
        raw_audit,
        source_start_seconds=start,
        source_end_seconds=end,
    )

    frames_scanned = raw_audit.get("frames_scanned")
    minimum_pixels = raw_audit.get("minimum_sampled_pixels_per_frame")
    if isinstance(frames_scanned, bool) or not isinstance(frames_scanned, int) or frames_scanned <= 0:
        raise RuntimeError("Slippi replay muxer content audit scanned no video frames")
    if isinstance(minimum_pixels, bool) or not isinstance(minimum_pixels, int) or minimum_pixels <= 0:
        raise RuntimeError("Slippi replay muxer content audit sampled no gameplay-region pixels")
    if raw_audit.get("decoded_width") != expected_width or raw_audit.get("decoded_height") != expected_height:
        raise RuntimeError("Slippi replay muxer content audit decoded the wrong source dimensions")
    for key in (
        "near_white_frame_count",
        "near_black_frame_count",
        "near_neutral_blank_frame_count",
        "low_motion_transition_count",
    ):
        value = raw_audit.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= frames_scanned:
            raise RuntimeError(f"Slippi replay muxer content audit reported an invalid {key}")

    first_frame = number("first_decoded_frame_seconds")
    last_frame = number("last_decoded_frame_seconds")
    if first_frame < start - 1e-3 or first_frame > start + 0.100:
        raise RuntimeError("Slippi replay muxer content audit missed the start of gameplay")
    if last_frame < start or last_frame > end + 1e-3:
        raise RuntimeError("Slippi replay muxer content audit decoded outside gameplay")

    allowed_frame_gap = number("maximum_allowed_frame_gap_seconds")
    maximum_frame_gap = number("maximum_frame_gap_seconds")
    if allowed_frame_gap != CONTENT_AUDIT_MAXIMUM_FRAME_GAP_SECONDS:
        raise RuntimeError("Slippi replay muxer content audit weakened its frame-gap coverage")
    if maximum_frame_gap < 0.0 or maximum_frame_gap > allowed_frame_gap:
        raise RuntimeError("Slippi replay muxer content audit found an uncovered frame gap")

    maximum_blank = number("maximum_sustained_blank_seconds")
    if not math.isclose(
        maximum_blank,
        CONTENT_AUDIT_MAXIMUM_SUSTAINED_BLANK_SECONDS,
        abs_tol=1e-9,
    ):
        raise RuntimeError("Slippi replay muxer content audit used the wrong blank interval limit")
    for key in (
        "longest_near_white_interval_seconds",
        "longest_near_black_interval_seconds",
        "longest_near_neutral_blank_interval_seconds",
    ):
        value = number(key)
        if value < 0.0 or value > maximum_blank + 1e-6:
            raise RuntimeError(f"Slippi replay muxer content audit admitted a sustained blank: {key}")
    longest_interval_keys = {
        "near-white": "longest_near_white_interval_seconds",
        "near-black": "longest_near_black_interval_seconds",
        "near-neutral-blank": "longest_near_neutral_blank_interval_seconds",
    }
    for frame_class, key in longest_interval_keys.items():
        expected_longest = max(
            (
                cast(float, interval["source_duration_seconds"])
                for interval in blank_intervals
                if interval["class"] == frame_class
            ),
            default=0.0,
        )
        if not math.isclose(number(key), expected_longest, abs_tol=1e-9):
            raise RuntimeError("Slippi replay muxer content audit blank interval duration disagrees")
    maximum_low_motion = number("maximum_sustained_low_motion_seconds")
    longest_low_motion = number("longest_low_motion_interval_seconds")
    if not math.isclose(maximum_low_motion, 2.0, abs_tol=1e-9):
        raise RuntimeError("Slippi replay muxer content audit weakened its low-motion limit")
    if longest_low_motion < 0.0 or longest_low_motion > maximum_low_motion + 1e-6:
        raise RuntimeError("Slippi replay muxer content audit admitted a frozen gameplay interval")
    if not math.isclose(number("maximum_low_motion_mean_luma_delta"), 0.5, abs_tol=1e-9):
        raise RuntimeError("Slippi replay muxer content audit weakened its motion threshold")

    if number("near_white_rgb_minimum") != 245.0:
        raise RuntimeError("Slippi replay muxer content audit weakened its white threshold")
    if number("near_black_rgb_maximum") != 10.0:
        raise RuntimeError("Slippi replay muxer content audit weakened its black threshold")
    if not math.isclose(number("required_blank_pixel_fraction"), 0.995, abs_tol=1e-9):
        raise RuntimeError("Slippi replay muxer content audit weakened its pixel fraction")
    region = raw_audit.get("region")
    expected_region = {
        "left_fraction": 0.125,
        "right_fraction": 0.875,
        "top_fraction": 1.0 / 6.0,
        "bottom_fraction": 5.0 / 6.0,
    }
    if not isinstance(region, Mapping) or any(
        isinstance(region.get(key), bool)
        or not isinstance(region.get(key), (int, float))
        or not math.isclose(float(cast(float, region.get(key))), value, abs_tol=1e-9)
        for key, value in expected_region.items()
    ):
        raise RuntimeError("Slippi replay muxer content audit used the wrong gameplay region")


def _quantize_blank_mapping_seconds(seconds: float) -> float:
    scaled = seconds * SOURCE_BLANK_INTERVAL_MAPPING_TIMESCALE
    return math.floor(scaled + 0.5) / SOURCE_BLANK_INTERVAL_MAPPING_TIMESCALE


def _map_source_blank_time(
    source_seconds: float,
    clock_landmarks: list[dict[str, float]],
) -> tuple[float, int]:
    first_source = clock_landmarks[0]["source_video_seconds"]
    last_source = clock_landmarks[-1]["source_video_seconds"]
    if source_seconds < first_source - 1e-9 or source_seconds > last_source + 1e-9:
        raise RuntimeError("Slippi replay muxer source blank interval lies outside clock landmarks")
    source_seconds = min(last_source, max(first_source, source_seconds))
    first_source_quantized = _quantize_blank_mapping_seconds(first_source)
    target_start = 0.0
    for index, (left, right) in enumerate(pairwise(clock_landmarks)):
        left_source = left["source_video_seconds"]
        right_source = right["source_video_seconds"]
        target_duration = _quantize_blank_mapping_seconds(
            right["audio_seconds"] - left["audio_seconds"]
        )
        if math.isclose(source_seconds, left_source, abs_tol=1e-9):
            return target_start, index
        if source_seconds <= right_source + 1e-9:
            if math.isclose(source_seconds, right_source, abs_tol=1e-9):
                return _quantize_blank_mapping_seconds(target_start + target_duration), index
            source_coordinate = _quantize_blank_mapping_seconds(
                source_seconds - first_source_quantized
            )
            segment_source_start = _quantize_blank_mapping_seconds(left_source - first_source)
            segment_source_duration = _quantize_blank_mapping_seconds(right_source - left_source)
            if segment_source_duration <= 0.0:
                raise RuntimeError("Slippi replay muxer blank mapping produced a zero source segment")
            fraction = (source_coordinate - segment_source_start) / segment_source_duration
            fraction = min(1.0, max(0.0, fraction))
            timeline_seconds = target_start + fraction * target_duration
            return _quantize_blank_mapping_seconds(timeline_seconds), index
        target_start = _quantize_blank_mapping_seconds(target_start + target_duration)
    raise RuntimeError("Slippi replay muxer source blank interval missed every clock segment")


def _normalized_blank_mapping_landmarks(
    raw_landmarks: list[dict[str, float]],
) -> list[dict[str, float]]:
    if len(raw_landmarks) < 2:
        raise RuntimeError("Slippi replay muxer blank mapping needs at least two clock landmarks")
    landmarks: list[dict[str, float]] = []
    previous_source = -math.inf
    previous_audio = -math.inf
    for index, raw in enumerate(raw_landmarks):
        if not isinstance(raw, Mapping):
            raise TypeError(f"Slippi replay muxer clock landmark {index} must be an object")
        point: dict[str, float] = {}
        for key in ("source_video_seconds", "audio_seconds"):
            value = raw.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"Slippi replay muxer clock landmark {index} {key} must be a number")
            number = float(value)
            if not math.isfinite(number) or number < 0.0:
                raise ValueError(f"Slippi replay muxer clock landmark {index} {key} is invalid")
            point[key] = number
        if index and (
            point["source_video_seconds"] <= previous_source
            or point["audio_seconds"] <= previous_audio
        ):
            raise RuntimeError("Slippi replay muxer clock landmarks are not strictly increasing")
        landmarks.append(point)
        previous_source = point["source_video_seconds"]
        previous_audio = point["audio_seconds"]
    return landmarks


def _validate_mux_output_content_audit(
    metadata: Mapping[str, Any],
    *,
    expected_gameplay_end_seconds: float,
    expected_internal_join_count: int,
    expected_width: int,
    expected_height: int,
    expected_clock_landmarks: list[dict[str, float]],
) -> None:
    """Require a clean sequential decode of the exported gameplay timeline."""
    raw_audit = metadata.get("output_content_audit")
    if not isinstance(raw_audit, Mapping):
        raise TypeError("Slippi replay muxer output content audit must be an object")
    if raw_audit.get("method") != OUTPUT_CONTENT_AUDIT_METHOD:
        raise RuntimeError("Slippi replay muxer used an unknown output content audit method")
    if raw_audit.get("passed") is not True:
        raise RuntimeError("Slippi replay muxer output content audit did not pass")
    if raw_audit.get("blank_frame_policy") != SOURCE_FAITHFUL_BLANK_FRAME_POLICY:
        raise RuntimeError("Slippi replay muxer output content audit weakened its blank-frame policy")

    def number(key: str) -> float:
        value = raw_audit.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Slippi replay muxer output content audit {key} must be a number")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"Slippi replay muxer output content audit {key} must be finite")
        return result

    start = number("timeline_interval_start_seconds")
    end = number("timeline_interval_end_seconds")
    if not math.isclose(start, 0.0, abs_tol=1e-9):
        raise RuntimeError("Slippi replay muxer output content audit did not start at timeline zero")
    if not math.isclose(end, expected_gameplay_end_seconds, abs_tol=1e-3):
        raise RuntimeError("Slippi replay muxer output content audit ended outside gameplay")
    join_bracketing_end = number("join_bracketing_interval_end_seconds")
    if join_bracketing_end < end or join_bracketing_end > end + 0.101:
        raise RuntimeError("Slippi replay muxer output content audit used an invalid join-bracketing tail")
    if raw_audit.get("join_bracketing_tail_policy") != "timestamps-only-no-content-classification":
        raise RuntimeError("Slippi replay muxer output content audit classified the terminal visual tail")
    first_frame = number("first_decoded_frame_seconds")
    last_frame = number("last_decoded_frame_seconds")
    if first_frame < -1e-3 or first_frame > 0.100:
        raise RuntimeError("Slippi replay muxer output content audit missed gameplay start")
    if last_frame < first_frame or last_frame > end + 1e-3:
        raise RuntimeError("Slippi replay muxer output content audit decoded outside gameplay")

    minimum_allowed_frame_gap = number("minimum_allowed_frame_gap_seconds")
    maximum_allowed_frame_gap = number("maximum_allowed_frame_gap_seconds")
    minimum_frame_gap = number("minimum_frame_gap_seconds")
    maximum_frame_gap = number("maximum_frame_gap_seconds")
    if not math.isclose(
        minimum_allowed_frame_gap,
        OUTPUT_CONTENT_AUDIT_MINIMUM_FRAME_GAP_SECONDS,
        abs_tol=1e-12,
    ) or not math.isclose(
        maximum_allowed_frame_gap,
        OUTPUT_CONTENT_AUDIT_MAXIMUM_FRAME_GAP_SECONDS,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Slippi replay muxer output content audit weakened its frame-cadence coverage")
    if (
        minimum_frame_gap < minimum_allowed_frame_gap
        or maximum_frame_gap > maximum_allowed_frame_gap
        or maximum_frame_gap < minimum_frame_gap
    ):
        raise RuntimeError("Slippi replay muxer output content audit found invalid frame cadence")

    frames_scanned = raw_audit.get("frames_scanned")
    minimum_pixels = raw_audit.get("minimum_sampled_pixels_per_frame")
    if isinstance(frames_scanned, bool) or not isinstance(frames_scanned, int) or frames_scanned <= 0:
        raise RuntimeError("Slippi replay muxer output content audit scanned no frames")
    if isinstance(minimum_pixels, bool) or not isinstance(minimum_pixels, int) or minimum_pixels <= 0:
        raise RuntimeError("Slippi replay muxer output content audit sampled no gameplay pixels")
    expected_frame_count = math.ceil(expected_gameplay_end_seconds * DEFAULT_FRAME_RATE - 1e-9)
    if raw_audit.get("expected_frame_rate") != DEFAULT_FRAME_RATE:
        raise RuntimeError("Slippi replay muxer output content audit used the wrong expected frame rate")
    if raw_audit.get("expected_frame_count") != expected_frame_count:
        raise RuntimeError("Slippi replay muxer output content audit used the wrong expected frame count")
    frame_count_delta = raw_audit.get("frame_count_delta")
    frame_count_tolerance = raw_audit.get("frame_count_tolerance")
    if (
        isinstance(frame_count_delta, bool)
        or not isinstance(frame_count_delta, int)
        or frame_count_delta != abs(frames_scanned - expected_frame_count)
        or isinstance(frame_count_tolerance, bool)
        or not isinstance(frame_count_tolerance, int)
        or frame_count_tolerance != 1
        or frame_count_delta > frame_count_tolerance
        or raw_audit.get("cadence_passed") is not True
    ):
        raise RuntimeError("Slippi replay muxer output content audit found the wrong frame count")
    if raw_audit.get("decoded_width") != expected_width or raw_audit.get("decoded_height") != expected_height:
        raise RuntimeError("Slippi replay muxer output content audit decoded the wrong output dimensions")
    landmarks = _normalized_blank_mapping_landmarks(expected_clock_landmarks)
    if len(landmarks) - 2 != expected_internal_join_count:
        raise RuntimeError("Slippi replay muxer blank mapping landmark count disagrees with joins")
    if not math.isclose(landmarks[0]["audio_seconds"], 0.0, abs_tol=1e-9) or not math.isclose(
        landmarks[-1]["audio_seconds"], end, abs_tol=1e-3
    ):
        raise RuntimeError("Slippi replay muxer blank mapping landmarks miss the gameplay timeline")
    content_audit = metadata.get("content_audit")
    if not isinstance(content_audit, Mapping):
        raise TypeError("Slippi replay muxer source content audit must accompany output blank mapping")
    source_intervals = _validated_source_blank_intervals(
        content_audit,
        source_start_seconds=landmarks[0]["source_video_seconds"],
        source_end_seconds=landmarks[-1]["source_video_seconds"],
    )
    if raw_audit.get("blank_interval_mapping_method") != SOURCE_BLANK_INTERVAL_MAPPING_METHOD:
        raise RuntimeError("Slippi replay muxer output content audit used the wrong blank mapping method")
    if (
        raw_audit.get("blank_interval_mapping_quantization")
        != SOURCE_BLANK_INTERVAL_MAPPING_QUANTIZATION
    ):
        raise RuntimeError(
            "Slippi replay muxer output content audit used the wrong blank mapping quantization"
        )
    if raw_audit.get("blank_interval_mapping_timescale") != SOURCE_BLANK_INTERVAL_MAPPING_TIMESCALE:
        raise RuntimeError("Slippi replay muxer output content audit used the wrong blank mapping timescale")
    if not math.isclose(
        number("blank_interval_mapping_numerical_slack_seconds"),
        SOURCE_BLANK_INTERVAL_MAPPING_NUMERICAL_SLACK_SECONDS,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Slippi replay muxer output content audit weakened blank mapping precision")
    if raw_audit.get("blank_interval_mapping_numerical_slack_role") != "metadata-reconciliation-only":
        raise RuntimeError("Slippi replay muxer output content audit applied mapping slack to membership")
    if not math.isclose(number("blank_interval_membership_tolerance_seconds"), 0.0, abs_tol=1e-12):
        raise RuntimeError("Slippi replay muxer output content audit weakened blank membership precision")
    raw_mapped_intervals = raw_audit.get("mapped_source_blank_intervals")
    if not isinstance(raw_mapped_intervals, list):
        raise TypeError("Slippi replay muxer mapped source blank intervals must be an array")
    if (
        raw_audit.get("source_blank_interval_count") != len(source_intervals)
        or raw_audit.get("mapped_source_blank_interval_count") != len(source_intervals)
        or len(raw_mapped_intervals) != len(source_intervals)
    ):
        raise RuntimeError("Slippi replay muxer output content audit lost source blank intervals")

    mapped_intervals: list[dict[str, float | int | str]] = []
    for index, (source_interval, raw_mapped) in enumerate(
        zip(source_intervals, raw_mapped_intervals, strict=True)
    ):
        if not isinstance(raw_mapped, Mapping):
            raise TypeError(f"Slippi replay muxer mapped blank interval {index} must be an object")

        def mapped_number(
            key: str,
            *,
            mapped: Mapping[str, Any] = raw_mapped,
            mapped_index: int = index,
        ) -> float:
            value = mapped.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"Slippi replay muxer mapped blank interval {mapped_index} {key} must be a number"
                )
            result = float(value)
            if not math.isfinite(result):
                raise ValueError(
                    f"Slippi replay muxer mapped blank interval {mapped_index} {key} must be finite"
                )
            return result

        expected_timeline_start, expected_start_segment = _map_source_blank_time(
            cast(float, source_interval["source_start_seconds"]), landmarks
        )
        expected_timeline_end, expected_end_segment = _map_source_blank_time(
            cast(float, source_interval["source_end_seconds"]), landmarks
        )
        expected_numbers = {
            "source_start_seconds": cast(float, source_interval["source_start_seconds"]),
            "source_end_seconds": cast(float, source_interval["source_end_seconds"]),
            "source_duration_seconds": cast(float, source_interval["source_duration_seconds"]),
            "timeline_start_seconds": expected_timeline_start,
            "timeline_end_seconds": expected_timeline_end,
            "timeline_duration_seconds": expected_timeline_end - expected_timeline_start,
        }
        if raw_mapped.get("class") != source_interval["class"]:
            raise RuntimeError("Slippi replay muxer mapped blank interval changed class")
        if raw_mapped.get("source_interval_index") != index:
            raise RuntimeError("Slippi replay muxer mapped blank interval changed source ordering")
        if raw_mapped.get("source_frame_count") != source_interval["source_frame_count"]:
            raise RuntimeError("Slippi replay muxer mapped blank interval changed source frame count")
        if (
            raw_mapped.get("start_clock_segment_index") != expected_start_segment
            or raw_mapped.get("end_clock_segment_index") != expected_end_segment
        ):
            raise RuntimeError("Slippi replay muxer mapped blank interval used the wrong clock segment")
        if any(
            not math.isclose(mapped_number(key), value, abs_tol=1e-9)
            for key, value in expected_numbers.items()
        ):
            raise RuntimeError("Slippi replay muxer mapped blank interval differs from the exact clock map")
        if expected_timeline_start < -1e-9 or expected_timeline_end > end + 1e-3:
            raise RuntimeError("Slippi replay muxer mapped blank interval lies outside gameplay")
        mapped_intervals.append(
            {
                "class": cast(str, source_interval["class"]),
                "timeline_start_seconds": expected_timeline_start,
                "timeline_end_seconds": expected_timeline_end,
                "source_interval_index": index,
            }
        )

    pts_by_class: dict[str, list[float]] = {}
    for frame_class, key in (
        ("near-white", "near_white_pts"),
        ("near-black", "near_black_pts"),
        ("near-neutral-blank", "split_neutral_pts"),
    ):
        raw_pts = raw_audit.get(key)
        if not isinstance(raw_pts, list):
            raise TypeError(f"Slippi replay muxer output content audit {key} must be an array")
        pts: list[float] = []
        for raw_pts_value in raw_pts:
            if isinstance(raw_pts_value, bool) or not isinstance(raw_pts_value, (int, float)):
                raise TypeError(f"Slippi replay muxer output content audit {key} must contain numbers")
            pts_value = float(raw_pts_value)
            if (
                not math.isfinite(pts_value)
                or pts_value < -1e-9
                or pts_value > end + 1e-9
                or (pts and pts_value <= pts[-1] + 1e-9)
            ):
                raise RuntimeError(f"Slippi replay muxer output content audit {key} is invalid")
            pts.append(pts_value)
        pts_by_class[frame_class] = pts
    count_keys = {
        "near-white": "near_white_frame_count",
        "near-black": "near_black_frame_count",
        "near-neutral-blank": "near_neutral_blank_frame_count",
    }
    for frame_class, key in count_keys.items():
        if raw_audit.get(key) != len(pts_by_class[frame_class]):
            raise RuntimeError("Slippi replay muxer output blank PTS count disagrees with frame count")

    raw_observations = raw_audit.get("blank_frame_observations")
    if not isinstance(raw_observations, list):
        raise TypeError("Slippi replay muxer output blank observations must be an array")
    if raw_audit.get("blank_frame_observation_count") != len(raw_observations):
        raise RuntimeError("Slippi replay muxer output blank observation count disagrees")
    observed_actual_pts: dict[str, list[float]] = {
        frame_class: [] for frame_class in SOURCE_BLANK_CLASSES
    }
    expected_source_blank_frames = 0
    source_faithful_blank_frames = 0
    unexpected_blank_frames = 0
    missing_source_blank_frames = 0
    blank_class_mismatch_frames = 0
    previous_observation_pts = -math.inf
    for index, observation in enumerate(raw_observations):
        if not isinstance(observation, Mapping):
            raise TypeError(f"Slippi replay muxer blank observation {index} must be an object")
        raw_pts_value = observation.get("timeline_seconds")
        if isinstance(raw_pts_value, bool) or not isinstance(raw_pts_value, (int, float)):
            raise TypeError(f"Slippi replay muxer blank observation {index} PTS must be a number")
        pts_value = float(raw_pts_value)
        if (
            not math.isfinite(pts_value)
            or pts_value < -1e-9
            or pts_value > end + 1e-9
            or pts_value <= previous_observation_pts + 1e-9
        ):
            raise RuntimeError("Slippi replay muxer blank observations are invalid or unordered")
        previous_observation_pts = pts_value
        expected_interval_index = next(
            (
                mapped_index
                for mapped_index, interval in enumerate(mapped_intervals)
                if cast(float, interval["timeline_start_seconds"])
                <= pts_value
                < cast(float, interval["timeline_end_seconds"])
            ),
            None,
        )
        expected_class = (
            cast(str, mapped_intervals[expected_interval_index]["class"])
            if expected_interval_index is not None
            else "active"
        )
        actual_class_value = observation.get("actual_class")
        if actual_class_value != "active" and actual_class_value not in SOURCE_BLANK_CLASSES:
            raise RuntimeError("Slippi replay muxer blank observation has an unknown actual class")
        actual_class = cast(str, actual_class_value)
        if observation.get("expected_class") != expected_class:
            raise RuntimeError("Slippi replay muxer blank observation differs from mapped source class")
        if observation.get("mapped_source_interval_index") != expected_interval_index:
            raise RuntimeError("Slippi replay muxer blank observation references the wrong source interval")
        source_faithful = actual_class == expected_class
        if observation.get("source_faithful") is not source_faithful:
            raise RuntimeError("Slippi replay muxer blank observation has an invalid faithfulness result")
        expected_blank = expected_class != "active"
        actual_blank = actual_class != "active"
        if not (expected_blank or actual_blank):
            raise RuntimeError("Slippi replay muxer emitted an irrelevant active blank observation")
        expected_source_blank_frames += expected_blank
        source_faithful_blank_frames += actual_blank and source_faithful
        unexpected_blank_frames += actual_blank and not source_faithful
        missing_source_blank_frames += expected_blank and not actual_blank
        blank_class_mismatch_frames += expected_blank and actual_blank and not source_faithful
        if actual_blank:
            observed_actual_pts[actual_class].append(pts_value)

    for frame_class in SOURCE_BLANK_CLASSES:
        if observed_actual_pts[frame_class] != pts_by_class[frame_class]:
            raise RuntimeError("Slippi replay muxer blank observations disagree with exact PTS arrays")
    expected_counts = {
        "expected_source_blank_frame_count": expected_source_blank_frames,
        "source_faithful_blank_frame_count": source_faithful_blank_frames,
        "unexpected_blank_frame_count": unexpected_blank_frames,
        "missing_source_blank_frame_count": missing_source_blank_frames,
        "blank_class_mismatch_frame_count": blank_class_mismatch_frames,
        "source_faithfulness_mismatch_frame_count": (
            unexpected_blank_frames + missing_source_blank_frames
        ),
    }
    for key, expected_value in expected_counts.items():
        if raw_audit.get(key) != expected_value:
            raise RuntimeError(f"Slippi replay muxer output content audit reported an invalid {key}")
    if unexpected_blank_frames or missing_source_blank_frames or blank_class_mismatch_frames:
        raise RuntimeError("Slippi replay muxer output content audit found source-unfaithful blank frames")
    low_motion_transitions = raw_audit.get("low_motion_transition_count")
    if (
        isinstance(low_motion_transitions, bool)
        or not isinstance(low_motion_transitions, int)
        or not 0 <= low_motion_transitions < frames_scanned
    ):
        raise RuntimeError("Slippi replay muxer output content audit reported invalid motion coverage")
    maximum_low_motion = number("maximum_sustained_low_motion_seconds")
    longest_low_motion = number("longest_low_motion_interval_seconds")
    if not math.isclose(maximum_low_motion, 2.0, abs_tol=1e-9):
        raise RuntimeError("Slippi replay muxer output content audit weakened its low-motion limit")
    if longest_low_motion < 0.0 or longest_low_motion > maximum_low_motion + 1e-6:
        raise RuntimeError("Slippi replay muxer output content audit admitted a frozen gameplay interval")
    if not math.isclose(number("maximum_low_motion_mean_luma_delta"), 0.5, abs_tol=1e-9):
        raise RuntimeError("Slippi replay muxer output content audit weakened its motion threshold")
    join_count = raw_audit.get("internal_piecewise_join_count")
    if (
        isinstance(join_count, bool)
        or not isinstance(join_count, int)
        or join_count != expected_internal_join_count
    ):
        raise RuntimeError("Slippi replay muxer output content audit missed piecewise joins")
    allowed_gap = number("maximum_allowed_join_sample_gap_seconds")
    actual_gap = number("maximum_join_sample_gap_seconds")
    if not math.isclose(
        allowed_gap,
        OUTPUT_CONTENT_AUDIT_MAXIMUM_JOIN_SAMPLE_GAP_SECONDS,
        abs_tol=1e-9,
    ):
        raise RuntimeError("Slippi replay muxer output content audit weakened its join coverage")
    if actual_gap < 0.0 or actual_gap > allowed_gap + 1e-6:
        raise RuntimeError("Slippi replay muxer output content audit found an uncovered join")


def _validate_recorder_capture_dimensions(
    metadata: Mapping[str, Any],
    *,
    minimum_width: int,
    minimum_height: int,
    expected_width: int | None,
    expected_height: int | None,
) -> tuple[int, int]:
    """Require the stable full-size playback window selected before capture."""
    values: dict[str, int] = {}
    for key in (
        "width",
        "height",
        "minimum_width",
        "minimum_height",
        "expected_width",
        "expected_height",
        "window_selection_stable_polls",
    ):
        value = metadata.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"isolated-window recorder {key} must be an integer")
        values[key] = value
    if values["minimum_width"] != minimum_width or values["minimum_height"] != minimum_height:
        raise RuntimeError("isolated-window recorder weakened its minimum capture dimensions")
    encoded_expected_width = expected_width or 0
    encoded_expected_height = expected_height or 0
    if (
        values["expected_width"] != encoded_expected_width
        or values["expected_height"] != encoded_expected_height
    ):
        raise RuntimeError("isolated-window recorder used the wrong expected capture dimensions")
    if values["width"] < minimum_width or values["height"] < minimum_height:
        raise RuntimeError("isolated-window recorder selected a transient undersized window")
    if expected_width is not None and (
        values["width"] != expected_width or values["height"] != expected_height
    ):
        raise RuntimeError("isolated-window recorder selected the wrong playback window dimensions")
    if values["window_selection_stable_polls"] < 4:
        raise RuntimeError("isolated-window recorder did not wait for a stable playback window")
    return values["width"], values["height"]


def _validate_recorder_capture_rate(metadata: Mapping[str, Any]) -> None:
    """Require the capture cadence and slowed Dolphin playback requested by Python."""
    requested_capture_fps = metadata.get("requested_capture_fps")
    if (
        isinstance(requested_capture_fps, bool)
        or not isinstance(requested_capture_fps, int)
        or requested_capture_fps != RENDER_CAPTURE_FPS
    ):
        raise RuntimeError("isolated-window recorder used the wrong requested capture frame rate")
    emulation_speed = metadata.get("playback_emulation_speed")
    if (
        isinstance(emulation_speed, bool)
        or not isinstance(emulation_speed, (int, float))
        or not math.isfinite(float(emulation_speed))
        or float(emulation_speed) != RENDER_CAPTURE_EMULATION_SPEED
    ):
        raise RuntimeError("isolated-window recorder used the wrong playback emulation speed")


def _validate_recorder_capture_delivery(metadata: Mapping[str, Any]) -> None:
    """Recompute and require the native recorder's lossless delivery contract."""
    raw_delivery = metadata.get("capture_delivery")
    if not isinstance(raw_delivery, Mapping):
        raise TypeError("isolated-window recorder capture delivery must be an object")
    delivery = raw_delivery

    def integer(container: Mapping[str, Any], key: str) -> int:
        value = container.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"capture delivery {key} must be an integer")
        return value

    def number(container: Mapping[str, Any], key: str) -> float:
        value = container.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"capture delivery {key} must be a number")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"capture delivery {key} must be finite")
        return result

    phased_startup = delivery.get("schema") == "sck-stream-output-avassetwriter-v4"
    if delivery.get("schema") != CAPTURE_DELIVERY_SCHEMA and not phased_startup:
        raise RuntimeError("isolated-window recorder used an unknown capture delivery schema")
    if delivery.get("status_source") != "SCStreamFrameInfoStatus":
        raise RuntimeError("capture delivery used the wrong frame-status source")
    if delivery.get("time_source") != "SCStreamFrameInfoDisplayTime-mach-absolute":
        raise RuntimeError("capture delivery used the wrong display-time source")
    if integer(delivery, "requested_capture_fps") != RENDER_CAPTURE_FPS:
        raise RuntimeError("capture delivery used the wrong requested capture frame rate")
    if integer(delivery, "queue_depth") != 8:
        raise RuntimeError("capture delivery weakened its ScreenCaptureKit queue depth")
    if delivery.get("pixel_format") != "420v" or delivery.get("writer_codec") != "avc1":
        raise RuntimeError("capture delivery used an unexpected pixel format or writer codec")
    if integer(delivery, "writer_expected_source_fps") != RENDER_CAPTURE_FPS:
        raise RuntimeError("capture delivery used the wrong writer source frame rate")

    expected_status_names = {
        "started",
        "complete",
        "idle",
        "blank",
        "suspended",
        "stopped",
        "unknown",
    }
    raw_status_counts = delivery.get("status_counts")
    if not isinstance(raw_status_counts, Mapping) or set(raw_status_counts) != expected_status_names:
        raise RuntimeError("capture delivery status counts have the wrong schema")
    status_counts = {name: integer(raw_status_counts, name) for name in expected_status_names}
    if any(value < 0 for value in status_counts.values()):
        raise RuntimeError("capture delivery status counts must be nonnegative")
    callback_count = integer(delivery, "callback_count")
    if callback_count < 1 or sum(status_counts.values()) != callback_count:
        raise RuntimeError("capture delivery callback accounting is inconsistent")
    if any(status_counts[name] != 0 for name in ("blank", "suspended", "unknown")):
        raise RuntimeError("capture delivery admitted a blank, suspended, or unknown frame status")
    if status_counts["started"] not in (0, 1) or status_counts["stopped"] not in (0, 1):
        raise RuntimeError("capture delivery contains duplicate Started or Stopped callbacks")
    if status_counts["complete"] < 1:
        raise RuntimeError("capture delivery contains no Complete image callback")

    zero_count_fields = (
        "drop_attachment_count",
        "writer_backpressure_count",
        "append_failure_count",
        "invalid_sample_count",
    )
    if any(integer(delivery, key) != 0 for key in zero_count_fields):
        raise RuntimeError("capture delivery reported a drop, backpressure, append, or sample failure")
    drop_reasons = delivery.get("drop_reasons")
    if not isinstance(drop_reasons, Mapping) or drop_reasons:
        raise RuntimeError("capture delivery reported a dropped-frame reason")

    expected_status_codes = {
        "S": "started",
        "C": "complete",
        "I": "idle",
        "B": "blank",
        "U": "suspended",
        "T": "stopped",
        "X": "unknown",
    }
    expected_class_codes = {
        "A": "active",
        "W": "near-white",
        "K": "near-black",
        "N": "neutral-blank",
        "X": "unknown",
    }
    expected_flag_bits = {
        "valid_sample": 1,
        "valid_status": 2,
        "valid_display_time": 4,
        "drop_attachment": 8,
        "valid_pixel": 16,
        "appended": 32,
        "idle_hold": 64,
        "stopped": 128,
        "gameplay_armed": 256,
        "active_content": 512,
        "stop_requested": 1024,
    }
    raw_callback_trace = delivery.get("callback_trace")
    if not isinstance(raw_callback_trace, Mapping):
        raise TypeError("capture delivery callback trace must be an object")
    if (
        raw_callback_trace.get("encoding")
        != "sequence,relative-display-us,delivery-lag-us,service-us,status,class,flags,visual-fnv64-v1"
        or raw_callback_trace.get("fields")
        != [
            "sequence",
            "relative_display_us",
            "delivery_lag_us",
            "service_us",
            "status_code",
            "content_class_code",
            "outcome_flags",
            "visual_signature",
        ]
        or raw_callback_trace.get("status_codes") != expected_status_codes
        or raw_callback_trace.get("content_class_codes") != expected_class_codes
        or raw_callback_trace.get("flag_bits") != expected_flag_bits
    ):
        raise RuntimeError("capture delivery callback trace schema is invalid")
    if integer(raw_callback_trace, "row_count") != callback_count:
        raise RuntimeError("capture delivery callback trace row count is inconsistent")
    callback_rows_text = raw_callback_trace.get("rows")
    if not isinstance(callback_rows_text, str) or not callback_rows_text.endswith("\n"):
        raise TypeError("capture delivery callback trace rows must be newline-terminated text")
    callback_lines = callback_rows_text.splitlines()
    if len(callback_lines) != callback_count:
        raise RuntimeError("capture delivery callback trace is incomplete")

    all_flag_mask = sum(expected_flag_bits.values())
    common_flags = (
        expected_flag_bits["valid_sample"]
        | expected_flag_bits["valid_status"]
        | expected_flag_bits["valid_display_time"]
    )
    status_code_counts = {code: 0 for code in expected_status_codes}
    parsed_callback_rows: list[tuple[int, int, int, int, str, str, int, str]] = []
    appended_callback_rows: list[tuple[int, str]] = []
    gameplay_armed_indices: list[int] = []
    previous_relative_us = -1
    previous_visual: tuple[str, str] | None = None
    stop_requested_seen = False
    for index, line in enumerate(callback_lines, start=1):
        columns = line.split(",")
        if len(columns) != 8:
            raise RuntimeError("capture delivery callback trace row has the wrong field count")
        try:
            sequence = int(columns[0])
            relative_us = int(columns[1])
            delivery_lag_us = int(columns[2])
            service_us = int(columns[3])
            flags = int(columns[6])
        except ValueError as error:
            raise RuntimeError("capture delivery callback trace contains a non-integer field") from error
        status_code, class_code, signature = columns[4], columns[5], columns[7]
        canonical = (
            f"{sequence},{relative_us},{delivery_lag_us},{service_us},"
            f"{status_code},{class_code},{flags},{signature}"
        )
        if line != canonical:
            raise RuntimeError("capture delivery callback trace is not canonically encoded")
        if sequence != index or relative_us < 0 or relative_us <= previous_relative_us:
            raise RuntimeError("capture delivery callback sequence or display time is invalid")
        if index == 1 and relative_us != 0:
            raise RuntimeError("capture delivery callback trace does not start at session time zero")
        if not 0 <= delivery_lag_us <= round(
            CAPTURE_DELIVERY_MAXIMUM_CALLBACK_LAG_SECONDS * 1_000_000
        ):
            raise RuntimeError("capture delivery callback lag exceeds its strict limit")
        if not 0 <= service_us <= round(
            CAPTURE_DELIVERY_MAXIMUM_CALLBACK_SERVICE_SECONDS * 1_000_000
        ):
            raise RuntimeError("capture delivery callback service time exceeds its strict limit")
        if status_code not in expected_status_codes or class_code not in expected_class_codes:
            raise RuntimeError("capture delivery callback trace contains an unknown code")
        if flags < 0 or flags & ~all_flag_mask or flags & common_flags != common_flags:
            raise RuntimeError("capture delivery callback trace contains invalid outcome flags")
        if flags & expected_flag_bits["drop_attachment"]:
            raise RuntimeError("capture delivery callback trace contains a dropped-frame attachment")
        if re.fullmatch(r"[0-9a-f]{16}", signature) is None:
            raise RuntimeError("capture delivery callback trace contains an invalid visual signature")

        appended = bool(flags & expected_flag_bits["appended"])
        idle_hold = bool(flags & expected_flag_bits["idle_hold"])
        stopped = bool(flags & expected_flag_bits["stopped"])
        valid_pixel = bool(flags & expected_flag_bits["valid_pixel"])
        active_content = bool(flags & expected_flag_bits["active_content"])
        stop_requested = bool(flags & expected_flag_bits["stop_requested"])
        if stop_requested_seen and not stop_requested:
            raise RuntimeError("capture delivery stop-request flags do not form a terminal suffix")
        stop_requested_seen = stop_requested_seen or stop_requested
        if flags & expected_flag_bits["gameplay_armed"]:
            gameplay_armed_indices.append(index - 1)

        if status_code in ("S", "C"):
            if not appended or idle_hold or stopped or not valid_pixel or class_code == "X":
                raise RuntimeError("capture delivery image callback has invalid append or pixel flags")
            if active_content != (class_code == "A"):
                raise RuntimeError("capture delivery active-content flag disagrees with its frame class")
            previous_visual = (class_code, signature)
            appended_callback_rows.append((relative_us, status_code))
        elif status_code == "I":
            if not appended or not idle_hold or stopped or previous_visual is None:
                raise RuntimeError("capture delivery idle callback is not a retained visual hold")
            if (class_code, signature) != previous_visual:
                raise RuntimeError("capture delivery idle hold changed its retained visual state")
            if active_content != (class_code == "A"):
                raise RuntimeError("capture delivery idle active-content flag disagrees with its frame class")
            appended_callback_rows.append((relative_us, status_code))
        elif status_code == "T":
            if (
                index != callback_count
                or appended
                or idle_hold
                or valid_pixel
                or not stopped
                or not stop_requested
                or active_content
                or flags & expected_flag_bits["gameplay_armed"]
                or class_code != "X"
                or signature != "0000000000000000"
            ):
                raise RuntimeError("capture delivery stopped callback has invalid terminal state")
        else:
            raise RuntimeError("capture delivery admitted an unsuccessful callback status")
        if status_code == "S" and index != 1:
            raise RuntimeError("capture delivery Started callback appeared after session origin")

        status_code_counts[status_code] += 1
        parsed_callback_rows.append(
            (
                sequence,
                relative_us,
                delivery_lag_us,
                service_us,
                status_code,
                class_code,
                flags,
                signature,
            )
        )
        previous_relative_us = relative_us

    if parsed_callback_rows[0][4] not in ("S", "C"):
        raise RuntimeError("capture delivery session origin is not an image callback")
    if status_code_counts["S"] == 1 and parsed_callback_rows[0][4] != "S":
        raise RuntimeError("capture delivery Started callback appeared after session origin")
    if not gameplay_armed_indices:
        raise RuntimeError("capture delivery trace never entered its gameplay interval")
    if gameplay_armed_indices != list(
        range(gameplay_armed_indices[0], gameplay_armed_indices[-1] + 1)
    ):
        raise RuntimeError("capture delivery gameplay callbacks do not form one contiguous interval")
    startup_callback_count = 0
    startup_source_us = -1
    if phased_startup:
        phase = delivery.get("startup_phase")
        sync = metadata.get("startup_sync")
        if not isinstance(phase, Mapping) or not isinstance(sync, Mapping):
            raise TypeError("capture delivery startup phase lacks its opening proof")
        capture_boundary = sync.get("capture_boundary")
        audio_boundary = sync.get("idle_audio_boundary")
        if (
            phase.get("method") != "validated-opening-preroll-v1"
            or phase.get("cutoff_immutable") is not True
            or phase.get("complete") is not True
            or sync.get("schema") != STARTUP_SYNC_SCHEMA
            or not isinstance(capture_boundary, Mapping)
            or not isinstance(audio_boundary, Mapping)
            or audio_boundary.get("passed") is not True
            or audio_boundary.get("visible_pcm_frames_unchanged_at_zero") is not True
            or audio_boundary.get("logical_dsp_barrier_frames") != 0
            or audio_boundary.get("logical_dtk_barrier_frames") != 0
            or any(capture_boundary.get(key) is not True for key in (
                "passed", "fresh_callback_after_second_stop", "complete_sample_advanced",
                "display_time_advanced", "visual_signature_concrete",
            ))
        ):
            raise RuntimeError("capture delivery startup phase lacks validated opening/zero-audio proof")
        for checkpoint in ("initial", "first_stop", "second_stop"):
            state = audio_boundary.get(checkpoint)
            if not isinstance(state, Mapping) or any(
                not isinstance(state.get(stream), Mapping)
                or state[stream].get("zero_audio_passed") is not True
                or state[stream].get("logical_data_frames") != 0
                for stream in ("dsp", "dtk")
            ):
                raise RuntimeError("capture delivery startup phase contains pre-boundary audio")
        startup_callback_count = integer(phase, "last_callback_sequence")
        if startup_callback_count != gameplay_armed_indices[0] or startup_callback_count < 1:
            raise RuntimeError("capture delivery startup cutoff differs from first armed callback")
        startup_source_us = parsed_callback_rows[startup_callback_count - 1][1]
        armed = capture_boundary.get("armed_after_opening")
        settled = capture_boundary.get("settled_after_second_stop")
        if not isinstance(armed, Mapping) or not isinstance(settled, Mapping):
            raise TypeError("capture delivery startup cutoff lacks atomic opening snapshots")
        settled_count = integer(settled, "callback_count")
        if (
            integer(armed, "callback_count") != startup_callback_count
            or not 1 <= settled_count <= startup_callback_count
            or any(not math.isclose(value, startup_source_us / 1_000_000, abs_tol=1.1e-6) for value in (
                number(phase, "source_video_seconds"),
                number(armed, "last_relative_display_seconds"),
                number(capture_boundary, "video_alignment_offset_seconds"),
                number(metadata, "video_alignment_offset_seconds"),
            ))
        ):
            raise RuntimeError("capture delivery startup cutoff is not the exact crop/arm boundary")
        settled_visual = (settled.get("last_content_class"), settled.get("last_visual_signature"))
        if settled_visual[0] not in ("A", "W", "K", "N") or settled_visual[1] == "0000000000000000":
            raise RuntimeError("capture delivery startup opening visual is not concrete")
        if any(
            (row[5], row[7]) != settled_visual
            for row in parsed_callback_rows[settled_count - 1:startup_callback_count]
        ):
            raise RuntimeError("capture delivery startup visual changed after the frozen opening proof")
    for code, name in expected_status_codes.items():
        if status_code_counts[code] != status_counts[name]:
            raise RuntimeError("capture delivery callback trace disagrees with status counts")

    raw_recovery = delivery.get("terminal_gap_recovery")
    if not isinstance(raw_recovery, Mapping):
        raise TypeError("capture delivery terminal gap recovery must be an object")
    recovery_encoding = (
        "recovery-index,prior-callback-sequence,current-callback-sequence,"
        "prior-relative-display-us,current-relative-display-us,"
        "inserted-relative-pts-us,prior-content-class-code,"
        "prior-visual-fnv64-v1"
    )
    if (
        raw_recovery.get("method") != "retained-prior-pixel-terminal-window-v1"
        or raw_recovery.get("encoding") != recovery_encoding
        or not math.isclose(
            number(raw_recovery, "maximum_allowed_normal_gap_seconds"),
            CAPTURE_DELIVERY_MAXIMUM_GAP_SECONDS,
            abs_tol=1e-12,
        )
        or not math.isclose(
            number(raw_recovery, "maximum_allowed_recoverable_gap_seconds"),
            CAPTURE_DELIVERY_TERMINAL_RECOVERY_CEILING_SECONDS,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError("capture delivery terminal gap recovery contract is invalid")
    recovery_interval_count = integer(raw_recovery, "recovery_interval_count")
    recovery_inserted_count = integer(raw_recovery, "inserted_sample_count")
    recovery_row_count = integer(raw_recovery, "row_count")
    if (
        recovery_interval_count not in (0, 1)
        or recovery_inserted_count != recovery_interval_count
        or recovery_row_count != recovery_interval_count
    ):
        raise RuntimeError("capture delivery admitted multiple or miscounted gap recoveries")
    recovery_rows_text = raw_recovery.get("rows")
    if not isinstance(recovery_rows_text, str) or (
        recovery_interval_count == 0 and recovery_rows_text != ""
    ) or (
        recovery_interval_count == 1 and not recovery_rows_text.endswith("\n")
    ):
        raise TypeError("capture delivery terminal gap recovery rows are malformed")
    recovery_lines = recovery_rows_text.splitlines()
    if len(recovery_lines) != recovery_interval_count:
        raise RuntimeError("capture delivery terminal gap recovery trace is incomplete")

    barrier_sequence = integer(raw_recovery, "terminal_barrier_callback_sequence")
    accepted_sequence = integer(
        raw_recovery, "accepted_terminal_complete_callback_sequence"
    )
    final_frozen_tail_sequence = integer(
        raw_recovery, "final_frozen_tail_callback_sequence"
    )
    barrier_relative_us = integer(
        raw_recovery, "terminal_barrier_relative_display_us"
    )
    final_frozen_tail_relative_us = integer(
        raw_recovery, "final_frozen_tail_relative_display_us"
    )
    if (
        barrier_sequence < 1
        or not barrier_sequence < accepted_sequence <= final_frozen_tail_sequence
        or final_frozen_tail_sequence > callback_count
        or barrier_relative_us < 0
        or final_frozen_tail_relative_us < barrier_relative_us
    ):
        raise RuntimeError("capture delivery terminal recovery window is invalid")

    raw_sync = metadata.get("startup_sync")
    if (
        not isinstance(raw_sync, Mapping)
        or raw_sync.get("schema") != STARTUP_SYNC_SCHEMA
        or raw_sync.get("method") != STARTUP_SYNC_SCHEMA
    ):
        raise TypeError("capture delivery recovery lacks startup synchronization proof")
    raw_terminal_apply = raw_sync.get("terminal_apply_boundary")
    if not isinstance(raw_terminal_apply, Mapping) or raw_terminal_apply.get(
        "method"
    ) != (
        "inclusive-current-frame-resume-exact-complete-audio-stop-"
        "retained-callback-interval-v7"
    ):
        raise TypeError("capture delivery recovery lacks terminal apply evidence")
    if isinstance(raw_terminal_apply, Mapping):
        raw_barrier = raw_terminal_apply.get("at_inclusive_frame_barrier")
        raw_accepted = raw_terminal_apply.get("accepted_post_resume_complete")
        raw_final_frozen_tail = raw_terminal_apply.get("after_final_frozen_tail")
        if not all(
            isinstance(snapshot, Mapping)
            for snapshot in (raw_barrier, raw_accepted, raw_final_frozen_tail)
        ):
            raise TypeError("capture delivery recovery terminal snapshots are malformed")
        assert isinstance(raw_barrier, Mapping)
        assert isinstance(raw_accepted, Mapping)
        assert isinstance(raw_final_frozen_tail, Mapping)
        expected_barrier_relative_us = math.floor(
            number(raw_barrier, "last_relative_display_seconds") * 1_000_000.0
            + 0.5
        )
        expected_final_relative_us = math.floor(
            number(raw_final_frozen_tail, "last_relative_display_seconds")
            * 1_000_000.0
            + 0.5
        )
        if (
            barrier_sequence != integer(raw_barrier, "callback_count")
            or accepted_sequence
            != integer(raw_accepted, "last_complete_callback_sequence")
            or final_frozen_tail_sequence
            != integer(raw_final_frozen_tail, "callback_count")
            or barrier_relative_us != expected_barrier_relative_us
            or final_frozen_tail_relative_us != expected_final_relative_us
        ):
            raise RuntimeError(
                "capture delivery recovery window disagrees with terminal snapshots"
            )

    recovery_pairs: set[tuple[int, int]] = set()
    recovery_inserted_pts_us: list[int] = []
    controlled_transition_count = 0
    retained_state_count = 0
    for recovery_index, line in enumerate(recovery_lines, start=1):
        columns = line.split(",")
        if len(columns) != 8:
            raise RuntimeError("capture delivery recovery row has the wrong field count")
        try:
            encoded_index = int(columns[0])
            prior_sequence = int(columns[1])
            current_sequence = int(columns[2])
            prior_relative_us = int(columns[3])
            current_relative_us = int(columns[4])
            inserted_relative_us = int(columns[5])
        except ValueError as error:
            raise RuntimeError("capture delivery recovery row is noncanonical") from error
        prior_class, prior_signature = columns[6], columns[7]
        canonical = (
            f"{encoded_index},{prior_sequence},{current_sequence},"
            f"{prior_relative_us},{current_relative_us},{inserted_relative_us},"
            f"{prior_class},{prior_signature}"
        )
        if line != canonical or encoded_index != recovery_index:
            raise RuntimeError("capture delivery recovery row is noncanonical")
        if (
            prior_sequence < 1
            or current_sequence != prior_sequence + 1
            or current_sequence > callback_count
        ):
            raise RuntimeError("capture delivery recovery endpoints are not adjacent callbacks")
        prior_row = parsed_callback_rows[prior_sequence - 1]
        current_row = parsed_callback_rows[current_sequence - 1]
        if (
            prior_relative_us != prior_row[1]
            or current_relative_us != current_row[1]
            or prior_class != prior_row[5]
            or prior_signature != prior_row[7]
        ):
            raise RuntimeError("capture delivery recovery row disagrees with callback trace")
        raw_gap_us = current_relative_us - prior_relative_us
        canonical_midpoint_us = prior_relative_us + raw_gap_us // 2
        if (
            raw_gap_us <= round(CAPTURE_DELIVERY_MAXIMUM_GAP_SECONDS * 1_000_000)
            or raw_gap_us
            > round(
                CAPTURE_DELIVERY_TERMINAL_RECOVERY_CEILING_SECONDS * 1_000_000
            )
            or inserted_relative_us != canonical_midpoint_us
            or inserted_relative_us - prior_relative_us
            > round(CAPTURE_DELIVERY_MAXIMUM_GAP_SECONDS * 1_000_000)
            or current_relative_us - inserted_relative_us
            > round(CAPTURE_DELIVERY_MAXIMUM_GAP_SECONDS * 1_000_000)
        ):
            raise RuntimeError("capture delivery recovery midpoint or raw gap is invalid")
        if (
            prior_sequence < barrier_sequence
            or current_sequence > final_frozen_tail_sequence
            or prior_relative_us < barrier_relative_us
            or current_relative_us > final_frozen_tail_relative_us
        ):
            raise RuntimeError("capture delivery recovery lies outside the terminal window")
        prior_status, current_status = prior_row[4], current_row[4]
        prior_flags, current_flags = prior_row[6], current_row[6]
        required_recovery_flags = 1 | 2 | 4 | 32 | 256
        forbidden_recovery_flags = 8 | 128 | 1024
        if (
            prior_status not in {"C", "I"}
            or current_status not in {"C", "I"}
            or prior_flags & required_recovery_flags != required_recovery_flags
            or current_flags & required_recovery_flags != required_recovery_flags
            or prior_flags & forbidden_recovery_flags
            or current_flags & forbidden_recovery_flags
            or prior_class not in {"A", "W", "K", "N"}
            or re.fullmatch(r"[0-9a-f]{16}", prior_signature) is None
            or prior_signature == "0000000000000000"
        ):
            raise RuntimeError("capture delivery recovery endpoint flags are invalid")
        retained = (prior_row[5], prior_row[7]) == (current_row[5], current_row[7])
        controlled_transition = (
            not retained
            and current_sequence == accepted_sequence
            and current_status == "C"
        )
        if retained:
            retained_state_count += 1
        elif controlled_transition:
            controlled_transition_count += 1
        else:
            raise RuntimeError("capture delivery recovery changed an unproven visual state")
        recovery_pairs.add((prior_sequence, current_sequence))
        recovery_inserted_pts_us.append(inserted_relative_us)

    if (
        controlled_transition_count > 1
        or integer(raw_recovery, "controlled_transition_recovery_count")
        != controlled_transition_count
        or integer(raw_recovery, "retained_state_recovery_count")
        != retained_state_count
        or raw_recovery.get("all_recoveries_within_terminal_window") is not True
        or raw_recovery.get("all_recoveries_visual_consistent") is not True
        or raw_recovery.get("encoded_gaps_split") is not True
        or raw_recovery.get("validated_after_stream_stop") is not True
        or raw_recovery.get("passed") is not True
    ):
        raise RuntimeError("capture delivery terminal gap recovery did not pass")

    display_gaps_us = [
        right[1] - left[1] for left, right in pairwise(parsed_callback_rows)
    ]
    observed_recovery_pairs = {
        (left[0], right[0])
        for left, right in pairwise(parsed_callback_rows)
        if right[0] > startup_callback_count and right[1] - left[1]
        > round(CAPTURE_DELIVERY_MAXIMUM_GAP_SECONDS * 1_000_000)
    }
    if observed_recovery_pairs != recovery_pairs:
        raise RuntimeError("capture delivery trace contains an uncovered display-time gap")
    if integer(raw_recovery, "raw_over_limit_gap_count") != len(
        observed_recovery_pairs
    ):
        raise RuntimeError("capture delivery raw over-limit gap count is inconsistent")
    if any(
        right[0] > startup_callback_count and right[1] - left[1]
        > round(CAPTURE_DELIVERY_TERMINAL_RECOVERY_CEILING_SECONDS * 1_000_000)
        for left, right in pairwise(parsed_callback_rows)
    ):
        raise RuntimeError("capture delivery display-time gap exceeds recovery ceiling")
    recomputed_display_gap = max(display_gaps_us, default=0) / 1_000_000.0
    recomputed_callback_lag = max(row[2] for row in parsed_callback_rows) / 1_000_000.0
    recomputed_callback_service = max(row[3] for row in parsed_callback_rows) / 1_000_000.0
    for key, recomputed in (
        ("maximum_display_time_gap_seconds", recomputed_display_gap),
        ("maximum_callback_delivery_lag_seconds", recomputed_callback_lag),
        ("maximum_callback_service_seconds", recomputed_callback_service),
    ):
        if not math.isclose(number(delivery, key), recomputed, abs_tol=1.1e-6):
            raise RuntimeError(f"capture delivery {key} disagrees with its callback trace")
    terminal_coverage_gap = number(delivery, "terminal_coverage_gap_seconds")
    if not 0.0 <= terminal_coverage_gap <= CAPTURE_DELIVERY_MAXIMUM_GAP_SECONDS:
        raise RuntimeError("capture delivery terminal coverage gap exceeds its strict limit")

    expected_callback_appends = len(appended_callback_rows)
    if integer(delivery, "appended_started_samples") != status_code_counts["S"]:
        raise RuntimeError("capture delivery Started sample accounting is inconsistent")
    if integer(delivery, "appended_complete_samples") != status_code_counts["C"]:
        raise RuntimeError("capture delivery Complete sample accounting is inconsistent")
    if integer(delivery, "appended_idle_hold_samples") != status_code_counts["I"]:
        raise RuntimeError("capture delivery Idle sample accounting is inconsistent")
    if (
        integer(delivery, "appended_terminal_recovery_samples")
        != recovery_inserted_count
    ):
        raise RuntimeError("capture delivery terminal recovery accounting is inconsistent")
    if integer(delivery, "terminal_hold_samples") != 1:
        raise RuntimeError("capture delivery must contain exactly one terminal hold sample")
    appended_sample_count = integer(delivery, "appended_sample_count")
    if appended_sample_count != expected_callback_appends + recovery_inserted_count + 1:
        raise RuntimeError("capture delivery appended sample accounting is inconsistent")

    raw_pts_trace = delivery.get("appended_pts_trace")
    if not isinstance(raw_pts_trace, Mapping):
        raise TypeError("capture delivery appended PTS trace must be an object")
    if raw_pts_trace.get("encoding") != "relative-pts-us,kind-code-v2":
        raise RuntimeError("capture delivery appended PTS trace has an unknown encoding")
    if integer(raw_pts_trace, "row_count") != appended_sample_count:
        raise RuntimeError("capture delivery appended PTS trace row count is inconsistent")
    pts_rows_text = raw_pts_trace.get("rows")
    if not isinstance(pts_rows_text, str) or not pts_rows_text.endswith("\n"):
        raise TypeError("capture delivery appended PTS rows must be newline-terminated text")
    pts_lines = pts_rows_text.splitlines()
    if len(pts_lines) != appended_sample_count:
        raise RuntimeError("capture delivery appended PTS trace is incomplete")
    parsed_pts_rows: list[tuple[int, str]] = []
    for index, line in enumerate(pts_lines):
        columns = line.split(",")
        if len(columns) != 2:
            raise RuntimeError("capture delivery appended PTS row has the wrong field count")
        try:
            pts_us = int(columns[0])
        except ValueError as error:
            raise RuntimeError("capture delivery appended PTS trace contains a non-integer PTS") from error
        kind = columns[1]
        if line != f"{pts_us},{kind}" or kind not in {"S", "C", "I", "R", "E"}:
            raise RuntimeError("capture delivery appended PTS trace is not canonically encoded")
        if pts_us < 0 or (parsed_pts_rows and pts_us <= parsed_pts_rows[-1][0]):
            raise RuntimeError("capture delivery appended PTS values must increase strictly")
        if index == 0 and pts_us != 0:
            raise RuntimeError("capture delivery appended PTS trace does not start at zero")
        parsed_pts_rows.append((pts_us, kind))
    callback_pts_rows = [row for row in parsed_pts_rows if row[1] != "R"]
    recovery_pts_rows = [row for row in parsed_pts_rows if row[1] == "R"]
    if (
        callback_pts_rows[:-1] != appended_callback_rows
        or callback_pts_rows[-1][1] != "E"
        or [row[0] for row in recovery_pts_rows] != recovery_inserted_pts_us
    ):
        raise RuntimeError("capture delivery appended PTS trace disagrees with callback outcomes")
    for inserted_pts_us in recovery_inserted_pts_us:
        inserted_index = parsed_pts_rows.index((inserted_pts_us, "R"))
        recovery_row = next(
            row
            for row in recovery_lines
            if int(row.split(",")[5]) == inserted_pts_us
        ).split(",")
        current_sequence = int(recovery_row[2])
        current_callback = parsed_callback_rows[current_sequence - 1]
        if (
            inserted_index + 1 >= len(parsed_pts_rows)
            or parsed_pts_rows[inserted_index + 1]
            != (current_callback[1], current_callback[4])
        ):
            raise RuntimeError(
                "capture delivery recovery PTS does not immediately precede its right callback"
            )
    appended_gaps_us = [right[0] - left[0] for left, right in pairwise(parsed_pts_rows)]
    strict_appended_gaps_us = [
        right[0] - left[0] for left, right in pairwise(parsed_pts_rows)
        if right[0] > startup_source_us
    ]
    if not strict_appended_gaps_us or max(strict_appended_gaps_us) > round(
        CAPTURE_DELIVERY_MAXIMUM_GAP_SECONDS * 1_000_000
    ):
        raise RuntimeError("capture delivery appended PTS trace contains an uncovered gap")
    if not math.isclose(number(delivery, "first_file_relative_pts_seconds"), 0.0, abs_tol=1e-9):
        raise RuntimeError("capture delivery first file PTS is not zero")
    if not math.isclose(
        number(delivery, "last_file_relative_pts_seconds"),
        parsed_pts_rows[-1][0] / 1_000_000.0,
        abs_tol=1.1e-6,
    ):
        raise RuntimeError("capture delivery last file PTS disagrees with its appended trace")

    raw_stop_ordering = delivery.get("stop_ordering")
    if not isinstance(raw_stop_ordering, Mapping):
        raise TypeError("capture delivery stop ordering must be an object")
    if raw_stop_ordering.get("stop_requested") is not True:
        raise RuntimeError("capture delivery lacks an explicit stop request")
    if raw_stop_ordering.get("stop_completion_observed") is not True:
        raise RuntimeError("capture delivery lacks the stopCapture completion")
    if raw_stop_ordering.get("stop_completion_succeeded") is not True:
        raise RuntimeError("capture delivery stopCapture completion reported failure")
    if raw_stop_ordering.get("stop_completion_followed_request") is not True:
        raise RuntimeError("capture delivery stopCapture completion preceded its request")
    stopped_status_expected = status_code_counts["T"] == 1
    if raw_stop_ordering.get("stopped_status_observed") is not stopped_status_expected:
        raise RuntimeError("capture delivery stopped-status telemetry disagrees with its trace")
    if raw_stop_ordering.get("stopped_status_followed_request") is not stopped_status_expected:
        raise RuntimeError("capture delivery stopped-status ordering disagrees with its trace")
    delegate_callback = raw_stop_ordering.get("delegate_stop_callback_observed")
    delegate_error = raw_stop_ordering.get("delegate_stop_error_observed")
    if not isinstance(delegate_callback, bool) or not isinstance(delegate_error, bool):
        raise TypeError("capture delivery delegate-stop telemetry must be boolean")
    if delegate_error and not delegate_callback:
        raise RuntimeError("capture delivery delegate-stop error lacks its callback")
    if delegate_error:
        raise RuntimeError("capture delivery delegate reported a stream-stop error")
    raw_session_start = delivery.get("session_start_host_time")
    if not isinstance(raw_session_start, Mapping):
        raise TypeError("capture delivery session start host time must be an object")
    if integer(raw_session_start, "value") <= 0 or integer(raw_session_start, "timescale") <= 0:
        raise RuntimeError("capture delivery session start host time is invalid")

    raw_content = delivery.get("content_classification")
    if not isinstance(raw_content, Mapping):
        raise TypeError("capture delivery content classification must be an object")
    if (
        raw_content.get("method") != "central-420v-luma-grid-v1"
        or integer(raw_content, "classified_frame_count")
        != status_code_counts["S"] + status_code_counts["C"]
        or integer(raw_content, "near_white_luma_minimum") != 226
        or integer(raw_content, "near_black_luma_maximum") != 25
        or not math.isclose(number(raw_content, "required_blank_pixel_fraction"), 0.995, abs_tol=1e-12)
        or not math.isclose(
            number(raw_content, "maximum_allowed_sustained_blank_seconds"),
            CAPTURE_DELIVERY_MAXIMUM_SUSTAINED_BLANK_SECONDS,
            abs_tol=1e-12,
        )
        or raw_content.get("passed") is not True
    ):
        raise RuntimeError("capture delivery content classification contract is invalid")
    for key in (
        "longest_near_white_seconds",
        "longest_near_black_seconds",
        "longest_neutral_blank_seconds",
    ):
        duration = number(raw_content, key)
        if not 0.0 <= duration <= CAPTURE_DELIVERY_MAXIMUM_SUSTAINED_BLANK_SECONDS:
            raise RuntimeError("capture delivery admitted a sustained blank gameplay interval")
    traced_blank_us = {"W": 0, "K": 0, "N": 0}
    blank_run_code: str | None = None
    blank_run_start_us = 0
    for index in gameplay_armed_indices:
        row = parsed_callback_rows[index]
        class_code = row[5]
        if class_code not in traced_blank_us:
            blank_run_code = None
            continue
        if class_code != blank_run_code:
            blank_run_code = class_code
            blank_run_start_us = row[1]
        traced_blank_us[class_code] = max(
            traced_blank_us[class_code], row[1] - blank_run_start_us
        )
    for code, key in {
        "W": "longest_near_white_seconds",
        "K": "longest_near_black_seconds",
        "N": "longest_neutral_blank_seconds",
    }.items():
        if number(raw_content, key) + 2e-6 < traced_blank_us[code] / 1_000_000.0:
            raise RuntimeError("capture delivery blank classification is below its callback trace")

    raw_health = delivery.get("health_gates")
    if not isinstance(raw_health, Mapping):
        raise TypeError("capture delivery health gates must be an object")
    if (
        not math.isclose(
            number(raw_health, "maximum_allowed_display_gap_seconds"),
            CAPTURE_DELIVERY_MAXIMUM_GAP_SECONDS,
            abs_tol=1e-12,
        )
        or not math.isclose(
            number(
                raw_health,
                "maximum_allowed_terminal_recovery_gap_seconds",
            ),
            CAPTURE_DELIVERY_TERMINAL_RECOVERY_CEILING_SECONDS,
            abs_tol=1e-12,
        )
        or not math.isclose(
            number(raw_health, "maximum_allowed_callback_delivery_lag_seconds"),
            CAPTURE_DELIVERY_MAXIMUM_CALLBACK_LAG_SECONDS,
            abs_tol=1e-12,
        )
        or not math.isclose(
            number(raw_health, "maximum_allowed_callback_service_seconds"),
            CAPTURE_DELIVERY_MAXIMUM_CALLBACK_SERVICE_SECONDS,
            abs_tol=1e-12,
        )
        or raw_health.get("sample_accounting_passed") is not True
        or raw_health.get("terminal_gap_recovery_passed") is not True
        or raw_health.get("passed") is not True
    ):
        raise RuntimeError("capture delivery health gates are incomplete or weakened")

    raw_post_write = delivery.get("post_write_encoded_audit")
    if not isinstance(raw_post_write, Mapping):
        raise TypeError("capture delivery post-write encoded audit must be an object")
    if raw_post_write.get("method") != (
        "avassetreader-encoded-media-samples-zero-size-markers-v2"
    ):
        raise RuntimeError("capture delivery used an unknown post-write audit method")
    expected_encoded_count = integer(raw_post_write, "expected_sample_count")
    encoded_count = integer(raw_post_write, "encoded_sample_count")
    if expected_encoded_count != appended_sample_count or encoded_count != appended_sample_count:
        raise RuntimeError("capture delivery post-write sample accounting is inconsistent")
    marker_buffer_count = integer(raw_post_write, "marker_buffer_count")
    reader_buffer_count = integer(raw_post_write, "reader_buffer_count")
    if marker_buffer_count < 0:
        raise RuntimeError("capture delivery post-write marker-buffer count is invalid")
    if reader_buffer_count != encoded_count + marker_buffer_count:
        raise RuntimeError("capture delivery post-write reader-buffer accounting is inconsistent")
    if raw_post_write.get("marker_buffers_validated") is not True:
        raise RuntimeError("capture delivery post-write marker buffers were not validated")
    allowed_sequence_error = number(raw_post_write, "maximum_allowed_pts_sequence_error_seconds")
    if not math.isclose(
        allowed_sequence_error,
        CAPTURE_ENCODED_PTS_SEQUENCE_ERROR_SECONDS,
        abs_tol=1e-12,
    ):
        raise RuntimeError("capture delivery post-write PTS sequence tolerance was weakened")
    maximum_sequence_error = number(raw_post_write, "maximum_pts_sequence_error_seconds")
    if not 0.0 <= maximum_sequence_error <= (
        allowed_sequence_error + CAPTURE_ENCODED_PTS_NUMERICAL_SLACK_SECONDS
    ):
        raise RuntimeError("capture delivery post-write PTS sequence does not match")
    allowed_encoded_gap = number(raw_post_write, "maximum_allowed_pts_gap_seconds")
    maximum_encoded_gap = number(raw_post_write, "maximum_pts_gap_seconds")
    maximum_strict_encoded_gap = maximum_encoded_gap
    if phased_startup:
        maximum_strict_encoded_gap = number(raw_post_write, "maximum_strict_pts_gap_seconds")
        if not math.isclose(
            number(raw_post_write, "strict_start_source_seconds"),
            startup_source_us / 1_000_000, abs_tol=1.1e-6,
        ):
            raise RuntimeError("capture delivery encoded audit uses a different startup cutoff")
    if not math.isclose(
        allowed_encoded_gap,
        CAPTURE_ENCODED_PTS_MAXIMUM_GAP_SECONDS,
        abs_tol=1e-12,
    ):
        raise RuntimeError("capture delivery post-write encoded gap limit was weakened")
    if not 0.0 <= maximum_strict_encoded_gap <= (
        CAPTURE_ENCODED_PTS_MAXIMUM_GAP_SECONDS
        + CAPTURE_ENCODED_PTS_NUMERICAL_SLACK_SECONDS
    ):
        raise RuntimeError("capture delivery post-write encoded PTS contains an uncovered gap")
    trace_rounding = number(raw_post_write, "pts_trace_rounding_seconds")
    if not math.isclose(
        trace_rounding,
        CAPTURE_ENCODED_PTS_TRACE_ROUNDING_SECONDS,
        abs_tol=1e-12,
    ):
        raise RuntimeError("capture delivery post-write PTS trace rounding contract changed")
    allowed_gap_reconciliation = number(
        raw_post_write,
        "maximum_allowed_pts_gap_reconciliation_error_seconds",
    )
    if not math.isclose(
        allowed_gap_reconciliation,
        CAPTURE_ENCODED_PTS_GAP_RECONCILIATION_ERROR_SECONDS,
        abs_tol=1e-12,
    ):
        raise RuntimeError("capture delivery post-write PTS gap reconciliation limit changed")
    maximum_gap_reconciliation = number(
        raw_post_write,
        "maximum_pts_gap_reconciliation_error_seconds",
    )
    if not 0.0 <= maximum_gap_reconciliation <= (
        allowed_gap_reconciliation + CAPTURE_ENCODED_PTS_NUMERICAL_SLACK_SECONDS
    ):
        raise RuntimeError("capture delivery post-write PTS gap reconciliation exceeded its bound")
    traced_maximum_gap = max(appended_gaps_us, default=0) / 1_000_000.0
    if abs(maximum_encoded_gap - traced_maximum_gap) > maximum_gap_reconciliation + 1e-12:
        raise RuntimeError("capture delivery post-write encoded gap disagrees with its appended trace")
    if (
        abs(maximum_strict_encoded_gap - max(strict_appended_gaps_us) / 1_000_000)
        > maximum_gap_reconciliation + 1e-12
    ):
        raise RuntimeError("capture delivery post-write strict gap disagrees with its appended trace")
    if not math.isclose(number(raw_post_write, "first_pts_seconds"), 0.0, abs_tol=1e-3):
        raise RuntimeError("capture delivery post-write encoded timeline does not start at zero")
    if not math.isclose(
        number(raw_post_write, "last_pts_seconds"),
        parsed_pts_rows[-1][0] / 1_000_000.0,
        abs_tol=(
            CAPTURE_ENCODED_PTS_ENDPOINT_RECONCILIATION_ERROR_SECONDS
            + CAPTURE_ENCODED_PTS_NUMERICAL_SLACK_SECONDS
        ),
    ):
        raise RuntimeError("capture delivery post-write final PTS disagrees with its appended trace")
    for key in (
        "sample_count_matched",
        "pts_strictly_increasing",
        "pts_sequence_matched",
        "pts_gap_reconciled",
        "passed",
    ):
        if raw_post_write.get(key) is not True:
            raise RuntimeError("capture delivery post-write encoded audit did not pass")


def _validate_terminal_stopped_hold_trace(
    metadata: Mapping[str, Any],
    *,
    terminal_stop: Mapping[str, Any],
    terminal_barrier: Mapping[str, Any],
    terminal_applied: Mapping[str, Any],
    terminal_stopped_hold: Mapping[str, Any],
    terminal_final_frozen_tail: Mapping[str, Any],
    terminal_apply: Mapping[str, Any],
) -> None:
    """Bind terminal snapshots to every retained callback through the stopped hold."""
    delivery = metadata.get("capture_delivery")
    if not isinstance(delivery, Mapping):
        raise TypeError("terminal stopped hold lacks capture-delivery trace evidence")
    trace = delivery.get("callback_trace")
    if not isinstance(trace, Mapping):
        raise TypeError("terminal stopped hold lacks a compact callback trace")
    rows_text = trace.get("rows")
    row_count = trace.get("row_count")
    if (
        not isinstance(rows_text, str)
        or not rows_text.endswith("\n")
        or isinstance(row_count, bool)
        or not isinstance(row_count, int)
    ):
        raise TypeError("terminal stopped-hold callback trace is malformed")
    lines = rows_text.splitlines()
    if len(lines) != row_count:
        raise RuntimeError("terminal stopped-hold callback trace is incomplete")

    parsed: list[tuple[int, int, str, str, int, str]] = []
    for index, line in enumerate(lines, start=1):
        columns = line.split(",")
        if len(columns) != 8:
            raise RuntimeError("terminal stopped-hold callback row has the wrong schema")
        try:
            sequence = int(columns[0])
            relative_us = int(columns[1])
            delivery_lag_us = int(columns[2])
            service_us = int(columns[3])
            flags = int(columns[6])
        except ValueError as error:
            raise RuntimeError("terminal stopped-hold callback row is noncanonical") from error
        canonical = (
            f"{sequence},{relative_us},{delivery_lag_us},{service_us},"
            f"{columns[4]},{columns[5]},{flags},{columns[7]}"
        )
        if line != canonical or sequence != index or relative_us < 0:
            raise RuntimeError("terminal stopped-hold callback row is noncanonical")
        parsed.append(
            (sequence, relative_us, columns[4], columns[5], flags, columns[7])
        )

    snapshots = (
        terminal_stop,
        terminal_barrier,
        terminal_applied,
        terminal_stopped_hold,
        terminal_final_frozen_tail,
    )
    maximum_snapshot_sequence = max(int(snapshot["callback_count"]) for snapshot in snapshots)
    if maximum_snapshot_sequence > len(parsed):
        raise RuntimeError("terminal stopped-hold snapshot exceeds the callback trace")

    def prefix_evidence(sequence_end: int) -> dict[str, Any]:
        complete_count = 0
        idle_count = 0
        visual_transitions = 0
        class_transitions = 0
        last_complete_sequence = 0
        previous_visual: tuple[str, str] | None = None
        for sequence, _, status, content_class, flags, signature in parsed[:sequence_end]:
            if status == "C":
                complete_count += 1
                last_complete_sequence = sequence
            elif status == "I":
                idle_count += 1
            if flags & 32:
                current_visual = (content_class, signature)
                if previous_visual is not None:
                    class_transitions += current_visual[0] != previous_visual[0]
                    visual_transitions += current_visual[1] != previous_visual[1]
                previous_visual = current_visual
        return {
            "complete_sample_count": complete_count,
            "idle_sample_count": idle_count,
            "visual_signature_transition_count": visual_transitions,
            "content_class_transition_count": class_transitions,
            "last_complete_callback_sequence": last_complete_sequence,
            "last_visual_signature": previous_visual[1] if previous_visual else None,
            "last_content_class": previous_visual[0] if previous_visual else None,
        }

    for snapshot in snapshots:
        evidence = prefix_evidence(int(snapshot["callback_count"]))
        for key, expected in evidence.items():
            if snapshot[key] != expected:
                raise RuntimeError(
                    f"terminal stopped-hold snapshot {key} disagrees with callback trace"
                )

    stop_sequence = int(terminal_stop["callback_count"])
    barrier_sequence = int(terminal_barrier["callback_count"])
    if stop_sequence < 1 or barrier_sequence <= stop_sequence:
        raise RuntimeError("terminal pre-resume frozen callback interval did not advance")
    frozen_interval = parsed[stop_sequence:barrier_sequence]
    frozen_signature = str(terminal_stop["last_visual_signature"])
    frozen_class = str(terminal_stop["last_content_class"])
    common_flags = 1 | 2 | 4
    allowed_flags = 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024
    frozen_complete_count = 0
    frozen_idle_count = 0
    for _, _, status, content_class, flags, signature in frozen_interval:
        if (
            status not in {"C", "I"}
            or content_class != frozen_class
            or signature != frozen_signature
            or flags < 0
            or flags & ~allowed_flags
            or flags & common_flags != common_flags
            or not flags & 32
            or not flags & 256
            or flags & (8 | 128 | 1024)
            or bool(flags & 512) != (frozen_class == "A")
        ):
            raise RuntimeError("terminal pre-resume frozen trace changed visual state")
        if status == "C":
            if not flags & 16 or flags & 64:
                raise RuntimeError("terminal pre-resume frozen Complete flags are invalid")
            frozen_complete_count += 1
        else:
            if not flags & 64:
                raise RuntimeError("terminal pre-resume frozen Idle flags are invalid")
            frozen_idle_count += 1
    if (
        terminal_barrier["last_visual_signature"] != frozen_signature
        or terminal_barrier["last_content_class"] != frozen_class
        or terminal_barrier["visual_signature_transition_count"]
        != terminal_stop["visual_signature_transition_count"]
        or terminal_barrier["content_class_transition_count"]
        != terminal_stop["content_class_transition_count"]
        or terminal_apply.get("pre_resume_frozen_callback_advanced") is not True
        or terminal_apply.get("pre_resume_frozen_only_retained_idle_or_complete")
        is not True
        or terminal_apply.get("pre_resume_frozen_transition_counters_unchanged")
        is not True
        or terminal_apply.get("pre_resume_frozen_visual_state_retained") is not True
        or terminal_apply.get("pre_resume_frozen_complete_sample_delta")
        != frozen_complete_count
        or terminal_apply.get("pre_resume_frozen_idle_sample_delta")
        != frozen_idle_count
        or len(frozen_interval) != frozen_complete_count + frozen_idle_count
    ):
        raise RuntimeError("terminal pre-resume frozen callback proof is inconsistent")

    first_sequence = int(terminal_applied["last_complete_callback_sequence"])
    last_sequence = int(terminal_stopped_hold["callback_count"])
    if first_sequence < 1 or last_sequence <= first_sequence:
        raise RuntimeError("terminal stopped-hold callback interval did not advance")
    interval = parsed[first_sequence - 1 : last_sequence]
    accepted_signature = str(terminal_applied["last_visual_signature"])
    accepted_class = str(terminal_applied["last_content_class"])
    common_flags = 1 | 2 | 4
    allowed_flags = 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024
    complete_count = 0
    idle_count = 0
    previous_relative_us = -1
    last_complete_sequence = 0
    for sequence, relative_us, status, content_class, flags, signature in interval:
        if relative_us <= previous_relative_us:
            raise RuntimeError("terminal stopped-hold display times do not increase strictly")
        previous_relative_us = relative_us
        if (
            status not in {"C", "I"}
            or content_class != accepted_class
            or signature != accepted_signature
            or flags < 0
            or flags & ~allowed_flags
            or flags & common_flags != common_flags
            or not flags & 32
            or not flags & 256
            or flags & (8 | 128 | 1024)
            or bool(flags & 512) != (accepted_class == "A")
        ):
            raise RuntimeError("terminal stopped-hold trace contains a non-retained callback")
        if status == "C":
            if not flags & 16 or flags & 64:
                raise RuntimeError("terminal stopped-hold Complete flags are invalid")
            complete_count += 1
            last_complete_sequence = sequence
        else:
            if not flags & 64:
                raise RuntimeError("terminal stopped-hold Idle flags are invalid")
            idle_count += 1
    if interval[0][2] != "C" or last_complete_sequence != int(
        terminal_stopped_hold["last_complete_callback_sequence"]
    ):
        raise RuntimeError("terminal stopped-hold Complete identity is inconsistent")

    post_accept_count = last_sequence - first_sequence
    expected_interval = {
        "method": "retained-complete-idle-callback-interval-v1",
        "first_callback_sequence": first_sequence,
        "last_callback_sequence": last_sequence,
        "row_count": len(interval),
        "post_accept_callback_count": post_accept_count,
        "complete_callback_count": complete_count,
        "idle_callback_count": idle_count,
        "appended_callback_count": len(interval),
        "forbidden_callback_count": 0,
        "first_relative_display_us": interval[0][1],
        "last_relative_display_us": interval[-1][1],
        "display_interval_us": interval[-1][1] - interval[0][1],
        "accepted_visual_signature": accepted_signature,
        "accepted_content_class": accepted_class,
        "visual_signature_transition_delta": 0,
        "content_class_transition_delta": 0,
        "passed": True,
    }
    reported_interval = terminal_apply.get("stopped_hold_callback_trace_interval")
    if not isinstance(reported_interval, Mapping) or dict(reported_interval) != expected_interval:
        raise RuntimeError("terminal stopped-hold trace interval metadata was forged")

    final_sequence = int(terminal_final_frozen_tail["callback_count"])
    if final_sequence <= last_sequence:
        raise RuntimeError("terminal final frozen-tail callback interval did not advance")
    final_interval = parsed[first_sequence - 1 : final_sequence]
    final_complete_count = 0
    final_idle_count = 0
    previous_relative_us = -1
    for _, relative_us, status, content_class, flags, signature in final_interval:
        if relative_us <= previous_relative_us:
            raise RuntimeError("terminal final frozen-tail display times do not increase")
        previous_relative_us = relative_us
        if (
            status not in {"C", "I"}
            or content_class != accepted_class
            or signature != accepted_signature
            or flags < 0
            or flags & ~allowed_flags
            or flags & common_flags != common_flags
            or not flags & 32
            or not flags & 256
            or flags & (8 | 128 | 1024)
            or bool(flags & 512) != (accepted_class == "A")
        ):
            raise RuntimeError("terminal final frozen-tail trace is not retained")
        if status == "C":
            if not flags & 16 or flags & 64:
                raise RuntimeError("terminal final frozen-tail Complete flags are invalid")
            final_complete_count += 1
        else:
            if not flags & 64:
                raise RuntimeError("terminal final frozen-tail Idle flags are invalid")
            final_idle_count += 1
    expected_final_interval = {
        "method": "retained-complete-idle-final-frozen-tail-interval-v1",
        "first_callback_sequence": first_sequence,
        "last_callback_sequence": final_sequence,
        "row_count": len(final_interval),
        "post_accept_callback_count": final_sequence - first_sequence,
        "complete_callback_count": final_complete_count,
        "idle_callback_count": final_idle_count,
        "appended_callback_count": len(final_interval),
        "forbidden_callback_count": 0,
        "first_relative_display_us": final_interval[0][1],
        "last_relative_display_us": final_interval[-1][1],
        "display_interval_us": final_interval[-1][1] - final_interval[0][1],
        "accepted_visual_signature": accepted_signature,
        "accepted_content_class": accepted_class,
        "visual_signature_transition_delta": 0,
        "content_class_transition_delta": 0,
        "passed": True,
    }
    reported_final_interval = terminal_apply.get(
        "final_frozen_tail_callback_trace_interval"
    )
    final_post_snapshot_count = final_sequence - int(terminal_applied["callback_count"])
    final_complete_delta = int(terminal_final_frozen_tail["complete_sample_count"]) - int(
        terminal_applied["complete_sample_count"]
    )
    final_idle_delta = int(terminal_final_frozen_tail["idle_sample_count"]) - int(
        terminal_applied["idle_sample_count"]
    )
    if (
        not isinstance(reported_final_interval, Mapping)
        or dict(reported_final_interval) != expected_final_interval
        or terminal_apply.get("final_frozen_tail_callback_advanced") is not True
        or terminal_apply.get("final_frozen_tail_post_accept_callback_delta")
        != final_sequence - first_sequence
        or terminal_apply.get("final_frozen_tail_post_snapshot_callback_delta")
        != final_post_snapshot_count
        or terminal_apply.get("final_frozen_tail_complete_sample_delta")
        != final_complete_delta
        or terminal_apply.get("final_frozen_tail_idle_sample_delta")
        != final_idle_delta
        or final_post_snapshot_count != final_complete_delta + final_idle_delta
        or terminal_apply.get("final_frozen_tail_only_retained_idle_or_complete")
        is not True
        or terminal_apply.get("final_frozen_tail_transition_counters_unchanged")
        is not True
        or terminal_apply.get("final_frozen_tail_visual_state_retained") is not True
        or terminal_final_frozen_tail["last_visual_signature"] != accepted_signature
        or terminal_final_frozen_tail["last_content_class"] != accepted_class
        or terminal_final_frozen_tail["visual_signature_transition_count"]
        != terminal_applied["visual_signature_transition_count"]
        or terminal_final_frozen_tail["content_class_transition_count"]
        != terminal_applied["content_class_transition_count"]
    ):
        raise RuntimeError("terminal final frozen-tail proof is inconsistent")

    suffix = parsed[final_sequence:]
    suffix_complete_count = 0
    suffix_idle_count = 0
    suffix_stopped_count = 0
    suffix_last_image_sequence = final_sequence
    suffix_first_relative_us = suffix[0][1] if suffix else -1
    suffix_last_relative_us = suffix[-1][1] if suffix else -1
    for suffix_index, (
        sequence,
        _,
        status,
        content_class,
        flags,
        signature,
    ) in enumerate(suffix):
        if flags & common_flags != common_flags or flags & (8 | 256):
            raise RuntimeError("terminal sealed-stop suffix has invalid common flags")
        if status in {"C", "I"}:
            if (
                content_class != accepted_class
                or signature != accepted_signature
                or not flags & 32
                or flags & 128
                or bool(flags & 512) != (accepted_class == "A")
            ):
                raise RuntimeError("terminal sealed-stop suffix changed visual state")
            if status == "C":
                if not flags & 16 or flags & 64:
                    raise RuntimeError("terminal sealed-stop Complete flags are invalid")
                suffix_complete_count += 1
            else:
                if not flags & 64:
                    raise RuntimeError("terminal sealed-stop Idle flags are invalid")
                suffix_idle_count += 1
            suffix_last_image_sequence = sequence
        elif status == "T":
            if (
                suffix_index != len(suffix) - 1
                or not flags & 128
                or not flags & 1024
                or flags & (16 | 32 | 64 | 512)
                or content_class != "X"
                or signature != "0000000000000000"
            ):
                raise RuntimeError("terminal sealed-stop Stopped flags are invalid")
            suffix_stopped_count += 1
        else:
            raise RuntimeError("terminal sealed-stop suffix contains a forbidden callback")
    expected_suffix = {
        "method": "retained-complete-idle-optional-stopped-sealed-suffix-v1",
        "boundary_callback_sequence": final_sequence,
        "first_callback_sequence": final_sequence + 1 if suffix else 0,
        "last_callback_sequence": len(parsed),
        "row_count": len(suffix),
        "complete_callback_count": suffix_complete_count,
        "idle_callback_count": suffix_idle_count,
        "stopped_callback_count": suffix_stopped_count,
        "appended_callback_count": suffix_complete_count + suffix_idle_count,
        "forbidden_callback_count": 0,
        "last_image_callback_sequence": suffix_last_image_sequence,
        "first_relative_display_us": suffix_first_relative_us,
        "last_relative_display_us": suffix_last_relative_us,
        "accepted_visual_signature": accepted_signature,
        "accepted_content_class": accepted_class,
        "passed": True,
    }
    reported_suffix = terminal_apply.get("sealed_stop_callback_suffix")
    if not isinstance(reported_suffix, Mapping) or dict(reported_suffix) != expected_suffix:
        raise RuntimeError("terminal sealed-stop callback suffix metadata was forged")


def _validate_recorder_startup_sync(
    metadata: Mapping[str, Any],
    *,
    expected_replay_path: Path,
    expected_start_frame: int,
    expected_inclusive_end_frame: int,
    minimum_tail_seconds: float,
    expected_audio_presentation_delay_seconds: float,
) -> None:
    """Require exact idle-command, zero-audio, and two-stop startup proof."""
    if (
        isinstance(expected_start_frame, bool)
        or not isinstance(expected_start_frame, int)
        or isinstance(expected_inclusive_end_frame, bool)
        or not isinstance(expected_inclusive_end_frame, int)
        or expected_inclusive_end_frame < expected_start_frame
        or isinstance(minimum_tail_seconds, bool)
        or not isinstance(minimum_tail_seconds, (int, float))
        or not math.isfinite(float(minimum_tail_seconds))
        or minimum_tail_seconds < 0.0
        or isinstance(expected_audio_presentation_delay_seconds, bool)
        or not isinstance(expected_audio_presentation_delay_seconds, (int, float))
        or not math.isfinite(float(expected_audio_presentation_delay_seconds))
        or expected_audio_presentation_delay_seconds < 0.0
    ):
        raise ValueError("expected replay startup frame interval is invalid")
    expected_exclusive_end = expected_inclusive_end_frame + 1
    expected_content_count = expected_inclusive_end_frame - expected_start_frame + 1
    expected_trace_count = expected_content_count
    expected_replay = str(expected_replay_path.resolve())

    raw_sync = metadata.get("startup_sync")
    if not isinstance(raw_sync, Mapping):
        raise TypeError("isolated-window recorder startup sync must be an object")
    sync = raw_sync

    def integer(container: Mapping[str, Any], key: str) -> int:
        value = container.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"recorder startup sync {key} must be an integer")
        return value

    def number(container: Mapping[str, Any], key: str) -> float:
        value = container.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"recorder startup sync {key} must be a number")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"recorder startup sync {key} must be finite")
        return result

    schema = STARTUP_SYNC_SCHEMA
    if sync.get("schema") != schema or sync.get("method") != schema:
        raise RuntimeError("recorder startup sync used an unknown schema or method")
    if (
        sync.get("command_boundary_semantics") != STARTUP_SYNC_COMMAND_BOUNDARY_SEMANTICS
        or sync.get("generation_boundary_semantics")
        != STARTUP_SYNC_GENERATION_BOUNDARY_SEMANTICS
    ):
        raise RuntimeError("recorder startup sync boundary semantics are invalid")
    expected_top_level = {
        "requested_start_frame": expected_start_frame,
        "requested_inclusive_end": expected_inclusive_end_frame,
        "command_exclusive_end": expected_exclusive_end,
        "expected_content_frame_count": expected_content_count,
        "expected_trace_frame_count": expected_trace_count,
    }
    if any(integer(sync, key) != expected for key, expected in expected_top_level.items()):
        raise RuntimeError("recorder startup sync frame interval differs from the retained replay")
    if sync.get("stop_targets") != [expected_start_frame, expected_start_frame + 1]:
        raise RuntimeError("recorder startup sync did not use exact start and start+1 stops")
    if (
        integer(sync, "terminal_stop_target") != expected_inclusive_end_frame
        or sync.get("terminal_stop_observed") is not True
    ):
        raise RuntimeError(
            "recorder startup sync did not stop at the inclusive last generated content frame"
        )
    if sync.get("passed") is not True:
        raise RuntimeError("recorder startup sync did not pass")
    if metadata.get("idle_command_capture_restart") is not True:
        raise RuntimeError("recorder startup did not use the idle-command capture restart")
    if "prewarmed_command_restart" in metadata:
        raise RuntimeError("recorder startup metadata contains the retired prewarmed contract")
    for key in ("dsp_audio_barrier_frames", "dtk_audio_barrier_frames"):
        if integer(metadata, key) != 0:
            raise RuntimeError("recorder startup sync retained a nonzero pre-boundary audio frame")

    raw_install = sync.get("command_install")
    if not isinstance(raw_install, Mapping):
        raise TypeError("recorder startup command install must be an object")
    install = raw_install
    if install.get("method") != "same-directory-rename-distinct-whole-second-mtime-v1":
        raise RuntimeError("recorder startup command install used an unknown method")
    active_path = install.get("active_path")
    template_path = install.get("capture_template_path")
    if (
        not isinstance(active_path, str)
        or not active_path
        or not isinstance(template_path, str)
        or not template_path
        or Path(active_path) == Path(template_path)
    ):
        raise RuntimeError("recorder startup command paths are invalid or indistinct")
    command_hashes: list[str] = []
    for key in (
        "idle_command_sha256",
        "capture_template_sha256",
        "installed_command_sha256",
    ):
        value = install.get(key)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise RuntimeError("recorder startup command identity hash is malformed")
        command_hashes.append(value)
    if len(set(command_hashes)) != len(command_hashes):
        raise RuntimeError("recorder startup idle, template, and installed commands are indistinct")
    idle_mtime = integer(install, "idle_command_mtime_seconds")
    idle_mtime_nanoseconds = integer(install, "idle_command_mtime_nanoseconds")
    installed_mtime = integer(install, "installed_command_mtime_seconds")
    installed_mtime_nanoseconds = integer(install, "installed_command_mtime_nanoseconds")
    if (
        idle_mtime == installed_mtime
        or abs(installed_mtime - idle_mtime) != 2
        or not 0 <= idle_mtime_nanoseconds < 1_000_000_000
        or installed_mtime_nanoseconds != 0
        or install.get("mtime_seconds_distinct") is not True
    ):
        raise RuntimeError(
            "recorder startup command mtimes are not concretely distinct: "
            f"idle={idle_mtime}.{idle_mtime_nanoseconds:09d} "
            f"installed={installed_mtime}.{installed_mtime_nanoseconds:09d} "
            f"distinct={install.get('mtime_seconds_distinct')!r}"
        )
    capture_command_id = install.get("capture_command_id")
    if not isinstance(capture_command_id, str) or not capture_command_id.startswith(
        "game-video-capture-"
    ):
        raise RuntimeError("recorder startup capture command has no unique identity")
    installed_replay = install.get("replay_path")
    if (
        not isinstance(installed_replay, str)
        or str(Path(installed_replay).resolve()) != expected_replay
    ):
        raise RuntimeError("recorder startup command installed the wrong replay")
    install_frame_values = {
        "requested_start_frame": expected_start_frame,
        "requested_inclusive_end": expected_inclusive_end_frame,
        "command_exclusive_end": expected_exclusive_end,
    }
    if any(integer(install, key) != expected for key, expected in install_frame_values.items()):
        raise RuntimeError("recorder startup command installed the wrong frame interval")
    if install.get("passed") is not True:
        raise RuntimeError("recorder startup command installation did not pass")

    raw_pty = sync.get("pty")
    if not isinstance(raw_pty, Mapping):
        raise TypeError("recorder startup PTY proof must be an object")
    pty_metadata = raw_pty
    if pty_metadata.get("schema") != schema:
        raise RuntimeError("recorder startup PTY proof used an unknown schema")
    if (
        pty_metadata.get("command_boundary_semantics")
        != STARTUP_SYNC_COMMAND_BOUNDARY_SEMANTICS
        or pty_metadata.get("generation_boundary_semantics")
        != STARTUP_SYNC_GENERATION_BOUNDARY_SEMANTICS
    ):
        raise RuntimeError("recorder startup PTY boundary semantics are invalid")
    if integer(pty_metadata, "pty_master_fd") < 0:
        raise RuntimeError("recorder startup PTY descriptor is invalid")
    log_path = pty_metadata.get("log_path")
    if not isinstance(log_path, str) or not log_path:
        raise RuntimeError("recorder startup PTY log path is missing")
    total_bytes = integer(pty_metadata, "total_pty_bytes")
    total_lines = integer(pty_metadata, "total_pty_lines")
    arm_byte_offset = integer(pty_metadata, "generation_arm_byte_offset")
    arm_line_offset = integer(pty_metadata, "generation_arm_line_offset")
    arm_pending_byte_count = integer(pty_metadata, "generation_arm_pending_byte_count")
    if (
        total_bytes <= 0
        or total_lines <= 0
        or not 0 <= arm_byte_offset < total_bytes
        or not 0 <= arm_line_offset < total_lines
        or arm_pending_byte_count != 0
    ):
        raise RuntimeError("recorder startup PTY offsets do not delimit a concrete generation")
    for key in ("expected_replay_path", "seen_replay_path"):
        value = pty_metadata.get(key)
        if not isinstance(value, str) or str(Path(value).resolve()) != expected_replay:
            raise RuntimeError("recorder startup PTY acknowledged the wrong replay")
    pty_frame_values = {
        "requested_start_frame": expected_start_frame,
        "requested_inclusive_end": expected_inclusive_end_frame,
        "command_exclusive_end": expected_exclusive_end,
        "seen_start_frame": expected_start_frame,
        "seen_end_frame": expected_exclusive_end,
        "generation_frame_count": expected_trace_count,
        "generation_first_frame": expected_start_frame,
        "generation_last_frame": expected_inclusive_end_frame,
    }
    if any(integer(pty_metadata, key) != expected for key, expected in pty_frame_values.items()):
        raise RuntimeError("recorder startup PTY frame acknowledgments are incomplete")
    seen_game_end_frame = integer(pty_metadata, "seen_game_end_frame")
    if seen_game_end_frame < expected_inclusive_end_frame:
        raise RuntimeError("recorder startup PTY game-end marker precedes retained content")
    file_marker_line = integer(pty_metadata, "file_marker_line")
    start_marker_line = integer(pty_metadata, "start_marker_line")
    game_end_marker_line = integer(pty_metadata, "game_end_marker_line")
    end_marker_line = integer(pty_metadata, "end_marker_line")
    if not (
        arm_line_offset
        < file_marker_line
        < start_marker_line
        < game_end_marker_line
        < end_marker_line
        <= total_lines
    ):
        raise RuntimeError("recorder startup PTY marker lines are missing or out of order")
    if integer(pty_metadata, "pre_start_frame_line_count") < 0:
        raise RuntimeError("recorder startup PTY pre-start context count is negative")
    exclusive_sentinel_observed = pty_metadata.get("exclusive_sentinel_observed")
    if not isinstance(exclusive_sentinel_observed, bool):
        raise TypeError("recorder startup PTY exclusive sentinel telemetry must be boolean")
    exclusive_sentinel_line = integer(pty_metadata, "exclusive_sentinel_line")
    if (
        pty_metadata.get("exclusive_sentinel_excluded_from_content_trace") is not True
        or (
            exclusive_sentinel_observed
            and not end_marker_line < exclusive_sentinel_line <= total_lines
        )
        or (not exclusive_sentinel_observed and exclusive_sentinel_line != 0)
    ):
        raise RuntimeError("recorder startup PTY optional exclusive sentinel telemetry is invalid")

    raw_frame_trace = pty_metadata.get("generation_frame_trace")
    if not isinstance(raw_frame_trace, Mapping):
        raise TypeError("recorder startup PTY generation frame trace must be an object")
    if raw_frame_trace.get("encoding") != "signed-frame-number-newline-v1":
        raise RuntimeError("recorder startup PTY generation frame trace has an unknown encoding")
    if integer(raw_frame_trace, "row_count") != expected_trace_count:
        raise RuntimeError("recorder startup PTY generation frame trace count is incomplete")
    frame_rows = raw_frame_trace.get("rows")
    expected_frame_rows = "".join(
        f"{frame}\n" for frame in range(expected_start_frame, expected_inclusive_end_frame + 1)
    )
    if not isinstance(frame_rows, str) or frame_rows != expected_frame_rows:
        raise RuntimeError(
            "recorder startup PTY frame trace is not consecutive through the inclusive last "
            "generated content frame"
        )
    frame_trace_sha256 = raw_frame_trace.get("sha256")
    recomputed_frame_trace_sha256 = hashlib.sha256(frame_rows.encode("utf-8")).hexdigest()
    if frame_trace_sha256 != recomputed_frame_trace_sha256:
        raise RuntimeError("recorder startup PTY frame trace hash does not match its embedded rows")

    raw_stop_events = pty_metadata.get("stop_events")
    if not isinstance(raw_stop_events, list) or len(raw_stop_events) != 3:
        raise RuntimeError("recorder startup PTY proof must contain exactly three frame stops")
    previous_line = end_marker_line
    previous_wall_time = -1.0
    stop_expectations = (
        (expected_start_frame, 1),
        (expected_start_frame + 1, 2),
        (expected_inclusive_end_frame, expected_trace_count),
    )
    for raw_event, (target, expected_generation_count) in zip(
        raw_stop_events, stop_expectations, strict=True
    ):
        if not isinstance(raw_event, Mapping):
            raise TypeError("recorder startup PTY stop event must be an object")
        if raw_event.get("kind") != "frame-stop" or integer(raw_event, "target_frame") != target:
            raise RuntimeError("recorder startup PTY stop event targeted the wrong frame")
        line_number = integer(raw_event, "line_number")
        if not previous_line < line_number <= total_lines:
            raise RuntimeError("recorder startup PTY stop event line ordering is invalid")
        if integer(raw_event, "generation_frame_count") != expected_generation_count:
            raise RuntimeError("recorder startup PTY stop event frame accounting is invalid")
        if raw_event.get("signal_succeeded") is not True:
            raise RuntimeError("recorder startup PTY stop signal did not succeed")
        wall_time = number(raw_event, "wall_time_seconds")
        if wall_time <= previous_wall_time:
            raise RuntimeError("recorder startup PTY stop event times do not increase strictly")
        previous_line = line_number
        previous_wall_time = wall_time
    if exclusive_sentinel_observed and exclusive_sentinel_line <= previous_line:
        raise RuntimeError("recorder startup PTY exclusive sentinel preceded the terminal frame stop")
    if (
        pty_metadata.get("markers_complete") is not True
        or pty_metadata.get("trace_complete") is not True
        or not isinstance(pty_metadata.get("reader_ended"), bool)
        or pty_metadata.get("failure") != ""
        or pty_metadata.get("passed") is not True
    ):
        raise RuntimeError("recorder startup PTY proof did not pass every acknowledgment gate")

    raw_idle_audio = sync.get("idle_audio_boundary")
    if not isinstance(raw_idle_audio, Mapping):
        raise TypeError("recorder startup idle audio boundary must be an object")
    idle_audio = raw_idle_audio
    if idle_audio.get("source_contract") != (
        "idle replay accepts zero WaveFile samples; current-frame equality guard excludes "
        "opening-frame audio before the second stop"
    ):
        raise RuntimeError("recorder startup idle audio boundary used an unknown source contract")
    stream_paths: dict[str, str] = {}
    for stage in ("initial", "first_stop", "second_stop"):
        raw_stage = idle_audio.get(stage)
        if not isinstance(raw_stage, Mapping):
            raise TypeError("recorder startup idle audio stage must be an object")
        for stream_name in ("dsp", "dtk"):
            raw_state = raw_stage.get(stream_name)
            if not isinstance(raw_state, Mapping):
                raise TypeError("recorder startup idle audio stream state must be an object")
            path = raw_state.get("path")
            if not isinstance(path, str) or not path:
                raise RuntimeError("recorder startup idle audio stream path is missing")
            if stream_name in stream_paths and path != stream_paths[stream_name]:
                raise RuntimeError("recorder startup idle audio stream path changed across stops")
            stream_paths[stream_name] = path
            if raw_state.get("exists") is not True:
                raise RuntimeError("recorder startup idle audio stream did not exist")
            if raw_state.get("regular_file") is not True:
                raise RuntimeError("recorder startup idle audio stream is not a regular file")
            size_bytes = integer(raw_state, "size_bytes")
            sample_rate = integer(raw_state, "sample_rate")
            block_align = integer(raw_state, "block_align")
            data_offset = integer(raw_state, "data_offset_bytes")
            data_frames = integer(raw_state, "data_frames")
            logical_data_frames = integer(raw_state, "logical_data_frames")
            layout_valid = raw_state.get("layout_valid")
            empty_file = raw_state.get("empty_file")
            header_only = raw_state.get("header_only_zero_data")
            if not all(isinstance(value, bool) for value in (layout_valid, empty_file, header_only)):
                raise TypeError("recorder startup idle audio state flags must be booleans")
            empty_state = (
                empty_file is True
                and header_only is False
                and layout_valid is False
                and size_bytes == 0
                and sample_rate == 0
                and block_align == 0
                and data_offset == 0
                and data_frames == -1
            )
            header_only_state = (
                empty_file is False
                and header_only is True
                and layout_valid is True
                and size_bytes == data_offset
                and size_bytes > 0
                and sample_rate > 0
                and block_align > 0
                and data_frames == 0
            )
            if (
                not (empty_state or header_only_state)
                or logical_data_frames != 0
                or raw_state.get("zero_audio_passed") is not True
            ):
                raise RuntimeError("recorder startup idle audio contains pre-boundary PCM bytes")
    if (
        integer(idle_audio, "logical_dsp_barrier_frames") != 0
        or integer(idle_audio, "logical_dtk_barrier_frames") != 0
        or idle_audio.get("visible_pcm_frames_unchanged_at_zero") is not True
        or idle_audio.get("passed") is not True
    ):
        raise RuntimeError("recorder startup logical audio boundary is nonzero or incomplete")

    raw_capture_boundary = sync.get("capture_boundary")
    if not isinstance(raw_capture_boundary, Mapping):
        raise TypeError("recorder startup capture boundary must be an object")
    capture_boundary = raw_capture_boundary

    def boundary_snapshot(key: str) -> dict[str, Any]:
        raw_snapshot = capture_boundary.get(key)
        if not isinstance(raw_snapshot, Mapping):
            raise TypeError("recorder startup capture boundary snapshot must be an object")
        callback_count = integer(raw_snapshot, "callback_count")
        complete_count = integer(raw_snapshot, "complete_sample_count")
        idle_count = integer(raw_snapshot, "idle_sample_count")
        display_time = integer(raw_snapshot, "last_display_time_mach")
        relative_seconds = number(raw_snapshot, "last_relative_display_seconds")
        signature = raw_snapshot.get("last_visual_signature")
        content_class = raw_snapshot.get("last_content_class")
        if (
            callback_count < 1
            or complete_count < 1
            or idle_count < 0
            or complete_count + idle_count > callback_count
            or display_time <= 0
            or relative_seconds < 0.0
            or not isinstance(signature, str)
            or re.fullmatch(r"[0-9a-f]{16}", signature) is None
            or signature == "0000000000000000"
            or content_class not in {"A", "W", "K", "N"}
        ):
            raise RuntimeError("recorder startup capture boundary snapshot is invalid")
        return {
            "callback_count": callback_count,
            "complete_sample_count": complete_count,
            "idle_sample_count": idle_count,
            "last_display_time_mach": display_time,
            "last_relative_display_seconds": relative_seconds,
            "last_visual_signature": signature,
            "last_content_class": content_class,
        }

    first_stop = boundary_snapshot("first_stop")
    before_advance = boundary_snapshot("before_opening_frame_advance")
    immediate_second = boundary_snapshot("immediate_after_second_stop")
    settled_second = boundary_snapshot("settled_after_second_stop")
    for earlier, later in (
        (first_stop, before_advance),
        (before_advance, immediate_second),
        (immediate_second, settled_second),
    ):
        for key in (
            "callback_count",
            "complete_sample_count",
            "idle_sample_count",
            "last_display_time_mach",
        ):
            if later[key] < earlier[key]:
                raise RuntimeError("recorder startup capture boundary counters moved backward")
        if later["last_relative_display_seconds"] < earlier["last_relative_display_seconds"]:
            raise RuntimeError("recorder startup capture boundary display time moved backward")
    signature_changed = capture_boundary.get("visual_signature_changed")
    if not isinstance(signature_changed, bool):
        raise TypeError("recorder startup visual signature change telemetry must be boolean")
    expected_signature_changed = (
        settled_second["last_visual_signature"] != before_advance["last_visual_signature"]
    )
    if signature_changed != expected_signature_changed:
        raise RuntimeError("recorder startup visual signature change telemetry is inconsistent")
    if (
        settled_second["callback_count"] <= immediate_second["callback_count"]
        or settled_second["complete_sample_count"] <= before_advance["complete_sample_count"]
        or settled_second["last_display_time_mach"] <= before_advance["last_display_time_mach"]
        or capture_boundary.get("fresh_callback_after_second_stop") is not True
        or capture_boundary.get("complete_sample_advanced") is not True
        or capture_boundary.get("display_time_advanced") is not True
        or capture_boundary.get("visual_signature_concrete") is not True
        or capture_boundary.get("passed") is not True
    ):
        raise RuntimeError("recorder startup capture boundary did not prove the opening frame")
    alignment_offset = number(capture_boundary, "video_alignment_offset_seconds")
    if not math.isclose(
        alignment_offset,
        _finite_number(metadata, "video_alignment_offset_seconds"),
        abs_tol=1e-9,
    ):
        raise RuntimeError("recorder startup capture boundary has the wrong video alignment offset")
    if alignment_offset + 1e-9 < settled_second["last_relative_display_seconds"]:
        raise RuntimeError("recorder startup video alignment does not cover the settled boundary")

    raw_terminal_apply = sync.get("terminal_apply_boundary")
    if not isinstance(raw_terminal_apply, Mapping):
        raise TypeError("recorder startup terminal apply boundary must be an object")
    if raw_terminal_apply.get("method") != (
        "inclusive-current-frame-resume-exact-complete-audio-stop-"
        "retained-callback-interval-v7"
    ):
        raise RuntimeError("recorder startup terminal apply boundary used an unknown method")

    def terminal_snapshot(key: str) -> dict[str, Any]:
        raw_snapshot = raw_terminal_apply.get(key)
        if not isinstance(raw_snapshot, Mapping):
            raise TypeError("recorder startup terminal apply snapshot must be an object")
        callback_count = integer(raw_snapshot, "callback_count")
        complete_count = integer(raw_snapshot, "complete_sample_count")
        idle_count = integer(raw_snapshot, "idle_sample_count")
        visual_transition_count = integer(
            raw_snapshot, "visual_signature_transition_count"
        )
        class_transition_count = integer(
            raw_snapshot, "content_class_transition_count"
        )
        display_time = integer(raw_snapshot, "last_display_time_mach")
        display_host_seconds = number(raw_snapshot, "last_display_host_seconds")
        complete_display_time = integer(raw_snapshot, "last_complete_display_time_mach")
        complete_display_host_seconds = number(
            raw_snapshot, "last_complete_display_host_seconds"
        )
        complete_callback_sequence = integer(
            raw_snapshot, "last_complete_callback_sequence"
        )
        relative_seconds = number(raw_snapshot, "last_relative_display_seconds")
        visual_signature = raw_snapshot.get("last_visual_signature")
        content_class = raw_snapshot.get("last_content_class")
        if (
            callback_count < 1
            or complete_count < 1
            or complete_count > callback_count
            or idle_count < 0
            or complete_count + idle_count > callback_count
            or visual_transition_count < 0
            or class_transition_count < 0
            or display_time <= 0
            or display_host_seconds <= 0.0
            or not 0 < complete_display_time <= display_time
            or not 0.0 < complete_display_host_seconds <= display_host_seconds
            or not 0 < complete_callback_sequence <= callback_count
            or relative_seconds < 0.0
            or not isinstance(visual_signature, str)
            or re.fullmatch(r"[0-9a-f]{16}", visual_signature) is None
            or visual_signature == "0000000000000000"
            or content_class not in {"A", "W", "K", "N"}
        ):
            raise RuntimeError("recorder startup terminal apply snapshot is invalid")
        return {
            "callback_count": callback_count,
            "complete_sample_count": complete_count,
            "idle_sample_count": idle_count,
            "visual_signature_transition_count": visual_transition_count,
            "content_class_transition_count": class_transition_count,
            "last_display_time_mach": display_time,
            "last_display_host_seconds": display_host_seconds,
            "last_complete_display_time_mach": complete_display_time,
            "last_complete_display_host_seconds": complete_display_host_seconds,
            "last_complete_callback_sequence": complete_callback_sequence,
            "last_relative_display_seconds": relative_seconds,
            "last_visual_signature": visual_signature,
            "last_content_class": content_class,
        }

    terminal_stop = terminal_snapshot("at_inclusive_frame_stop")
    terminal_barrier = terminal_snapshot("at_inclusive_frame_barrier")
    terminal_applied = terminal_snapshot("accepted_post_resume_complete")
    terminal_stopped_hold = terminal_snapshot("after_stopped_display_hold")
    terminal_final_frozen_tail = terminal_snapshot("after_final_frozen_tail")
    resume_host_time = integer(raw_terminal_apply, "resume_host_time_mach")
    resume_host_seconds = number(raw_terminal_apply, "resume_host_time_seconds")
    maximum_application_seconds = number(
        raw_terminal_apply, "maximum_application_proof_seconds"
    )
    accepted_complete_callback_sequence = integer(
        raw_terminal_apply, "accepted_complete_callback_sequence"
    )
    accepted_complete_display_time = integer(
        raw_terminal_apply, "accepted_complete_display_time_mach"
    )
    observed_resume_to_complete_wall = number(
        raw_terminal_apply, "observed_resume_to_complete_wall_seconds"
    )
    required_stopped_display_hold = number(
        raw_terminal_apply, "required_stopped_display_hold_seconds"
    )
    observed_stopped_display_hold = number(
        raw_terminal_apply, "observed_stopped_display_hold_seconds"
    )
    expected_stopped_display_hold = 1.0 / (
        DEFAULT_FRAME_RATE * _finite_number(metadata, "playback_emulation_speed")
    )
    recomputed_stopped_display_hold = (
        terminal_stopped_hold["last_display_host_seconds"]
        - terminal_applied["last_complete_display_host_seconds"]
    )
    post_accept_callback_delta = (
        terminal_stopped_hold["callback_count"] - accepted_complete_callback_sequence
    )
    post_snapshot_callback_delta = (
        terminal_stopped_hold["callback_count"] - terminal_applied["callback_count"]
    )
    complete_delta = (
        terminal_stopped_hold["complete_sample_count"]
        - terminal_applied["complete_sample_count"]
    )
    idle_delta = (
        terminal_stopped_hold["idle_sample_count"]
        - terminal_applied["idle_sample_count"]
    )
    complete_identity_advanced = complete_delta > 0
    complete_identity_consistent = (
        complete_delta == 0
        and terminal_stopped_hold["last_complete_callback_sequence"]
        == terminal_applied["last_complete_callback_sequence"]
        and terminal_stopped_hold["last_complete_display_time_mach"]
        == terminal_applied["last_complete_display_time_mach"]
        and math.isclose(
            terminal_stopped_hold["last_complete_display_host_seconds"],
            terminal_applied["last_complete_display_host_seconds"],
            abs_tol=1e-12,
        )
    ) or (
        complete_delta > 0
        and terminal_stopped_hold["last_complete_callback_sequence"]
        > terminal_applied["last_complete_callback_sequence"]
        and terminal_stopped_hold["last_complete_display_time_mach"]
        > terminal_applied["last_complete_display_time_mach"]
        and terminal_stopped_hold["last_complete_display_host_seconds"]
        > terminal_applied["last_complete_display_host_seconds"]
    )
    transition_counters_unchanged = (
        terminal_stopped_hold["visual_signature_transition_count"]
        == terminal_applied["visual_signature_transition_count"]
        and terminal_stopped_hold["content_class_transition_count"]
        == terminal_applied["content_class_transition_count"]
    )
    if (
        terminal_applied["callback_count"] <= terminal_barrier["callback_count"]
        or terminal_applied["complete_sample_count"]
        <= terminal_barrier["complete_sample_count"]
        or terminal_applied["last_complete_callback_sequence"]
        <= terminal_barrier["callback_count"]
        or terminal_applied["last_complete_display_time_mach"] <= resume_host_time
        or accepted_complete_callback_sequence
        != terminal_applied["last_complete_callback_sequence"]
        or accepted_complete_display_time
        != terminal_applied["last_complete_display_time_mach"]
        or resume_host_time <= terminal_barrier["last_display_time_mach"]
        or resume_host_seconds <= terminal_barrier["last_display_host_seconds"]
        or terminal_applied["last_relative_display_seconds"]
        <= terminal_barrier["last_relative_display_seconds"]
        or not 0.0 < maximum_application_seconds <= 1.0
        or not 0.0 <= observed_resume_to_complete_wall
        <= maximum_application_seconds + 0.010
        or not math.isclose(
            required_stopped_display_hold,
            expected_stopped_display_hold,
            abs_tol=1e-12,
        )
        or observed_stopped_display_hold < required_stopped_display_hold
        or not math.isclose(
            observed_stopped_display_hold,
            recomputed_stopped_display_hold,
            abs_tol=1e-9,
        )
        or terminal_stopped_hold["callback_count"]
        <= terminal_applied["callback_count"]
        or post_accept_callback_delta < 1
        or post_snapshot_callback_delta < 1
        or complete_delta < 0
        or idle_delta < 0
        or post_snapshot_callback_delta != complete_delta + idle_delta
        or not complete_identity_consistent
        or not transition_counters_unchanged
        or terminal_stopped_hold["last_visual_signature"]
        != terminal_applied["last_visual_signature"]
        or terminal_stopped_hold["last_content_class"]
        != terminal_applied["last_content_class"]
        or raw_terminal_apply.get("fresh_callback_after_resume") is not True
        or raw_terminal_apply.get("complete_sample_advanced") is not True
        or raw_terminal_apply.get("complete_callback_sequence_advanced") is not True
        or raw_terminal_apply.get("complete_display_time_after_resume") is not True
        or raw_terminal_apply.get("stopped_hold_callback_advanced") is not True
        or integer(
            raw_terminal_apply, "stopped_hold_callback_sequence_start_inclusive"
        )
        != accepted_complete_callback_sequence
        or integer(raw_terminal_apply, "stopped_hold_callback_sequence_end_inclusive")
        != terminal_stopped_hold["callback_count"]
        or integer(raw_terminal_apply, "stopped_hold_post_accept_callback_delta")
        != post_accept_callback_delta
        or integer(raw_terminal_apply, "stopped_hold_post_snapshot_callback_delta")
        != post_snapshot_callback_delta
        or integer(raw_terminal_apply, "stopped_hold_complete_sample_delta")
        != complete_delta
        or integer(raw_terminal_apply, "stopped_hold_idle_sample_delta") != idle_delta
        or raw_terminal_apply.get("stopped_hold_complete_identity_advanced")
        is not complete_identity_advanced
        or raw_terminal_apply.get("stopped_hold_only_retained_idle_or_complete") is not True
        or raw_terminal_apply.get("stopped_hold_transition_counters_unchanged") is not True
        or raw_terminal_apply.get("stopped_hold_complete_identity_consistent") is not True
        or raw_terminal_apply.get("stopped_hold_display_interval_covered") is not True
        or raw_terminal_apply.get("stopped_hold_visual_state_retained") is not True
        or raw_terminal_apply.get("visible_audio_within_pre_stop_bounds") is not True
        or raw_terminal_apply.get("parent_stop_signal_succeeded") is not True
        or raw_terminal_apply.get("passed") is not True
    ):
        raise RuntimeError(
            "recorder startup terminal apply boundary did not prove the inclusive final frame"
        )
    _validate_terminal_stopped_hold_trace(
        metadata,
        terminal_stop=terminal_stop,
        terminal_barrier=terminal_barrier,
        terminal_applied=terminal_applied,
        terminal_stopped_hold=terminal_stopped_hold,
        terminal_final_frozen_tail=terminal_final_frozen_tail,
        terminal_apply=raw_terminal_apply,
    )

    raw_terminal_audio = sync.get("terminal_audio_boundary")
    if not isinstance(raw_terminal_audio, Mapping):
        raise TypeError("recorder startup terminal audio boundary must be an object")
    if raw_terminal_audio.get("method") != "physical-wave-seal-parent-stop-no-growth-v4":
        raise RuntimeError("recorder startup terminal audio boundary used an unknown method")
    audio_delay = number(raw_terminal_audio, "audio_presentation_delay_seconds")
    if not math.isclose(
        audio_delay,
        float(expected_audio_presentation_delay_seconds),
        abs_tol=1e-12,
    ):
        raise RuntimeError("recorder startup terminal audio boundary used the wrong delay")
    expected_raw_endpoint = expected_content_count / DEFAULT_FRAME_RATE - audio_delay
    if expected_raw_endpoint <= 0.0 or not math.isclose(
        number(raw_terminal_audio, "target_content_endpoint_seconds"),
        expected_raw_endpoint,
        abs_tol=1e-9,
    ):
        raise RuntimeError("recorder startup terminal raw audio endpoint is invalid")
    maximum_overshoot = number(
        raw_terminal_audio, "maximum_physical_overshoot_seconds"
    )
    if not math.isclose(
        maximum_overshoot,
        AUDIO_PHYSICAL_MAXIMUM_OVERSHOOT_SECONDS,
        abs_tol=1e-12,
    ):
        raise RuntimeError("recorder startup terminal audio overshoot bound is invalid")
    maximum_undershoot = number(
        raw_terminal_audio, "maximum_physical_undershoot_seconds"
    )
    if not math.isclose(
        maximum_undershoot,
        AUDIO_PHYSICAL_MAXIMUM_UNDERSHOOT_SECONDS,
        abs_tol=1e-12,
    ):
        raise RuntimeError("recorder startup terminal audio undershoot bound is invalid")
    finalized_buffer_allowance = number(
        raw_terminal_audio, "finalized_buffer_allowance_seconds"
    )
    if not math.isclose(finalized_buffer_allowance, 0.100, abs_tol=1e-12):
        raise RuntimeError("recorder startup terminal finalized buffer allowance is invalid")
    lower_bound = expected_raw_endpoint - maximum_undershoot
    upper_bound = expected_raw_endpoint + maximum_overshoot
    if (
        not math.isclose(
            number(raw_terminal_audio, "physical_lower_bound_seconds"),
            lower_bound,
            abs_tol=1e-12,
        )
        or not math.isclose(
            number(raw_terminal_audio, "physical_upper_bound_seconds"),
            upper_bound,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError("recorder startup terminal visible-audio bounds are invalid")
    for stream_name in ("dsp", "dtk"):
        sample_rate = integer(raw_terminal_audio, f"{stream_name}_sample_rate")
        before_frames = integer(raw_terminal_audio, f"{stream_name}_frames_before_parent_stop")
        baseline_frames = integer(raw_terminal_audio, f"{stream_name}_frames_at_stop_baseline")
        after_frames = integer(raw_terminal_audio, f"{stream_name}_frames_after_observation")
        if sample_rate <= 0 or min(before_frames, baseline_frames, after_frames) < 0:
            raise RuntimeError("recorder startup terminal audio frame accounting is invalid")
        before_endpoint = before_frames / sample_rate
        if not lower_bound <= before_endpoint <= upper_bound:
            raise RuntimeError("recorder startup terminal visible audio was outside its stop bounds")
        endpoint = number(
            raw_terminal_audio, f"{stream_name}_physical_endpoint_seconds"
        )
        endpoint_offset = number(
            raw_terminal_audio, f"{stream_name}_physical_offset_seconds"
        )
        recomputed_endpoint = after_frames / sample_rate
        baseline_endpoint = baseline_frames / sample_rate
        recomputed_offset = recomputed_endpoint - expected_raw_endpoint
        if (
            not math.isclose(endpoint, recomputed_endpoint, abs_tol=1e-12)
            or not math.isclose(endpoint_offset, recomputed_offset, abs_tol=1e-12)
            or not lower_bound <= endpoint <= upper_bound
            or not lower_bound <= baseline_endpoint <= upper_bound
            or baseline_frames < before_frames
            or baseline_frames != after_frames
            or metadata.get(f"sealed_{stream_name}_audio_frames") != after_frames
        ):
            raise RuntimeError("recorder startup terminal audio post-stop proof is invalid")
    if not math.isclose(
        number(raw_terminal_audio, "stop_stability_interval_seconds"),
        0.025,
        abs_tol=1e-12,
    ) or not math.isclose(
        number(raw_terminal_audio, "post_stop_observation_seconds"),
        0.050,
        abs_tol=1e-12,
    ):
        raise RuntimeError("recorder startup terminal audio no-growth window is invalid")
    if (
        raw_terminal_audio.get("visible_audio_within_bounds_before_parent_stop") is not True
        or raw_terminal_audio.get("post_stop_no_growth") is not True
        or raw_terminal_audio.get("physical_pre_sigint_seal_proven") is not True
        or raw_terminal_audio.get("post_sigint_revalidation_and_content_seal_required")
        is not True
        or raw_terminal_audio.get("passed") is not True
    ):
        raise RuntimeError("recorder startup terminal audio proof did not pass")

    raw_terminal_tail = sync.get("terminal_capture_tail")
    if not isinstance(raw_terminal_tail, Mapping):
        raise TypeError("recorder startup terminal capture tail must be an object")
    required_terminal_tail = max(minimum_tail_seconds + 0.100, 0.350)
    reported_required_tail = number(raw_terminal_tail, "required_source_seconds")
    observed_source_tail = number(raw_terminal_tail, "observed_source_seconds")
    observed_wall_tail = number(raw_terminal_tail, "observed_wall_seconds")
    if not math.isclose(reported_required_tail, required_terminal_tail, abs_tol=1e-9):
        raise RuntimeError("recorder startup terminal capture tail used the wrong requirement")
    if observed_source_tail < required_terminal_tail:
        raise RuntimeError("recorder startup terminal source capture tail is incomplete")
    if observed_wall_tail + CAPTURE_DELIVERY_MAXIMUM_GAP_SECONDS < observed_source_tail:
        raise RuntimeError("recorder startup terminal wall capture tail is incomplete")
    if (
        raw_terminal_tail.get("capture_callbacks_continued") is not True
        or raw_terminal_tail.get("passed") is not True
    ):
        raise RuntimeError("recorder startup terminal capture callbacks did not continue")


def _validate_mux_output_frame_rate(metadata: Mapping[str, Any]) -> None:
    """Require a normal-cadence 60 fps exported replay."""
    output_frame_rate = metadata.get("output_frame_rate")
    if (
        isinstance(output_frame_rate, bool)
        or not isinstance(output_frame_rate, (int, float))
        or not math.isfinite(float(output_frame_rate))
        or float(output_frame_rate) != DEFAULT_FRAME_RATE
    ):
        raise RuntimeError("Slippi replay muxer used the wrong output frame rate")
    measured_output_frame_rate = metadata.get("measured_output_frame_rate")
    if (
        isinstance(measured_output_frame_rate, bool)
        or not isinstance(measured_output_frame_rate, (int, float))
        or not math.isfinite(float(measured_output_frame_rate))
        or not math.isclose(float(measured_output_frame_rate), DEFAULT_FRAME_RATE, abs_tol=0.001)
    ):
        raise RuntimeError("Slippi replay muxer produced the wrong measured output frame rate")


def _validate_recorder_timing_metadata(
    metadata: Mapping[str, Any],
    *,
    expected_audio_duration_seconds: float,
    minimum_tail_seconds: float,
) -> list[dict[str, float]]:
    """Validate the complete audio-clock capture contract used for replay retiming."""
    if metadata.get("completion_method") != "dolphin-audio-clock-stable-end":
        raise RuntimeError("recorder timing completion method is not audio-clock based")
    if metadata.get("audio_end_observed") is not True:
        raise RuntimeError("recorder timing did not observe a stable audio end")

    expected = _finite_number(metadata, "expected_audio_duration_seconds")
    if not math.isclose(expected, expected_audio_duration_seconds, abs_tol=1e-6):
        raise RuntimeError("recorder timing expected duration differs from the requested replay duration")
    observed = _finite_number(metadata, "observed_audio_duration_seconds")
    if observed < 0.0:
        raise ValueError("recorder timing observed duration must be nonnegative")
    observed_dsp = _finite_number(metadata, "observed_dsp_duration_seconds")
    observed_dtk = _finite_number(metadata, "observed_dtk_duration_seconds")
    if observed_dsp < 0.0 or observed_dtk < 0.0:
        raise ValueError("recorder timing stream durations must be nonnegative")
    tolerance = _finite_number(metadata, "completion_tolerance_seconds")
    if tolerance < 0.0 or tolerance > 0.5:
        raise RuntimeError("recorder timing completion tolerance exceeds 0.5 seconds")
    if observed + 0.5 < expected_audio_duration_seconds:
        raise RuntimeError("recorder timing audio coverage falls more than 0.5 seconds short of the replay")
    if (
        observed_dsp + 0.5 < expected_audio_duration_seconds
        or observed_dtk + 0.5 < expected_audio_duration_seconds
    ):
        raise RuntimeError("recorder timing per-stream audio coverage is incomplete")
    tail = _finite_number(metadata, "post_audio_tail_seconds")
    if tail < minimum_tail_seconds:
        raise RuntimeError("recorder timing terminal visual tail is shorter than the configured minimum")

    video_alignment_offset = _finite_number(metadata, "video_alignment_offset_seconds")
    if video_alignment_offset < 0.0:
        raise ValueError("recorder timing video alignment offset must be nonnegative")
    raw_landmarks = metadata.get("clock_landmarks")
    if not isinstance(raw_landmarks, list) or len(raw_landmarks) < 2:
        raise TypeError("recorder timing clock landmarks must contain at least two points")

    landmarks: list[dict[str, float]] = []
    previous_source = -1.0
    previous_audio = -1.0
    for index, raw_landmark in enumerate(raw_landmarks):
        if not isinstance(raw_landmark, Mapping):
            raise TypeError("recorder timing landmark entries must be objects")
        try:
            source = _finite_number(raw_landmark, "source_video_seconds")
            audio = _finite_number(raw_landmark, "audio_seconds")
        except (TypeError, ValueError) as error:
            raise ValueError(f"recorder timing landmark {index} is invalid: {error}") from error
        if source < 0.0 or audio < 0.0:
            raise ValueError("recorder timing landmark coordinates must be nonnegative")
        if index == 0:
            if not math.isclose(source, video_alignment_offset, abs_tol=1e-3):
                raise RuntimeError("recorder timing first landmark differs from the video alignment offset")
            if not math.isclose(audio, 0.0, abs_tol=1e-6):
                raise RuntimeError("recorder timing first landmark must start at audio zero")
        elif source <= previous_source or audio <= previous_audio:
            raise RuntimeError("recorder timing landmark coordinates must increase strictly on both axes")
        landmarks.append(
            {
                "source_video_seconds": source,
                "audio_seconds": audio,
            }
        )
        previous_source = source
        previous_audio = audio

    if not math.isclose(landmarks[-1]["audio_seconds"], observed, abs_tol=1e-3):
        raise RuntimeError("recorder timing final landmark differs from the observed audio duration")
    return landmarks


def _extend_landmarks_to_finalized_audio(
    landmarks: list[dict[str, float]],
    *,
    finalized_audio_duration_seconds: float,
    capture_emulation_speed: float = NORMAL_OUTPUT_PLAYBACK_SPEED,
    maximum_buffer_extension_seconds: float = 0.100,
) -> tuple[list[dict[str, float]], float]:
    """Account for Dolphin's final flushed stdio block on the recent clock slope."""
    if len(landmarks) < 2:
        raise ValueError("finalized audio retiming requires at least two clock landmarks")
    if not math.isfinite(finalized_audio_duration_seconds) or finalized_audio_duration_seconds <= 0.0:
        raise ValueError("finalized Dolphin audio duration must be finite and positive")
    if not math.isfinite(maximum_buffer_extension_seconds) or maximum_buffer_extension_seconds < 0.0:
        raise ValueError("maximum Dolphin audio buffer extension must be finite and nonnegative")
    if not math.isfinite(capture_emulation_speed) or capture_emulation_speed <= 0.0:
        raise ValueError("capture emulation speed must be finite and positive")

    result = [dict(landmark) for landmark in landmarks]
    sealed_duration = result[-1]["audio_seconds"]
    extension = finalized_audio_duration_seconds - sealed_duration
    sample_period = 1.0 / 48_000
    if extension < -sample_period:
        raise RuntimeError("finalized Dolphin audio is shorter than its sealed clock boundary")
    if extension > maximum_buffer_extension_seconds:
        raise RuntimeError("finalized Dolphin audio added an unexpectedly large unlandmarked buffer")
    if abs(extension) <= sample_period:
        result[-1]["audio_seconds"] = finalized_audio_duration_seconds
        return result, extension

    final = result[-1]
    recent = result[-6:]
    recent_slopes = []
    for left, right in pairwise(recent):
        recent_slopes.append(
            (right["source_video_seconds"] - left["source_video_seconds"])
            / (right["audio_seconds"] - left["audio_seconds"])
        )
    slope = statistics.median(recent_slopes)
    expected_slope = NORMAL_OUTPUT_PLAYBACK_SPEED / capture_emulation_speed
    minimum_slope = expected_slope * 0.75
    maximum_slope = expected_slope * 1.50
    if not math.isfinite(slope) or slope < minimum_slope or slope > maximum_slope:
        raise RuntimeError(
            "recent replay wall/audio clock slope is invalid for the capture emulation speed"
        )
    result.append(
        {
            "source_video_seconds": final["source_video_seconds"] + extension * slope,
            "audio_seconds": finalized_audio_duration_seconds,
        }
    )
    return result, extension


def _seal_landmarks_to_content_audio(
    landmarks: list[dict[str, float]],
    *,
    content_audio_duration_seconds: float,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    """Clip a physical audio-clock trace at the deterministic mixed endpoint."""
    if len(landmarks) < 2:
        raise ValueError("content audio retiming requires at least two clock landmarks")
    if (
        not math.isfinite(content_audio_duration_seconds)
        or content_audio_duration_seconds <= 0.0
    ):
        raise ValueError("content audio duration must be finite and positive")
    physical = [dict(landmark) for landmark in landmarks]
    for index, landmark in enumerate(physical):
        source = landmark.get("source_video_seconds")
        audio = landmark.get("audio_seconds")
        if (
            isinstance(source, bool)
            or not isinstance(source, (int, float))
            or not math.isfinite(float(source))
            or isinstance(audio, bool)
            or not isinstance(audio, (int, float))
            or not math.isfinite(float(audio))
        ):
            raise TypeError("content audio clock landmarks must contain finite numbers")
        if index and (
            float(source) <= float(physical[index - 1]["source_video_seconds"])
            or float(audio) <= float(physical[index - 1]["audio_seconds"])
        ):
            raise RuntimeError("content audio clock landmarks must increase strictly")
    physical_endpoint = float(physical[-1]["audio_seconds"])
    if content_audio_duration_seconds > physical_endpoint + 1e-12:
        raise RuntimeError("physical audio clock does not reach the deterministic content seal")

    sealed: list[dict[str, float]] = []
    operation = "exact"
    interpolated = False
    for index, landmark in enumerate(physical):
        audio = float(landmark["audio_seconds"])
        if audio < content_audio_duration_seconds - 1e-12:
            sealed.append(
                {
                    "source_video_seconds": float(landmark["source_video_seconds"]),
                    "audio_seconds": audio,
                }
            )
            continue
        if math.isclose(audio, content_audio_duration_seconds, abs_tol=1e-12):
            sealed.append(
                {
                    "source_video_seconds": float(landmark["source_video_seconds"]),
                    "audio_seconds": content_audio_duration_seconds,
                }
            )
            operation = "exact" if index == len(physical) - 1 else "trim"
            break
        if not sealed:
            raise RuntimeError("content audio seal precedes the first physical clock interval")
        left = sealed[-1]
        right_source = float(landmark["source_video_seconds"])
        right_audio = audio
        fraction = (
            (content_audio_duration_seconds - left["audio_seconds"])
            / (right_audio - left["audio_seconds"])
        )
        source = left["source_video_seconds"] + fraction * (
            right_source - left["source_video_seconds"]
        )
        sealed.append(
            {
                "source_video_seconds": source,
                "audio_seconds": content_audio_duration_seconds,
            }
        )
        operation = "trim"
        interpolated = True
        break
    if len(sealed) < 2 or not math.isclose(
        sealed[-1]["audio_seconds"], content_audio_duration_seconds, abs_tol=1e-12
    ):
        raise RuntimeError("could not seal the physical clock at the content endpoint")
    adjustment = content_audio_duration_seconds - physical_endpoint
    metadata = {
        "schema": "dolphin-audio-content-clock-seal-v1",
        "method": "dolphin-audio-content-clock-seal-v1",
        "physical_endpoint_seconds": physical_endpoint,
        "content_endpoint_seconds": content_audio_duration_seconds,
        "signed_adjustment_seconds": adjustment,
        "trimmed_physical_seconds": max(0.0, -adjustment),
        "operation": operation,
        "input_landmark_count": len(physical),
        "output_landmark_count": len(sealed),
        "terminal_landmark_interpolated": interpolated,
        "landmarks": sealed,
        "raw_source_terminal_visual_tail_preserved": True,
        "passed": True,
    }
    return sealed, metadata


def _retain_video_capture_failure(
    project_root: Path,
    replay_path: Path,
    output_path: Path,
    recorded: subprocess.CompletedProcess[str],
    dolphin_log_tail: str,
) -> str:
    """Keep bounded capture diagnostics outside the disposable render workspace."""
    directory = project_root / "artifacts/integration/video-render-failures"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix="capture-", suffix=".json", dir=directory)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({
                "schema": "replay-video-capture-failure-v1",
                "source_replay": str(replay_path.resolve()),
                "requested_output": str(output_path.resolve()),
                "returncode": recorded.returncode,
                "stdout_tail": recorded.stdout[-16_384:],
                "stderr_tail": recorded.stderr[-32_768:],
                "dolphin_log_tail": dolphin_log_tail,
            }, stream, indent=2, sort_keys=True)
            stream.write("\n")
        return name
    except OSError as error:
        return f"could not retain diagnostics: {error}"


def render_replay_video(
    replay_path: Path,
    output_path: Path,
    context: VideoRenderContext,
) -> Mapping[str, Any]:
    """Capture one retained SLP slowly, then export normal-speed Dolphin video and audio."""
    if platform.system() != "Darwin":
        raise RuntimeError("save_video is currently supported only on macOS")
    settings = _video_config(context)
    if settings["startup_timeout_seconds"] <= 0.0:
        raise ValueError("video.startup_timeout_seconds must be positive")
    if settings["tail_padding_seconds"] < MINIMUM_TERMINAL_TAIL_SECONDS:
        raise ValueError(
            f"video.tail_padding_seconds must be at least {MINIMUM_TERMINAL_TAIL_SECONDS:.1f} seconds"
        )
    if settings["resume_delay_seconds"] < 0.0:
        raise ValueError("video.resume_delay_seconds cannot be negative")
    if not math.isfinite(settings["audio_presentation_delay_seconds"]):
        raise ValueError("video.audio_presentation_delay_seconds must be finite")
    if settings["audio_presentation_delay_seconds"] < 0.0:
        raise ValueError("video.audio_presentation_delay_seconds cannot be negative")
    if not math.isclose(
        cast(float, settings["frame_rate"]), DEFAULT_FRAME_RATE, abs_tol=1e-12
    ):
        raise ValueError("video.frame_rate must be exactly 60 for replay frame fidelity")
    configured_audio_delay = cast(float, settings["audio_presentation_delay_seconds"])
    if not math.isclose(
        configured_audio_delay,
        float(f"{configured_audio_delay:.9f}"),
        abs_tol=1e-12,
    ):
        raise ValueError("video.audio_presentation_delay_seconds requires nanosecond precision")
    if settings["width"] < 2 or settings["height"] < 2:
        raise ValueError("video width and height must be at least two pixels")
    expected_capture_width = settings["expected_capture_width"]
    expected_capture_height = settings["expected_capture_height"]
    if (expected_capture_width is None) != (expected_capture_height is None):
        raise ValueError("video expected capture width and height must be configured together")
    if expected_capture_width is not None and (expected_capture_width < 2 or expected_capture_height < 2):
        raise ValueError("video expected capture dimensions must be at least two pixels")
    if not replay_path.is_file():
        raise FileNotFoundError(f"video source replay is missing: {replay_path}")
    if not context.iso_path.is_file():
        raise FileNotFoundError(f"video game image is missing: {context.iso_path}")

    application = _resolve_project_path(settings["playback_application"], context.project_root)
    executable = application / "Contents" / "MacOS" / "Slippi Dolphin"
    if not executable.is_file():
        raise FileNotFoundError(f"Slippi playback executable is missing: {executable}")
    recorder, recorder_source_sha256 = _compile_native_recorder(context.project_root)
    muxer, muxer_source_sha256 = _compile_native_muxer(context.project_root)
    source_first_frame = context.first_frame
    capture_start_frame, capture_end_frame, frame_count = _capture_frame_interval(
        source_first_frame,
        context.last_frame,
    )
    capture_context = replace(context, first_frame=capture_start_frame)
    game_seconds = frame_count / cast(float, settings["frame_rate"])
    player_labels = (
        _player_video_label(context.players[0], 1),
        _player_video_label(context.players[1], 2),
    )
    minimum_capture_width = max(
        2,
        round(cast(int, settings["width"]) * MINIMUM_CAPTURE_DIMENSION_FRACTION),
    )
    minimum_capture_height = max(
        2,
        round(cast(int, settings["height"]) * MINIMUM_CAPTURE_DIMENSION_FRACTION),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".video-render-", dir=output_path.parent) as temporary:
        workspace = Path(temporary)
        user_directory = workspace / "User"
        dump_directory = user_directory / "Dump"
        _write_playback_user(
            user_directory,
            width=cast(int, settings["width"]),
            height=cast(int, settings["height"]),
            dump_directory=dump_directory,
        )
        playback_command = workspace / "playback.json"
        capture_playback_command = workspace / "capture-playback.json"
        _write_idle_playback_command(playback_command, replay_path, capture_context)
        _write_playback_command(capture_playback_command, replay_path, capture_context)
        dolphin_log = workspace / "dolphin.log"
        raw_video_path = workspace / "isolated-window.mp4"
        environment = os.environ.copy()
        environment["SLP_METAL_DOUBLE_BUFFER"] = "1"
        launch = _playback_launch_command(
            executable,
            user_directory,
            playback_command,
            context.iso_path,
        )
        timeout = _playback_recorder_timeout_seconds(
            startup_timeout_seconds=cast(float, settings["startup_timeout_seconds"]),
            game_seconds=game_seconds,
            tail_padding_seconds=cast(float, settings["tail_padding_seconds"]),
            resume_delay_seconds=cast(float, settings["resume_delay_seconds"]),
        )
        recorded, shutdown_method = _run_playback_recorder(
            launch,
            environment,
            _playback_recorder_command(
                recorder,
                startup_timeout_seconds=cast(float, settings["startup_timeout_seconds"]),
                game_seconds=game_seconds,
                tail_padding_seconds=cast(float, settings["tail_padding_seconds"]),
                resume_delay_seconds=cast(float, settings["resume_delay_seconds"]),
                dsp_audio_path=dump_directory / "Audio" / "dspdump.wav",
                dtk_audio_path=dump_directory / "Audio" / "dtkdump.wav",
                minimum_capture_width=minimum_capture_width,
                minimum_capture_height=minimum_capture_height,
                expected_capture_width=expected_capture_width,
                expected_capture_height=expected_capture_height,
                audio_presentation_delay_seconds=cast(
                    float, settings["audio_presentation_delay_seconds"]
                ),
                output_path=raw_video_path,
            ),
            active_playback_command=playback_command,
            capture_playback_command=capture_playback_command,
            dolphin_log=dolphin_log,
            requested_start_frame=capture_start_frame,
            requested_inclusive_end_frame=capture_end_frame,
            timeout_seconds=timeout,
        )
        if recorded.returncode != 0:
            details = recorded.stderr.strip() or recorded.stdout.strip()
            dolphin_tail = _last_log_lines(dolphin_log)
            failure_receipt = _retain_video_capture_failure(
                context.project_root, replay_path, output_path, recorded, dolphin_tail,
            )
            raise RuntimeError(
                f"Slippi replay video recording failed: {details}; Dolphin log: {dolphin_tail}; "
                f"failure receipt: {failure_receipt}"
            )

        if shutdown_method == "sigkill":
            raise RuntimeError("Slippi Dolphin required SIGKILL, so its audio dumps may be incomplete")

        if not raw_video_path.is_file() or raw_video_path.stat().st_size == 0:
            raise RuntimeError("Slippi replay recorder produced no isolated-window MP4")
        output_lines = recorded.stdout.splitlines()
        try:
            recorder_metadata = json.loads(output_lines[-1])
        except (IndexError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"macOS replay recorder returned invalid metadata: {recorded.stdout!r}"
            ) from error
        if not isinstance(recorder_metadata, dict):
            raise TypeError("macOS replay recorder metadata must be an object")
        if recorder_metadata.get("has_video") is not True:
            raise RuntimeError(f"isolated-window MP4 is missing video: {recorder_metadata}")
        capture_width, capture_height = _validate_recorder_capture_dimensions(
            recorder_metadata,
            minimum_width=minimum_capture_width,
            minimum_height=minimum_capture_height,
            expected_width=expected_capture_width,
            expected_height=expected_capture_height,
        )
        _validate_recorder_capture_rate(recorder_metadata)
        _validate_recorder_startup_sync(
            recorder_metadata,
            expected_replay_path=replay_path,
            expected_start_frame=capture_start_frame,
            expected_inclusive_end_frame=capture_end_frame,
            minimum_tail_seconds=cast(float, settings["tail_padding_seconds"]),
            expected_audio_presentation_delay_seconds=cast(
                float, settings["audio_presentation_delay_seconds"]
            ),
        )
        _validate_recorder_capture_delivery(recorder_metadata)
        video_alignment_offset = recorder_metadata.get("video_alignment_offset_seconds")
        if not isinstance(video_alignment_offset, (int, float)) or isinstance(video_alignment_offset, bool):
            raise TypeError("isolated-window recorder did not report a video alignment offset")
        if video_alignment_offset < 0.0:
            raise RuntimeError("isolated-window recorder reported a negative video alignment offset")
        barrier_frames: dict[str, int] = {}
        sealed_frames: dict[str, int] = {}
        for stream_name in ("dsp", "dtk"):
            key = f"{stream_name}_audio_barrier_frames"
            value = recorder_metadata.get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"isolated-window recorder did not report integer {key}")
            if value < 0:
                raise RuntimeError(f"isolated-window recorder reported negative {key}: {value}")
            barrier_frames[stream_name] = value
            sealed_key = f"sealed_{stream_name}_audio_frames"
            sealed_value = recorder_metadata.get(sealed_key)
            if isinstance(sealed_value, bool) or not isinstance(sealed_value, int):
                raise TypeError(f"isolated-window recorder did not report integer {sealed_key}")
            if sealed_value < value:
                raise RuntimeError(f"isolated-window recorder reported {sealed_key} before its barrier")
            sealed_frames[stream_name] = sealed_value
        if recorder_metadata.get("alignment_method") != "dolphin-audio-sample-barrier":
            raise RuntimeError("isolated-window recorder did not use the required audio sample barrier")
        clock_landmarks = _validate_recorder_timing_metadata(
            recorder_metadata,
            expected_audio_duration_seconds=game_seconds,
            minimum_tail_seconds=cast(float, settings["tail_padding_seconds"]),
        )

        audio_directory = dump_directory / "Audio"
        dsp_audio_path, dtk_audio_path = _validate_dolphin_audio_inventory(audio_directory)
        dsp_source_frames, dsp_source_rate = _validated_pcm16_stereo_layout(dsp_audio_path)
        dtk_source_frames, dtk_source_rate = _validated_pcm16_stereo_layout(dtk_audio_path)
        content_seal = _derive_audio_content_seal(
            expected_game_frames=frame_count,
            audio_presentation_delay_seconds=cast(
                float, settings["audio_presentation_delay_seconds"]
            ),
            dsp_source_rate=dsp_source_rate,
            dtk_source_rate=dtk_source_rate,
            dsp_barrier_frames=barrier_frames["dsp"],
            dtk_barrier_frames=barrier_frames["dtk"],
        )
        physical_seal = _build_audio_physical_seal(
            content_seal=content_seal,
            dsp_source_frames=dsp_source_frames,
            dtk_source_frames=dtk_source_frames,
            recorder_sealed_dsp_frames=sealed_frames["dsp"],
            recorder_sealed_dtk_frames=sealed_frames["dtk"],
        )
        content_streams = cast(Mapping[str, Mapping[str, Any]], content_seal["streams"])
        mixed_audio_path = workspace / "dolphin-mixed.wav"
        audio_metadata = _mix_dolphin_audio(
            dsp_audio_path,
            dtk_audio_path,
            mixed_audio_path,
            dsp_barrier_frames=barrier_frames["dsp"],
            dtk_barrier_frames=barrier_frames["dtk"],
            dsp_content_start_frames=int(
                content_streams["dsp"]["content_start_frames"]
            ),
            dtk_content_start_frames=int(
                content_streams["dtk"]["content_start_frames"]
            ),
            dsp_content_end_frames=int(content_streams["dsp"]["content_end_frames"]),
            dtk_content_end_frames=int(content_streams["dtk"]["content_end_frames"]),
            target_output_frames=int(content_seal["output_frames"]),
        )
        audio_metadata["content_seal"] = content_seal
        audio_metadata["physical_seal"] = physical_seal
        for stream_name in ("dsp", "dtk"):
            audio_metadata[f"recorder_sealed_{stream_name}_frames"] = sealed_frames[
                stream_name
            ]
        shared_barrier = Fraction(
            int(content_seal["shared_barrier_numerator"]),
            int(content_seal["shared_barrier_denominator"]),
        )
        physical_finalized_duration = max(
            Fraction(dsp_source_frames, dsp_source_rate) - shared_barrier,
            Fraction(dtk_source_frames, dtk_source_rate) - shared_barrier,
        )
        physical_clock_landmarks, finalized_buffer_extension = (
            _extend_landmarks_to_finalized_audio(
            clock_landmarks,
            finalized_audio_duration_seconds=float(physical_finalized_duration),
            capture_emulation_speed=RENDER_CAPTURE_EMULATION_SPEED,
            )
        )
        clock_landmarks, content_clock_seal = _seal_landmarks_to_content_audio(
            physical_clock_landmarks,
            content_audio_duration_seconds=float(audio_metadata["duration_seconds"]),
        )
        audio_metadata["content_clock_seal"] = content_clock_seal
        audio_metadata["physical_finalized_clock_duration_seconds"] = float(
            physical_finalized_duration
        )
        audio_metadata["finalized_clock_extension_seconds"] = finalized_buffer_extension
        audio_metadata["finalized_buffer_extension_warning"] = finalized_buffer_extension > 0.050
        _validate_finalized_audio_durations(
            audio_metadata,
            expected_game_frames=frame_count,
            audio_presentation_delay_seconds=cast(
                float,
                settings["audio_presentation_delay_seconds"],
            ),
            recorder_clock_landmarks=cast(
                list[dict[str, float]], recorder_metadata["clock_landmarks"]
            ),
        )
        clock_landmarks_path = workspace / "clock-landmarks.json"
        clock_landmarks_path.write_text(
            json.dumps({"clock_landmarks": clock_landmarks}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        mux_command = [
            str(muxer),
            str(raw_video_path),
            str(mixed_audio_path),
            str(output_path),
            f"{video_alignment_offset:.9f}",
            f"{settings['audio_presentation_delay_seconds']:.9f}",
            player_labels[0],
            player_labels[1],
            str(clock_landmarks_path),
        ]
        muxed = subprocess.run(
            mux_command,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(120.0, game_seconds * 2.0),
        )
        if muxed.returncode != 0:
            raise RuntimeError(
                "could not mux isolated-window video with Dolphin audio: "
                f"{muxed.stderr.strip() or muxed.stdout.strip()}"
            )
        try:
            mux_metadata = json.loads(muxed.stdout.splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as error:
            raise RuntimeError(f"macOS replay muxer returned invalid metadata: {muxed.stdout!r}") from error
        if not isinstance(mux_metadata, dict):
            raise TypeError("macOS replay muxer metadata must be an object")
        if mux_metadata.get("has_video") is not True or mux_metadata.get("has_audio") is not True:
            raise RuntimeError(f"final MP4 is missing video or audio: {mux_metadata}")
        _validate_mux_output_frame_rate(mux_metadata)
        if mux_metadata.get("labels_burned_in") is not True:
            raise RuntimeError(f"final MP4 is missing burned-in player labels: {mux_metadata}")
        if mux_metadata.get("player_labels") != list(player_labels):
            raise RuntimeError(
                f"Slippi replay muxer did not preserve the requested player labels: {mux_metadata}"
            )
        if mux_metadata.get("timing_method") != "dolphin-audio-clock-piecewise":
            raise RuntimeError(f"Slippi replay muxer did not use piecewise timing: {mux_metadata}")
        if mux_metadata.get("clock_landmark_count") != len(clock_landmarks):
            raise RuntimeError(f"Slippi replay muxer did not consume every timing landmark: {mux_metadata}")
        if mux_metadata.get("piecewise_segment_count") != len(clock_landmarks) - 1:
            raise RuntimeError(f"Slippi replay muxer did not retime every timing segment: {mux_metadata}")
        _validate_mux_content_audit(
            mux_metadata,
            expected_source_start_seconds=clock_landmarks[0]["source_video_seconds"],
            expected_source_end_seconds=clock_landmarks[-1]["source_video_seconds"],
            expected_width=capture_width,
            expected_height=capture_height,
        )
        _validate_mux_output_content_audit(
            mux_metadata,
            expected_gameplay_end_seconds=clock_landmarks[-1]["audio_seconds"],
            expected_internal_join_count=max(0, len(clock_landmarks) - 2),
            expected_width=capture_width,
            expected_height=capture_height,
            expected_clock_landmarks=clock_landmarks,
        )
        if mux_metadata.get("audio_fully_preserved") is not True:
            raise RuntimeError(f"Slippi replay muxer truncated Dolphin audio: {mux_metadata}")
        mux_audio_duration = mux_metadata.get("audio_inserted_duration_seconds")
        if (
            isinstance(mux_audio_duration, bool)
            or not isinstance(mux_audio_duration, (int, float))
            or not math.isclose(
                float(mux_audio_duration),
                clock_landmarks[-1]["audio_seconds"],
                abs_tol=2.0 / 48_000,
            )
        ):
            raise RuntimeError(f"Slippi replay muxer did not insert the complete audio clock: {mux_metadata}")
        retimed_game_duration = mux_metadata.get("retimed_game_duration_seconds")
        if (
            isinstance(retimed_game_duration, bool)
            or not isinstance(retimed_game_duration, (int, float))
            or not math.isclose(
                float(retimed_game_duration),
                clock_landmarks[-1]["audio_seconds"],
                abs_tol=1e-9,
            )
        ):
            raise RuntimeError(
                f"Slippi replay muxer reported the wrong retimed game duration: {mux_metadata}"
            )
        raw_visual_tail = mux_metadata.get("raw_visual_tail_seconds")
        if (
            isinstance(raw_visual_tail, bool)
            or not isinstance(raw_visual_tail, (int, float))
            or float(raw_visual_tail) + 1.0 / cast(float, settings["frame_rate"])
            < cast(float, settings["tail_padding_seconds"])
        ):
            raise RuntimeError(
                f"Slippi replay muxer did not preserve the terminal visual tail: {mux_metadata}"
            )
        mux_output_duration = mux_metadata.get("duration_seconds")
        if (
            isinstance(mux_output_duration, bool)
            or not isinstance(mux_output_duration, (int, float))
            or not math.isclose(
                float(mux_output_duration),
                float(retimed_game_duration) + float(raw_visual_tail),
                abs_tol=2.0 / cast(float, settings["frame_rate"]) + 0.001,
            )
        ):
            raise RuntimeError(f"Slippi replay muxer output duration violates its timeline: {mux_metadata}")
        mux_delay = mux_metadata.get("audio_presentation_delay_seconds")
        if (
            not isinstance(mux_delay, (int, float))
            or isinstance(mux_delay, bool)
            or not math.isclose(
                float(mux_delay),
                cast(float, settings["audio_presentation_delay_seconds"]),
                abs_tol=1e-6,
            )
        ):
            raise RuntimeError(
                "Slippi replay muxer did not preserve the configured audio presentation delay: "
                f"{mux_metadata}"
            )
        mux_audio_end = mux_metadata.get("audio_end_seconds")
        expected_mux_audio_end = (
            clock_landmarks[-1]["audio_seconds"] + float(mux_delay)
        )
        if (
            isinstance(mux_audio_end, bool)
            or not isinstance(mux_audio_end, (int, float))
            or not math.isclose(
                float(mux_audio_end),
                expected_mux_audio_end,
                abs_tol=2.0 / 48_000,
            )
        ):
            raise RuntimeError("Slippi replay muxer reported the wrong normalized audio endpoint")
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError("Slippi replay muxer produced no final MP4")
        return {
            "method": "postgame-slippi-playback-isolated-window",
            **_render_speed_metadata(),
            "desktop_captured": False,
            "cursor_captured": False,
            "microphone_captured": False,
            "audio_scope": "Dolphin replay DSP and DTK dumps only",
            "playback_application": str(application),
            "source_replay": str(replay_path.resolve()),
            "frame_rate": settings["frame_rate"],
            "source_first_frame": source_first_frame,
            "capture_first_frame": capture_start_frame,
            "first_frame": capture_start_frame,
            "last_frame": capture_end_frame,
            "frame_count": frame_count,
            "game_seconds": game_seconds,
            "tail_padding_seconds": settings["tail_padding_seconds"],
            "video_alignment": "Dolphin audio sample barrier with piecewise audio-clock retiming",
            "audio_presentation_delay_seconds": settings["audio_presentation_delay_seconds"],
            "dolphin_shutdown_method": shutdown_method,
            "labels": {
                "burned_in": True,
                "p1": player_labels[0],
                "p2": player_labels[1],
            },
            "recorder_source_sha256": recorder_source_sha256,
            "muxer_source_sha256": muxer_source_sha256,
            "audio": audio_metadata,
            "native": {
                "recording": recorder_metadata,
                "mux": mux_metadata,
            },
        }


__all__ = ["DEFAULT_PLAYBACK_APPLICATION", "render_replay_video"]
