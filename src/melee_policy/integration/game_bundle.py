"""Atomic postgame export of one replay and its rendered video."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GAME_BUNDLE_SCHEMA_VERSION = "integration.game_bundle.v1"


@dataclass(frozen=True, slots=True)
class VideoRenderContext:
    """Inputs that bind a video render to the completed policy game."""

    project_root: Path
    config: Mapping[str, Any]
    iso_path: Path
    first_frame: int
    last_frame: int
    players: tuple[dict[str, Any], dict[str, Any]]


VideoRenderer = Callable[[Path, Path, VideoRenderContext], Mapping[str, Any]]


def export_artifact_label(
    artifact_label: str | None,
    *,
    save_slp: bool,
    save_video: bool,
) -> str | None:
    """Allocate a unique outer directory when a postgame export was requested."""
    if artifact_label is not None or not (save_slp or save_video):
        return artifact_label
    timestamp = datetime.now(UTC).strftime("%Y%m%dt%H%M%S")
    return f"game-{timestamp}-{uuid.uuid4().hex[:8]}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _file_identity(path: Path, project_root: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise FileNotFoundError(f"game export must not be a symlink: {path}")
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"game export is missing: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        before = os.fstat(stream.fileno())
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise RuntimeError(f"game export changed while hashing: {resolved}")
    current = resolved.stat()
    if any(getattr(current, field) != getattr(after, field) for field in stable_fields):
        raise RuntimeError(f"game export path changed while hashing: {resolved}")
    return {
        "path": _display_path(resolved, project_root),
        "sha256": digest.hexdigest(),
        "byte_length": after.st_size,
    }


def _stable_json_file(
    path: Path,
    project_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"game summary is missing or invalid: {path}")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        payload = stream.read()
        after = os.fstat(stream.fileno())
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise RuntimeError(f"game summary changed while being read: {path}")
    current = path.stat()
    if any(getattr(current, field) != getattr(after, field) for field in stable_fields):
        raise RuntimeError(f"game summary path changed while being read: {path}")
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"game summary is invalid JSON: {path}") from error
    summary = _mapping(value, "on-disk game summary")
    return summary, {
        "path": _display_path(path, project_root),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "byte_length": len(payload),
    }


def _resolve_record_path(value: object, project_root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise TypeError(f"artifact path must be a nonempty string, got {value!r}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return {str(key): item for key, item in value.items()}


def _artifact_root(summary: Mapping[str, Any], project_root: Path) -> Path:
    artifacts = _mapping(summary.get("artifacts"), "summary.artifacts")
    summary_value = artifacts.get("summary")
    if isinstance(summary_value, str) and summary_value:
        summary_path = _resolve_record_path(summary_value, project_root)
    else:
        trace = _mapping(artifacts.get("trace"), "summary.artifacts.trace")
        summary_path = _resolve_record_path(trace.get("path"), project_root).parent / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"completed-game summary is missing: {summary_path}")
    return summary_path.parent.resolve()


def _selected_replay(
    summary: Mapping[str, Any],
    project_root: Path,
    artifact_root: Path,
) -> tuple[Path, dict[str, Any]]:
    artifacts = _mapping(summary.get("artifacts"), "summary.artifacts")
    replay_values = artifacts.get("replays")
    if not isinstance(replay_values, Sequence) or isinstance(replay_values, (str, bytes)):
        raise TypeError("summary.artifacts.replays must be a list")
    records = [
        _mapping(value, f"summary.artifacts.replays[{index}]")
        for index, value in enumerate(replay_values)
    ]
    selected = [record for record in records if record.get("tournament_result_replay") is True]
    if len(selected) != 1:
        raise RuntimeError(
            "game export requires exactly one replay marked tournament_result_replay; "
            f"found {len(selected)}"
        )
    record = selected[0]
    replay_path = _resolve_record_path(record.get("path"), project_root)
    if not replay_path.is_file():
        raise FileNotFoundError(f"selected scoring replay is missing: {replay_path}")
    if not replay_path.is_relative_to(artifact_root):
        raise ValueError(
            f"selected scoring replay escapes its completed-game artifact directory: {replay_path}"
        )
    observed_size = replay_path.stat().st_size
    expected_size = record.get("byte_length")
    if expected_size is not None and expected_size != observed_size:
        raise RuntimeError(
            f"selected replay byte length changed: {observed_size} != {expected_size}"
        )
    observed_sha256 = _sha256_file(replay_path)
    expected_sha256 = record.get("sha256")
    if expected_sha256 is not None and expected_sha256 != observed_sha256:
        raise RuntimeError(
            f"selected replay SHA-256 changed: {observed_sha256} != {expected_sha256}"
        )
    return replay_path, record


def _replay_frame_range(record: Mapping[str, Any]) -> tuple[int, int]:
    validation = _mapping(record.get("validation"), "selected replay validation")
    first = validation.get("first_parsed_frame")
    last = validation.get("last_parsed_frame")
    if isinstance(first, bool) or not isinstance(first, int):
        raise TypeError("selected replay first_parsed_frame must be an integer")
    if isinstance(last, bool) or not isinstance(last, int):
        raise TypeError("selected replay last_parsed_frame must be an integer")
    if last < first:
        raise ValueError(f"selected replay frame range is reversed: {first} to {last}")
    return first, last


def _player_record(summary: Mapping[str, Any], port: int) -> dict[str, Any]:
    configuration = summary.get("configuration")
    if isinstance(configuration, Mapping):
        configured = configuration.get(f"player_{port}")
        if isinstance(configured, Mapping):
            return {
                "port": port,
                "model": configured.get("model"),
                "character": configured.get("character"),
            }
    slots = summary.get("slots")
    if isinstance(slots, Mapping):
        slot = slots.get(f"p{port}")
        if isinstance(slot, Mapping):
            return {
                "port": port,
                "model": slot.get("model"),
                "character": slot.get("character", slot.get("requested_character")),
            }
    return {"port": port, "model": None, "character": None}


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _default_video_renderer(
    replay_path: Path,
    output_path: Path,
    context: VideoRenderContext,
) -> Mapping[str, Any]:
    from melee_policy.integration.replay_video import render_replay_video

    return render_replay_video(replay_path, output_path, context)


def finalize_game_bundle(
    summary: Mapping[str, Any],
    *,
    project_root: Path,
    config: Mapping[str, Any],
    iso_path: Path,
    save_slp: bool,
    save_video: bool,
    video_renderer: VideoRenderer | None = None,
    gameplay_provenance: Mapping[str, Any] | None = None,
    preserve_source_summary: bool = False,
) -> dict[str, Any]:
    """Publish ``game/`` only after every requested output is complete."""
    if not isinstance(save_slp, bool) or not isinstance(save_video, bool):
        raise TypeError("save_slp and save_video must be booleans")
    if not isinstance(preserve_source_summary, bool):
        raise TypeError("preserve_source_summary must be a boolean")
    if preserve_source_summary and gameplay_provenance is None:
        raise ValueError(
            "preserving a source summary requires explicit gameplay provenance"
        )
    provenance = (
        None
        if gameplay_provenance is None
        else copy.deepcopy(_mapping(gameplay_provenance, "gameplay_provenance"))
    )
    if not save_slp and not save_video:
        return copy.deepcopy(dict(summary))

    project_root = project_root.resolve()
    artifact_root = _artifact_root(summary, project_root)
    if not artifact_root.is_relative_to(project_root):
        raise ValueError(f"completed-game artifact directory escapes the project: {artifact_root}")
    final_directory = artifact_root / "game"
    if final_directory.exists():
        raise FileExistsError(
            f"game export already exists and will not be overwritten: {final_directory}"
        )
    preserved_summary_identity: dict[str, Any] | None = None
    if preserve_source_summary:
        summary_identity = _mapping(
            provenance.get("summary") if provenance is not None else None,
            "gameplay_provenance.summary",
        )
        bound_summary, preserved_summary_identity = _stable_json_file(
            artifact_root / "summary.json", project_root
        )
        if preserved_summary_identity != summary_identity:
            raise RuntimeError("source summary differs from its gameplay provenance")
        if bound_summary != _mapping(summary, "caller-supplied game summary"):
            raise RuntimeError("caller-supplied summary differs from the bound source summary")

    replay_path, replay_record = _selected_replay(summary, project_root, artifact_root)
    first_frame, last_frame = _replay_frame_range(replay_record)
    players = (_player_record(summary, 1), _player_record(summary, 2))
    renderer = _default_video_renderer if video_renderer is None else video_renderer
    staging_directory = Path(
        tempfile.mkdtemp(prefix=".game-staging-", dir=artifact_root)
    ).resolve()
    published = False
    try:
        slp_path = staging_directory / "game.slp"
        video_path = staging_directory / "game.mp4"
        render_metadata: dict[str, Any] | None = None
        if save_slp:
            shutil.copy2(replay_path, slp_path)
            if _sha256_file(slp_path) != _sha256_file(replay_path):
                raise RuntimeError("copied game.slp does not match the selected scoring replay")
        if save_video:
            context = VideoRenderContext(
                project_root=project_root,
                config=config,
                iso_path=iso_path.resolve(),
                first_frame=first_frame,
                last_frame=last_frame,
                players=players,
            )
            render_metadata = dict(renderer(replay_path, video_path, context))
            if not video_path.is_file() or video_path.stat().st_size == 0:
                raise RuntimeError("video renderer did not produce a nonempty game.mp4")

        final_slp_path = final_directory / "game.slp"
        final_video_path = final_directory / "game.mp4"
        manifest: dict[str, Any] = {
            "schema_version": GAME_BUNDLE_SCHEMA_VERSION,
            "save_slp": save_slp,
            "save_video": save_video,
            "players": list(players),
            "source_replay": _file_identity(replay_path, project_root),
            "frame_range": {
                "first": first_frame,
                "last": last_frame,
                "count": last_frame - first_frame + 1,
            },
            "outputs": {
                "slp": (
                    {
                        "path": _display_path(final_slp_path, project_root),
                        "sha256": _sha256_file(slp_path),
                        "byte_length": slp_path.stat().st_size,
                    }
                    if save_slp
                    else None
                ),
                "video": (
                    {
                        "path": _display_path(final_video_path, project_root),
                        "sha256": _sha256_file(video_path),
                        "byte_length": video_path.stat().st_size,
                        "render": render_metadata,
                    }
                    if save_video
                    else None
                ),
            },
        }
        if provenance is not None:
            manifest["gameplay_provenance"] = provenance
        manifest_path = staging_directory / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if preserve_source_summary and _file_identity(
            artifact_root / "summary.json", project_root
        ) != preserved_summary_identity:
            raise RuntimeError("source summary changed during game bundle publication")
        staging_directory.replace(final_directory)
        published = True

        updated = copy.deepcopy(dict(summary))
        if not preserve_source_summary:
            artifacts = _mapping(updated.get("artifacts"), "summary.artifacts")
            artifacts["game"] = {
                "path": _display_path(final_directory, project_root),
                "manifest": _file_identity(
                    final_directory / "manifest.json", project_root
                ),
                "slp": _file_identity(final_slp_path, project_root) if save_slp else None,
                "video": _file_identity(final_video_path, project_root) if save_video else None,
            }
            updated["artifacts"] = artifacts
            _atomic_write_json(artifact_root / "summary.json", updated)
        return updated
    except BaseException:
        if published:
            shutil.rmtree(final_directory)
        raise
    finally:
        if staging_directory.exists():
            shutil.rmtree(staging_directory)


__all__ = [
    "GAME_BUNDLE_SCHEMA_VERSION",
    "VideoRenderContext",
    "export_artifact_label",
    "finalize_game_bundle",
]
