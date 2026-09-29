from __future__ import annotations

import copy
import dataclasses
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import melee_policy.integration.slippi_compatibility as compatibility_module
from melee_policy.integration.slippi_ai_policy import (
    MEDIUM_V2_CHECKPOINT_BYTES,
    MEDIUM_V2_CHECKPOINT_SHA256,
    SLIPPI_AI_SOURCE_REVISION,
    CanonicalControllerCommand,
)
from melee_policy.integration.slippi_compatibility import (
    CHECKPOINT_MANIFEST_SCHEMA,
    COMPATIBILITY_SCHEMA,
    E010CompatibilityConfig,
    execute_single_run,
    validate_artifacts,
    write_artifacts,
)


class _FakeConsole:
    def __init__(self, states: list[Any]) -> None:
        self._states = iter(states)
        self.stopped = False

    def connect(self) -> bool:
        return True

    def step(self) -> Any | None:
        return next(self._states, None)

    def stop(self) -> None:
        self.stopped = True


class _FakeSession:
    def __init__(self) -> None:
        self.started = False
        self.closed = False
        self.frames: list[int] = []

    def start(self) -> None:
        self.started = True

    def step(self, gamestate: Any) -> CanonicalControllerCommand:
        assert self.started
        self.frames.append(int(gamestate.frame))
        if len(self.frames) <= 19:
            return CanonicalControllerCommand(
                main_stick=(0.0, 0.0),
                c_stick=(0.0, 0.0),
                analog_l=0.0,
                analog_r=0.0,
            )
        return CanonicalControllerCommand(
            main_stick=(17 / 32, 16 / 32),
            c_stick=(0.0, 32 / 32),
            analog_l=3 / 10,
            analog_r=0.0,
            buttons=("A", "R"),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": "melee_policy.slippi_ai_policy.metadata.v1",
            "runtime_contract": {
                "sample_temperature": 1.0,
                "effective_policy_delay_frames": 19,
            },
            "upstream_runtime": {
                "parser_class": "slippi_db.parse_libmelee.Parser",
                "observation_filter_class": "slippi_ai.observations.ChainObservationFilter",
                "controller_head_class": "slippi_ai.tf.controller_heads.AutoregressiveHead",
                "platform": "tf",
                "recurrent_state_class": "LSTMState",
                "checkpoint_config": {
                    "network": {
                        "name": "tx_like",
                        "tx_like": {"recurrent_layer": "lstm"},
                    }
                },
                "policy_delay_frames": 21,
                "effective_policy_delay_frames": 19,
                "variable_count": 141,
                "parameter_count": 23_887_032,
                "state_assignment": {
                    "shape_mismatches": [],
                    "dtype_mismatches": [],
                    "value_mismatches": [],
                    "nonfinite_variables": [],
                },
            },
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            "schema_version": "melee_policy.slippi_ai_policy.diagnostics.v1",
            "frames_total": len(self.frames),
            "capture_decoder_assertions": len(self.frames),
            "capture_decoder_mismatches": 0,
        }

    def close(self) -> None:
        self.closed = True


def _enum(name: str, value: int) -> Any:
    return SimpleNamespace(name=name, value=value)


def _states(count: int) -> list[Any]:
    return [
        SimpleNamespace(
            frame=-123 + index,
            stage=_enum("BATTLEFIELD", 31),
            players={
                1: SimpleNamespace(character=_enum("FOX", 1)),
                2: SimpleNamespace(character=_enum("YOSHI", 14)),
            },
        )
        for index in range(count)
    ]


def _asset_identity(config: E010CompatibilityConfig) -> dict[str, Any]:
    return {
        "source": {
            "repository_url": config.source_repository_url,
            "revision": config.source_revision,
            "tracked_tree_clean": True,
        },
        "checkpoint": {
            "release": config.checkpoint_name,
            "sha256": config.checkpoint_sha256,
            "byte_length": config.checkpoint_byte_length,
        },
    }


def test_fake_trace_writes_stable_artifact_schema(
    monkeypatch: Any, tmp_path: Path
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    loaded = E010CompatibilityConfig.load(project_root / "configs/e010_slippi.toml")
    replay = tmp_path / "canary.slp"
    replay.write_bytes(b"valid-fake-slp-bytes")
    checkpoint = tmp_path / "medium-v2"
    checkpoint.write_bytes(b"fake-checkpoint")
    replay_sha256 = hashlib.sha256(replay.read_bytes()).hexdigest()
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    e000_manifest_path = tmp_path / "sample_manifest.csv"
    e000_manifest_path.write_bytes(b"fake-manifest-for-injected-verifier\n")
    e000_manifest_identity = {
        "path": "artifacts/e000/sample_manifest.csv",
        "sha256": hashlib.sha256(e000_manifest_path.read_bytes()).hexdigest(),
        "byte_length": e000_manifest_path.stat().st_size,
    }
    config = dataclasses.replace(
        loaded,
        replay_path=replay,
        replay_sha256=replay_sha256,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_byte_length=checkpoint.stat().st_size,
        frames=24,
    )

    consoles: list[_FakeConsole] = []
    sessions: list[_FakeSession] = []

    def console_factory(_path: Path) -> _FakeConsole:
        console = _FakeConsole(_states(config.frames))
        consoles.append(console)
        return console

    def session_factory(_config: Any) -> _FakeSession:
        session = _FakeSession()
        sessions.append(session)
        return session

    result = execute_single_run(
        config,
        console_factory=console_factory,
        session_factory=session_factory,
        asset_verifier=lambda _config: _asset_identity(config),
        tensorflow_setup=lambda seed: {
            "tensorflow": "fake-tensorflow",
            "seed": seed,
            "physical_gpu_count": 0,
            "logical_gpu_count": 0,
        },
        manifest_verifier=lambda _config: {
            **e000_manifest_identity,
            "checks": {
                "manifest_version": True,
                "raw_byte_length": True,
                "libmelee_status": True,
                "peppi_status": True,
                "supported": True,
            },
            "dataset_url": "https://example.invalid/dataset",
            "dataset_revision": "2" * 40,
            "source_archive": 2,
            "source_shard": "fake.tar.gz",
            "source_member": "canary.slp",
            "slippi_version": "3.15.0",
            "stage_id": 31,
            "libmelee_status": "ok",
            "peppi_status": "ok",
            "supported": True,
        },
    )
    assert consoles[0].stopped
    assert sessions[0].closed
    assert result["gate"] == "pass"
    assert result["perspective"]["stage"] == {"name": "BATTLEFIELD", "value": 31}
    assert result["native_dummy_prefix"]["frames"] == 19
    assert result["native_dummy_prefix"]["exact"] is True
    assert all(result["checks"].values())

    first = copy.deepcopy(result)
    second = copy.deepcopy(result)
    first["process_id"] = 1001
    second["process_id"] = 1002
    first["performance"] = {"wall_seconds": 1.25, "maximum_resident_set_raw": 100}
    second["performance"] = {"wall_seconds": 9.75, "maximum_resident_set_raw": 999}
    first["policy_diagnostics"]["step_timing"] = {"total_seconds": 0.25}
    second["policy_diagnostics"]["step_timing"] = {"total_seconds": 5.0}
    first["policy_diagnostics"]["upstream"] = {
        "state_queue_profiler": {"cumulative_seconds": 0.1},
        "step_profiler": {"cumulative_seconds": 0.2},
        "effective_policy_delay_frames": 19,
    }
    second["policy_diagnostics"]["upstream"] = {
        "state_queue_profiler": {"cumulative_seconds": 3.0},
        "step_profiler": {"cumulative_seconds": 4.0},
        "effective_policy_delay_frames": 19,
    }
    monkeypatch.setattr(
        compatibility_module,
        "_source_manifest",
        lambda _config: {
            "repository_url": loaded.source_repository_url,
            "branch": "main",
            "revision": SLIPPI_AI_SOURCE_REVISION,
            "tracked_tree_clean": True,
        },
    )
    monkeypatch.setattr(
        compatibility_module,
        "_verify_e000_manifest",
        lambda _config: result["e000_manifest"],
    )

    monkeypatch.setattr(
        compatibility_module,
        "_git",
        lambda _root, *args: "0" * 40 if args[:1] == ("rev-parse",) else "",
    )
    original_file_record = compatibility_module._project_file_record
    monkeypatch.setattr(
        compatibility_module,
        "_project_file_record",
        lambda root, relative: (
            e000_manifest_identity
            if relative == compatibility_module.E000_SAMPLE_MANIFEST_PATH
            else original_file_record(root, relative)
        ),
    )
    output_a = tmp_path / "artifacts-a"
    output_b = tmp_path / "artifacts-b"
    written = write_artifacts(config, first, second, output_directory=output_a)
    write_artifacts(config, first, second, output_directory=output_b)
    assert written["gate"] == "pass"
    assert (output_a / "checkpoint_manifest.json").read_bytes() == (
        output_b / "checkpoint_manifest.json"
    ).read_bytes()
    assert (output_a / "compatibility.json").read_bytes() == (
        output_b / "compatibility.json"
    ).read_bytes()
    assert str(project_root) not in (output_a / "checkpoint_manifest.json").read_text(
        encoding="utf-8"
    )
    assert str(project_root) not in (output_a / "compatibility.json").read_text(
        encoding="utf-8"
    )

    manifest = compatibility_module._read_json(output_a / "checkpoint_manifest.json")
    compatibility = compatibility_module._read_json(output_a / "compatibility.json")
    assert manifest["schema_version"] == CHECKPOINT_MANIFEST_SCHEMA
    assert {
        record["path"] for record in manifest["project"]["implementation_files"]
    } == {
        "src/melee_policy/integration/slippi_ai_policy.py",
        "src/melee_policy/integration/slippi_compatibility.py",
        "src/melee_policy/integration/slippi_match.py",
    }
    assert all(
        len(record["sha256"]) == 64 and record["byte_length"] > 0
        for record in manifest["project"]["implementation_files"]
    )
    assert {
        key: manifest["e000_manifest"][key] for key in ("path", "sha256", "byte_length")
    } == e000_manifest_identity
    assert compatibility["schema_version"] == COMPATIBILITY_SCHEMA
    assert compatibility["reload_checks"] == {
        "asset_identity_equal": True,
        "both_single_process_gates_pass": True,
        "complete_command_trace_equal": True,
        "distinct_processes": True,
        "replay_identity_equal": True,
        "trace_sha256_equal": True,
    }
    assert "performance" not in compatibility["verified_trace"]
    assert "step_timing" not in compatibility["verified_trace"]["policy_diagnostics"]
    assert compatibility["verified_trace"]["policy_diagnostics"]["upstream"] == {
        "effective_policy_delay_frames": 19
    }
    validation = validate_artifacts(config, output_directory=output_a)
    assert validation["checks"]["gate"] is True
    assert validation["checks"]["e000_sample_manifest_identity"] is True
    assert validation["checks"]["implementation_slippi_ai_policy_identity"] is True
    assert validation["checks"]["implementation_slippi_compatibility_identity"] is True
    assert validation["checks"]["implementation_slippi_match_identity"] is True

    manifest["project"]["implementation_files"][0]["sha256"] = "0" * 64
    compatibility_module._write_json(output_a / "checkpoint_manifest.json", manifest)
    with pytest.raises(RuntimeError, match="implementation_slippi_ai_policy_identity"):
        validate_artifacts(config, output_directory=output_a)

    manifest_b = compatibility_module._read_json(output_b / "checkpoint_manifest.json")
    manifest_b["e000_manifest"]["sha256"] = "0" * 64
    compatibility_module._write_json(output_b / "checkpoint_manifest.json", manifest_b)
    with pytest.raises(RuntimeError, match="e000_sample_manifest_identity"):
        validate_artifacts(config, output_directory=output_b)


def test_frozen_config_keeps_exact_public_identities() -> None:
    project_root = Path(__file__).resolve().parents[2]
    config = E010CompatibilityConfig.load(project_root / "configs/e010_slippi.toml")
    assert config.source_revision == SLIPPI_AI_SOURCE_REVISION
    assert config.checkpoint_sha256 == MEDIUM_V2_CHECKPOINT_SHA256
    assert config.checkpoint_byte_length == MEDIUM_V2_CHECKPOINT_BYTES
    assert config.effective_output_delay_frames == 19
    assert config.frames == 64
