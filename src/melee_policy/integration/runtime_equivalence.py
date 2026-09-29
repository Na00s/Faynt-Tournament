"""Audit policy-visible libmelee input equivalence across tournament runtimes."""

from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from melee_policy.integration.match_runtime import (
    _activate_sources,
    _load_config,
    _sha256_file,
)

SCHEMA_VERSION = "integration.runtime_equivalence.v2"
REPLAY_SHA256 = "09a0d7d468df588b3ca0d1a6e763c4c31385e3a0564dba41998cb24a7a6ed498"
REPLAY_BYTE_LENGTH = 4644162
FRAME_COUNT = 1000
EXPECTED_DIGESTS = {
    "mimic_port_1": "76e7d18c24701ff2e91750e46c15fe24ac3ca0a51eefae258d90b4caadbbe5b1",
    "mimic_port_2": "cff4b514814fc91a097b92d833be2b15da24b5aa25a4e4e2e357ad9d3d1dbe8f",
}
_PREVIOUS_COMMAND = {
    "main_x": 0.25,
    "main_y": 0.75,
    "c_x": 1.0,
    "c_y": 0.0,
    "l_shldr": 0.4,
    "r_shldr": 0.0,
    "btn_BUTTON_A": 1,
    "btn_BUTTON_Z": 1,
}


def _update_digest(digest: Any, tensors: Mapping[str, Any]) -> None:
    for key in sorted(tensors.keys()):
        value = tensors[key].detach().cpu().contiguous()
        digest.update(key.encode() + b"\x00")
        digest.update(str(value.dtype).encode() + b"\x00")
        digest.update(str(tuple(value.shape)).encode() + b"\x00")
        digest.update(value.numpy().tobytes())


def evaluate(config_path: Path, expected_melee_version: str) -> dict[str, Any]:
    """Hash the same 1,000 policy-visible frames in one isolated runtime."""
    config, project_root = _load_config(config_path)
    source_checks = _activate_sources(config, project_root)
    import melee
    from tools.inference_utils import (
        build_frame,
        build_frame_p2,
        load_inference_context,
    )

    replay_path = project_root / ".e000-cache" / "raw" / f"{REPLAY_SHA256}.slp"
    replay_identity = {
        "path": str(replay_path.relative_to(project_root)),
        "sha256": _sha256_file(replay_path),
        "byte_length": replay_path.stat().st_size,
    }
    context = load_inference_context(project_root / config["mimic"]["asset_directory"])
    digests = {"mimic_port_1": hashlib.sha256(), "mimic_port_2": hashlib.sha256()}
    console = melee.Console(
        path=str(replay_path), is_dolphin=False, allow_old_version=True
    )
    if not console.connect():
        raise RuntimeError("libmelee could not open the runtime-equivalence replay")
    processed_frames = 0
    first_frame: int | None = None
    last_frame: int | None = None
    try:
        while processed_frames < FRAME_COUNT:
            gamestate = console.step()
            if gamestate is None:
                raise RuntimeError(
                    "runtime-equivalence replay ended before the required frame count"
                )
            if len(gamestate.players) != 2 or gamestate.stage == melee.Stage.NO_STAGE:
                continue
            mimic_port_1 = build_frame(gamestate, _PREVIOUS_COMMAND, context)
            mimic_port_2 = build_frame_p2(gamestate, _PREVIOUS_COMMAND, context)
            if mimic_port_1 is None or mimic_port_2 is None:
                raise RuntimeError("MIMIC returned no frame for a valid gameplay state")
            _update_digest(digests["mimic_port_1"], mimic_port_1)
            _update_digest(digests["mimic_port_2"], mimic_port_2)
            frame = int(gamestate.frame)
            first_frame = frame if first_frame is None else first_frame
            last_frame = frame
            processed_frames += 1
    finally:
        console.stop()
    observed_digests = {name: digest.hexdigest() for name, digest in digests.items()}
    melee_version = importlib.metadata.version("melee")
    checks = {
        "replay_sha256_exact": replay_identity["sha256"] == REPLAY_SHA256,
        "replay_byte_length_exact": replay_identity["byte_length"]
        == REPLAY_BYTE_LENGTH,
        "frame_count_exact": processed_frames == FRAME_COUNT,
        "melee_version_exact": melee_version == expected_melee_version,
        "policy_visible_digests_exact": observed_digests == EXPECTED_DIGESTS,
        "pinned_sources_clean": all(
            (
                identity.get("tracked_tree_clean") is True
                for identity in source_checks["repositories"].values()
            )
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "policy-visible parser and input-builder equivalence canary; this does not claim floating-point inference or controller-transport equivalence",
        "decision": "pass" if not failures else "fail",
        "checks": checks,
        "failures": failures,
        "python_executable": str(
            Path(sys.executable).absolute().relative_to(project_root)
        ),
        "melee_version": melee_version,
        "replay": replay_identity,
        "processed_frames": processed_frames,
        "first_frame": first_frame,
        "last_frame": last_frame,
        "digests": observed_digests,
        "source_revisions": source_checks["revisions"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/integration.toml"))
    parser.add_argument("--expected-melee-version", required=True)
    arguments = parser.parse_args()
    print(
        "RUNTIME_EQUIVALENCE_JSON="
        + json.dumps(
            evaluate(arguments.config, arguments.expected_melee_version),
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
