from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from melee_policy.integration import frisson_policy, play


ROOT = Path(__file__).resolve().parents[2]
SUBDIRECTORY = f"sources/faynt/{frisson_policy.FRISSON_SOURCE_REVISION}"


def test_bundled_source_materializes_without_the_original_git_repository(
    tmp_path, monkeypatch
):
    def reject_git(*args):
        raise AssertionError("the authenticated source bundle must be self-contained")

    monkeypatch.setattr(frisson_policy, "_git_output", reject_git)
    target = frisson_policy.materialize_pinned_model_source(
        repository=ROOT,
        revision=frisson_policy.FRISSON_SOURCE_REVISION,
        source_subdirectory=SUBDIRECTORY,
        destination=tmp_path / "model-source",
    )
    for filename, digest in frisson_policy._pinned_model_source_files().values():
        assert frisson_policy._sha256_file(target / filename) == digest


@pytest.mark.parametrize("tamper", ["source", "manifest"])
def test_bundled_source_rejects_tampering(tmp_path, tamper):
    destination = tmp_path / SUBDIRECTORY
    shutil.copytree(ROOT / SUBDIRECTORY, destination)
    if tamper == "source":
        with (destination / "model.py").open("ab") as stream:
            stream.write(b"\n# changed source\n")
    else:
        manifest = destination / "source-manifest.json"
        value = json.loads(manifest.read_text())
        value["runtime_source_revision"] = "0" * 40
        manifest.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="bundled Faynt source"):
        frisson_policy._verified_pinned_model_blobs(
            tmp_path, frisson_policy.FRISSON_SOURCE_REVISION, SUBDIRECTORY
        )


def test_public_faynt_alias_uses_the_existing_runtime():
    args = play._build_parser().parse_args(
        [
            "--p1",
            "faynt",
            "--p2",
            "mimic",
            "--p1-checkpoint",
            "/path/to/Faynt-10M-Base/checkpoint.pt",
            "--stage",
            "FINAL_DESTINATION",
            "--seed",
            "0",
        ]
    )
    assert play._normalize_model(args.player_1) == "frisson-ai"
