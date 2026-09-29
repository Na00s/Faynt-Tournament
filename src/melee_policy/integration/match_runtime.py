"""Shared match transport, provenance checks, and native MIMIC runtime."""

from __future__ import annotations
import argparse
import configparser
import copy
import ctypes
import ctypes.util
import errno
import hashlib
import importlib.util
import json
import os
import platform
import random
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import types
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TextIO, cast
import numpy as np
import torch
from tensordict import TensorDict
from melee_policy.integration.frame_watchdog import InGameNoFrameWatchdog
from melee_policy.integration.mimic_bundle_manifest import (
    MIMIC_BUNDLE_ASSET_NAMES,
    MIMIC_NATIVE_BUNDLES,
)
from melee_policy.integration.natural_game_end import has_decisive_zero_stock
from melee_policy.integration.runtime_identity import runtime_environment_record
from melee_policy.integration.state_identity import state_dictionary_sha256

MIMIC_SOURCE_REVISION = "70c925b7675202d853472c4e54332790ad76efe7"
MIMIC_SOURCE_REPOSITORY = "https://github.com/erickfm/MIMIC.git"
MIMIC_SEQUENCE_LENGTH = 180
FIRST_GAMEPLAY_FRAME = -123
EXPECTED_DOLPHIN_VERSION = "3.6.4"
EXPECTED_DOLPHIN_SOURCE_REVISION = "e7711b104b339a99385f2bb12b472d46140a7bc7"
EXPECTED_DOLPHIN_FRAME_SYNC_PATCH_SHA256 = (
    "bb0e8885b33e6f3bb5a43e5ef936fce4a0b79459cde926a97d02aed9a7fd388e"
)
EMULATOR_RELEASE_ATTESTATION_SCHEMA_VERSION = (
    "integration.emulator_release_attestation.v1"
)
ATTESTED_DOLPHIN_LAUNCH_SCHEMA_VERSION = "integration.attested_dolphin_launch.v1"
MACOS_PERSISTENCE_IGNORE_ARGUMENTS = ("-ApplePersistenceIgnoreState", "YES")
DOLPHIN_HOTKEY_SECTION = "Hotkeys1"
DOLPHIN_STOP_HOTKEY = "General/Stop/Hide OSD chat"
MIMIC_DECODE_STRATEGY = "categorical-sampling"
INTEGRATION_SCHEMA_VERSION = "integration.match_runtime.v1"
SAME_FAMILY_CONTROLLER_REPLAY_LAG_FRAMES = {"mimic": 1}
_LIBMELEE_VERSION_ATTESTATION_LOCK = threading.Lock()
CHARACTER_LABELS = (
    "MARIO",
    "FOX",
    "CPTFALCON",
    "DK",
    "KIRBY",
    "BOWSER",
    "LINK",
    "SHEIK",
    "NESS",
    "PEACH",
    "POPO",
    "NANA",
    "PIKACHU",
    "SAMUS",
    "YOSHI",
    "JIGGLYPUFF",
    "MEWTWO",
    "LUIGI",
    "MARTH",
    "ZELDA",
    "YLINK",
    "DOC",
    "FALCO",
    "PICHU",
    "GAMEANDWATCH",
    "GANONDORF",
    "ROY",
)
LAUNCHABLE_CHARACTERS = tuple(
    (character for character in CHARACTER_LABELS if character != "NANA")
)
LEGAL_STAGES = (
    "FINAL_DESTINATION",
    "BATTLEFIELD",
    "POKEMON_STADIUM",
    "DREAMLAND",
    "FOUNTAIN_OF_DREAMS",
    "YOSHIS_STORY",
)
HEAD_ORDER = ("shoulder", "c_stick", "main_stick", "buttons")
POLICY_TORCH_RNG_DOMAIN = "melee-policy.integration.policy-torch-rng.v1"
POLICY_TORCH_RNG_IDENTITIES = frozenset(("mimic",))
_POLICY_TORCH_RNG_LOCK = threading.RLock()


class _CaseSensitiveRawConfigParser(configparser.RawConfigParser):
    def optionxform(self, optionstr: str) -> str:
        return optionstr


MIMIC_ASSET_NAMES = MIMIC_BUNDLE_ASSET_NAMES
MIMIC_KNOWN_BUNDLES: dict[str, dict[str, Any]] = {
    "d2a317763bf21038ef70f369d4047506800fdeeea45a57a1b5fec866065c2f0c": {
        "name": "fox-master",
        "character": "FOX",
        "run_name": "fox-mastfox-20260625",
        "repository": "https://huggingface.co/erickfm/MIMIC",
        "revision": "e9a805cefc590f3f06eb1d520fc6a33ea9c66c07",
        "assets": {
            "model.pt": (
                "d2a317763bf21038ef70f369d4047506800fdeeea45a57a1b5fec866065c2f0c",
                345241127,
            ),
            "config.json": (
                "214c18de0e047249bfe1f255d3c741b665552df901b46338f6c98d6db960ca65",
                1146,
            ),
            "metadata.json": (
                "5643f3ee85b9d067d2c23fb8a86cd8b2308d4a6fa07ecb3723330baa8e488125",
                372,
            ),
            "mimic_norm.json": (
                "087dffcd816be8e5a40c1bebf63a016feafe9ddd831bb22f7d16019f52584e70",
                2038,
            ),
            "controller_combos.json": (
                "939f65efb7823836ca2d38df2ffb39aafd0018eea2066f399424c2e4ba75a01b",
                129,
            ),
            "cat_maps.json": (
                "fa696f39c3a4b6e0bccbfe2296a86962ffd8bef8e9c35ec4bb9e179637384648",
                2132,
            ),
            "stick_clusters.json": (
                "f1dc69b7620189674064ca8661ac0fe50b79f7ad3db60f73892cc3f7ba9ad36c",
                1424,
            ),
            "norm_stats.json": (
                "31022db33d94ceabd18e0ab07d16b05426fb067058be372b0a4e0142be94dd12",
                8021,
            ),
        },
    },
    "be15082845c3e037bd6332f8a2bbd41468511789e63ebb0f299aeef73b4fd82a": {
        "name": "marth-20260420-baseline",
        "character": "MARTH",
        "run_name": "marth-20260420-baseline",
        "repository": "https://huggingface.co/erickfm/MIMIC",
        "revision": "e9a805cefc590f3f06eb1d520fc6a33ea9c66c07",
        "assets": {
            "model.pt": (
                "be15082845c3e037bd6332f8a2bbd41468511789e63ebb0f299aeef73b4fd82a",
                265254050,
            ),
            "config.json": (
                "93198d34f5af96eb9e4490bb91c345fade3fb4913faa2a2f5a7b23a65bc01d79",
                1149,
            ),
            "metadata.json": (
                "65fae39a731a2607b202585c322826c5f79c15237cea66d1fb3e4083b80fa6e0",
                368,
            ),
            "mimic_norm.json": (
                "5ddb565a0d7db9642f422ae7ee6ca1a13f34fbdec30ece69484f9a3874c289ea",
                2035,
            ),
            "controller_combos.json": (
                "939f65efb7823836ca2d38df2ffb39aafd0018eea2066f399424c2e4ba75a01b",
                129,
            ),
            "cat_maps.json": (
                "aa49cb19705875571c9df309ab863df15e63915e17f0296977afd29f1ac74079",
                2263,
            ),
            "stick_clusters.json": (
                "f1dc69b7620189674064ca8661ac0fe50b79f7ad3db60f73892cc3f7ba9ad36c",
                1424,
            ),
            "norm_stats.json": (
                "e98cde99ef2cdbfcd02158a2c011564740a3e2e23f4f0ae0d9a30555fb5ac3a9",
                9876,
            ),
        },
    },
    "f51174be09c879b93e0095f394e35b26b9d461711f73927a22dcb60e72b1d87d": {
        "name": "falco-20260420-baseline",
        "character": "FALCO",
        "run_name": "falco-20260420-baseline",
        "repository": "https://huggingface.co/erickfm/MIMIC",
        "revision": "e9a805cefc590f3f06eb1d520fc6a33ea9c66c07",
        "assets": {
            "model.pt": (
                "f51174be09c879b93e0095f394e35b26b9d461711f73927a22dcb60e72b1d87d",
                265254050,
            ),
            "config.json": (
                "855f0be1617b49a1b043d5bf4e916f05ad0eefcd041f80ff65c4f18705c34a64",
                1149,
            ),
            "metadata.json": (
                "8763d598cd2005a06a212ca62527aece220d2826148543d7214b06dfa0d7bdb8",
                368,
            ),
            "mimic_norm.json": (
                "94fda5656f0034bafaf0ab7c91c0ff2289d740347177c31c195ce39562f30d0a",
                2027,
            ),
            "controller_combos.json": (
                "939f65efb7823836ca2d38df2ffb39aafd0018eea2066f399424c2e4ba75a01b",
                129,
            ),
            "cat_maps.json": (
                "fed315f87a2db15447e792c417b6912feb72cd4b9083f1195a487e23fff291a7",
                2235,
            ),
            "stick_clusters.json": (
                "f1dc69b7620189674064ca8661ac0fe50b79f7ad3db60f73892cc3f7ba9ad36c",
                1424,
            ),
            "norm_stats.json": (
                "b1f3dcc8cd9b8d2b327675398ef2e1a85cdd08f3bfe9f8626943b61b1edb9df2",
                9846,
            ),
        },
    },
    "a1efc35af523203bc7ae6878dd09458bf7ab2df95e57f368cff3e82574f0fdca": {
        "name": "sheik-20260420-baseline",
        "character": "SHEIK",
        "run_name": "sheik-20260420-baseline",
        "repository": "https://huggingface.co/erickfm/MIMIC",
        "revision": "e9a805cefc590f3f06eb1d520fc6a33ea9c66c07",
        "assets": {
            "model.pt": (
                "a1efc35af523203bc7ae6878dd09458bf7ab2df95e57f368cff3e82574f0fdca",
                265254050,
            ),
            "config.json": (
                "fcd323580458b46489e6d47ee43904a6d1e11e2f106972994420718366f8641e",
                1149,
            ),
            "metadata.json": (
                "9ec9836b5c91e8ed147f18fe9c8729d5a26dd1ebd691ee4e28984a7a73dd3965",
                368,
            ),
            "mimic_norm.json": (
                "c8ee050a02d1930401e5dc01b746de9b0c7d65a57dc1560c697f692276d90930",
                2025,
            ),
            "controller_combos.json": (
                "939f65efb7823836ca2d38df2ffb39aafd0018eea2066f399424c2e4ba75a01b",
                129,
            ),
            "cat_maps.json": (
                "75c1e83db3e050d29e141bd6dc2a619ae200d0387d2695adba97ebabb00a3240",
                2301,
            ),
            "stick_clusters.json": (
                "f1dc69b7620189674064ca8661ac0fe50b79f7ad3db60f73892cc3f7ba9ad36c",
                1424,
            ),
            "norm_stats.json": (
                "017f20b71f08f5cff39f5f3face564353c62af54d0c7db09c088b50600095d56",
                9852,
            ),
        },
    },
    "1d0b82dfa07aeebea650ff284ffdf44901c153caba0f06850f8c3de415585dd9": {
        "name": "cptfalcon-20260420-baseline",
        "character": "CPTFALCON",
        "run_name": "cptfalcon-20260420-baseline",
        "repository": "https://huggingface.co/erickfm/MIMIC",
        "revision": "e9a805cefc590f3f06eb1d520fc6a33ea9c66c07",
        "assets": {
            "model.pt": (
                "1d0b82dfa07aeebea650ff284ffdf44901c153caba0f06850f8c3de415585dd9",
                265256230,
            ),
            "config.json": (
                "ae3f966198176f115696d7ba2b6d95952ad7987326d6e180040510c95e517f06",
                1153,
            ),
            "metadata.json": (
                "b89a6cb8de497095262c62bb73db8818bad81aa4175e3b22b623e1d41ad7526c",
                379,
            ),
            "mimic_norm.json": (
                "79d165049c6e30b9183722a8abf2450dafa01fa174cc252665b64d2dcfd6e63b",
                2040,
            ),
            "controller_combos.json": (
                "939f65efb7823836ca2d38df2ffb39aafd0018eea2066f399424c2e4ba75a01b",
                129,
            ),
            "cat_maps.json": (
                "fe02bcd5cc0ac4a2985cd07b1c0d0809d277273bf209b3c8b2b86c8719229c45",
                2160,
            ),
            "stick_clusters.json": (
                "f1dc69b7620189674064ca8661ac0fe50b79f7ad3db60f73892cc3f7ba9ad36c",
                1424,
            ),
            "norm_stats.json": (
                "5c73e4ba3245f140116ce56456bfe6bdd7b8febed378342b9e04ab1740a93b15",
                9890,
            ),
        },
    },
    "3be7d6034095e33cac7e693fc648662a1ff058969e95854fb517e2ccc331676d": {
        "name": "luigi-20260420-baseline",
        "character": "LUIGI",
        "run_name": "luigi-20260420-baseline",
        "repository": "https://huggingface.co/erickfm/MIMIC",
        "revision": "e9a805cefc590f3f06eb1d520fc6a33ea9c66c07",
        "assets": {
            "model.pt": (
                "3be7d6034095e33cac7e693fc648662a1ff058969e95854fb517e2ccc331676d",
                265254050,
            ),
            "config.json": (
                "a2c8daa0303b2b64c2e50e75a3d81c920904b0f8518882361c520782d0578ef9",
                1149,
            ),
            "metadata.json": (
                "13459e9e9be8cddccb535a0c6368ac05acaf88a60ebe9dd3ac7f4e0b39f97089",
                361,
            ),
            "mimic_norm.json": (
                "9123c6d8c0333057393033e467311667d5ee0e88c1446b5cc9e25b4fcda0a21d",
                2042,
            ),
            "controller_combos.json": (
                "939f65efb7823836ca2d38df2ffb39aafd0018eea2066f399424c2e4ba75a01b",
                129,
            ),
            "cat_maps.json": (
                "3ca7eef5657491f5a1b21e22b57ecc59407473ca1854bff97670675d36fa2a99",
                2143,
            ),
            "stick_clusters.json": (
                "f1dc69b7620189674064ca8661ac0fe50b79f7ad3db60f73892cc3f7ba9ad36c",
                1424,
            ),
            "norm_stats.json": (
                "1552d322b89dd6a84387e185542e2da084d49459c9e6e28f318e91cfe30fc050",
                9906,
            ),
        },
    },
}
for _legacy_mimic_bundle in MIMIC_KNOWN_BUNDLES.values():
    _legacy_mimic_bundle.update(
        {
            "source_repository": MIMIC_SOURCE_REPOSITORY,
            "source_revision": MIMIC_SOURCE_REVISION,
            "source_directory": ".e001-cache/mimic-source",
            "allowlist_manifest": {
                "schema_version": "integration.mimic_inline_bundle_allowlist.v1",
                "path": "src/melee_policy/integration/match_runtime.py",
            },
        }
    )
_mimic_bundle_hash_collisions = set(MIMIC_KNOWN_BUNDLES) & set(MIMIC_NATIVE_BUNDLES)
if _mimic_bundle_hash_collisions:
    raise RuntimeError(
        f"MIMIC legacy/native bundle checkpoint collision: {sorted(_mimic_bundle_hash_collisions)}"
    )
MIMIC_KNOWN_BUNDLES.update(copy.deepcopy(MIMIC_NATIVE_BUNDLES))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return cast(dict[str, Any], value)


def _validate_mimic_bundle(
    checkpoint: Path, asset_directory: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Fail closed unless the checkpoint and all seven assets form a known bundle."""
    checkpoint = checkpoint.resolve()
    asset_directory = asset_directory.resolve()
    checkpoint_sha256 = _sha256_file(checkpoint)
    known = MIMIC_KNOWN_BUNDLES.get(checkpoint_sha256)
    if known is None:
        raise RuntimeError(
            "unmanifested MIMIC checkpoint: its SHA-256 is not in the exact bundle allowlist"
        )
    expected_assets = cast(dict[str, tuple[str, int]], known["assets"])
    observed_assets: list[dict[str, Any]] = []
    failures: dict[str, dict[str, Any]] = {}
    for name in MIMIC_ASSET_NAMES:
        path = asset_directory / name
        expected_sha256, expected_size = expected_assets[name]
        if not path.is_file():
            failures[name] = {"error": "missing"}
            continue
        observed_size = path.stat().st_size
        observed_sha256 = (
            checkpoint_sha256 if path.resolve() == checkpoint else _sha256_file(path)
        )
        observed_assets.append(
            {"name": name, "sha256": observed_sha256, "byte_length": observed_size}
        )
        if observed_sha256 != expected_sha256 or observed_size != expected_size:
            failures[name] = {
                "expected_sha256": expected_sha256,
                "observed_sha256": observed_sha256,
                "expected_byte_length": expected_size,
                "observed_byte_length": observed_size,
            }
    if failures:
        raise RuntimeError(
            f"MIMIC checkpoint/assets bundle identity mismatch: {failures}"
        )
    released_config = _load_json_object(asset_directory / "config.json")
    metadata = _load_json_object(asset_directory / "metadata.json")
    semantic_checks = {
        "checkpoint_matches_asset_model": checkpoint_sha256
        == expected_assets["model.pt"][0],
        "metadata_character": metadata.get("melee_enum") == known["character"],
        "metadata_run_name": metadata.get("run_name") == known["run_name"],
        "config_run_name": released_config.get("run_name") == known["run_name"],
        "metadata_config_run_name": metadata.get("run_name")
        == released_config.get("run_name"),
        "sequence_length": released_config.get("max_seq_len") == MIMIC_SEQUENCE_LENGTH,
        "controller_vocabulary": metadata.get("n_controller_combos") == 7
        and released_config.get("n_controller_combos") == 7,
        "architecture": released_config.get("encoder_type") == "mimic_flat"
        and released_config.get("model_preset") == "mimic"
        and (released_config.get("num_stages") == 6)
        and (released_config.get("num_characters") == 27)
        and (released_config.get("num_actions") == 396)
        and (released_config.get("n_stick_clusters") == 37)
        and (released_config.get("n_shoulder_bins") == 3)
        and (released_config.get("num_c_dirs") == 9)
        and (released_config.get("no_opp_inputs") is True),
    }
    if not all(semantic_checks.values()):
        raise RuntimeError(f"MIMIC bundle semantic mismatch: {semantic_checks}")
    identity = {
        "name": known["name"],
        "repository": known["repository"],
        "revision": known["revision"],
        "source_repository": known["source_repository"],
        "source_revision": known["source_revision"],
        "source_directory": known["source_directory"],
        "allowlist_manifest": copy.deepcopy(known["allowlist_manifest"]),
        "checkpoint_sha256": checkpoint_sha256,
        "character": known["character"],
        "run_name": known["run_name"],
        "assets": observed_assets,
        "checks": semantic_checks,
    }
    return (identity, released_config, metadata)


def _git_output(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _activate_mimic_bundle_source(
    project_root: Path, bundle_identity: Mapping[str, Any]
) -> dict[str, Any]:
    """Activate the exact upstream source revision declared by a validated bundle."""
    directory_value = bundle_identity.get("source_directory")
    repository = bundle_identity.get("source_repository")
    revision = bundle_identity.get("source_revision")
    if not isinstance(directory_value, str) or Path(directory_value).is_absolute():
        raise RuntimeError("MIMIC bundle has an invalid source directory")
    if not isinstance(repository, str) or not isinstance(revision, str):
        raise RuntimeError("MIMIC bundle has no exact source repository and revision")
    source = (project_root / directory_value).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"pinned MIMIC source is missing: {source}")
    identity = {
        "repository_url": _git_output(source, "remote", "get-url", "origin"),
        "revision": _git_output(source, "rev-parse", "HEAD"),
        "tracked_tree_clean": not bool(
            _git_output(source, "status", "--short", "--untracked-files=all")
        ),
        "directory": str(source),
    }
    checks = {
        "repository_url_exact": identity["repository_url"] == repository,
        "revision_exact": identity["revision"] == revision,
        "tracked_tree_clean": identity["tracked_tree_clean"] is True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"pinned MIMIC source identity mismatch: {identity}")
    conflicting_modules: dict[str, str] = {}
    for name, module in sys.modules.items():
        if (
            name != "tools"
            and (not name.startswith("tools."))
            and (name != "mimic")
            and (not name.startswith("mimic."))
        ):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        module_path = Path(module_file).resolve()
        if not module_path.is_relative_to(source):
            conflicting_modules[name] = str(module_path)
    if conflicting_modules:
        raise RuntimeError(
            f"MIMIC source revision cannot change after upstream modules were imported: {conflicting_modules}"
        )
    known_source_paths = {
        str((project_root / str(bundle["source_directory"])).resolve())
        for bundle in MIMIC_KNOWN_BUNDLES.values()
    }
    sys.path[:] = [
        entry
        for entry in sys.path
        if str(Path(entry).resolve()) not in known_source_paths
    ]
    sys.path.insert(0, str(source))
    return {
        **identity,
        "checks": checks,
        "bundle_character": bundle_identity.get("character"),
        "bundle_checkpoint_sha256": bundle_identity.get("checkpoint_sha256"),
    }


def _load_config(path: Path) -> tuple[dict[str, Any], Path]:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    project_root = path.resolve().parent.parent
    required = {
        ("integration", "schema_version"): INTEGRATION_SCHEMA_VERSION,
        ("integration", "seed"): 42,
        ("integration", "device"): "cpu",
        ("integration", "inference_mode"): "synchronous-concurrent",
        ("integration", "maximum_pending_snapshots_per_model"): 1,
        ("game_image", "byte_length"): 1459978240,
        (
            "game_image",
            "sha256",
        ): "0de05981a34156b9cedcef73c73d4244ac05cf6149ab3c9cfed917698819e464",
        ("emulator", "release"): EXPECTED_DOLPHIN_VERSION,
        ("emulator", "source_revision"): EXPECTED_DOLPHIN_SOURCE_REVISION,
        ("emulator", "frame_sync_protocol"): "FRAME_SYNC sequence kind",
        (
            "emulator",
            "frame_sync_patch",
        ): "patches/slippi-dolphin-two-pipe-frame-sync.patch",
        (
            "emulator",
            "frame_sync_patch_sha256",
        ): EXPECTED_DOLPHIN_FRAME_SYNC_PATCH_SHA256,
        ("emulator", "application_file_count"): 717,
        ("emulator", "application_byte_length"): 40565896,
        (
            "emulator",
            "application_tree_sha256",
        ): "b2213750930fe5752a6ce42263f9e94f2d8b144cf42bcf82fcdf84612581ecb7",
        (
            "emulator",
            "executable_sha256",
        ): "33ad9ec4d439e5b0493bd8488d92d91f4d3b5787b4d36a97332e1a551134c7a8",
        ("emulator", "slippi_port"): 51442,
        ("mimic", "source_revision"): MIMIC_SOURCE_REVISION,
        ("mimic", "decode_strategy"): MIMIC_DECODE_STRATEGY,
        ("mimic", "temperature"): 1.0,
        ("mimic", "top_k"): 0,
        ("mimic", "top_p"): 0.0,
    }
    for (section, field), expected in required.items():
        observed = config[section][field]
        if observed != expected:
            raise ValueError(
                f"config mismatch for {section}.{field}: {observed!r} != {expected!r}"
            )
    frame_sync_patch = project_root / str(config["emulator"]["frame_sync_patch"])
    if not frame_sync_patch.is_file():
        raise FileNotFoundError(
            f"Slippi Dolphin frame-sync patch is missing: {frame_sync_patch}"
        )
    observed_patch_sha256 = _sha256_file(frame_sync_patch)
    if observed_patch_sha256 != EXPECTED_DOLPHIN_FRAME_SYNC_PATCH_SHA256:
        raise ValueError(
            f"Slippi Dolphin frame-sync patch identity mismatch: {observed_patch_sha256} != {EXPECTED_DOLPHIN_FRAME_SYNC_PATCH_SHA256}"
        )
    return (config, project_root)


def _activate_sources(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Activate a user-provided MIMIC checkout after verifying its provenance."""
    source = (project_root / config["mimic"]["source_directory"]).resolve()
    identity = {
        "repository_url": _git_output(source, "remote", "get-url", "origin"),
        "revision": _git_output(source, "rev-parse", "HEAD"),
        "tracked_tree_clean": not bool(
            _git_output(source, "status", "--short", "--untracked-files=all")
        ),
        "directory": str(source),
    }
    if (
        identity["repository_url"] != MIMIC_SOURCE_REPOSITORY
        or identity["revision"] != MIMIC_SOURCE_REVISION
        or (not identity["tracked_tree_clean"])
    ):
        raise RuntimeError(f"pinned MIMIC source identity mismatch: {identity}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return {
        "revisions": {"mimic": identity["revision"]},
        "repositories": {"mimic": identity},
    }


def _validate_evaluation_seed(seed: int) -> int:
    """Return one seed supported by Python, NumPy, Torch, and TensorFlow."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be between 0 and 4294967295 inclusive")
    return seed


def _effective_evaluation_seed(config: dict[str, Any], override: int | None) -> int:
    configured = int(config["integration"]["seed"])
    return _validate_evaluation_seed(configured if override is None else override)


def _natural_end_requirement_met(*, required: bool, observed: bool) -> bool:
    """Allow canary frame limits unless a complete game was explicitly required."""
    return not required or observed


def _policy_torch_seed(evaluation_seed: int, policy_identity: str) -> int:
    """Derive a stable CPU sampling seed by policy family, never physical port."""
    base_seed = _validate_evaluation_seed(evaluation_seed)
    identity = policy_identity.strip().lower()
    if identity not in POLICY_TORCH_RNG_IDENTITIES:
        raise ValueError(f"unsupported Torch policy RNG identity: {policy_identity!r}")
    digest = hashlib.sha256(
        f"{POLICY_TORCH_RNG_DOMAIN}:{base_seed}:{identity}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") & (1 << 63) - 1


class _PolicyTorchRNG:
    """Run an unchanged CPU Torch sampler on an isolated, persistent RNG stream."""

    def __init__(self, evaluation_seed: int, policy_identity: str) -> None:
        self.evaluation_seed = _validate_evaluation_seed(evaluation_seed)
        self.policy_identity = policy_identity.strip().lower()
        self.derived_seed = _policy_torch_seed(
            self.evaluation_seed, self.policy_identity
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.derived_seed)
        self._state = generator.get_state()

    def invoke(self, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call a sampler while preserving both its local state and global state."""
        with _POLICY_TORCH_RNG_LOCK:
            global_state = torch.random.get_rng_state()
            torch.random.set_rng_state(self._state)
            try:
                return callback(*args, **kwargs)
            finally:
                self._state = torch.random.get_rng_state()
                torch.random.set_rng_state(global_state)

    def record(self) -> dict[str, Any]:
        return {
            "domain": POLICY_TORCH_RNG_DOMAIN,
            "evaluation_seed": self.evaluation_seed,
            "policy_identity": self.policy_identity,
            "derived_torch_seed": self.derived_seed,
            "port_independent": True,
            "global_torch_rng_isolated": True,
        }


@dataclass
class MimicRuntime:
    model: torch.nn.Module
    model_config: Any
    context: dict[str, Any]
    state: Any
    checkpoint_path: Path
    checkpoint_sha256: str
    asset_directory: Path
    bundle_identity: dict[str, Any]
    source_identity: dict[str, Any]
    metadata: dict[str, Any]
    controlled_character: str


def load_mimic_runtime(
    config: dict[str, Any],
    project_root: Path,
    checkpoint_override: Path | None = None,
    asset_directory_override: Path | None = None,
) -> MimicRuntime:
    checkpoint = (
        project_root / config["mimic"]["checkpoint"]
        if checkpoint_override is None
        else checkpoint_override.expanduser().resolve()
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"MIMIC checkpoint is missing: {checkpoint}")
    asset_directory = (
        project_root / config["mimic"]["asset_directory"]
        if asset_directory_override is None
        else asset_directory_override.expanduser().resolve()
    )
    bundle_identity, released_config, metadata = _validate_mimic_bundle(
        checkpoint, asset_directory
    )
    source_identity = _activate_mimic_bundle_source(project_root, bundle_identity)
    from tools.inference_utils import (
        PlayerState,
        load_inference_context,
        load_mimic_model,
    )

    checkpoint_sha256 = str(bundle_identity["checkpoint_sha256"])
    if (
        checkpoint_override is None
        and checkpoint_sha256 != config["mimic"]["checkpoint_sha256"]
    ):
        raise RuntimeError(
            "default MIMIC checkpoint hash does not match its configured identity"
        )
    controlled_character = bundle_identity["character"]
    if checkpoint_override is None and controlled_character != str(
        config["mimic"]["character"]
    ):
        raise RuntimeError(
            f"default MIMIC bundle does not match the configured character: {controlled_character} != {config['mimic']['character']}"
        )
    model, model_config = load_mimic_model(str(checkpoint), "cpu")
    checkpoint_payload = cast(
        dict[str, Any], torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    embedded_config = checkpoint_payload.get("config")
    if embedded_config != released_config:
        raise RuntimeError(
            "MIMIC embedded checkpoint configuration differs from config.json"
        )
    state_dictionary = checkpoint_payload.get("model_state_dict")
    if not isinstance(state_dictionary, dict):
        raise RuntimeError("MIMIC checkpoint has no model_state_dict")
    checkpoint_keys = {str(key).removeprefix("_orig_mod.") for key in state_dictionary}
    model_keys = set(model.state_dict())
    missing_keys = sorted(model_keys - checkpoint_keys)
    unexpected_keys = sorted(checkpoint_keys - model_keys)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"MIMIC state-dictionary mismatch after strict upstream load: missing={missing_keys}, unexpected={unexpected_keys}"
        )
    bundle_identity["state_dictionary"] = {
        "strict": True,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
    }
    del checkpoint_payload
    if int(model_config.max_seq_len) != MIMIC_SEQUENCE_LENGTH:
        raise RuntimeError(
            f"MIMIC sequence length mismatch: {model_config.max_seq_len}"
        )
    context = cast(dict[str, Any], load_inference_context(asset_directory))
    state = PlayerState(model, model_config.max_seq_len, "cpu", ctx=context)
    return MimicRuntime(
        model=model,
        model_config=model_config,
        context=context,
        state=state,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        asset_directory=asset_directory,
        bundle_identity=bundle_identity,
        source_identity=source_identity,
        metadata=metadata,
        controlled_character=controlled_character,
    )


class MimicLivePolicy:
    """MIMIC live inference through its released per-port frame builders."""

    def __init__(
        self,
        runtime: MimicRuntime,
        port: int,
        online_delay_frames: int = 0,
        *,
        evaluation_seed: int = 42,
    ) -> None:
        if port not in (1, 2):
            raise ValueError(f"MIMIC port must be 1 or 2, got {port}")
        if online_delay_frames < 0:
            raise ValueError("online delay must be nonnegative")
        self.runtime = runtime
        self.port = port
        self.online_delay_frames = online_delay_frames
        self._sent_commands: dict[int, dict[str, Any]] = {}
        self.previous_executed_frame: int | None = None
        self._policy_torch_rng = _PolicyTorchRNG(evaluation_seed, "mimic")

    @property
    def policy_rng(self) -> dict[str, Any]:
        return self._policy_torch_rng.record()

    def record_decoded_command(self, decision_frame: int, sent: dict[str, Any]) -> None:
        """Record the controller submitted after observing ``decision_frame``."""
        command = {str(key): value for key, value in sent.items()}
        self._sent_commands[int(decision_frame)] = command
        self.runtime.state.prev_sent = command

    def previous_executed_command(self, observed_frame: int) -> dict[str, Any] | None:
        """Return MIMIC's latest sent command eligible for this observation.

        Upstream MIMIC stores every decoded command in ``PlayerState.prev_sent``
        and supplies it to the next frame builder, including the command sent
        after observing frame -123.
        """
        target = int(observed_frame) - self.online_delay_frames - 1
        eligible = [frame for frame in self._sent_commands if frame <= target]
        if not eligible:
            self.previous_executed_frame = None
            return None
        selected = max(eligible)
        self.previous_executed_frame = selected
        return self._sent_commands[selected]

    def observe(self, gamestate: Any) -> bool:
        from tools.inference_utils import build_frame, build_frame_p2

        builder = build_frame if self.port == 1 else build_frame_p2
        previous = self.previous_executed_command(int(gamestate.frame))
        frame = builder(gamestate, previous, self.runtime.context)
        if frame is None:
            return False
        self.runtime.state.push_frame(frame)
        return True

    def infer(self) -> dict[str, torch.Tensor]:
        return self.infer_snapshot(self.snapshot())

    def snapshot(self) -> dict[str, torch.Tensor]:
        frames = list(self.runtime.state._frame_cache)
        if not frames:
            raise RuntimeError("cannot snapshot an empty MIMIC context")
        return {
            key: torch.cat([frame[key] for frame in frames], dim=0).unsqueeze(0)
            for key in frames[0]
        }

    def infer_snapshot(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        with torch.inference_mode():
            return cast(dict[str, torch.Tensor], self.runtime.model(batch))

    def decode_and_press(
        self,
        controller: Any,
        prediction: dict[str, torch.Tensor],
        previous: dict[str, Any] | None,
        *,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> tuple[dict[str, Any], list[str], list[str]]:
        """Invoke MIMIC's released decoder on this policy family's RNG stream."""
        from tools.inference_utils import decode_and_press

        adapted_controller = (
            None if controller is None else _MimicCompleteControllerFrame(controller)
        )
        sent, pressed, button_names = self._policy_torch_rng.invoke(
            decode_and_press,
            adapted_controller,
            prediction,
            previous,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        if adapted_controller is not None:
            adapted_controller.require_exactly_one_flush()
        return (
            cast(dict[str, Any], sent),
            cast(list[str], pressed),
            cast(list[str], button_names),
        )

    def predict(self, gamestate: Any) -> dict[str, torch.Tensor] | None:
        if not self.observe(gamestate):
            return None
        return self.infer()


class _MimicCompleteControllerFrame:
    """Preserve MIMIC's sender while neutralizing its undeclared analog R axis."""

    def __init__(self, controller: Any) -> None:
        self._controller = controller
        self._flushes = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._controller, name)

    def flush(self) -> None:
        import melee

        self._controller.press_shoulder(melee.Button.BUTTON_R, 0.0)
        self._controller.flush()
        self._flushes += 1

    def require_exactly_one_flush(self) -> None:
        if self._flushes != 1:
            raise RuntimeError(
                f"released MIMIC decoder must flush exactly one complete controller frame; observed {self._flushes} flushes"
            )


def _finite_prediction(prediction: Any) -> bool:
    values = prediction.values() if hasattr(prediction, "values") else []
    return all(
        (
            not value.is_floating_point() or bool(torch.isfinite(value).all())
            for value in values
        )
    )


class _LatestInferenceWorker:
    """Run one model off-thread, with optional latest-state or exact waiting."""

    def __init__(self, name: str, infer: Callable[[Any], Any]) -> None:
        self.name = name
        self._infer = infer
        self._condition = threading.Condition()
        self._pending: tuple[int, Any] | None = None
        self._completed: tuple[int, Any, float] | None = None
        self._failure: BaseException | None = None
        self._stopping = False
        self._thread = threading.Thread(
            target=self._run, name=f"{name}-latest-inference", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and (not self._stopping):
                    self._condition.wait()
                if self._stopping:
                    return
                source_frame, snapshot = cast(tuple[int, Any], self._pending)
                self._pending = None
            started_at = time.perf_counter()
            try:
                result = self._infer(snapshot)
            except BaseException as error:
                with self._condition:
                    self._failure = error
                    self._stopping = True
                    self._condition.notify_all()
                return
            elapsed = time.perf_counter() - started_at
            with self._condition:
                self._completed = (source_frame, result, elapsed)
                self._condition.notify_all()

    def submit(self, source_frame: int, snapshot: Any) -> None:
        self.raise_if_failed()
        with self._condition:
            if self._stopping:
                raise RuntimeError(f"{self.name} inference worker is stopped")
            self._pending = (source_frame, snapshot)
            self._condition.notify()

    def take_completed(self) -> tuple[int, Any, float] | None:
        self.raise_if_failed()
        with self._condition:
            completed = self._completed
            self._completed = None
            return completed

    def wait_completed(
        self, source_frame: int, timeout: float = 30.0
    ) -> tuple[int, Any, float]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._completed is None and self._failure is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"{self.name} inference timed out at frame {source_frame}"
                    )
                self._condition.wait(timeout=remaining)
            if self._failure is not None:
                error = self._failure
                raise RuntimeError(
                    f"{self.name} inference failed: {type(error).__name__}: {error}"
                ) from error
            completed = cast(tuple[int, Any, float], self._completed)
            self._completed = None
        if completed[0] != source_frame:
            raise RuntimeError(
                f"{self.name} returned frame {completed[0]} while waiting for {source_frame}"
            )
        return completed

    def raise_if_failed(self) -> None:
        with self._condition:
            failure = self._failure
        if failure is not None:
            raise RuntimeError(
                f"{self.name} asynchronous inference failed: {type(failure).__name__}: {failure}"
            ) from failure

    @property
    def failure(self) -> BaseException | None:
        with self._condition:
            return self._failure

    def close(self) -> None:
        with self._condition:
            self._stopping = True
            self._pending = None
            self._condition.notify_all()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RuntimeError(f"{self.name} inference worker did not stop")


def _neutral_mimic_command() -> dict[str, Any]:
    command: dict[str, Any] = {
        "main_x": 0.5,
        "main_y": 0.5,
        "c_x": 0.5,
        "c_y": 0.5,
        "l_shldr": 0.0,
        "r_shldr": 0.0,
    }
    for button in ("A", "B", "X", "Y", "Z", "L", "R"):
        command[f"btn_BUTTON_{button}"] = 0
    return command


def _json_controller(decoded: dict[str, Any]) -> dict[str, Any]:
    return {
        "main_stick": [float(value) for value in decoded["main_stick"]],
        "c_stick": [float(value) for value in decoded["c_stick"]],
        "shoulder": float(decoded.get("shoulder", 0.0)),
        "buttons": [str(value) for value in decoded.get("buttons", [])],
    }


def _send_mimic_controller_inputs(controller: Any, sent: dict[str, Any]) -> None:
    """Repeat a decoded MIMIC controller command without another model pass."""
    import melee
    from tools.inference_utils import ALL_ACTION_BUTTONS

    controller.tilt_analog(
        melee.Button.BUTTON_MAIN, float(sent["main_x"]), float(sent["main_y"])
    )
    controller.tilt_analog(
        melee.Button.BUTTON_C, float(sent["c_x"]), float(sent["c_y"])
    )
    for button in ALL_ACTION_BUTTONS:
        controller.release_button(button)
    controller.press_shoulder(melee.Button.BUTTON_L, float(sent["l_shldr"]))
    controller.press_shoulder(melee.Button.BUTTON_R, float(sent["r_shldr"]))
    for button in ALL_ACTION_BUTTONS:
        if int(sent.get(f"btn_{button.name}", 0)):
            controller.press_button(button)
    controller.flush()


def _dispatch_mimic_controller_inputs(
    controller: Any,
    policy: MimicLivePolicy,
    prediction: dict[str, torch.Tensor] | None,
    previous: dict[str, Any] | None,
    held: dict[str, Any],
    held_pressed: list[str],
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> tuple[dict[str, Any], list[str]]:
    """Send exactly once, decoding a fresh result or repeating the held command."""
    if prediction is None:
        _send_mimic_controller_inputs(controller, held)
        return (held, list(held_pressed))
    sent, pressed, _button_names = policy.decode_and_press(
        controller,
        prediction,
        previous,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    return (sent, pressed)


def _game_image_identity(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        header = stream.read(8)
    game_id = header[:6].decode("ascii", errors="replace")
    return {
        "filename": path.name,
        "byte_length": path.stat().st_size,
        "sha256": _sha256_file(path),
        "disc_game_id": game_id,
        "disc_number": header[6] if len(header) > 6 else None,
        "disc_revision": header[7] if len(header) > 7 else None,
    }


def _resolve_game_image_path(
    config: dict[str, Any], project_root: Path, command_line_path: Path | None
) -> Path:
    if command_line_path is not None:
        path = command_line_path
    else:
        variable = str(config["game_image"]["environment_variable"])
        configured = os.environ.get(variable)
        path = (
            Path(configured)
            if configured
            else project_root / config["game_image"]["default_path"]
        )
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"game image is missing: {path}; provide --iso-path or set {config['game_image']['environment_variable']}"
        )
    identity = _game_image_identity(path)
    expected_id = str(config["game_image"]["disc_game_id"])
    expected_revision = int(config["game_image"]["disc_revision"])
    if (
        identity["disc_game_id"] != expected_id
        or identity["disc_revision"] != expected_revision
    ):
        raise ValueError(
            f"game image is not Melee NTSC 1.02: disc_game_id={identity['disc_game_id']!r}, disc_revision={identity['disc_revision']!r}"
        )
    expected_byte_length = int(config["game_image"]["byte_length"])
    expected_sha256 = str(config["game_image"]["sha256"])
    if (
        identity["byte_length"] != expected_byte_length
        or identity["sha256"] != expected_sha256
    ):
        raise ValueError(
            f"game image does not match the frozen tournament image: byte_length={identity['byte_length']!r}, sha256={identity['sha256']!r}"
        )
    return path


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _file_identity(path: Path, project_root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"reproducibility input is missing: {resolved}")
    return {
        "path": _display_path(resolved, project_root),
        "sha256": _sha256_file(resolved),
        "byte_length": resolved.stat().st_size,
    }


def _emulator_application_identity(
    config: dict[str, Any], project_root: Path
) -> dict[str, Any]:
    application = (
        project_root
        / config["emulator"]["directory"]
        / config["emulator"]["application"]
    ).resolve()
    if not application.is_dir():
        raise FileNotFoundError(f"Slippi Dolphin application is missing: {application}")
    paths = sorted(
        (path for path in application.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(application).as_posix(),
    )
    files = [
        {
            "path": path.relative_to(application).as_posix(),
            "sha256": _sha256_file(path),
            "byte_length": path.stat().st_size,
        }
        for path in paths
    ]
    tree_sha256 = hashlib.sha256(
        json.dumps(files, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    executable = application / "Contents" / "MacOS" / "Slippi Dolphin"
    executable_identity = _file_identity(executable, project_root)
    identity: dict[str, Any] = {
        "path": _display_path(application, project_root),
        "file_count": len(files),
        "byte_length": sum((int(record["byte_length"]) for record in files)),
        "tree_manifest_sha256": tree_sha256,
        "executable": executable_identity,
    }
    expected = {
        "file_count": int(config["emulator"]["application_file_count"]),
        "byte_length": int(config["emulator"]["application_byte_length"]),
        "tree_manifest_sha256": str(config["emulator"]["application_tree_sha256"]),
        "executable_sha256": str(config["emulator"]["executable_sha256"]),
    }
    observed = {
        "file_count": identity["file_count"],
        "byte_length": identity["byte_length"],
        "tree_manifest_sha256": identity["tree_manifest_sha256"],
        "executable_sha256": executable_identity["sha256"],
    }
    if observed != expected:
        raise RuntimeError(
            f"Slippi Dolphin application identity mismatch: {observed} != {expected}"
        )
    identity["matches_frozen_configuration"] = True
    identity["release_attestation"] = _emulator_release_attestation(config, identity)
    return identity


def _emulator_release_attestation(
    config: Mapping[str, Any], application_identity: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind the release label to the frozen application without launching its GUI."""
    emulator = config.get("emulator")
    if not isinstance(emulator, Mapping):
        raise RuntimeError(
            "Slippi Dolphin release attestation requires emulator configuration"
        )
    executable = application_identity.get("executable")
    executable_identity = executable if isinstance(executable, Mapping) else {}
    release = emulator.get("release")
    source_revision = emulator.get("source_revision")
    checks = {
        "configuration_release_exact": release == EXPECTED_DOLPHIN_VERSION,
        "configuration_source_revision_exact": source_revision
        == EXPECTED_DOLPHIN_SOURCE_REVISION,
        "application_identity_gate_passed": application_identity.get(
            "matches_frozen_configuration"
        )
        is True,
        "application_file_count_exact": application_identity.get("file_count")
        == emulator.get("application_file_count"),
        "application_byte_length_exact": application_identity.get("byte_length")
        == emulator.get("application_byte_length"),
        "application_tree_sha256_exact": application_identity.get(
            "tree_manifest_sha256"
        )
        == emulator.get("application_tree_sha256"),
        "executable_sha256_exact": executable_identity.get("sha256")
        == emulator.get("executable_sha256"),
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(f"Slippi Dolphin release attestation failed: {failures}")
    return {
        "schema_version": EMULATOR_RELEASE_ATTESTATION_SCHEMA_VERSION,
        "decision": "pass",
        "release": release,
        "source_revision": source_revision,
        "method": "pinned-configuration-and-full-application-tree-identity",
        "runtime_gui_version_probe_used": False,
        "checks": checks,
    }


def _attested_emulator_release(application_identity: Mapping[str, Any]) -> str:
    """Return a release only from a passing cryptographic attestation record."""
    attestation = application_identity.get("release_attestation")
    if not isinstance(attestation, Mapping):
        raise RuntimeError(
            "Slippi Dolphin application identity has no release attestation"
        )
    if (
        attestation.get("schema_version") != EMULATOR_RELEASE_ATTESTATION_SCHEMA_VERSION
        or attestation.get("decision") != "pass"
        or attestation.get("runtime_gui_version_probe_used") is not False
    ):
        raise RuntimeError("Slippi Dolphin release attestation is invalid")
    release = attestation.get("release")
    if release != EXPECTED_DOLPHIN_VERSION:
        raise RuntimeError(f"Slippi Dolphin release attestation mismatch: {release!r}")
    return str(release)


def _create_attested_dolphin_console(
    config: Mapping[str, Any],
    project_root: Path,
    application_identity: Mapping[str, Any],
    **console_options: Any,
) -> Any:
    """Construct libmelee Console using the already-verified executable identity."""
    import melee.console as melee_console

    emulator = config.get("emulator")
    if not isinstance(emulator, Mapping):
        raise RuntimeError(
            "attested libmelee Console construction requires emulator configuration"
        )
    directory = emulator.get("directory")
    application = emulator.get("application")
    if not isinstance(directory, str) or not isinstance(application, str):
        raise RuntimeError(
            "attested libmelee Console construction requires emulator paths"
        )
    emulator_directory = (project_root / directory).resolve()
    expected_executable = (
        emulator_directory / application / "Contents" / "MacOS" / "Slippi Dolphin"
    ).resolve()
    executable = application_identity.get("executable")
    executable_identity = executable if isinstance(executable, Mapping) else {}
    embedded_attestation = application_identity.get("release_attestation")
    recomputed_attestation = _emulator_release_attestation(config, application_identity)
    if embedded_attestation != recomputed_attestation:
        raise RuntimeError(
            "attested libmelee Console construction failed: ['embedded_release_attestation_exact']"
        )
    attested_release = _attested_emulator_release(application_identity)
    recorded_path = executable_identity.get("path")
    recorded_executable = (
        Path(recorded_path)
        if isinstance(recorded_path, str) and Path(recorded_path).is_absolute()
        else project_root / str(recorded_path)
    ).resolve()
    current_executable_identity = _file_identity(expected_executable, project_root)
    checks = {
        "application_identity_gate_passed": application_identity.get(
            "matches_frozen_configuration"
        )
        is True,
        "recorded_executable_path_exact": recorded_executable == expected_executable,
        "recorded_executable_sha256_exact": executable_identity.get("sha256")
        == current_executable_identity.get("sha256")
        == emulator.get("executable_sha256"),
        "recorded_executable_byte_length_exact": executable_identity.get("byte_length")
        == current_executable_identity.get("byte_length"),
        "attested_release_exact": attested_release == EXPECTED_DOLPHIN_VERSION,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(f"attested libmelee Console construction failed: {failures}")
    if "path" in console_options:
        raise RuntimeError(
            "attested libmelee Console construction owns the executable path"
        )
    if "is_dolphin" in console_options and console_options["is_dolphin"] is not True:
        raise RuntimeError(
            "attested libmelee Console construction requires is_dolphin=True"
        )
    console_options["is_dolphin"] = True

    def attested_get_dolphin_version(path: str) -> Any:
        requested_executable = Path(path).resolve()
        if requested_executable != expected_executable:
            raise RuntimeError(
                f"libmelee requested an executable outside the attested application: {requested_executable}"
            )
        return melee_console.DolphinVersion(
            mainline=False,
            version=attested_release,
            build=melee_console.DolphinBuild.NETPLAY,
        )

    with _LIBMELEE_VERSION_ATTESTATION_LOCK:
        original_get_dolphin_version = melee_console.get_dolphin_version
        melee_console.get_dolphin_version = attested_get_dolphin_version
        try:
            console = melee_console.Console(
                path=str(emulator_directory), **console_options
            )
        finally:
            melee_console.get_dolphin_version = original_get_dolphin_version
    console._melee_policy_attested_launch = types.MappingProxyType(
        {
            "schema_version": ATTESTED_DOLPHIN_LAUNCH_SCHEMA_VERSION,
            "executable_path": str(expected_executable),
            "executable_sha256": current_executable_identity["sha256"],
            "executable_byte_length": current_executable_identity["byte_length"],
        }
    )
    return console


def _macos_processes_for_exact_executable(
    executable: Path,
) -> tuple[tuple[int, str], ...]:
    """Return live Darwin processes whose kernel-reported executable is exact."""
    if platform.system() != "Darwin":
        return ()
    library_path = ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib"
    libproc = ctypes.CDLL(library_path, use_errno=True)
    proc_listpids = libproc.proc_listpids
    proc_listpids.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    proc_listpids.restype = ctypes.c_int
    proc_pidpath = libproc.proc_pidpath
    proc_pidpath.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    proc_pidpath.restype = ctypes.c_int
    proc_name = libproc.proc_name
    proc_name.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    proc_name.restype = ctypes.c_int
    proc_all_pids = 1
    maximum_path_bytes = 4096
    required_bytes = proc_listpids(proc_all_pids, 0, None, 0)
    if required_bytes <= 0:
        error_number = ctypes.get_errno()
        raise RuntimeError(
            f"could not enumerate Darwin processes before Dolphin launch: errno={error_number}"
        )
    capacity = required_bytes // ctypes.sizeof(ctypes.c_int) + 32
    pid_buffer = (ctypes.c_int * capacity)()
    populated_bytes = proc_listpids(
        proc_all_pids,
        0,
        ctypes.cast(pid_buffer, ctypes.c_void_p),
        ctypes.sizeof(pid_buffer),
    )
    if populated_bytes <= 0:
        error_number = ctypes.get_errno()
        raise RuntimeError(
            f"could not enumerate Darwin process IDs before Dolphin launch: errno={error_number}"
        )
    expected = executable.resolve()
    matches: list[tuple[int, str]] = []
    for pid in pid_buffer[: populated_bytes // ctypes.sizeof(ctypes.c_int)]:
        if pid <= 0 or pid == os.getpid():
            continue
        name_buffer = ctypes.create_string_buffer(maximum_path_bytes)
        if proc_name(pid, name_buffer, maximum_path_bytes) <= 0:
            continue
        if os.fsdecode(name_buffer.value) != expected.name:
            continue
        path_buffer = ctypes.create_string_buffer(maximum_path_bytes)
        ctypes.set_errno(0)
        path_length = proc_pidpath(pid, path_buffer, maximum_path_bytes)
        if path_length <= 0:
            error_number = ctypes.get_errno()
            if error_number == errno.ESRCH:
                continue
            raise RuntimeError(
                f"could not resolve a candidate Slippi Dolphin executable path: pid={pid} errno={error_number}"
            )
        observed = Path(os.fsdecode(path_buffer.value)).resolve()
        if observed == expected:
            matches.append((int(pid), str(observed)))
    return tuple(sorted(matches))


def _disable_attested_dolphin_stop_hotkey(console: Any) -> Path:
    """Clear the pinned Ishiiruka Stop binding in this run's isolated profile."""
    config_directory = Path(str(console._get_dolphin_home_path())) / "Config"
    config_directory.mkdir(parents=True, exist_ok=True)
    hotkeys_path = config_directory / "Hotkeys.ini"
    hotkeys = _CaseSensitiveRawConfigParser()
    if hotkeys_path.is_file():
        with hotkeys_path.open(encoding="utf-8") as hotkeys_file:
            hotkeys.read_file(hotkeys_file)
    if not hotkeys.has_section(DOLPHIN_HOTKEY_SECTION):
        hotkeys.add_section(DOLPHIN_HOTKEY_SECTION)
    hotkeys.set(DOLPHIN_HOTKEY_SECTION, DOLPHIN_STOP_HOTKEY, "")
    with hotkeys_path.open("w", encoding="utf-8", newline="\n") as hotkeys_file:
        hotkeys.write(hotkeys_file)
    return hotkeys_path


def _launch_and_connect_attested_dolphin(console: Any, iso_path: str | Path) -> bool:
    """Launch the exact attested Dolphin with a process-scoped AppKit guard."""
    launch_contract = getattr(console, "_melee_policy_attested_launch", None)
    if not isinstance(launch_contract, Mapping):
        raise RuntimeError("live Dolphin launch requires an attested console")
    if launch_contract.get("schema_version") != ATTESTED_DOLPHIN_LAUNCH_SCHEMA_VERSION:
        raise RuntimeError("live Dolphin launch attestation schema mismatch")
    if getattr(console, "is_dolphin", None) is not True:
        raise RuntimeError("attested Dolphin launch requires is_dolphin=True")
    if getattr(console, "_process", None) is not None:
        raise RuntimeError("attested Dolphin console already owns a live process")
    expected_executable = Path(str(launch_contract.get("executable_path"))).resolve()
    console_executable = Path(str(getattr(console, "exe_path", ""))).resolve()
    current_identity = {
        "sha256": _sha256_file(expected_executable),
        "byte_length": expected_executable.stat().st_size,
    }
    checks = {
        "console_executable_path_exact": console_executable == expected_executable,
        "executable_sha256_exact": current_identity["sha256"]
        == launch_contract.get("executable_sha256"),
        "executable_byte_length_exact": current_identity["byte_length"]
        == launch_contract.get("executable_byte_length"),
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(f"attested Dolphin launch failed: {failures}")
    stale_processes = _macos_processes_for_exact_executable(expected_executable)
    if stale_processes:
        details = ", ".join((f"pid={pid} path={path}" for pid, path in stale_processes))
        raise RuntimeError(
            f"exact Slippi Dolphin executable is already running: {details}"
        )
    _disable_attested_dolphin_stop_hotkey(console)
    if platform.system() != "Darwin":
        console.run(iso_path=str(iso_path))
        return bool(console.connect())
    base_command = [
        str(expected_executable),
        "-e",
        str(iso_path),
        "-u",
        str(console._get_dolphin_home_path()),
    ]
    command = [*base_command, *MACOS_PERSISTENCE_IGNORE_ARGUMENTS]
    process = subprocess.Popen(command, env=os.environ.copy())
    console._process = process
    return bool(console.connect())


def _runtime_reproducibility_record(
    project_root: Path,
    config_path: Path,
    dependency_lock: str,
    implementation_paths: tuple[str, ...],
) -> dict[str, Any]:
    """Hash every local input that determines one live integration run."""
    lock_path = project_root / dependency_lock
    return {
        "configuration": _file_identity(config_path, project_root),
        "dependency_lock": _file_identity(lock_path, project_root),
        "implementation_files": [
            _file_identity(project_root / relative, project_root)
            for relative in implementation_paths
        ],
        "runtime_environment": runtime_environment_record(project_root, lock_path),
    }


def _require_unused_artifact_label(
    output_directory: Path, artifact_label: str | None
) -> None:
    """Prevent an explicit run label from overwriting or mixing prior evidence."""
    if artifact_label is None or not output_directory.exists():
        return
    if any(output_directory.iterdir()):
        raise FileExistsError(
            f"artifact label {artifact_label!r} already contains run output: {output_directory}; choose a new label"
        )


def _stop_console(console: Any, replay_finalize_timeout_seconds: float = 30.0) -> str:
    if replay_finalize_timeout_seconds <= 0:
        raise ValueError("replay finalize timeout must be positive")
    shutdown_method = "libmelee-kill"
    process = getattr(console, "_process", None)
    controllers = tuple(getattr(console, "controllers", ()))
    managed_controllers = [
        controller
        for controller in controllers
        if isinstance(controller, _ControllerFlushProxy)
    ]
    if managed_controllers:
        if len(managed_controllers) != len(controllers):
            raise RuntimeError(
                "cannot seal a mixed managed and unmanaged controller transport"
            )
        transports = {
            id(controller._transport): controller._transport
            for controller in controllers
        }
        if len(transports) != 1:
            raise RuntimeError(
                "cannot seal more than one controller transport for one console"
            )
        next(iter(transports.values())).seal_benchmark_audit()
    if process is not None and process.poll() is None:
        process.send_signal(signal.SIGINT)
        deadline = time.monotonic() + replay_finalize_timeout_seconds
        exited_cleanly = False
        while time.monotonic() < deadline:
            for controller in controllers:
                with suppress(BrokenPipeError, OSError, RuntimeError):
                    controller.flush()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                process.wait(timeout=min(1.0 / 60.0, remaining))
                exited_cleanly = True
                break
            except subprocess.TimeoutExpired:
                continue
        for controller in controllers:
            controller.disconnect()
        if exited_cleanly:
            shutdown_method = "child-sigint-input-drain"
        else:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=10)
                shutdown_method = "child-double-sigint"
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
                shutdown_method = "child-double-sigint-then-kill"
        console._process = None
    try:
        console.stop()
    except AssertionError:
        temporary_home = getattr(console, "temp_dir", None)
        if temporary_home:
            shutil.rmtree(temporary_home, ignore_errors=True)
            console.temp_dir = None
    return shutdown_method


class _ControllerFlushProxy:
    """Delegate a controller while giving one owner its blocking-pipe commits."""

    __slots__ = ("_port", "_raw", "_transport")

    def __init__(self, transport: _ControllerPipeLockstep, port: int, raw: Any) -> None:
        object.__setattr__(self, "_transport", transport)
        object.__setattr__(self, "_port", port)
        object.__setattr__(self, "_raw", raw)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)

    def flush(self) -> None:
        self._transport._flush(self._port)


class _ControllerPipeLockstep:
    """Commit both controller pipes as one identified frame transaction.

    The pinned tournament emulator combines its two strict human-controller
    pipes at one frame barrier. Each physical ``FLUSH`` must be preceded by a
    ``FRAME_SYNC`` marker. Both ports receive the same positive sequence and
    transaction kind so the emulator cannot advance with one stale command.

    Each tournament boundary is explicit. Native controller senders keep
    their complete-frame ``flush()`` calls, while queued Slippi-AI commands mark
    a transport-owned boundary. The first per-port flush in ``Console.step`` is
    libmelee's duplicate preamble and is suppressed. The proxy pairs policy,
    genuine internal, and otherwise unscoped requests before forwarding either
    port.
    """

    SCHEMA_VERSION = "integration.controller_pipe_lockstep.v7"
    VERIFIED_PIPE_DEVICE_ORDER_SOURCE = (
        "stable-filesystem-readdir-order-matching-dolphin-source"
    )
    FRAME_SYNC_KINDS = frozenset(
        {"menu", "gameplay", "preamble", "internal", "unscoped"}
    )

    def __init__(self, console: Any, raw_controllers: dict[int, Any]) -> None:
        self.console = console
        self.raw_controllers = dict(raw_controllers)
        self.controllers: dict[int, _ControllerFlushProxy] = {}
        self._registration_order: tuple[int, ...] = ()
        self._pipe_device_order: tuple[int, ...] = ()
        self._pipe_device_order_source = "unavailable"
        self._inside_console_step = False
        self._seen_inside_step: set[int] = set()
        self._internal_pending: set[int] = set()
        self._unscoped_pending: set[int] = set()
        self._active_boundary: dict[str, Any] | None = None
        self._installed = False
        self._prime_calls = 0
        self._step_calls = 0
        self._none_step_results = 0
        self._boundaries_started = 0
        self._boundaries_committed = 0
        self._group_commits = 0
        self._group_commit_reasons: dict[str, int] = {}
        self._frame_sync_sequence = 0
        self._frame_sync_last_completed_sequence = 0
        self._frame_sync_kind_commits = {
            kind: 0 for kind in sorted(self.FRAME_SYNC_KINDS)
        }
        self._frame_sync_markers = {port: 0 for port in self.raw_controllers}
        self._native_boundary_requests = {port: 0 for port in self.raw_controllers}
        self._unscoped_outside_forwarded = {port: 0 for port in self.raw_controllers}
        self._preamble_suppressed = {port: 0 for port in self.raw_controllers}
        self._internal_requests = {port: 0 for port in self.raw_controllers}
        self._group_flushes = {port: 0 for port in self.raw_controllers}
        self._held_group_flushes = {port: 0 for port in self.raw_controllers}
        self._scheduled_created = {port: 0 for port in self.raw_controllers}
        self._scheduled_consumed = {port: 0 for port in self.raw_controllers}
        self._sealed_audit_record: dict[str, Any] | None = None
        self._sealed_gate_checks: dict[str, bool] | None = None

    @classmethod
    def install(
        cls, console: Any, raw_controllers: dict[int, Any]
    ) -> _ControllerPipeLockstep:
        if set(raw_controllers) != {1, 2}:
            raise RuntimeError(
                "controller transport requires exactly physical ports 1 and 2"
            )
        if len({id(controller) for controller in raw_controllers.values()}) != 2:
            raise RuntimeError(
                "controller transport requires two distinct controller objects"
            )
        for port, controller in raw_controllers.items():
            if int(getattr(controller, "port", -1)) != port:
                raise RuntimeError(
                    f"controller key {port} does not match its physical port"
                )
            if not callable(getattr(controller, "_write", None)):
                raise RuntimeError(
                    f"controller port {port} does not expose libmelee's raw pipe writer; FRAME_SYNC cannot be emitted before every FLUSH"
                )
        registered = tuple(getattr(console, "controllers", ()))
        expected_ids = {id(controller) for controller in raw_controllers.values()}
        if (
            len(registered) != 2
            or {id(controller) for controller in registered} != expected_ids
        ):
            raise RuntimeError(
                "libmelee controller registration does not exactly match the two tournament ports"
            )
        transport = cls(console, raw_controllers)
        by_raw_id: dict[int, _ControllerFlushProxy] = {}
        for port, raw in raw_controllers.items():
            proxy = _ControllerFlushProxy(transport, port, raw)
            transport.controllers[port] = proxy
            by_raw_id[id(raw)] = proxy
        transport._registration_order = tuple(
            (int(controller.port) for controller in registered)
        )
        transport._pipe_device_order, transport._pipe_device_order_source = (
            transport._discover_pipe_device_order()
        )
        console.controllers = [by_raw_id[id(controller)] for controller in registered]
        transport._installed = True
        return transport

    def _discover_pipe_device_order(self) -> tuple[tuple[int, ...], str]:
        """Recover the order used by Dolphin's non-recursive pipe scan."""
        pipe_paths: dict[int, Path] = {}
        missing_path_ports: list[int] = []
        for port, controller in self.raw_controllers.items():
            value = getattr(controller, "pipe_path", None)
            if not isinstance(value, (str, os.PathLike)):
                missing_path_ports.append(port)
                continue
            pipe_paths[port] = Path(value)
        if missing_path_ports:
            raise RuntimeError(
                f"cannot verify Dolphin controller-pipe scan order because controller ports {missing_path_ports} do not expose pipe_path; libmelee registration order is not a safe fallback"
            )
        resolved_paths = {port: path.resolve() for port, path in pipe_paths.items()}
        if len(set(resolved_paths.values())) != len(self.raw_controllers):
            raise RuntimeError(
                "cannot verify Dolphin controller-pipe scan order because physical ports do not map to distinct pipe paths"
            )
        parents = {path.parent.resolve() for path in pipe_paths.values()}
        missing_paths = [str(path) for path in pipe_paths.values() if not path.exists()]
        if len(parents) != 1 or missing_paths:
            raise RuntimeError(
                f"cannot verify Dolphin controller-pipe scan order from one live Pipes directory; parents={sorted((str(parent) for parent in parents))}, missing={missing_paths}; libmelee registration order is not a safe fallback"
            )
        parent = next(iter(parents))
        name_to_port = {path.name: port for port, path in pipe_paths.items()}
        with os.scandir(parent) as entries:
            first_scan = list(entries)
        unexpected = [
            entry.name for entry in first_scan if entry.name not in name_to_port
        ]
        non_fifos = [
            entry.name
            for entry in first_scan
            if not stat.S_ISFIFO(entry.stat(follow_symlinks=False).st_mode)
        ]
        if unexpected or non_fifos:
            raise RuntimeError(
                f"controller Pipes directory is not the fresh two-FIFO tournament directory; unexpected={unexpected}, non_fifos={non_fifos}"
            )
        order = tuple((name_to_port[entry.name] for entry in first_scan))
        with os.scandir(parent) as entries:
            repeated_order = tuple((name_to_port[entry.name] for entry in entries))
        if repeated_order != order:
            raise RuntimeError(
                "controller-pipe scan order changed during transport installation"
            )
        if len(order) != len(self.raw_controllers) or set(order) != set(
            self.raw_controllers
        ):
            raise RuntimeError(
                "could not recover Dolphin's complete controller-pipe scan order"
            )
        return (order, self.VERIFIED_PIPE_DEVICE_ORDER_SOURCE)

    def _registration_is_exact(self) -> bool:
        registered = tuple(getattr(self.console, "controllers", ()))
        expected = tuple((self.controllers[port] for port in self._registration_order))
        return len(registered) == len(expected) and all(
            (
                actual is wanted
                for actual, wanted in zip(registered, expected, strict=True)
            )
        )

    def prime(self) -> None:
        """Place one ordered initial neutral/held command on both blocking pipes."""
        if not self._installed or not self._registration_is_exact():
            raise RuntimeError("controller transport is not installed exactly")
        if self._prime_calls != 0 or self._step_calls != 0:
            raise RuntimeError(
                "controller transport may be primed exactly once before stepping"
            )
        self._prime_calls += 1
        self._forward_group(
            reason="initial-prime", kind="unscoped", held_ports=set(self.controllers)
        )

    def begin_boundary(self, *, reason: str, game_frame: int | None) -> None:
        """Open one two-port command transaction after all current work is ready."""
        if not self._installed or not self._registration_is_exact():
            raise RuntimeError("controller transport is not installed exactly")
        if self._inside_console_step:
            raise RuntimeError("cannot open a policy boundary from inside Console.step")
        if self._active_boundary is not None:
            raise RuntimeError("a controller boundary transaction is already active")
        self._active_boundary = {
            "reason": str(reason),
            "game_frame": None if game_frame is None else int(game_frame),
            "native": set(),
            "scheduled": {},
        }
        self._boundaries_started += 1

    def schedule_next_boundary(
        self, port: int, *, reason: str, game_frame: int
    ) -> None:
        """Account for a queued or held port without claiming policy dispatch."""
        if port not in self.controllers:
            raise RuntimeError(f"cannot schedule unknown controller port {port}")
        if self._inside_console_step:
            raise RuntimeError(
                "cannot schedule a controller boundary from inside Console.step"
            )
        if self._active_boundary is None:
            raise RuntimeError(
                "cannot schedule a controller port without an active boundary"
            )
        native = cast(set[int], self._active_boundary["native"])
        scheduled = cast(dict[int, dict[str, Any]], self._active_boundary["scheduled"])
        if port in native:
            raise RuntimeError(
                f"controller port {port} already requested a native boundary flush"
            )
        if port in scheduled:
            previous = scheduled[port]
            raise RuntimeError(
                f"controller port {port} already has a pending boundary from game frame {previous['game_frame']} ({previous['reason']})"
            )
        scheduled[port] = {"reason": str(reason), "game_frame": int(game_frame)}
        self._scheduled_created[port] += 1

    def commit_boundary(self) -> None:
        """Make both complete pipe batches visible before releasing Dolphin."""
        if self._active_boundary is None:
            raise RuntimeError("cannot commit without an active controller boundary")
        native = cast(set[int], self._active_boundary["native"])
        scheduled = cast(dict[int, dict[str, Any]], self._active_boundary["scheduled"])
        accounted = native | set(scheduled)
        expected = set(self.controllers)
        if accounted != expected:
            missing = sorted(expected - accounted)
            unexpected = sorted(accounted - expected)
            raise RuntimeError(
                f"controller boundary does not account for both ports; missing={missing}, unexpected={unexpected}"
            )
        reason = str(self._active_boundary["reason"])
        self._forward_group(
            reason=f"policy:{reason}",
            kind="menu" if reason == "menu" else "gameplay",
            held_ports=set(scheduled),
        )
        for port in scheduled:
            self._scheduled_consumed[port] += 1
        self._active_boundary = None
        self._boundaries_committed += 1

    def _forward_group(self, *, reason: str, kind: str, held_ports: set[int]) -> None:
        expected = set(self.raw_controllers)
        if set(self._pipe_device_order) != expected or len(
            self._pipe_device_order
        ) != len(expected):
            raise RuntimeError("controller pipe order is incomplete")
        if kind not in self.FRAME_SYNC_KINDS:
            raise RuntimeError(f"unsupported controller FRAME_SYNC kind: {kind}")
        if not held_ports <= expected:
            raise RuntimeError(
                f"held controller ports are not physical ports: {sorted(held_ports)}"
            )
        self._frame_sync_sequence += 1
        sequence = self._frame_sync_sequence
        marker = f"FRAME_SYNC {sequence} {kind}\n"
        for port in reversed(self._pipe_device_order):
            raw = self.raw_controllers[port]
            raw._write(marker)
            self._frame_sync_markers[port] += 1
            raw.flush()
            self._group_flushes[port] += 1
            if port in held_ports:
                self._held_group_flushes[port] += 1
        self._group_commits += 1
        self._frame_sync_last_completed_sequence = sequence
        self._frame_sync_kind_commits[kind] += 1
        self._group_commit_reasons[reason] = (
            self._group_commit_reasons.get(reason, 0) + 1
        )

    def step(self) -> Any | None:
        """Read one state after suppressing libmelee's duplicate preamble flushes."""
        if not self._installed or not self._registration_is_exact():
            raise RuntimeError("controller proxy registration changed unexpectedly")
        if self._prime_calls != 1:
            raise RuntimeError("controller transport must be primed before stepping")
        if self._inside_console_step:
            raise RuntimeError("nested Console.step calls are forbidden")
        if self._active_boundary is not None:
            raise RuntimeError("cannot step with an uncommitted controller boundary")
        if self._internal_pending or self._unscoped_pending:
            raise RuntimeError("cannot step with an incomplete controller flush group")
        self._step_calls += 1
        self._seen_inside_step = set()
        self._internal_pending = set()
        self._inside_console_step = True
        try:
            gamestate = self.console.step()
            if gamestate is None:
                self._none_step_results += 1
            return gamestate
        finally:
            self._inside_console_step = False
            missing_preamble_ports = sorted(
                set(self.controllers) - self._seen_inside_step
            )
            if missing_preamble_ports:
                raise RuntimeError(
                    f"Console.step did not issue exactly one suppressible preamble flush for controller ports {missing_preamble_ports}"
                )
            if self._internal_pending:
                pending = sorted(self._internal_pending)
                self._internal_pending = set()
                raise RuntimeError(
                    f"Console.step ended with an incomplete internal controller flush group: {pending}"
                )
            self._seen_inside_step = set()

    def _flush(self, port: int) -> None:
        if not self._inside_console_step:
            if self._active_boundary is not None:
                native = cast(set[int], self._active_boundary["native"])
                scheduled = cast(
                    dict[int, dict[str, Any]], self._active_boundary["scheduled"]
                )
                if port in scheduled:
                    raise RuntimeError(
                        f"controller port {port} was scheduled and also requested a native flush"
                    )
                if port in native:
                    raise RuntimeError(
                        f"controller port {port} requested more than one native flush in one boundary"
                    )
                native.add(port)
                self._native_boundary_requests[port] += 1
                return
            if port in self._unscoped_pending:
                raise RuntimeError(
                    f"controller port {port} issued duplicate unscoped flushes before its peer"
                )
            self._unscoped_pending.add(port)
            if self._unscoped_pending == set(self.controllers):
                self._forward_group(
                    reason="console-unscoped", kind="unscoped", held_ports=set()
                )
                for grouped_port in self.controllers:
                    self._unscoped_outside_forwarded[grouped_port] += 1
                self._unscoped_pending = set()
            return
        if port not in self._seen_inside_step:
            self._seen_inside_step.add(port)
            self._preamble_suppressed[port] += 1
            return
        if port in self._internal_pending:
            raise RuntimeError(
                f"controller port {port} issued duplicate internal flushes before its peer"
            )
        self._internal_pending.add(port)
        self._internal_requests[port] += 1
        if self._internal_pending == set(self.controllers):
            self._forward_group(
                reason="console-internal", kind="internal", held_ports=set()
            )
            self._internal_pending = set()

    def audit_record(self) -> dict[str, Any]:
        if self._sealed_audit_record is not None:
            return copy.deepcopy(self._sealed_audit_record)
        return {
            "schema_version": self.SCHEMA_VERSION,
            "installed": self._installed,
            "console_registration_exact": self._registration_is_exact(),
            "registration_order": list(self._registration_order),
            "dolphin_pipe_device_order": list(self._pipe_device_order),
            "dolphin_pipe_device_order_source": self._pipe_device_order_source,
            "dolphin_pipe_device_order_verified": self._pipe_device_order_source
            == self.VERIFIED_PIPE_DEVICE_ORDER_SOURCE
            and len(self._pipe_device_order) == len(self.controllers)
            and (set(self._pipe_device_order) == set(self.controllers)),
            "global_gate_semantics": "the pinned tournament emulator combines both strict human pipes by matching FRAME_SYNC sequence and kind before advancing one frame",
            "frame_sync": {
                "protocol": "FRAME_SYNC <positive-shared-sequence> <kind> before every raw FLUSH",
                "allowed_kinds": sorted(self.FRAME_SYNC_KINDS),
                "last_allocated_sequence": self._frame_sync_sequence,
                "last_completed_sequence": self._frame_sync_last_completed_sequence,
                "kind_commits": dict(sorted(self._frame_sync_kind_commits.items())),
                "markers_written": {
                    f"p{port}": self._frame_sync_markers[port]
                    for port in sorted(self.controllers)
                },
            },
            "prime_calls": self._prime_calls,
            "step_calls": self._step_calls,
            "none_step_results": self._none_step_results,
            "boundaries_started": self._boundaries_started,
            "boundaries_committed": self._boundaries_committed,
            "group_commits": self._group_commits,
            "group_commit_reasons": dict(sorted(self._group_commit_reasons.items())),
            "active_boundary": None
            if self._active_boundary is None
            else {
                "reason": self._active_boundary["reason"],
                "game_frame": self._active_boundary["game_frame"],
                "native_ports": sorted(cast(set[int], self._active_boundary["native"])),
                "scheduled_ports": sorted(
                    cast(dict[int, dict[str, Any]], self._active_boundary["scheduled"])
                ),
            },
            "pending_internal_ports": sorted(self._internal_pending),
            "pending_unscoped_ports": sorted(self._unscoped_pending),
            "ports": {
                f"p{port}": {
                    "native_boundary_flush_requests": self._native_boundary_requests[
                        port
                    ],
                    "unscoped_outside_flushes_forwarded": self._unscoped_outside_forwarded[
                        port
                    ],
                    "scheduled_boundaries_created": self._scheduled_created[port],
                    "scheduled_boundaries_consumed_at_group_commit": self._scheduled_consumed[
                        port
                    ],
                    "step_preamble_duplicate_flushes_suppressed": self._preamble_suppressed[
                        port
                    ],
                    "later_internal_flush_requests": self._internal_requests[port],
                    "ordered_group_flushes_forwarded": self._group_flushes[port],
                    "held_or_queued_group_flushes": self._held_group_flushes[port],
                }
                for port in sorted(self.controllers)
            },
            "ownership": {
                "native_explicit_flush": "defer inside an explicit two-port policy boundary",
                "queued_policy_flush": "join the same two-port boundary without policy dispatch",
                "frame_sync_marker": "FRAME_SYNC sequence kind before every physical FLUSH",
                "group_commit_order": "reverse Dolphin pipe-device scan order",
                "step_preamble_duplicate": "suppress the first per-port flush in every Console.step",
                "game_start_or_rollback_flush": "pair and group-commit every later in-step flush",
                "none_state_poll": "do not synthesize or resend a controller command",
            },
        }

    def gate_checks(self) -> dict[str, bool]:
        if self._sealed_gate_checks is not None:
            return dict(self._sealed_gate_checks)
        record = self.audit_record()
        every_schedule_consumed_once = all(
            (
                self._scheduled_created[port] == self._scheduled_consumed[port]
                for port in self.controllers
            )
        )
        one_preamble_decision_per_step = all(
            (
                self._preamble_suppressed[port] == self._step_calls
                for port in self.controllers
            )
        )
        every_group_flushed_both_ports = all(
            (
                self._group_flushes[port] == self._group_commits
                for port in self.controllers
            )
        )
        frame_sync_protocol_exact = (
            self._frame_sync_last_completed_sequence == self._group_commits
            and self._frame_sync_sequence == self._group_commits
            and (sum(self._frame_sync_kind_commits.values()) == self._group_commits)
            and all(
                (
                    self._frame_sync_markers[port] == self._group_commits
                    for port in self.controllers
                )
            )
        )
        return {
            "controller_pipe_lockstep_installed": record["installed"] is True,
            "controller_pipe_registration_exact": record["console_registration_exact"]
            is True,
            "controller_pipe_primed_exactly_once": record["prime_calls"] == 1,
            "controller_pipe_device_order_complete": len(self._pipe_device_order)
            == len(self.controllers)
            and set(self._pipe_device_order) == set(self.controllers)
            and (record["dolphin_pipe_device_order_verified"] is True),
            "controller_pipe_no_pending_boundaries_after_shutdown": record[
                "active_boundary"
            ]
            is None
            and (not record["pending_internal_ports"])
            and (not record["pending_unscoped_ports"]),
            "controller_pipe_every_boundary_committed_exactly_once": self._boundaries_started
            == self._boundaries_committed,
            "controller_pipe_every_scheduled_boundary_consumed_exactly_once": every_schedule_consumed_once,
            "controller_pipe_every_group_flushes_both_ports": every_group_flushed_both_ports
            and frame_sync_protocol_exact,
            "controller_pipe_one_preamble_decision_per_step": one_preamble_decision_per_step,
        }

    def seal_benchmark_audit(self) -> tuple[dict[str, Any], dict[str, bool]]:
        """Freeze game-scope evidence before SIGINT input-drain transactions."""
        if self._sealed_audit_record is None:
            record = self.audit_record()
            checks = self.gate_checks()
            record["benchmark_scope"] = {
                "sealed_before_shutdown_drain": True,
                "teardown_transactions_excluded": True,
                "last_in_scope_group_commit": record["group_commits"],
                "last_in_scope_frame_sync_sequence": record["frame_sync"][
                    "last_completed_sequence"
                ],
            }
            self._sealed_audit_record = copy.deepcopy(record)
            self._sealed_gate_checks = dict(checks)
        return (self.audit_record(), self.gate_checks())


def _assert_udp_port_available(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as error:
            raise RuntimeError(
                f"Slippi UDP port {port} is already in use; close the stale emulator first"
            ) from error


def _validate_saved_replay(
    path: Path,
    *,
    required_first_frame: int | None = None,
    required_last_frame: int | None = None,
) -> dict[str, Any]:
    if (required_first_frame is None) != (required_last_frame is None):
        raise ValueError("saved replay coverage requires both trace frame bounds")
    if (
        required_first_frame is not None
        and required_last_frame is not None
        and (required_last_frame < required_first_frame)
    ):
        raise ValueError("saved replay trace frame bounds are reversed")
    import melee

    console = melee.Console(path=str(path), is_dolphin=False, allow_old_version=True)
    parsed_states = 0
    first_parsed_frame: int | None = None
    last_parsed_frame: int | None = None
    previous_parsed_frame: int | None = None
    parsed_frames_strictly_consecutive = True
    parsed_frame_delta_counts: dict[int, int] = {}
    required_frames_seen: set[int] = set()
    parse_error: str | None = None
    try:
        if not console.connect():
            raise RuntimeError("libmelee did not connect to saved replay")
        while (gamestate := console.step()) is not None:
            frame = int(gamestate.frame)
            parsed_states += 1
            if first_parsed_frame is None:
                first_parsed_frame = frame
            if previous_parsed_frame is not None:
                delta = frame - previous_parsed_frame
                parsed_frame_delta_counts[delta] = (
                    parsed_frame_delta_counts.get(delta, 0) + 1
                )
                parsed_frames_strictly_consecutive = (
                    parsed_frames_strictly_consecutive and delta == 1
                )
            previous_parsed_frame = frame
            last_parsed_frame = frame
            if (
                required_first_frame is not None
                and required_last_frame is not None
                and (required_first_frame <= frame <= required_last_frame)
            ):
                required_frames_seen.add(frame)
    except Exception as error:
        parse_error = f"{type(error).__name__}: {error}"
    finally:
        with suppress(AssertionError):
            console.stop()
    required_trace_coverage: dict[str, Any] | None = None
    if required_first_frame is not None and required_last_frame is not None:
        required_frame_count = required_last_frame - required_first_frame + 1
        missing_required_frame_count = required_frame_count - len(required_frames_seen)
        first_missing_required_frames: list[int] = []
        if missing_required_frame_count:
            for frame in range(required_first_frame, required_last_frame + 1):
                if frame not in required_frames_seen:
                    first_missing_required_frames.append(frame)
                    if len(first_missing_required_frames) == 10:
                        break
        required_trace_coverage = {
            "first_frame": required_first_frame,
            "last_frame": required_last_frame,
            "required_frame_count": required_frame_count,
            "covered_frame_count": len(required_frames_seen),
            "missing_frame_count": missing_required_frame_count,
            "first_missing_frames": first_missing_required_frames,
            "complete": parse_error is None and missing_required_frame_count == 0,
        }
    parseable = parse_error is None and parsed_states > 0
    return {
        "libmelee_parseable": parseable,
        "parsed_game_states": parsed_states,
        "first_parsed_frame": first_parsed_frame,
        "last_parsed_frame": last_parsed_frame,
        "parsed_frames_strictly_consecutive": parsed_frames_strictly_consecutive,
        "parsed_frame_delta_counts": {
            str(delta): count
            for delta, count in sorted(parsed_frame_delta_counts.items())
        },
        "required_trace_coverage": required_trace_coverage,
        "error": parse_error
        if parse_error is not None
        else None
        if parseable
        else "replay contained no parsed game states",
    }


def _require_exact_player_ports(gamestate: Any, game_frame: int) -> None:
    observed_ports = sorted((int(port) for port in gamestate.players))
    if observed_ports != [1, 2]:
        raise RuntimeError(
            f"policy frame {game_frame} must contain exactly player ports [1, 2], got {observed_ports}"
        )


def _classify_trace_covering_replays(
    replay_records: list[dict[str, Any]], *, sudden_death_transition_observed: bool
) -> None:
    candidates: list[dict[str, Any]] = []
    for record in replay_records:
        validation = record.get("validation")
        required_trace = (
            validation.get("required_trace_coverage")
            if isinstance(validation, dict)
            else None
        )
        if (
            isinstance(validation, dict)
            and validation.get("libmelee_parseable") is True
            and isinstance(required_trace, dict)
            and (required_trace.get("complete") is True)
        ):
            candidates.append(record)
    for record in replay_records:
        selected = len(candidates) == 1 and record is candidates[0]
        record["tournament_result_replay"] = selected
        record["role"] = (
            "base-game-result"
            if selected
            else "sudden-death-transition-auxiliary"
            if sudden_death_transition_observed
            else "unexpected-auxiliary"
        )


def _legacy_replay_gate_checks(
    replay_records: list[dict[str, Any]],
    *,
    sudden_death_transition_observed: bool = False,
) -> dict[str, bool]:
    if replay_records and (
        not any(("tournament_result_replay" in record for record in replay_records))
    ):
        _classify_trace_covering_replays(
            replay_records,
            sudden_death_transition_observed=sudden_death_transition_observed,
        )
    selected = [
        record
        for record in replay_records
        if record.get("tournament_result_replay") is True
    ]
    single_replay = len(selected) == 1
    validation: dict[str, Any] = {}
    if single_replay:
        candidate = selected[0].get("validation")
        if isinstance(candidate, dict):
            validation = candidate
    required_trace = validation.get("required_trace_coverage")
    trace_complete = (
        isinstance(required_trace, dict) and required_trace.get("complete") is True
    )
    parseable = validation.get("libmelee_parseable") is True
    return {
        "exactly_one_tournament_result_replay": single_replay,
        "no_unexpected_auxiliary_replays": len(replay_records) == 1
        or (sudden_death_transition_observed and len(replay_records) == 2),
        "parseable_current_run_replay": single_replay and parseable,
        "current_run_replay_covers_every_policy_frame": single_replay
        and parseable
        and trace_complete,
    }


def _audit_selected_controller_boundary(
    trace_path: Path,
    replay_paths: list[Path],
    replay_records: list[dict[str, Any]],
    project_root: Path,
    *,
    lag_frames: int,
) -> dict[str, Any]:
    """Run the full replay-backed gate in the parser-pinned E010 environment."""
    from melee_policy.integration.slippi_match import _unavailable_controller_boundary

    selected_indexes = [
        index
        for index, record in enumerate(replay_records)
        if record.get("tournament_result_replay") is True
    ]
    if len(selected_indexes) != 1:
        return _unavailable_controller_boundary(
            trace_path,
            project_root,
            "controller boundary audit requires exactly one replay covering the base-game trace",
        )
    index = selected_indexes[0]
    if index >= len(replay_paths):
        return _unavailable_controller_boundary(
            trace_path,
            project_root,
            "selected replay record has no corresponding replay path",
        )
    interpreter = project_root / ".e010-env" / "bin" / "python"
    dependency_lock = project_root / "requirements-e010.lock"
    if not interpreter.is_file() or not dependency_lock.is_file():
        missing = interpreter if not interpreter.is_file() else dependency_lock
        return _unavailable_controller_boundary(
            trace_path,
            project_root,
            f"pinned controller-audit dependency is missing: {_display_path(missing, project_root)}",
        )
    marker = "MELEE_POLICY_CONTROLLER_AUDIT_JSON="
    audit_program = f"import json,sys\nfrom pathlib import Path\nsys.path.insert(0, sys.argv[1])\nfrom melee_policy.integration.slippi_match import _audit_controller_boundary_candidate\nresult = _audit_controller_boundary_candidate(\n    Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), lag_frames=int(sys.argv[5])\n)\nprint({marker!r} + json.dumps(result, sort_keys=True))\n"
    command = [
        str(interpreter),
        "-I",
        "-c",
        audit_program,
        str(project_root / "src"),
        str(trace_path),
        str(replay_paths[index]),
        str(project_root),
        str(lag_frames),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=300.0,
        )
    except Exception as error:
        return _unavailable_controller_boundary(
            trace_path,
            project_root,
            f"pinned E010 controller audit failed to execute: {type(error).__name__}: {error}",
        )
    payload_lines = [
        line.removeprefix(marker)
        for line in completed.stdout.splitlines()
        if line.startswith(marker)
    ]
    if completed.returncode != 0 or len(payload_lines) != 1:
        stderr_tail = completed.stderr[-2000:].strip()
        reason = f"pinned E010 controller audit exited {completed.returncode}; JSON records={len(payload_lines)}"
        if stderr_tail:
            reason += f"; stderr tail: {stderr_tail}"
        return _unavailable_controller_boundary(trace_path, project_root, reason)
    try:
        audit = json.loads(payload_lines[0])
    except json.JSONDecodeError as error:
        return _unavailable_controller_boundary(
            trace_path,
            project_root,
            f"pinned E010 controller audit emitted invalid JSON: {error}",
        )
    if not isinstance(audit, dict):
        return _unavailable_controller_boundary(
            trace_path,
            project_root,
            "pinned E010 controller audit did not emit an object",
        )
    audit["auditor"] = {
        "execution": "isolated pinned E010 subprocess",
        "interpreter": _file_identity(interpreter, project_root),
        "dependency_lock": _file_identity(dependency_lock, project_root),
        "lag_frames": lag_frames,
    }
    return cast(dict[str, Any], audit)


def _controller_boundary_lag_candidate_summary(
    lag_frames: int, audit: dict[str, Any]
) -> dict[str, Any]:
    """Keep alternate-lag evidence compact while retaining every hard-gate result."""
    slots = cast(dict[str, Any], audit.get("slots", {}))
    mismatch_counts: dict[str, dict[str, int]] = {}
    for slot_name in ("p1", "p2"):
        slot = cast(dict[str, Any], slots.get(slot_name, {}))
        digital = cast(dict[str, Any], slot.get("digital_buttons", {}))
        physical = cast(dict[str, Any], digital.get("physical", {}))
        processed = cast(
            dict[str, Any], digital.get("processed_upstream_observation", {})
        )
        raw_main = cast(dict[str, Any], slot.get("intended_raw_main_stick", {}))
        c_stick = cast(dict[str, Any], slot.get("processed_c_stick", {}))
        shoulders = cast(dict[str, Any], slot.get("physical_analog_shoulders", {}))
        mismatch_counts[slot_name] = {
            "physical_button_frames": int(physical.get("mismatch_frames", 0)),
            "processed_button_frames": int(processed.get("mismatch_frames", 0)),
            "raw_main_components": int(raw_main.get("mismatch_components", 0)),
            "processed_c_stick_components": int(c_stick.get("mismatch_components", 0)),
            "physical_analog_shoulder_components": int(
                shoulders.get("mismatch_components", 0)
            ),
        }
    gate = cast(dict[str, Any], audit.get("gate", {}))
    return {
        "lag_frames": lag_frames,
        "decision": gate.get("decision"),
        "checks": dict(cast(dict[str, Any], gate.get("checks", {}))),
        "mismatch_counts": mismatch_counts,
    }


def _audit_selected_controller_boundary_unique_lag(
    trace_path: Path,
    replay_paths: list[Path],
    replay_records: list[dict[str, Any]],
    project_root: Path,
    *,
    lag_frames_candidates: tuple[int, ...],
) -> dict[str, Any]:
    """Accept only one full-trace-exact lag candidate and fail on ambiguity."""
    candidates = tuple(dict.fromkeys(lag_frames_candidates))
    if not candidates:
        raise ValueError(
            "controller boundary lag selection requires at least one candidate"
        )
    if len(candidates) != len(lag_frames_candidates):
        raise ValueError("controller boundary lag candidates must be unique")
    audits = {
        lag_frames: _audit_selected_controller_boundary(
            trace_path,
            replay_paths,
            replay_records,
            project_root,
            lag_frames=lag_frames,
        )
        for lag_frames in candidates
    }
    passing_lags = [
        lag_frames
        for lag_frames in candidates
        if cast(dict[str, Any], audits[lag_frames].get("gate", {})).get("decision")
        == "pass"
    ]
    selected_lag = passing_lags[0] if len(passing_lags) == 1 else None
    result_lag = selected_lag if selected_lag is not None else candidates[0]
    result = audits[result_lag]
    result["lag_selection"] = {
        "mode": "unique-full-trace-exact-match",
        "allowed_lag_frames": list(candidates),
        "passing_lag_frames": passing_lags,
        "selected_lag_frames": selected_lag,
        "unique_exact_lag_selected": selected_lag is not None,
        "failure_reason": None
        if selected_lag is not None
        else "no candidate passed every controller boundary check"
        if not passing_lags
        else "multiple candidates passed, so controller timing is ambiguous",
        "candidates": [
            _controller_boundary_lag_candidate_summary(lag_frames, audits[lag_frames])
            for lag_frames in candidates
        ],
        "safety_rule": "A candidate passes only with complete causal overlap and zero hard-gated button, raw-main, processed-C-stick, and physical-shoulder mismatches on both ports. Zero or multiple passing candidates fail the game.",
    }
    result_gate = cast(dict[str, Any], result.get("gate", {}))
    result_checks = cast(dict[str, bool], result_gate.get("checks", {}))
    result_checks["unique_exact_lag_selected"] = selected_lag is not None
    result_gate["checks"] = result_checks
    result_gate["decision"] = "pass" if all(result_checks.values()) else "fail"
    result["gate"] = result_gate
    return result


def _controller_boundary_gate_checks(boundary: dict[str, Any]) -> dict[str, bool]:
    gate = cast(dict[str, Any], boundary.get("gate", {}))
    checks = cast(dict[str, Any], gate.get("checks", {}))
    return {
        "controller_boundary_gate_pass": gate.get("decision") == "pass",
        "controller_boundary_physical_buttons_exact": checks.get(
            "both_slots_physical_buttons_exact"
        )
        is True,
        "controller_boundary_processed_buttons_exact": checks.get(
            "both_slots_processed_upstream_buttons_exact"
        )
        is True,
        "controller_boundary_raw_main_exact": checks.get(
            "both_slots_intended_raw_main_stick_exact"
        )
        is True,
        "controller_boundary_processed_c_stick_exact": checks.get(
            "both_slots_processed_c_stick_within_tolerance"
        )
        is True,
        "controller_boundary_physical_shoulders_exact": checks.get(
            "both_slots_physical_analog_shoulders_within_tolerance"
        )
        is True,
    }


@dataclass(frozen=True)
class PlayRequest:
    player_1_model: str = "mimic"
    player_2_model: str = "mimic"
    player_1_character: str = "FOX"
    player_2_character: str = "FOX"
    stage: str = "BATTLEFIELD"
    player_1_checkpoint: Path | None = None
    player_2_checkpoint: Path | None = None
    player_1_assets: Path | None = None
    player_2_assets: Path | None = None
    max_game_frames: int | None = None
    inference_mode: str | None = None
    artifact_label: str | None = None
    seed: int | None = None
    require_natural_end: bool = False
    save_slp: bool = False
    save_video: bool = False

    def validate(self) -> None:
        models = (self.player_1_model.lower(), self.player_2_model.lower())
        if any((model not in ("mimic",) for model in models)):
            raise ValueError("this runner requires 'mimic' in each slot")
        for label, character in (
            ("player 1", self.player_1_character),
            ("player 2", self.player_2_character),
        ):
            if not character or character != character.upper():
                raise ValueError(
                    f"{label} character must be an uppercase libmelee enum name"
                )
            if character not in LAUNCHABLE_CHARACTERS:
                raise ValueError(
                    f"{label} character is not a launchable standard character: {character}"
                )
        if not self.stage or self.stage != self.stage.upper():
            raise ValueError("stage must be an uppercase libmelee enum name")
        if self.stage not in LEGAL_STAGES:
            raise ValueError(f"stage must be one of {sorted(LEGAL_STAGES)}")
        if self.max_game_frames is not None and self.max_game_frames < 1:
            raise ValueError("max_game_frames must be positive")
        if self.seed is not None:
            _validate_evaluation_seed(self.seed)
        if self.inference_mode not in (
            None,
            "synchronous-concurrent",
            "asynchronous-latest",
        ):
            raise ValueError(f"unsupported inference mode: {self.inference_mode}")
        if not isinstance(self.save_slp, bool) or not isinstance(self.save_video, bool):
            raise TypeError("save_slp and save_video must be booleans")
        if self.artifact_label is not None and (
            not self.artifact_label
            or any(
                (
                    character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
                    for character in self.artifact_label
                )
            )
        ):
            raise ValueError(
                "artifact_label must contain only lowercase letters, digits, underscores, or hyphens"
            )

    def port_for(self, model_name: str) -> int:
        matches = [
            port
            for port, model in ((1, self.player_1_model), (2, self.player_2_model))
            if model.lower() == model_name
        ]
        if len(matches) != 1:
            raise ValueError(
                f"request does not contain exactly one {model_name!r} side"
            )
        return matches[0]

    def checkpoint_for(self, model_name: str) -> Path | None:
        return (
            self.player_1_checkpoint
            if self.port_for(model_name) == 1
            else self.player_2_checkpoint
        )

    def assets_for(self, model_name: str) -> Path | None:
        return (
            self.player_1_assets
            if self.port_for(model_name) == 1
            else self.player_2_assets
        )

    def character_for(self, model_name: str) -> str:
        return (
            self.player_1_character
            if self.port_for(model_name) == 1
            else self.player_2_character
        )


def _validate_first_game_context(
    gamestate: Any, request: PlayRequest
) -> dict[str, Any]:
    """Bind the policy stream to the requested first gameplay context."""
    observed_ports = sorted((int(port) for port in gamestate.players))
    observed_characters = {
        f"p{port}": gamestate.players[port].character.name
        for port in (1, 2)
        if port in gamestate.players
    }
    observed_costumes = {
        f"p{port}": int(gamestate.players[port].costume)
        for port in (1, 2)
        if port in gamestate.players
    }
    observed_stage = gamestate.stage.name
    checks = {
        "first_frame": int(gamestate.frame) == FIRST_GAMEPLAY_FRAME,
        "two_standard_ports": observed_ports == [1, 2],
        "stage": observed_stage == request.stage,
        "player_1_character": observed_characters.get("p1")
        == request.player_1_character,
        "player_2_character": observed_characters.get("p2")
        == request.player_2_character,
    }
    result = {
        "required_first_frame": FIRST_GAMEPLAY_FRAME,
        "observed_first_frame": int(gamestate.frame),
        "observed_ports": observed_ports,
        "requested_stage": request.stage,
        "observed_stage": observed_stage,
        "requested_characters": {
            "p1": request.player_1_character,
            "p2": request.player_2_character,
        },
        "observed_characters": observed_characters,
        "observed_costumes": observed_costumes,
        "checks": checks,
        "passed": all(checks.values()),
    }
    if not result["passed"]:
        raise RuntimeError(f"first gameplay context mismatch: {result}")
    return result


def _load_same_family_runtimes(
    config: dict[str, Any], project_root: Path, request: PlayRequest
) -> dict[int, MimicRuntime]:
    """Load one isolated native runtime per physical slot, honoring each override."""
    model_name = request.player_1_model.lower()
    if model_name != request.player_2_model.lower():
        raise ValueError("same-family runtime loader requires matching slot models")
    if model_name not in ("mimic",):
        raise ValueError(f"unsupported same-family model: {model_name}")
    runtimes: dict[int, MimicRuntime] = {}
    selections: dict[int, tuple[Path | None, Path | None]] = {}
    source_requirements: set[tuple[str, str, str]] = set()
    for port in (1, 2):
        checkpoint = (
            request.player_1_checkpoint if port == 1 else request.player_2_checkpoint
        )
        assets = request.player_1_assets if port == 1 else request.player_2_assets
        if checkpoint is not None and assets is None:
            assets = checkpoint.expanduser().resolve().parent
        selected_checkpoint = (
            (project_root / config["mimic"]["checkpoint"]).resolve()
            if checkpoint is None
            else checkpoint.expanduser().resolve()
        )
        selected_assets = (
            (project_root / config["mimic"]["asset_directory"]).resolve()
            if assets is None
            else assets.expanduser().resolve()
        )
        identity, _, _ = _validate_mimic_bundle(selected_checkpoint, selected_assets)
        source_requirements.add(
            (
                str(identity["source_repository"]),
                str(identity["source_revision"]),
                str(identity["source_directory"]),
            )
        )
        selections[port] = (checkpoint, assets)
    if len(source_requirements) != 1:
        raise RuntimeError(
            f"same-family MIMIC slots require one exact upstream source revision; received {sorted(source_requirements)}"
        )
    for port in (1, 2):
        checkpoint, assets = selections[port]
        runtimes[port] = load_mimic_runtime(
            config,
            project_root,
            checkpoint_override=checkpoint,
            asset_directory_override=assets,
        )
    if runtimes[1] is runtimes[2]:
        raise RuntimeError(
            "same-family slots unexpectedly share one policy runtime object"
        )
    return runtimes


def _run_same_family_console(
    config: dict[str, Any],
    project_root: Path,
    iso_path: Path,
    source_checks: dict[str, Any],
    runtimes: dict[int, MimicRuntime],
    request: PlayRequest,
    effective_seed: int,
) -> dict[str, Any]:
    """Run two independent instances of one native windowed policy family."""
    import melee

    model_name = request.player_1_model.lower()
    if model_name != request.player_2_model.lower() or model_name not in ("mimic",):
        raise ValueError("same-family runner requires matching MIMIC slots")
    if request.inference_mode not in (None, "synchronous-concurrent"):
        raise ValueError(
            "same-family matches require exact synchronous-concurrent inference"
        )
    characters = {1: request.player_1_character, 2: request.player_2_character}
    character_contracts: dict[str, dict[str, Any]] = {}
    for port in (1, 2):
        runtime = runtimes[port]
        requested = characters[port]
        checkpoint_character = cast(MimicRuntime, runtime).controlled_character
        match = requested == checkpoint_character
        character_contracts[f"p{port}"] = {
            "model": model_name,
            "requested_character": requested,
            "checkpoint_character": checkpoint_character,
            "match": match,
            "evidence": "MIMIC asset metadata.json melee_enum",
        }
        if not match:
            raise ValueError(
                f"{model_name.upper()} cannot control {requested} on port {port}"
            )
    output_directory = project_root / config["outputs"]["directory"]
    replay_directory = project_root / config["outputs"]["replay_directory"]
    trace_path = project_root / config["outputs"]["trace"]
    summary_path = project_root / config["outputs"]["summary"]
    _require_unused_artifact_label(output_directory, request.artifact_label)
    output_directory.mkdir(parents=True, exist_ok=True)
    replay_directory.mkdir(parents=True, exist_ok=True)
    existing_replays = {path.resolve() for path in replay_directory.rglob("*.slp")}
    emulator_application = _emulator_application_identity(config, project_root)
    _attested_emulator_release(emulator_application)
    slippi_port = int(config["emulator"]["slippi_port"])
    _assert_udp_port_available(slippi_port)
    console = _create_attested_dolphin_console(
        config,
        project_root,
        emulator_application,
        is_dolphin=True,
        tmp_home_directory=True,
        copy_home_directory=False,
        blocking_input=True,
        polling_mode=True,
        polling_timeout=1.0,
        online_delay=0,
        setup_gecko_codes=True,
        fullscreen=False,
        gfx_backend="",
        disable_audio=False,
        use_exi_inputs=False,
        enable_ffw=False,
        save_replays=True,
        replay_dir=str(replay_directory),
        replay_monthly_folders=False,
        slippi_port=slippi_port,
    )
    controllers: dict[int, Any] = {
        port: melee.Controller(
            console=console, port=port, type=melee.ControllerType.STANDARD
        )
        for port in (1, 2)
    }
    policies: dict[int, MimicLivePolicy] = {
        port: MimicLivePolicy(
            cast(MimicRuntime, runtimes[port]), port, evaluation_seed=effective_seed
        )
        for port in (1, 2)
    }
    workers = {
        port: _LatestInferenceWorker(
            f"{model_name.upper()}-P{port}", policies[port].infer_snapshot
        )
        for port in (1, 2)
    }
    menu_helpers = {1: melee.MenuHelper(), 2: melee.MenuHelper()}
    try:
        character_enums = {port: melee.Character[characters[port]] for port in (1, 2)}
        stage = melee.Stage[request.stage]
    except KeyError as error:
        for worker in workers.values():
            worker.close()
        raise ValueError(f"unknown character or stage enum: {error}") from error
    maximum_frames = (
        int(config["integration"]["max_game_frames"])
        if request.max_game_frames is None
        else request.max_game_frames
    )
    started_at = time.time()
    menu_timeout = float(config["integration"]["menu_timeout_seconds"])
    trace_stream: TextIO | None = None
    exception: BaseException | None = None
    in_game = False
    natural_game_end = False
    sudden_death_transition_observed = False
    termination = "not-started"
    shutdown_method = "not-started"
    processed_frames = 0
    first_game_frame: int | None = None
    last_game_frame: int | None = None
    previous_game_frame: int | None = None
    frame_delta_counts: dict[int, int] = {}
    first_context_validation: dict[str, Any] | None = None
    finite_outputs = True
    inference_counts = {1: 0, 2: 0}
    dispatch_counts = {1: 0, 2: 0}
    dispatch_frames: dict[int, list[int]] = {1: [], 2: []}
    observe_counts = {1: 0, 2: 0}
    command_ages: dict[int, list[int]] = {1: [], 2: []}
    warmup_frames: dict[int, list[int]] = {1: [], 2: []}
    commands: dict[int, dict[str, Any]] = {
        port: _neutral_mimic_command() for port in (1, 2)
    }
    pressed: dict[int, list[str]] = {1: [], 2: []}
    has_command = {1: False, 2: False}
    transport: _ControllerPipeLockstep | None = None
    menu_transport_flushes = {1: 0, 2: 0}
    warmup_transport_schedule_frames: dict[int, list[int]] = {1: [], 2: []}
    stocks = {"p1": 4, "p2": 4}
    no_frame_watchdog = InGameNoFrameWatchdog()

    def interrupt(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGINT, interrupt)
    try:
        trace_stream = trace_path.open("w", encoding="utf-8")
        if not _launch_and_connect_attested_dolphin(console, iso_path):
            raise RuntimeError("libmelee could not connect to Slippi Dolphin")
        if not all((controller.connect() for controller in controllers.values())):
            raise RuntimeError("libmelee could not connect both virtual controllers")
        transport = _ControllerPipeLockstep.install(console, controllers)
        controllers = dict(transport.controllers)
        transport.prime()
        while processed_frames < maximum_frames:
            gamestate = transport.step()
            no_frame_watchdog.observe_step_result(gamestate, gameplay_started=in_game)
            if gamestate is None:
                if not in_game and time.time() - started_at > menu_timeout:
                    raise TimeoutError(
                        "Slippi did not provide a menu or game state before timeout"
                    )
                continue
            if in_game and gamestate.menu_state == melee.Menu.SUDDEN_DEATH:
                natural_game_end = True
                sudden_death_transition_observed = True
                termination = "natural-game-end"
                break
            if gamestate.menu_state not in (
                melee.Menu.IN_GAME,
                melee.Menu.SUDDEN_DEATH,
            ):
                if in_game:
                    natural_game_end = True
                    termination = "natural-game-end"
                    break
                if time.time() - started_at > menu_timeout:
                    raise TimeoutError(
                        "automatic menu navigation did not start a match before timeout"
                    )
                transport.begin_boundary(reason="menu", game_frame=None)
                for port in (1, 2):
                    menu_helpers[port].menu_helper_simple(
                        gamestate,
                        controllers[port],
                        character_enums[port],
                        stage,
                        cpu_level=0,
                        autostart=port == 2,
                        frozen_stadium=True,
                    )
                    controllers[port].flush()
                    menu_transport_flushes[port] += 1
                transport.commit_boundary()
                continue
            in_game = True
            game_frame = int(gamestate.frame)
            if first_game_frame is None:
                first_game_frame = game_frame
                first_context_validation = _validate_first_game_context(
                    gamestate, request
                )
            if previous_game_frame is not None:
                delta = game_frame - previous_game_frame
                frame_delta_counts[delta] = frame_delta_counts.get(delta, 0) + 1
                if delta != 1:
                    raise RuntimeError(
                        f"rendered policy frame stream is not exact: {previous_game_frame} to {game_frame}"
                    )
            previous_game_frame = game_frame
            last_game_frame = game_frame
            _require_exact_player_ports(gamestate, game_frame)
            ready: dict[int, bool] = {}
            for port in (1, 2):
                ready[port] = policies[port].observe(gamestate)
                observe_counts[port] += 1
                if not ready[port]:
                    warmup_frames[port].append(game_frame)
                if ready[port]:
                    workers[port].submit(game_frame, policies[port].snapshot())
            results: dict[int, tuple[int, Any, float] | None] = {
                port: workers[port].wait_completed(game_frame) if ready[port] else None
                for port in (1, 2)
            }
            transport.begin_boundary(reason="gameplay", game_frame=game_frame)
            slot_payloads: dict[str, dict[str, Any]] = {}
            for port in (1, 2):
                inference_result = results[port]
                source_frame: int | None = None
                prediction: Any = None
                if inference_result is not None:
                    source_frame, prediction, _duration = inference_result
                    if not _finite_prediction(prediction):
                        finite_outputs = False
                        raise RuntimeError(
                            f"non-finite {model_name} output at game frame {game_frame}"
                        )
                    inference_counts[port] += 1
                    command_ages[port].append(game_frame - source_frame)
                dispatched = False
                mimic_policy = cast(MimicLivePolicy, policies[port])
                mimic_runtime = cast(MimicRuntime, runtimes[port])
                if inference_result is not None:
                    commands[port], pressed[port], _button_names = (
                        mimic_policy.decode_and_press(
                            controllers[port],
                            prediction,
                            mimic_runtime.state.prev_sent,
                            temperature=float(config["mimic"]["temperature"]),
                            top_k=int(config["mimic"]["top_k"]),
                            top_p=float(config["mimic"]["top_p"]),
                        )
                    )
                else:
                    _send_mimic_controller_inputs(controllers[port], commands[port])
                mimic_policy.record_decoded_command(game_frame, commands[port])
                dispatched = True
                if dispatched:
                    dispatch_counts[port] += 1
                    dispatch_frames[port].append(game_frame)
                player = gamestate.players[port]
                stocks[f"p{port}"] = int(player.stock)
                command = {**commands[port], "pressed": list(pressed[port])}
                inference: dict[str, Any] = {
                    "called": inference_result is not None,
                    "source_frame": source_frame,
                    "command_age_frames": None
                    if source_frame is None
                    else game_frame - source_frame,
                }
                inference["previous_executed_controller_frame"] = cast(
                    MimicLivePolicy, policies[port]
                ).previous_executed_frame
                slot_payloads[f"p{port}"] = {
                    "port": port,
                    "model": model_name,
                    "requested_character": characters[port],
                    "state": {
                        "character": player.character.name,
                        "action": int(player.action.value),
                        "stocks": int(player.stock),
                        "percent": float(player.percent),
                        "position": [
                            float(player.position.x),
                            float(player.position.y),
                        ],
                    },
                    "command": command,
                    "inference": inference,
                    "controller_dispatch": {
                        "called": dispatched,
                        "sender": f"native {model_name.upper()} complete-frame sender",
                        "explicit_flush": dispatched,
                        "implicit_console_step_flush_suppressed": True,
                        "held_transport_flush_without_dispatch": not dispatched,
                    },
                }
            transport.commit_boundary()
            trace_stream.write(
                json.dumps(
                    {"game_frame": game_frame, "slots": slot_payloads}, sort_keys=True
                )
                + "\n"
            )
            trace_stream.flush()
            processed_frames += 1
            if has_decisive_zero_stock(gamestate, (1, 2)):
                natural_game_end = True
                termination = "natural-game-end"
                break
        if in_game and processed_frames >= maximum_frames:
            termination = "frame-limit-graceful-stop"
    except BaseException as caught:
        exception = caught
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        for worker in workers.values():
            try:
                worker.close()
            except BaseException as worker_error:
                if exception is None:
                    exception = worker_error
            if worker.failure is not None and exception is None:
                exception = RuntimeError(
                    f"{worker.name} inference failed: {worker.failure}"
                )
        if trace_stream is not None:
            trace_stream.close()
        shutdown_method = _stop_console(
            console, float(config["integration"]["replay_finalize_timeout_seconds"])
        )
    replay_paths = sorted(
        (
            path
            for path in replay_directory.rglob("*.slp")
            if path.is_file() and path.resolve() not in existing_replays
        )
    )
    replay_records = [
        {
            "path": _display_path(path, project_root),
            "byte_length": path.stat().st_size,
            "sha256": _sha256_file(path),
            "validation": _validate_saved_replay(
                path,
                required_first_frame=first_game_frame,
                required_last_frame=last_game_frame,
            ),
        }
        for path in replay_paths
    ]
    _classify_trace_covering_replays(
        replay_records,
        sudden_death_transition_observed=sudden_death_transition_observed,
    )
    replay_checks = _legacy_replay_gate_checks(
        replay_records,
        sudden_death_transition_observed=sudden_death_transition_observed,
    )
    expected_inferences = processed_frames
    expected_dispatches = expected_inferences
    expected_first_dispatch_frame = FIRST_GAMEPLAY_FRAME + 0
    controller_replay_lag_frames = SAME_FAMILY_CONTROLLER_REPLAY_LAG_FRAMES[model_name]
    controller_boundary = _audit_selected_controller_boundary(
        trace_path,
        replay_paths,
        replay_records,
        project_root,
        lag_frames=controller_replay_lag_frames,
    )
    slot_checks: dict[str, bool] = {}
    for port in (1, 2):
        slot_checks.update(
            {
                f"p{port}_observed_every_frame": observe_counts[port]
                == processed_frames,
                f"p{port}_inference_count_exact": inference_counts[port]
                == expected_inferences,
                f"p{port}_current_frame_inference_barrier": len(command_ages[port])
                == expected_inferences
                and all((age == 0 for age in command_ages[port])),
                f"p{port}_dispatch_count_exact": dispatch_counts[port]
                == expected_dispatches,
                f"p{port}_native_first_dispatch_frame_exact": bool(
                    dispatch_frames[port]
                )
                and dispatch_frames[port][0] == expected_first_dispatch_frame,
                f"p{port}_native_warmup_exact": warmup_frames[port] == [],
            }
        )
    transport_record = (
        transport.audit_record() if transport is not None else {"installed": False}
    )
    transport_checks = (
        transport.gate_checks()
        if transport is not None
        else {
            "controller_pipe_lockstep_installed": False,
            "controller_pipe_registration_exact": False,
            "controller_pipe_primed_exactly_once": False,
            "controller_pipe_device_order_complete": False,
            "controller_pipe_no_pending_boundaries_after_shutdown": False,
            "controller_pipe_every_boundary_committed_exactly_once": False,
            "controller_pipe_every_scheduled_boundary_consumed_exactly_once": False,
            "controller_pipe_every_group_flushes_both_ports": False,
            "controller_pipe_one_preamble_decision_per_step": False,
        }
    )
    gate_checks = {
        "entered_gameplay": in_game,
        "processed_policy_frames": processed_frames > 0,
        "independent_policy_runtime_objects": runtimes[1] is not runtimes[2],
        "first_gameplay_frame_minus_123": first_game_frame == FIRST_GAMEPLAY_FRAME,
        "first_game_context_exact": bool(
            first_context_validation is not None
            and first_context_validation.get("passed") is True
        ),
        "strict_consecutive_frame_order": all(
            (delta == 1 for delta in frame_delta_counts)
        ),
        "all_outputs_finite": finite_outputs,
        "both_policies_finish_current_step_before_advance": all(
            (
                len(command_ages[port]) == expected_inferences
                and all((age == 0 for age in command_ages[port]))
                for port in (1, 2)
            )
        ),
        "both_current_frame_inference_barriers_exact": all(
            (
                inference_counts[port] == expected_inferences
                and len(command_ages[port]) == expected_inferences
                and all((age == 0 for age in command_ages[port]))
                for port in (1, 2)
            )
        ),
        "same_family_controller_replay_lag_contract_exact": controller_replay_lag_frames
        == 1,
        **transport_checks,
        "warmup_transport_schedule_without_policy_dispatch_exact": all(
            (warmup_transport_schedule_frames[port] == [] for port in (1, 2))
        ),
        **_controller_boundary_gate_checks(controller_boundary),
        **slot_checks,
        **replay_checks,
        "natural_game_end_requirement_met": _natural_end_requirement_met(
            required=request.require_natural_end, observed=natural_game_end
        ),
    }
    runtime_environment = cast(dict[str, Any], config["_runtime_reproducibility"])[
        "runtime_environment"
    ]
    gate_checks["runtime_environment_lock_gate_passed"] = (
        cast(dict[str, Any], runtime_environment)["dependency_lock_validation"][
            "environment_lock_gate_passed"
        ]
        is True
    )
    if exception is None and (not all(gate_checks.values())):
        exception = RuntimeError(
            f"same-family {model_name} integration gate failed: {[name for name, passed in gate_checks.items() if not passed]}"
        )
    slots: dict[str, dict[str, Any]] = {}
    for port in (1, 2):
        runtime = runtimes[port]
        identity = {
            "classification": "released MIMIC checkpoint and native inference path",
            "source": {
                "repository_url": cast(MimicRuntime, runtime).bundle_identity[
                    "source_repository"
                ],
                "revision": cast(MimicRuntime, runtime).bundle_identity[
                    "source_revision"
                ],
            },
            "checkpoint": {
                "path": _display_path(
                    cast(MimicRuntime, runtime).checkpoint_path, project_root
                ),
                "sha256": cast(MimicRuntime, runtime).checkpoint_sha256,
                "byte_length": cast(MimicRuntime, runtime)
                .checkpoint_path.stat()
                .st_size,
            },
            "bundle_identity": cast(MimicRuntime, runtime).bundle_identity,
            "controlled_character": cast(MimicRuntime, runtime).controlled_character,
            "policy_rng": policies[port].policy_rng,
        }
        slots[f"p{port}"] = {
            "port": port,
            "model": model_name,
            "requested_character": characters[port],
            "character": characters[port],
            "character_contract": character_contracts[f"p{port}"],
            "identity": identity,
            "timing": {
                "inference_count": inference_counts[port],
                "dispatch_count": dispatch_counts[port],
                "mean_command_age_frames": sum(command_ages[port])
                / len(command_ages[port])
                if command_ages[port]
                else None,
                "maximum_command_age_frames": max(command_ages[port])
                if command_ages[port]
                else None,
            },
        }
    result = "complete" if exception is None and all(gate_checks.values()) else "failed"
    summary = {
        "schema_version": config["integration"]["schema_version"],
        "classification": f"frame-exact time-dilated {model_name}-versus-{model_name} integration; two independent native policy runtimes",
        "result": result,
        "error": None
        if exception is None
        else f"{type(exception).__name__}: {exception}",
        "gate": {
            "decision": "pass" if result == "complete" else "fail",
            "checks": gate_checks,
        },
        "configuration": {
            "seed": effective_seed,
            "stage": request.stage,
            "inference_mode": "synchronous-concurrent",
            "controller_replay_lag_frames": controller_replay_lag_frames,
            "maximum_game_frames": maximum_frames,
            "require_natural_end": request.require_natural_end,
            "player_1": {"model": model_name, "character": characters[1]},
            "player_2": {"model": model_name, "character": characters[2]},
        },
        "slots": slots,
        "source": source_checks,
        "reproducibility": config["_runtime_reproducibility"],
        "emulator_application": emulator_application,
        "game_image": _game_image_identity(iso_path),
        "controller_boundary_audit": controller_boundary,
        "execution": {
            "entered_gameplay": in_game,
            "processed_policy_frames": processed_frames,
            "first_game_frame": first_game_frame,
            "last_game_frame": last_game_frame,
            "frame_delta_counts": {
                str(delta): count for delta, count in sorted(frame_delta_counts.items())
            },
            "last_stocks": stocks,
            "natural_game_end": natural_game_end,
            "game_end_observed": natural_game_end,
            "sudden_death_transition_observed": sudden_death_transition_observed,
            "termination": termination,
            "shutdown_method": shutdown_method,
            "controller_transport": {
                **transport_record,
                "menu_flushes": {
                    f"p{port}": menu_transport_flushes[port] for port in (1, 2)
                },
                "warmup_held_frame_scheduled_from_frames": {
                    f"p{port}": warmup_transport_schedule_frames[port]
                    for port in (1, 2)
                },
                "policy_for_auxiliary_poll": "no inference, dispatch, or synthetic reflush",
            },
        },
        "artifacts": {
            "trace": {
                **_file_identity(trace_path, project_root),
                "rows": processed_frames,
            },
            "replays": replay_records,
        },
    }
    _write_json(summary_path, summary)
    if exception is not None:
        raise RuntimeError(summary["error"])
    return summary


def run(
    config_path: Path, iso_path: Path | None, request: PlayRequest | None = None
) -> dict[str, Any]:
    config, project_root = _load_config(config_path)
    config["_runtime_reproducibility"] = _runtime_reproducibility_record(
        project_root,
        config_path.resolve(),
        "requirements-e001.lock",
        (
            "src/melee_policy/integration/game_bundle.py",
            "src/melee_policy/integration/match_runtime.py",
            "src/melee_policy/integration/native/macos_mux_replay_audio.m",
            "src/melee_policy/integration/native/macos_replay_recorder.m",
            "src/melee_policy/integration/play.py",
            "src/melee_policy/integration/replay_video.py",
            "src/melee_policy/integration/runtime_identity.py",
            "src/melee_policy/integration/state_identity.py",
            "patches/slippi-dolphin-two-pipe-frame-sync.patch",
            "scripts/play",
        ),
    )
    request = PlayRequest() if request is None else request
    if request.artifact_label is None and (request.save_slp or request.save_video):
        from melee_policy.integration.game_bundle import export_artifact_label

        request = replace(
            request,
            artifact_label=export_artifact_label(
                None, save_slp=request.save_slp, save_video=request.save_video
            ),
        )
    request.validate()
    if request.artifact_label is not None:
        output_directory = (
            Path(config["outputs"]["directory"]).parent / request.artifact_label
        )
        config["outputs"] = {
            "directory": str(output_directory),
            "summary": str(output_directory / "summary.json"),
            "trace": str(output_directory / "controller_trace.jsonl"),
            "replay_directory": str(output_directory / "replays"),
        }
    seed = _effective_evaluation_seed(config, request.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    iso_path = _resolve_game_image_path(config, project_root, iso_path)
    source_checks = _activate_sources(config, project_root)
    if request.player_1_model.lower() == request.player_2_model.lower():
        runtimes = _load_same_family_runtimes(config, project_root, request)
        if request.player_1_model.lower() == "mimic":
            mimic_source = cast(MimicRuntime, runtimes[1]).source_identity
            if cast(MimicRuntime, runtimes[2]).source_identity != mimic_source:
                raise RuntimeError("same-family MIMIC source identities differ")
            source_checks["revisions"]["mimic"] = mimic_source["revision"]
            source_checks["repositories"]["mimic"] = mimic_source
        summary = _run_same_family_console(
            config, project_root, iso_path, source_checks, runtimes, request, seed
        )
    else:
        raise ValueError("Use the match dispatcher for mixed model families")
    if request.save_slp or request.save_video:
        from melee_policy.integration.game_bundle import finalize_game_bundle

        summary = finalize_game_bundle(
            summary,
            project_root=project_root,
            config=config,
            iso_path=iso_path,
            save_slp=request.save_slp,
            save_video=request.save_video,
        )
    return summary
