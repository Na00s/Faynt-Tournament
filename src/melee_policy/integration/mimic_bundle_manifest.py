"""Strict loader for versioned external MIMIC bundle allowlists."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, cast

MIMIC_NATIVE_BUNDLE_MANIFEST_SCHEMA_VERSION = (
    "integration.mimic_external_bundle_manifest.v1"
)
MIMIC_NATIVE_BUNDLE_MANIFEST_PATH = Path(__file__).with_name(
    "mimic_native_bundles.v1.json"
)
MIMIC_NATIVE_BUNDLE_MANIFEST_SHA256 = (
    "adb8fa9ef8c6fd5f054fb68b22552c2c92cacce3f8cc0ac0ca2f72bed9ebcef3"
)
MIMIC_NATIVE_ARTIFACT_REVISION = "0629eb174bb7548038469f538d98b3691ea647be"
MIMIC_NATIVE_SOURCE_REVISION = "01eb974962c8338147518dd360ddbf4b9d4c48e3"
MIMIC_NATIVE_BUNDLE_CHARACTERS = frozenset(
    {
        "BOWSER",
        "CPTFALCON",
        "DK",
        "DOC",
        "FALCO",
        "FOX",
        "GAMEANDWATCH",
        "GANONDORF",
        "JIGGLYPUFF",
        "LINK",
        "LUIGI",
        "MARIO",
        "MARTH",
        "MEWTWO",
        "NESS",
        "PEACH",
        "PIKACHU",
        "POPO",
        "ROY",
        "SAMUS",
        "SHEIK",
        "YLINK",
        "YOSHI",
    }
)
MIMIC_BUNDLE_ASSET_NAMES = (
    "model.pt",
    "config.json",
    "metadata.json",
    "mimic_norm.json",
    "controller_combos.json",
    "cat_maps.json",
    "stick_clusters.json",
    "norm_stats.json",
)
_MANIFEST_KEYS = {
    "schema_version",
    "artifact_repository",
    "asset_directory",
    "source_repository",
    "source_directory",
    "bundles",
}
_BUNDLE_KEYS = {
    "name",
    "character",
    "run_name",
    "artifact_revision",
    "source_revision",
    "assets",
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise RuntimeError(f"duplicate key in MIMIC bundle manifest: {key!r}")
        value[key] = item
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_revision(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def load_mimic_native_bundle_manifest(
    path: Path = MIMIC_NATIVE_BUNDLE_MANIFEST_PATH,
    *,
    expected_sha256: str = MIMIC_NATIVE_BUNDLE_MANIFEST_SHA256,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load the exact checked-in manifest and normalize it by checkpoint hash."""

    payload = path.read_bytes()
    observed_sha256 = _sha256_bytes(payload)
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            "MIMIC native bundle manifest identity mismatch: "
            f"{observed_sha256} != {expected_sha256}"
        )
    decoded = json.loads(
        payload,
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(decoded, dict):
        raise RuntimeError("MIMIC native bundle manifest must be a JSON object")
    manifest = cast(dict[str, Any], decoded)
    if set(manifest) != _MANIFEST_KEYS:
        raise RuntimeError(
            "MIMIC native bundle manifest fields changed: "
            f"{sorted(manifest)} != {sorted(_MANIFEST_KEYS)}"
        )
    if manifest.get("schema_version") != MIMIC_NATIVE_BUNDLE_MANIFEST_SCHEMA_VERSION:
        raise RuntimeError("unsupported MIMIC native bundle manifest schema")
    artifact_repository = manifest.get("artifact_repository")
    asset_directory = manifest.get("asset_directory")
    source_repository = manifest.get("source_repository")
    source_directory = manifest.get("source_directory")
    if artifact_repository != "https://huggingface.co/erickfm/MIMIC":
        raise RuntimeError("MIMIC native artifact repository changed")
    if asset_directory != ".e012-cache/mimic-native-0629eb17":
        raise RuntimeError("MIMIC native asset directory changed")
    if source_repository != "https://github.com/erickfm/MIMIC.git":
        raise RuntimeError("MIMIC native source repository changed")
    if source_directory != ".e012-cache/mimic-source-current":
        raise RuntimeError("MIMIC native source directory changed")
    rows = manifest.get("bundles")
    if not isinstance(rows, list) or len(rows) != len(MIMIC_NATIVE_BUNDLE_CHARACTERS):
        raise RuntimeError("MIMIC native bundle manifest must contain exactly 23 bundles")

    bundles: dict[str, dict[str, Any]] = {}
    names: set[str] = set()
    characters: set[str] = set()
    for index, untyped_row in enumerate(rows):
        if not isinstance(untyped_row, dict):
            raise RuntimeError(f"MIMIC native bundle row {index} must be an object")
        row = cast(dict[str, Any], untyped_row)
        if set(row) != _BUNDLE_KEYS:
            raise RuntimeError(f"MIMIC native bundle row {index} fields changed")
        name = row.get("name")
        character = row.get("character")
        run_name = row.get("run_name")
        artifact_revision = row.get("artifact_revision")
        source_revision = row.get("source_revision")
        if not isinstance(name, str) or not name or name in names:
            raise RuntimeError(f"invalid or duplicate MIMIC native bundle name: {name!r}")
        if (
            not isinstance(character, str)
            or character not in MIMIC_NATIVE_BUNDLE_CHARACTERS
            or character in characters
        ):
            raise RuntimeError(
                f"invalid or duplicate MIMIC native bundle character: {character!r}"
            )
        if not isinstance(run_name, str) or not run_name:
            raise RuntimeError(f"invalid MIMIC native run name for {name}")
        if artifact_revision != MIMIC_NATIVE_ARTIFACT_REVISION or not _is_revision(
            artifact_revision
        ):
            raise RuntimeError(f"MIMIC native artifact revision changed for {name}")
        if source_revision != MIMIC_NATIVE_SOURCE_REVISION or not _is_revision(
            source_revision
        ):
            raise RuntimeError(f"MIMIC native source revision changed for {name}")

        untyped_assets = row.get("assets")
        if not isinstance(untyped_assets, dict) or set(untyped_assets) != set(
            MIMIC_BUNDLE_ASSET_NAMES
        ):
            raise RuntimeError(f"MIMIC native asset set changed for {name}")
        assets: dict[str, tuple[str, int]] = {}
        for asset_name in MIMIC_BUNDLE_ASSET_NAMES:
            asset = untyped_assets.get(asset_name)
            if not isinstance(asset, list) or len(asset) != 2:
                raise RuntimeError(f"invalid MIMIC native asset record: {name}/{asset_name}")
            digest, byte_length = asset
            if not _is_sha256(digest) or not isinstance(byte_length, int) or byte_length <= 0:
                raise RuntimeError(f"invalid MIMIC native asset identity: {name}/{asset_name}")
            assets[asset_name] = (cast(str, digest), byte_length)
        checkpoint_sha256 = assets["model.pt"][0]
        if checkpoint_sha256 in bundles:
            raise RuntimeError(f"duplicate MIMIC native checkpoint: {checkpoint_sha256}")
        bundles[checkpoint_sha256] = {
            "name": name,
            "character": character,
            "run_name": run_name,
            "repository": artifact_repository,
            "revision": artifact_revision,
            "source_repository": source_repository,
            "source_revision": source_revision,
            "source_directory": source_directory,
            "asset_directory": asset_directory,
            "assets": assets,
            "allowlist_manifest": {
                "schema_version": MIMIC_NATIVE_BUNDLE_MANIFEST_SCHEMA_VERSION,
                "path": path.name,
                "sha256": observed_sha256,
                "byte_length": len(payload),
            },
        }
        names.add(name)
        characters.add(character)
    if characters != set(MIMIC_NATIVE_BUNDLE_CHARACTERS):
        raise RuntimeError("MIMIC native bundle character coverage changed")
    identity = {
        "schema_version": MIMIC_NATIVE_BUNDLE_MANIFEST_SCHEMA_VERSION,
        "path": str(path),
        "sha256": observed_sha256,
        "byte_length": len(payload),
        "bundle_count": len(bundles),
    }
    return bundles, identity


MIMIC_NATIVE_BUNDLES, MIMIC_NATIVE_BUNDLE_MANIFEST_IDENTITY = (
    load_mimic_native_bundle_manifest()
)


def mimic_native_bundle(character: str) -> dict[str, Any]:
    """Return an isolated copy of the one current native bundle for a character."""

    matches = [
        bundle
        for bundle in MIMIC_NATIVE_BUNDLES.values()
        if bundle["character"] == character
    ]
    if len(matches) != 1:
        raise KeyError(f"no unique current native MIMIC bundle for {character!r}")
    return copy.deepcopy(matches[0])


__all__ = [
    "MIMIC_BUNDLE_ASSET_NAMES",
    "MIMIC_NATIVE_ARTIFACT_REVISION",
    "MIMIC_NATIVE_BUNDLES",
    "MIMIC_NATIVE_BUNDLE_CHARACTERS",
    "MIMIC_NATIVE_BUNDLE_MANIFEST_IDENTITY",
    "MIMIC_NATIVE_BUNDLE_MANIFEST_PATH",
    "MIMIC_NATIVE_BUNDLE_MANIFEST_SCHEMA_VERSION",
    "MIMIC_NATIVE_BUNDLE_MANIFEST_SHA256",
    "MIMIC_NATIVE_SOURCE_REVISION",
    "load_mimic_native_bundle_manifest",
    "mimic_native_bundle",
]
