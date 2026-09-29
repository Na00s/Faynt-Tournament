"""Record and validate the exact Python environment used by a live match."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _canonical_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _canonical_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _display_path(path: Path, project_root: Path) -> str:
    absolute = path.expanduser().absolute()
    try:
        return str(absolute.relative_to(project_root.resolve()))
    except ValueError:
        return str(absolute)


def runtime_environment_identity_sha256(record: Mapping[str, Any]) -> str:
    """Hash every field except the self-describing digest itself."""

    material = {key: value for key, value in record.items() if key != "identity_sha256"}
    return hashlib.sha256(_canonical_json(material).encode("utf-8")).hexdigest()


def _installed_distributions() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            continue
        direct_url_text = distribution.read_text("direct_url.json")
        direct_url: Any = None
        if direct_url_text:
            try:
                direct_url = json.loads(direct_url_text)
            except json.JSONDecodeError:
                direct_url = {"invalid_json": direct_url_text}
        records.append(
            {
                "name": _canonical_distribution_name(raw_name),
                "version": distribution.version,
                "direct_url": direct_url,
            }
        )
    records.sort(key=_canonical_json)
    return records


def _lock_validation(
    lock_path: Path,
    installed_distributions: list[dict[str, Any]],
) -> dict[str, Any]:
    installed_versions: dict[str, set[str]] = {}
    for distribution in installed_distributions:
        installed_versions.setdefault(str(distribution["name"]), set()).add(str(distribution["version"]))

    exact_pins: list[dict[str, Any]] = []
    non_exact_entries: list[str] = []
    for raw_line in lock_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line or line.startswith("-e "):
            non_exact_entries.append(line)
            continue
        raw_name, expected_version = line.split("==", maxsplit=1)
        name = _canonical_distribution_name(raw_name.strip())
        actual_versions = sorted(installed_versions.get(name, set()))
        exact_pins.append(
            {
                "name": name,
                "expected_version": expected_version.strip(),
                "installed_versions": actual_versions,
                "match": actual_versions == [expected_version.strip()],
            }
        )
    exact_pins.sort(key=lambda value: str(value["name"]))
    pinned_names = {str(value["name"]) for value in exact_pins}
    allowed_local_distributions = {"melee-policy-e000", "slippi-ai"}
    unpinned_distributions = sorted(set(installed_versions) - pinned_names)
    unpinned_external_distributions = sorted(set(unpinned_distributions) - allowed_local_distributions)
    exact_pins_satisfied = bool(exact_pins) and all(bool(value["match"]) for value in exact_pins)
    return {
        "exact_pins": exact_pins,
        "exact_pins_satisfied": exact_pins_satisfied,
        "non_exact_entries": non_exact_entries,
        "installed_distributions_not_exactly_pinned": unpinned_distributions,
        "allowed_local_distributions": sorted(allowed_local_distributions),
        "unpinned_external_distributions": unpinned_external_distributions,
        "all_external_distributions_pinned": not unpinned_external_distributions,
        "environment_lock_gate_passed": exact_pins_satisfied and not unpinned_external_distributions,
        "scope": (
            "Every name==version lock entry must be the only installed version for that name. "
            "Editable VCS entries are recorded here and validated by the launcher's pinned-source gate."
        ),
    }


def runtime_environment_record(project_root: Path, dependency_lock: Path) -> dict[str, Any]:
    """Return a canonical identity for the interpreter and all installed distributions."""

    root = project_root.expanduser().resolve()
    lock_path = dependency_lock.expanduser().resolve()
    if not lock_path.is_file():
        raise FileNotFoundError(f"runtime dependency lock is missing: {lock_path}")
    distributions = _installed_distributions()
    record: dict[str, Any] = {
        "python_executable": _display_path(Path(sys.executable), root),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "host_platform": platform.platform(),
        "host_machine": platform.machine(),
        "host_os_build": platform.version(),
        "installed_distributions": distributions,
        "installed_distributions_sha256": hashlib.sha256(
            _canonical_json(distributions).encode("utf-8")
        ).hexdigest(),
        "dependency_lock_validation": _lock_validation(lock_path, distributions),
    }
    record["identity_sha256"] = runtime_environment_identity_sha256(record)
    return record
