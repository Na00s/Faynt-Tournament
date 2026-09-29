"""New D0 checkpoint admission, identity refusal and local inference canaries."""
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from melee_policy.integration import frisson_policy as policy
from melee_policy.integration.faynt_d0_checkpoints import CHECKPOINTS, FORMAT, ROOT, require_identity
from melee_policy.integration.post_rl_checkpoints import CHECKPOINTS as P21_CHECKPOINTS


def _identity(profile: str) -> dict[str, Any]:
    record = CHECKPOINTS[profile]
    return {
        **record,
        "path": str(ROOT / record["relative_path"]),
        "deployed_actor": policy._pinned_evaluation_actor_contract(profile),
    }


@pytest.mark.parametrize("profile", tuple(CHECKPOINTS))
def test_catalog_identity_has_separate_unchanged_runtime(profile: str) -> None:
    identity = _identity(profile)
    assert require_identity(identity) is CHECKPOINTS[profile]
    assert identity["format"] == FORMAT
    assert identity["format"] != P21_CHECKPOINTS[profile]["format"]
    assert identity["relative_path"] != P21_CHECKPOINTS[profile]["relative_path"]
    assert identity["sha256"] != P21_CHECKPOINTS[profile]["sha256"]
    assert {name: row["step"] for name, row in P21_CHECKPOINTS.items()} == {"10m": 1318, "75m": 222}


@pytest.mark.parametrize("profile", tuple(CHECKPOINTS))
@pytest.mark.parametrize("field,value", [
    ("step", 1318), ("sha256", "0" * 64), ("byte_length", 1),
    ("processed_target_frames", 8337096704), ("trained_delay", 18),
    ("trained_delay", False), ("parameter_count", 1), ("state_tensor_count", 1),
    ("path", "/tmp/wrong.pt"), ("format", "melee_policy.post_rl_ali_checkpoint.v1"),
])
def test_registry_refuses_changed_identity(profile: str, field: str, value: Any) -> None:
    identity = _identity(profile)
    identity[field] = value
    with pytest.raises(ValueError, match="Faynt D0 checkpoint"):
        require_identity(identity)


@pytest.mark.parametrize("section,field,value", [
    ("actor", "delay_frames", 18), ("actor", "context_mode", "prefix"),
    ("actor", "context_frames", 256), ("actor", "temperature", 0.0),
    ("actor", "batch_steps", 2), ("policy", "compute_dtype", "bfloat16"),
    ("policy", "cache_dtype", "bfloat16"),
    ("actor", "delay_frames", False),
])
def test_registry_refuses_changed_deployment(section: str, field: str, value: Any) -> None:
    identity = _identity("10m")
    identity["deployed_actor"][section][field] = value
    with pytest.raises(ValueError, match=r"deployed .* contract differs"):
        require_identity(identity)


def _metadata_payload(profile: str, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    record = CHECKPOINTS[profile]
    monkeypatch.setattr(policy, "_posttraining_checkpoint_contract", lambda payload: {
        "profile": profile, "processed_target_frames": 12795772928,
        "resolved_training": {"precision": "bfloat16"},
    })
    return {
        "training_state": {"optimizer_steps": 195248},
        "metadata": {
            "trained_delay": 0,
            "rl_post_training": {
                "step": record["step"], "wandb_run": record["wandb_run_id"],
                "source_checkpoint": record["source_checkpoint"],
                "init_checkpoint": f"melee-rl-runs:/bc/frisson-melee-{profile}-posttrained-best-val.pt",
                "exported": record["exported"], "delay_frames": 0,
            },
        },
    }


@pytest.mark.parametrize("profile", tuple(CHECKPOINTS))
@pytest.mark.parametrize("location,key", [("metadata", "trained_delay"), ("rl", "delay_frames")])
@pytest.mark.parametrize("value", [18, False, "0", None])
def test_payload_refuses_nonzero_or_noninteger_delay(
    profile: str, location: str, key: str, value: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _metadata_payload(profile, monkeypatch)
    mapping = payload["metadata"] if location == "metadata" else payload["metadata"]["rl_post_training"]
    mapping[key] = value
    with pytest.raises(RuntimeError, match=f"{key}|integer"):
        policy._faynt_d0_checkpoint_contract(payload)


def test_10m_requires_both_explicit_delay_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _metadata_payload("10m", monkeypatch)
    del payload["metadata"]["trained_delay"]
    with pytest.raises(RuntimeError, match="trained_delay"):
        policy._faynt_d0_checkpoint_contract(payload)
    payload = _metadata_payload("10m", monkeypatch)
    del payload["metadata"]["rl_post_training"]["delay_frames"]
    with pytest.raises(RuntimeError, match="delay_frames"):
        policy._faynt_d0_checkpoint_contract(payload)


def test_75m_missing_delay_is_bound_to_verified_native_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _metadata_payload("75m", monkeypatch)
    del payload["metadata"]["trained_delay"]
    del payload["metadata"]["rl_post_training"]["delay_frames"]
    contract = policy._faynt_d0_checkpoint_contract(payload)
    assert contract["trained_delay"] == 0
    assert contract["expected_checkpoint_sha256"] == CHECKPOINTS["75m"]["sha256"]
    assert contract["rl_provenance"]["native_checkpoint_sha256"] == (
        "8bd4345a294760ceb3853014bf36bf5ef56f0817a1a1f5947b9362d6d6454ddf"
    )


@pytest.mark.parametrize("field", ["step", "source_checkpoint", "wandb_run", "init_checkpoint", "exported"])
def test_payload_refuses_crossed_rl_provenance(field: str, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _metadata_payload("10m", monkeypatch)
    payload["metadata"]["rl_post_training"][field] = "wrong"
    with pytest.raises(RuntimeError, match=f"provenance mismatch: {field}"):
        policy._faynt_d0_checkpoint_contract(payload)


@pytest.mark.parametrize("profile", tuple(CHECKPOINTS))
def test_only_exact_hash_can_allow_frozenset(profile: str, monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    observed_loads = []
    monkeypatch.setattr(
        torch.serialization, "get_unsafe_globals_in_checkpoint", lambda path: ["builtins.frozenset"],
    )
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: observed_loads.append(kwargs) or {})
    monkeypatch.setattr(policy, "_sha256_file", lambda path: CHECKPOINTS[profile]["sha256"])
    assert policy._safe_load_checkpoint_payload(Path("unused.pt"), torch=torch) == {}
    assert observed_loads == [{"map_location": "cpu", "weights_only": True, "mmap": True}]
    monkeypatch.setattr(policy, "_sha256_file", lambda path: "0" * 64)
    with pytest.raises(RuntimeError, match="content-addressed trusted"):
        policy._safe_load_checkpoint_payload(Path("unused.pt"), torch=torch)
    assert len(observed_loads) == 1
    monkeypatch.setattr(
        torch.serialization, "get_unsafe_globals_in_checkpoint", lambda path: ["unsafe.Custom"],
    )
    with pytest.raises(RuntimeError, match="unsafe globals"):
        policy._safe_load_checkpoint_payload(Path("unused.pt"), torch=torch)


@pytest.fixture(scope="module", params=tuple(CHECKPOINTS))
def real_checkpoint(request: pytest.FixtureRequest) -> tuple[dict[str, Any], Path, Any, Any]:
    torch = pytest.importorskip("torch")
    record = CHECKPOINTS[request.param]
    path = ROOT / record["relative_path"]
    if directory := os.environ.get("MELEE_FAYNT_D0_CHECKPOINT_DIR"):
        path = Path(directory) / path.name
    if not path.is_file():
        pytest.skip(f"local Faynt D0 export is unavailable: {path}")
    payload = policy._safe_load_checkpoint_payload(path, torch=torch)
    return record, path, payload, torch


def test_real_identity_separates_rl_weights_from_inherited_optimizer(real_checkpoint: Any) -> None:
    record, path, payload, torch = real_checkpoint
    state, identity, _ = policy._inspect_loaded_checkpoint(path, payload, torch=torch)
    assert identity["format"] == FORMAT
    assert identity["step"] == record["step"]
    assert identity["sha256"] == record["sha256"]
    assert identity["trained_delay"] == 0
    assert identity["processed_target_frames"] is None
    assert identity["post_rl"]["initialization_training_state"]["optimizer_steps"] != record["step"]
    assert identity["post_rl"]["inherited_optimizer_state_used_for_inference"] is False
    assert identity["initialization"]["wandb_run_id"] != record["wandb_run_id"]
    assert identity["initialization"]["metadata"] == {
        key: policy._json_safe(value) for key, value in payload["metadata"].items()
        if key not in {"rl_post_training", "trained_delay"}
    }
    assert state is payload["model"]
    changed = {**payload, "optimizer_bundle": object()}
    assert policy._faynt_d0_checkpoint_contract(changed)["step"] == record["step"]
    # The portable inspection boundary accepts this exact export in any location.
    # Match launch admission additionally requires the registry's fixed local path.
    require_identity({**identity, "path": str(ROOT / record["relative_path"])})


def test_real_identity_refuses_wrong_hash(real_checkpoint: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _, path, payload, torch = real_checkpoint
    monkeypatch.setattr(policy, "_sha256_file", lambda path: "0" * 64)
    with pytest.raises(RuntimeError, match="content identity mismatch"):
        policy._inspect_loaded_checkpoint(path, payload, torch=torch)


def test_real_safe_strict_load_and_rolling_inference_smoke(real_checkpoint: Any) -> None:
    record, path, _, torch = real_checkpoint
    source = ROOT / ".e010-cache" / "frisson-ai-source" / policy.FRISSON_SOURCE_REVISION
    if not source.is_dir():
        pytest.skip("pinned benchmark model source is unavailable")
    for filename, digest in policy._pinned_model_source_files().values():
        assert policy._sha256_file(source / filename) == digest
    config = replace(
        policy.FrissonPolicyConfig.from_project_root(ROOT, port=1, opponent_port=2),
        model_source_directory=source,
        checkpoint_path=path,
    )
    config.validate()
    modules = policy._activate_runtime_sources(config)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model, identity = policy._load_checkpoint(path, torch=torch, model_module=modules["model"])
        assert identity["strict_state_dict"] is True
        assert identity["parameter_count"] == record["parameter_count"]
        assert identity["runtime_model_config"]["action_offset_frames"] == 1
        assert identity["runtime_model_config"]["compute_dtype"] == "float32"
        assert identity["runtime_model_config"]["cache_dtype"] == "float32"
        assert identity["model_config_compatibility"]["training_compute_dtype"] == "float32"
        assert identity["post_rl"]["initialization_training_compute_dtype"] == "bfloat16"
        fixture = policy._load_module_from_path(
            "_faynt_d0_existing_parser_fixture", ROOT / "tests/integration/test_frisson_policy.py",
        )
        batch = policy._tensorize_parsed_game(
            fixture._parsed_game(), tensor_batch=modules["tensor_batch"], torch=torch,
            device=torch.device("cpu"),
        )
        cache = model.backbone.init_cache(1, device="cpu")
        generator = torch.Generator(device="cpu").manual_seed(42)
        with torch.inference_mode():
            for frame in range(260):
                command, cache, _ = policy._sample_policy_frame(
                    policy=model, cache=cache, game_batch=batch, reset=frame == 0,
                    generator=generator, torch=torch,
                )
                assert all(0.0 <= value <= 1.0 for value in (*command.main_stick, *command.c_stick))
        assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    finally:
        torch.set_num_threads(previous_threads)
