"""Frozen September 15 zero-added-delay exports for a separate benchmark.

The Ali envelopes retain the supervised initializer's training and optimizer
state. Only their policy weights are used by inference. The 75M Ali export omits
delay fields; its delay is attested by the bit-identical native checkpoint.
"""
from collections.abc import Mapping
from pathlib import Path
from typing import Any

FORMAT = "melee_policy.faynt_d0_ali_checkpoint.v1"
ROOT = Path(__file__).resolve().parents[3]
CHECKPOINTS: dict[str, dict[str, Any]] = {
    "10m": {
        "profile": "10m",
        "step": 632,
        "relative_path": ".e010-cache/faynt-d0-632-980/frisson-melee-10m-rl-leash-step632-d0-ali.pt",
        "sha256": "fd884f19c53b7353add18c3819a1821c325ad3b4818cee83304ef57055b1af92",
        "byte_length": 96_453_467,
        "parameter_count": 10_163_629,
        "state_tensor_count": 109,
        "wandb_run_id": "tbrekji3",
        "source_checkpoint": "melee-rl-runs:/selfplay/10m_leash/kl0p003/step_00000632.pt",
        "policy_export_sha256": "c8a810c0d33b72ae448eec0e0b7d0538beacfaeb9a2e357eddd7be4fc0ce286b",
        "policy_export_byte_length": 40_698_232,
        "native_path": None,
        "native_sha256": None,
        "native_byte_length": None,
        "created": None,
        "exported": "2026-09-12T01:11:58Z",
        "delay_evidence": "Ali metadata and bit-identical policy-only export explicitly declare zero",
        "explicit_ali_delay_required": True,
        "native_context_mode": None,
        "rl_frames_seen": None,
    },
    "75m": {
        "profile": "75m",
        "step": 980,
        "relative_path": ".e010-cache/faynt-d0-632-980/frisson-melee-75m-rl-step980-ali.pt",
        "sha256": "1f630ac2c7f731a9142bac2c48e5e4a134b313f7b16c24d2108d45605752806e",
        "byte_length": 644_283_955,
        "parameter_count": 75_305_709,
        "state_tensor_count": 205,
        "wandb_run_id": "kiml5sl5",
        "source_checkpoint": "melee-rl-runs:/selfplay/75m_pt_stages/latest.pt",
        "policy_export_sha256": "6d2474608869b360238fb15598aa95eed34e0b28bee83a57b59890abead180f8",
        "policy_export_byte_length": 301_298_093,
        "native_path": ".e010-cache/faynt-d0-632-980/native-75m-rl-step980.pt",
        "native_sha256": "8bd4345a294760ceb3853014bf36bf5ef56f0817a1a1f5947b9362d6d6454ddf",
        "native_byte_length": 932_974_219,
        "created": "2026-09-09T01:31:36Z",
        "exported": "2026-09-09T23:50:46Z",
        "delay_evidence": "Bit-identical native checkpoint declares zero checkpoint, actor and console delay",
        "explicit_ali_delay_required": False,
        "native_context_mode": "prefix",
        "rl_frames_seen": None,
    },
}

for _record in CHECKPOINTS.values():
    _record.update(
        format=FORMAT,
        processed_target_frames=None,
        validation_nll=None,
        trained_delay=0,
        action_offset_frames=1,
        compute_dtype="float32",
        cache_dtype="float32",
        context_mode="ring",
        actor_context_frames=128,
        temperature=1.0,
    )


def require_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Admit only a pinned export with its unchanged zero-delay runtime."""
    expected = CHECKPOINTS.get(identity.get("profile"))
    if expected is None:
        raise ValueError("unknown Faynt D0 profile")
    for key in (
        "format", "sha256", "byte_length", "step", "processed_target_frames",
        "parameter_count", "state_tensor_count", "trained_delay",
    ):
        observed = identity.get(key)
        if observed != expected[key] or (isinstance(expected[key], int) and isinstance(observed, bool)):
            raise ValueError(f"Faynt D0 checkpoint identity differs: {key}")
    path = identity.get("path")
    if not isinstance(path, str) or Path(path).resolve() != (ROOT / expected["relative_path"]).resolve():
        raise ValueError("Faynt D0 checkpoint path differs")
    deployed = identity.get("deployed_actor")
    if not isinstance(deployed, Mapping):
        raise ValueError("Faynt D0 deployed actor is missing")
    for section, required in {
        "policy": {"profile": expected["profile"], "compute_dtype": "float32", "cache_dtype": "float32"},
        "actor": {"delay_frames": 0, "context_mode": "ring", "context_frames": 128,
                  "batch_steps": 1, "temperature": 1.0},
    }.items():
        observed = deployed.get(section)
        if not isinstance(observed, Mapping) or any(
            observed.get(key) != value
            or (isinstance(value, (int, float)) and isinstance(observed.get(key), bool))
            for key, value in required.items()
        ):
            raise ValueError(f"Faynt D0 deployed {section} contract differs")
    return expected
