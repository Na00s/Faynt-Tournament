"""Content-addressed P21 RL exports, separate from inherited BC metadata."""
from pathlib import Path

FORMAT = "melee_policy.post_rl_ali_checkpoint.v1"
ROOT = Path(__file__).resolve().parents[3]
CHECKPOINTS = {
    "10m": {
        "profile": "10m", "step": 1318,
        "relative_path": ".e010-cache/post-rl-p21/frisson-melee-10m-rl-step1318.pt",
        "sha256": "da1179d6319d2f9a805b1697d8176b66a455aa9f37b51e46ad5f2a3d3537a7da",
        "byte_length": 96448859, "parameter_count": 10163629,
        "state_tensor_count": 109, "wandb_run_id": "oymejnsq",
        "native_path": ".e010-cache/post-rl-p21/native-10m-best_1318.pt",
        "native_sha256": "1178f615459349319c77fd4bc31b467f7aad93c41ff19e8c1d51caf3b3ae123d",
        "source_checkpoint": "melee-rl-runs:/selfplay/10m_pt_stages/best_1318.pt",
        "rl_frames_seen": 453476352, "created": "2026-09-07T04:45:16Z",
    },
    "75m": {
        "profile": "75m", "step": 222,
        "relative_path": ".e010-cache/post-rl-p21/frisson-melee-75m-rl-step222.pt",
        "sha256": "860b0d06789a5e52fcf44bf497c9f2838e387d7c20689bb455434048732883e0",
        "byte_length": 644283507, "parameter_count": 75305709,
        "state_tensor_count": 205, "wandb_run_id": "kiml5sl5",
        "native_path": ".e010-cache/post-rl-p21/native-75m-best_222.pt",
        "native_sha256": "190c7689ae3fbc1e4e320ae44cf1ac0f4f83d9af87b58364fbefa8bcb130f14c",
        "source_checkpoint": "melee-rl-runs:/selfplay/75m_pt_stages/best_222.pt",
        "rl_frames_seen": 76382208, "created": "2026-09-06T06:47:14Z",
    },
}

for _record in CHECKPOINTS.values():
    _record.update(format=FORMAT, processed_target_frames=None, validation_nll=None)


def require_identity(identity):
    expected = CHECKPOINTS.get(identity.get("profile"))
    if expected is None:
        raise ValueError("unknown P21 RL profile")
    for key in ("format", "sha256", "byte_length", "step", "processed_target_frames", "parameter_count"):
        if identity.get(key) != expected[key]:
            raise ValueError(f"P21 RL checkpoint identity differs: {key}")
    if Path(identity.get("path", "")).resolve() != (ROOT / expected["relative_path"]).resolve():
        raise ValueError("P21 RL checkpoint path differs")
    return expected
