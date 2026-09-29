"""Public command for launching a local two-policy Melee match."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL_CHOICES = (
    "faynt",
    "mimic",
    "slippi-ai",
    "slippi",
    "frisson-ai",
    "frisson_ai",
    "frisson",
    "cpu",
)


def _normalize_model(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    if normalized == "slippi":
        return "slippi-ai"
    if normalized in ("frisson", "faynt"):
        return "frisson-ai"
    return normalized


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/integration.toml"),
    )
    parser.add_argument(
        "--iso-path",
        type=Path,
        help="NTSC 1.02 image; defaults to MELEE_ISO_PATH or the private cache path",
    )
    parser.add_argument("--player-1", "--p1", choices=MODEL_CHOICES, default="mimic")
    parser.add_argument("--player-2", "--p2", choices=MODEL_CHOICES, default="slippi-ai")
    parser.add_argument("--player-1-checkpoint", "--p1-checkpoint", type=Path)
    parser.add_argument("--player-2-checkpoint", "--p2-checkpoint", type=Path)
    parser.add_argument("--player-1-assets", "--p1-assets", type=Path)
    parser.add_argument("--player-2-assets", "--p2-assets", type=Path)
    parser.add_argument("--player-1-character", "--p1-character", default="FOX")
    parser.add_argument("--player-2-character", "--p2-character", default="FOX")
    parser.add_argument("--player-1-name", "--p1-name")
    parser.add_argument("--player-2-name", "--p2-name")
    parser.add_argument(
        "--player-1-slippi-release",
        "--p1-slippi-release",
        choices=("medium-v2", "dk_d18_imitation_v2", "doc_d18_imitation_v3"),
        default="medium-v2",
        help="exact official Slippi-AI release contract for P1",
    )
    parser.add_argument(
        "--player-2-slippi-release",
        "--p2-slippi-release",
        choices=("medium-v2", "dk_d18_imitation_v2", "doc_d18_imitation_v3"),
        default="medium-v2",
        help="exact official Slippi-AI release contract for P2",
    )
    parser.add_argument("--player-1-temperature", "--p1-temperature", type=float)
    parser.add_argument("--player-2-temperature", "--p2-temperature", type=float)
    parser.add_argument(
        "--cpu-level",
        type=int,
        choices=(9,),
        help="native Melee CPU level; currently supported only as P2 CPU level 9",
    )
    parser.add_argument("--stage", default="BATTLEFIELD")
    parser.add_argument("--max-game-frames", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--require-natural-end",
        action="store_true",
        help="fail the run unless the saved game reaches a natural end",
    )
    parser.add_argument(
        "--require-formal-game-end",
        action="store_true",
        help="drain paired neutral inputs until libmelee observes the formal GAME_END event",
    )
    parser.add_argument(
        "--inference-mode",
        choices=("exact",),
        default="exact",
        help="exact processes every policy frame",
    )
    parser.add_argument(
        "--artifact-label",
        help="write under a new labeled artifact directory; reused labels fail closed",
    )
    parser.add_argument(
        "--save-slp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="copy the validated scoring replay to game/game.slp after the match",
    )
    parser.add_argument(
        "--save-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="render the validated scoring replay to game/game.mp4 after the match",
    )
    parser.add_argument(
        "--allow-p1-ood-character",
        action="store_true",
        help=(
            "explicitly run a pinned P1 Slippi-AI release on a physical character outside "
            "its released training roster"
        ),
    )
    parser.add_argument(
        "--allow-p2-ood-character",
        action="store_true",
        help=(
            "explicitly run a pinned P2 checkpoint on a physical character outside its "
            "released training roster"
        ),
    )
    return parser


def main() -> None:
    arguments = _build_parser().parse_args()
    player_1_model = _normalize_model(arguments.player_1)
    player_2_model = _normalize_model(arguments.player_2)
    if arguments.player_1_slippi_release != "medium-v2" and player_1_model != "slippi-ai":
        raise ValueError("specialist --p1-slippi-release requires Slippi-AI on P1")
    if arguments.player_2_slippi_release != "medium-v2" and player_2_model != "slippi-ai":
        raise ValueError(
            "specialist --p2-slippi-release requires Slippi-AI on P2"
        )
    if arguments.allow_p1_ood_character and player_1_model != "slippi-ai":
        raise ValueError("--allow-p1-ood-character requires Slippi-AI on P1")
    if arguments.allow_p2_ood_character and (
        player_1_model not in {"frisson-ai", "slippi-ai"}
        or player_2_model not in {"mimic", "slippi-ai"}
    ):
        raise ValueError(
            "--allow-p2-ood-character requires Frisson-AI on P1 with MIMIC or "
            "Slippi-AI on P2, or Slippi-AI on P1 with MIMIC or Slippi-AI on P2"
        )
    if arguments.require_formal_game_end and (
        player_1_model,
        player_2_model,
    ) != ("slippi-ai", "mimic"):
        raise ValueError(
            "--require-formal-game-end is currently scoped to Slippi-AI P1 versus MIMIC P2"
        )
    inference_mode = (
        "synchronous-concurrent" if arguments.inference_mode == "exact" else "asynchronous-latest"
    )

    if "cpu" in (player_1_model, player_2_model):
        if (player_1_model, player_2_model) not in {
            ("frisson-ai", "cpu"),
            ("slippi-ai", "cpu"),
        }:
            raise ValueError("CPU matches require Frisson-AI or Slippi-AI as P1 and native CPU as P2")
        if arguments.cpu_level != 9:
            raise ValueError("policy versus CPU requires explicit --cpu-level 9")
        from melee_policy.integration.frisson_match import MATCH_SEED
        cpu_fields = {
            "player_1_model": player_1_model,
            "player_2_model": player_2_model,
            "player_1_character": arguments.player_1_character,
            "player_2_character": arguments.player_2_character,
            "player_1_checkpoint": arguments.player_1_checkpoint,
            "player_2_checkpoint": arguments.player_2_checkpoint,
            "player_1_assets": arguments.player_1_assets,
            "player_2_assets": arguments.player_2_assets,
            "player_1_name": arguments.player_1_name,
            "player_2_name": arguments.player_2_name,
            "player_1_temperature": arguments.player_1_temperature,
            "player_2_temperature": arguments.player_2_temperature,
            "stage": arguments.stage,
            "max_game_frames": arguments.max_game_frames,
            "seed": MATCH_SEED if arguments.seed is None else arguments.seed,
            "require_natural_end": arguments.require_natural_end,
            "inference_mode": inference_mode,
            "artifact_label": arguments.artifact_label,
            "save_slp": arguments.save_slp,
            "save_video": arguments.save_video,
            "cpu_level": arguments.cpu_level,
        }
        if player_1_model == "frisson-ai":
            from melee_policy.integration.frisson_cpu_match import (
                FrissonCpuMatchRequest,
                run_frisson_cpu_match,
            )

            summary = run_frisson_cpu_match(
                arguments.config,
                arguments.iso_path,
                FrissonCpuMatchRequest(**cpu_fields),
            )
        else:
            from melee_policy.integration.slippi_cpu_match import (
                SlippiCpuMatchRequest,
                run_slippi_cpu_match,
            )

            summary = run_slippi_cpu_match(
                arguments.config,
                arguments.iso_path,
                SlippiCpuMatchRequest(
                    **cpu_fields,
                    player_1_slippi_release=arguments.player_1_slippi_release,
                    allow_player_1_ood_character=arguments.allow_p1_ood_character,
                ),
            )
    elif (player_1_model, player_2_model) == ("frisson-ai", "slippi-ai"):
        if arguments.cpu_level is not None:
            raise ValueError("--cpu-level requires --p2 cpu")
        from melee_policy.integration.frisson_match import MATCH_SEED
        from melee_policy.integration.frisson_slippi_match import (
            FrissonSlippiMatchRequest,
            run_frisson_slippi_match,
        )

        frisson_slippi_request = FrissonSlippiMatchRequest(
            player_1_model=player_1_model,
            player_2_model=player_2_model,
            player_1_character=arguments.player_1_character,
            player_2_character=arguments.player_2_character,
            player_1_checkpoint=arguments.player_1_checkpoint,
            player_2_checkpoint=arguments.player_2_checkpoint,
            player_2_slippi_release=arguments.player_2_slippi_release,
            player_1_assets=arguments.player_1_assets,
            player_2_assets=arguments.player_2_assets,
            player_1_name=arguments.player_1_name,
            player_2_name=arguments.player_2_name,
            player_1_temperature=arguments.player_1_temperature,
            player_2_temperature=arguments.player_2_temperature,
            stage=arguments.stage,
            max_game_frames=arguments.max_game_frames,
            seed=MATCH_SEED if arguments.seed is None else arguments.seed,
            require_natural_end=arguments.require_natural_end,
            inference_mode=inference_mode,
            artifact_label=arguments.artifact_label,
            save_slp=arguments.save_slp,
            save_video=arguments.save_video,
            allow_player_2_ood_character=arguments.allow_p2_ood_character,
        )
        summary = run_frisson_slippi_match(
            arguments.config,
            arguments.iso_path,
            frisson_slippi_request,
        )
    elif {player_1_model, player_2_model} == {"frisson-ai", "slippi-ai"}:
        raise ValueError("Frisson-versus-Slippi requires Frisson-AI on P1 and Slippi-AI on P2")
    elif "frisson-ai" in (player_1_model, player_2_model):
        if arguments.cpu_level is not None:
            raise ValueError("--cpu-level requires --p2 cpu")
        from melee_policy.integration.frisson_match import (
            MATCH_SEED,
            FrissonMatchRequest,
            run_frisson_match,
        )

        frisson_request = FrissonMatchRequest(
            player_1_model=player_1_model,
            player_2_model=player_2_model,
            player_1_character=arguments.player_1_character,
            player_2_character=arguments.player_2_character,
            player_1_checkpoint=arguments.player_1_checkpoint,
            player_2_checkpoint=arguments.player_2_checkpoint,
            player_1_assets=arguments.player_1_assets,
            player_2_assets=arguments.player_2_assets,
            player_1_name=arguments.player_1_name,
            player_2_name=arguments.player_2_name,
            player_1_temperature=arguments.player_1_temperature,
            player_2_temperature=arguments.player_2_temperature,
            stage=arguments.stage,
            max_game_frames=arguments.max_game_frames,
            seed=MATCH_SEED if arguments.seed is None else arguments.seed,
            require_natural_end=arguments.require_natural_end,
            inference_mode=inference_mode,
            artifact_label=arguments.artifact_label,
            save_slp=arguments.save_slp,
            save_video=arguments.save_video,
            allow_player_2_ood_character=arguments.allow_p2_ood_character,
        )
        summary = run_frisson_match(arguments.config, arguments.iso_path, frisson_request)
    elif "slippi-ai" in (player_1_model, player_2_model):
        if arguments.cpu_level is not None:
            raise ValueError("--cpu-level requires --p2 cpu")
        from melee_policy.integration.slippi_match import (
            SlippiMatchRequest,
            run_slippi_match,
        )

        slippi_request = SlippiMatchRequest(
            player_1_model=player_1_model,
            player_2_model=player_2_model,
            player_1_character=arguments.player_1_character,
            player_2_character=arguments.player_2_character,
            player_1_checkpoint=arguments.player_1_checkpoint,
            player_2_checkpoint=arguments.player_2_checkpoint,
            player_1_assets=arguments.player_1_assets,
            player_2_assets=arguments.player_2_assets,
            player_1_name=arguments.player_1_name,
            player_2_name=arguments.player_2_name,
            player_1_temperature=arguments.player_1_temperature,
            player_2_temperature=arguments.player_2_temperature,
            player_1_slippi_release=arguments.player_1_slippi_release,
            player_2_slippi_release=arguments.player_2_slippi_release,
            stage=arguments.stage,
            max_game_frames=arguments.max_game_frames,
            seed=arguments.seed,
            require_natural_end=arguments.require_natural_end,
            require_formal_game_end=arguments.require_formal_game_end,
            inference_mode=inference_mode,
            artifact_label=arguments.artifact_label,
            save_slp=arguments.save_slp,
            save_video=arguments.save_video,
            allow_player_1_ood_character=arguments.allow_p1_ood_character,
            allow_player_2_ood_character=arguments.allow_p2_ood_character,
        )
        summary = run_slippi_match(arguments.config, arguments.iso_path, slippi_request)
    else:
        if arguments.cpu_level is not None:
            raise ValueError("--cpu-level requires --p2 cpu")
        from melee_policy.integration.match_runtime import PlayRequest, run

        if any(
            value is not None
            for value in (
                arguments.player_1_name,
                arguments.player_2_name,
                arguments.player_1_temperature,
                arguments.player_2_temperature,
            )
        ):
            raise ValueError("policy name and per-player temperature are Slippi-AI options")
        legacy_request = PlayRequest(
            player_1_model=player_1_model,
            player_2_model=player_2_model,
            player_1_character=arguments.player_1_character,
            player_2_character=arguments.player_2_character,
            player_1_checkpoint=arguments.player_1_checkpoint,
            player_2_checkpoint=arguments.player_2_checkpoint,
            player_1_assets=arguments.player_1_assets,
            player_2_assets=arguments.player_2_assets,
            stage=arguments.stage,
            max_game_frames=arguments.max_game_frames,
            seed=arguments.seed,
            require_natural_end=arguments.require_natural_end,
            inference_mode=inference_mode,
            artifact_label=arguments.artifact_label,
            save_slp=arguments.save_slp,
            save_video=arguments.save_video,
        )
        summary = run(arguments.config, arguments.iso_path, legacy_request)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
