from __future__ import annotations
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
import pytest
from melee_policy.integration import tournament as tournament_module
from melee_policy.integration.runtime_identity import runtime_environment_identity_sha256
from melee_policy.integration.tournament import DEFAULT_ENTRANTS, DISPLAY_IDENTITIES, IMPLEMENTATION_PATHS, ChildProcessTimeout, TournamentRequest, TournamentRunError, _default_process_runner, _expected_child_summary_path, aggregate_results, build_schedule, run_tournament, wilson_interval_95

def test_schedule_covers_every_pair_seed_stage_and_mirrored_port_order() -> None:
    request = TournamentRequest(seeds=(11, 22), stages=('BATTLEFIELD', 'YOSHIS_STORY'), games_per_block=4, order_seed=91)
    schedule = build_schedule(request)
    assert len(schedule) == 1 * 2 * 2 * 4
    assert [game['execution_index'] for game in schedule] == list(range(1, len(schedule) + 1))
    assert all((game['character'] == 'FOX' for game in schedule))
    assert all((game['expected_replay_costumes'] == {'p1': 1, 'p2': 0} for game in schedule))
    assert all(('auto-selection' in game['replay_costume_assignment'] for game in schedule))
    assert all(('policy/evaluation sampling only' in game['policy_sampling_seed_scope'] for game in schedule))
    matched_blocks: dict[str, list[dict[str, Any]]] = {}
    for game in schedule:
        matched_blocks.setdefault(game['matched_port_block_id'], []).append(game)
    assert len(matched_blocks) == 1 * 2 * 2 * 2
    for block in matched_blocks.values():
        assert len(block) == 2
        assert {game['port_order'] for game in block} == {1, 2}
        assert block[0]['player_1_model'] == block[1]['player_2_model']
        assert block[0]['player_2_model'] == block[1]['player_1_model']
        assert len({game['policy_sampling_seed'] for game in block}) == 1
    seeds_by_derivation_input: dict[tuple[int, int], set[int]] = {}
    for game in schedule:
        key = (game['base_policy_sampling_seed'], game['mirror_repetition'])
        seeds_by_derivation_input.setdefault(key, set()).add(game['policy_sampling_seed'])
    assert all((len(effective_seeds) == 1 for effective_seeds in seeds_by_derivation_input.values()))
    assert len({next(iter(values)) for values in seeds_by_derivation_input.values()}) == len(seeds_by_derivation_input)
    assert build_schedule(request) == schedule
    other_order = build_schedule(TournamentRequest(seeds=request.seeds, stages=request.stages, games_per_block=request.games_per_block, order_seed=92))
    assert [game['game_id'] for game in other_order] != [game['game_id'] for game in schedule]

def test_request_rejects_unmatched_or_duplicate_configuration() -> None:
    with pytest.raises(ValueError, match='even integer'):
        TournamentRequest(games_per_block=3).validate()
    with pytest.raises(ValueError, match='distinct'):
        TournamentRequest(entrants=('mimic', 'mimic')).validate()
    with pytest.raises(ValueError, match='unique'):
        TournamentRequest(seeds=(42, 42)).validate()
    with pytest.raises(ValueError, match='unsupported tournament stages'):
        TournamentRequest(stages=('NO_STAGE',)).validate()
    with pytest.raises(ValueError, match='fixed tournament MIMIC entrant'):
        TournamentRequest(character='MARTH').validate()
    with pytest.raises(ValueError, match='child_wall_timeout_seconds'):
        TournamentRequest(child_wall_timeout_seconds=59.0).validate()
    TournamentRequest().validate()

def test_production_watchdog_kills_child_process_group_when_leader_exits(tmp_path: Path) -> None:
    process_groups: list[int] = []
    child_code = "import subprocess,sys; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])"
    with pytest.raises(ChildProcessTimeout) as captured:
        _default_process_runner([sys.executable, '-c', child_code], tmp_path, 0.1, on_start=process_groups.append)
    assert len(process_groups) == 1
    assert 'watchdog_shutdown=' in captured.value.stderr
    assert not tournament_module._process_group_exists(process_groups[0])

def test_report_lock_rejects_concurrent_orchestrator(tmp_path: Path) -> None:
    report_path = tmp_path / 'report.json'
    with tournament_module._exclusive_report_lock(report_path), pytest.raises(TournamentRunError, match='already locked'), tournament_module._exclusive_report_lock(report_path):
        raise AssertionError('a second orchestrator must not acquire the report lock')

def test_derived_policy_seed_collision_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tournament_module, '_derive_policy_sampling_seed', lambda _base_seed, _repetition: 7)
    with pytest.raises(RuntimeError, match='derived policy sampling seed collision'):
        build_schedule(TournamentRequest(seeds=(1, 2)))

def test_wilson_interval_documents_and_excludes_draws() -> None:
    interval = wilson_interval_95(5, 5)
    assert interval['estimate'] == 0.5
    assert interval['lower'] == pytest.approx(0.2365930905)
    assert interval['upper'] == pytest.approx(0.7634069095)
    assert 'draws excluded' in interval['method']
    empty = wilson_interval_95(0, 0)
    assert empty['sample_size'] == 0
    assert empty['estimate'] is None

def test_aggregate_schema_keeps_draw_fields_and_matched_blocks() -> None:
    request = TournamentRequest(entrants=('mimic', 'slippi-ai'))
    schedule = build_schedule(request)
    games = {game['game_id']: {'status': 'accepted', 'outcome': {'winner_model': game['player_1_model'], 'natural_game_end': True}} for game in schedule}
    aggregate = aggregate_results(request.entrants, schedule, games)
    pairing = aggregate['pairings']['mimic__slippi-ai']
    assert pairing['draws'] == 0
    assert pairing['matched_port_blocks']['complete'] == 1
    assert aggregate['matched_port_blocks']
