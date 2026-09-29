from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

import melee_policy.integration.frisson_policy as frisson
from melee_policy.integration.frisson_policy import (
    EXPECTED_ACTOR_CONTEXT_FRAMES,
    EXPECTED_MODEL_CONTEXT_LENGTH,
    EXPECTED_PARAMETER_COUNT,
    FRISSON_BC_CHECKPOINT_FORMAT,
    FRISSON_CHECKPOINT_FORMAT,
    FRISSON_POLICY_IDENTITY,
    FrissonPolicyConfig,
    FrissonPolicySession,
    capture_frisson_controller_command,
)
from melee_policy.integration.slippi_ai_policy import CanonicalControllerCommand


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _workspace_config(*, port: int = 1, opponent_port: int = 2) -> FrissonPolicyConfig:
    return FrissonPolicyConfig.from_project_root(
        _project_root(),
        port=port,
        opponent_port=opponent_port,
    )


def _parsed_controller(value: float = 0.5) -> Any:
    return SimpleNamespace(
        main_stick=SimpleNamespace(x=value, y=0.75),
        c_stick=SimpleNamespace(x=0.25, y=value),
        shoulder=0.35,
        buttons=SimpleNamespace(
            A=True,
            B=False,
            X=False,
            Y=True,
            Z=False,
            L=True,
            R=False,
            D_UP=False,
        ),
    )


def _parsed_nana(*, exists: bool) -> Any:
    return SimpleNamespace(
        exists=exists,
        percent=0,
        facing=True,
        x=0.0,
        y=0.0,
        action=0,
        invulnerable=False,
        character=0,
        jumps_left=0,
        shield_strength=0.0,
        on_ground=False,
    )


def _parsed_player(*, percent: int, controller_value: float) -> Any:
    return SimpleNamespace(
        percent=percent,
        facing=True,
        x=1.25,
        y=-2.5,
        action=14,
        invulnerable=False,
        character=1,
        jumps_left=2,
        shield_strength=60.0,
        on_ground=True,
        controller=_parsed_controller(controller_value),
        nana=_parsed_nana(exists=False),
    )


def _parsed_game() -> Any:
    items = {
        f"item_{index}": SimpleNamespace(
            exists=index == 3,
            type=index,
            state=index % 4,
            x=float(index),
            y=float(-index),
        )
        for index in range(15)
    }
    return SimpleNamespace(
        p0=_parsed_player(percent=17, controller_value=0.1),
        p1=_parsed_player(percent=83, controller_value=0.9),
        stage=32,
        randall=SimpleNamespace(x=0.0, y=0.0),
        fod_platforms=SimpleNamespace(left=1.0, right=2.0),
        items=SimpleNamespace(**items),
    )


def _runtime_modules() -> tuple[Any, dict[str, Any]]:
    torch = pytest.importorskip("torch")
    config = _workspace_config()
    if not config.model_source_directory.is_dir():
        pytest.skip("Frisson sibling source is unavailable")
    if not config.slippi_ai_source_directory.is_dir():
        pytest.skip("pinned Slippi-AI source is unavailable")
    return torch, frisson._activate_runtime_sources(config)


def test_config_freezes_the_deployed_actor_contract() -> None:
    config = _workspace_config()
    config.validate()

    assert config.sample_temperature == 1.0
    assert config.evaluation_seed == 0
    assert config.fast_step is True
    assert config.deterministic_algorithms is True
    assert config.context_mode == "ring"
    assert config.actor_context_frames == EXPECTED_ACTOR_CONTEXT_FRAMES == 128
    assert config.model_context_length == EXPECTED_MODEL_CONTEXT_LENGTH == 256
    assert config.delay_frames == 0
    assert config.batch_steps == 1

    with pytest.raises(ValueError, match=r"ports? must differ"):
        replace(config, opponent_port=config.port).validate()
    with pytest.raises(ValueError, match="actor contract mismatch"):
        replace(config, actor_context_frames=256).validate()
    with pytest.raises(ValueError, match="actor contract mismatch"):
        replace(config, sample_temperature=0.0).validate()
    replace(config, evaluation_seed=123).validate()
    with pytest.raises(ValueError, match="evaluation_seed"):
        replace(config, evaluation_seed=-1).validate()
    with pytest.raises(ValueError, match="provided together"):
        replace(config, model_source_repository=config.model_source_directory).validate()
    with pytest.raises(ValueError, match="without traversal"):
        replace(
            config,
            model_source_repository=config.model_source_directory,
            model_source_revision="a" * 40,
            model_source_subdirectory="../source",
        ).validate()


def test_pinned_source_materialization_uses_commit_not_dirty_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    source = repository / "frisson-source"
    source.mkdir(parents=True)
    committed = {
        "model.py": b"MODEL = 'committed'\n",
        "controller_codec.py": b"CODEC = 'committed'\n",
        "tensor_batch.py": b"BATCH = 'committed'\n",
    }
    for filename, content in committed.items():
        (source / filename).write_bytes(content)
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "frisson-source"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Frisson Test",
            "-c",
            "user.email=frisson-test@example.invalid",
            "commit",
            "-qm",
            "pin runtime",
        ],
        check=True,
    )
    revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(frisson, "MODEL_SOURCE_SHA256", hashlib.sha256(committed["model.py"]).hexdigest())
    monkeypatch.setattr(
        frisson,
        "CONTROLLER_CODEC_SOURCE_SHA256",
        hashlib.sha256(committed["controller_codec.py"]).hexdigest(),
    )
    monkeypatch.setattr(
        frisson,
        "TENSOR_BATCH_SOURCE_SHA256",
        hashlib.sha256(committed["tensor_batch.py"]).hexdigest(),
    )

    for filename in committed:
        (source / filename).write_bytes(b"dirty working tree\n")
    destination = tmp_path / "cache" / revision
    resolved = frisson.materialize_pinned_model_source(
        repository=repository,
        revision=revision,
        source_subdirectory="frisson-source",
        destination=destination,
    )

    assert resolved == destination.resolve()
    assert {filename: (resolved / filename).read_bytes() for filename in committed} == committed
    (resolved / "model.py").chmod(0o644)
    (resolved / "model.py").write_bytes(b"corrupt cache\n")
    with pytest.raises(RuntimeError, match="snapshot identity mismatch"):
        frisson.materialize_pinned_model_source(
            repository=repository,
            revision=revision,
            source_subdirectory="frisson-source",
            destination=destination,
        )


def test_scalar_parser_game_becomes_exact_sibling_batch_dataclasses() -> None:
    torch, modules = _runtime_modules()
    batch = frisson._tensorize_parsed_game(
        _parsed_game(),
        tensor_batch=modules["tensor_batch"],
        torch=torch,
        device=torch.device("cpu"),
    )

    assert type(batch).__name__ == "GameStateBatch"
    assert tuple(batch.stage.shape) == (1, 1)
    assert tuple(batch.p0.percent.shape) == (1, 1)
    assert tuple(batch.p0.controller.main_stick.x.shape) == (1, 1)
    assert tuple(batch.items.exists.shape) == (1, 1, 15)
    assert tuple(batch.items.type.shape) == (1, 1, 15)
    assert batch.p0.percent.item() == 17
    assert batch.p1.percent.item() == 83
    assert batch.items.exists[0, 0, 3].item() is True
    assert batch.items.type[0, 0].tolist() == list(range(15))


def test_custom_v1_capture_maps_shared_shoulder_to_l_and_neutral_r() -> None:
    torch = pytest.importorskip("torch")
    decoded = SimpleNamespace(
        main_stick=SimpleNamespace(x=torch.tensor([0.2]), y=torch.tensor([0.8])),
        c_stick=SimpleNamespace(x=torch.tensor([0.4]), y=torch.tensor([0.6])),
        shoulder=torch.tensor([0.35]),
        buttons=SimpleNamespace(
            A=torch.tensor([True]),
            B=torch.tensor([False]),
            X=torch.tensor([False]),
            Y=torch.tensor([True]),
            Z=torch.tensor([False]),
            L=torch.tensor([True]),
            R=torch.tensor([False]),
            D_UP=torch.tensor([True]),
        ),
    )

    command = capture_frisson_controller_command(decoded)

    assert command.main_stick == pytest.approx((0.2, 0.8))
    assert command.c_stick == pytest.approx((0.4, 0.6))
    assert command.analog_l == pytest.approx(0.35)
    assert command.analog_r == 0.0
    assert command.buttons == ("A", "Y", "L", "D_UP")


class _FakeBackbone:
    def __init__(self, torch: Any, calls: list[str]) -> None:
        self.torch = torch
        self.calls = calls
        self.init_count = 0

    def init_cache(self, batch_size: int, *, device: str) -> Any:
        self.calls.append("init_cache")
        self.init_count += 1
        return SimpleNamespace(
            generation=self.init_count,
            batch_size=batch_size,
            device=device,
            capacity=256,
            valid_length=self.torch.zeros(1, dtype=self.torch.long),
            write_position=self.torch.zeros(1, dtype=self.torch.long),
            next_position=self.torch.zeros(1, dtype=self.torch.long),
        )

    def step(self, encoded: Any, cache: Any, *, reset_mask: Any) -> tuple[Any, Any]:
        self.calls.append("backbone.step")
        assert tuple(encoded.shape) == (1, 4)
        assert tuple(reset_mask.shape) == (1,)
        return self.torch.ones((1, 4)), cache


class _FakeHead:
    def __init__(self, torch: Any, calls: list[str]) -> None:
        self.torch = torch
        self.calls = calls
        self.generators: list[Any] = []

    def sample(
        self,
        hidden: Any,
        controller: Any,
        *,
        temperature: float,
        generator: Any,
    ) -> Any:
        self.calls.append("controller_head.sample")
        self.generators.append(generator)
        assert tuple(hidden.shape) == (1, 4)
        assert tuple(controller.shoulder.shape) == (1,)
        assert temperature == 1.0
        self.torch.rand((), generator=generator)
        false = self.torch.tensor([False])
        true = self.torch.tensor([True])
        controller_state = SimpleNamespace(
            main_stick=SimpleNamespace(
                x=self.torch.tensor([0.5]),
                y=self.torch.tensor([0.75]),
            ),
            c_stick=SimpleNamespace(
                x=self.torch.tensor([0.25]),
                y=self.torch.tensor([0.5]),
            ),
            shoulder=self.torch.tensor([0.35]),
            buttons=SimpleNamespace(
                A=true,
                B=false,
                X=false,
                Y=false,
                Z=false,
                L=true,
                R=false,
                D_UP=false,
            ),
        )
        return SimpleNamespace(
            controller_state=controller_state,
            logits={
                "buttons": self.torch.zeros((1, 728)),
                "main_stick": self.torch.zeros((1, 85)),
            },
        )


class _FakePolicy:
    def __init__(self, torch: Any, calls: list[str]) -> None:
        self.torch = torch
        self.calls = calls
        self.backbone = _FakeBackbone(torch, calls)
        self.controller_head = _FakeHead(torch, calls)

    def encoder(self, game: Any, controller: Any) -> Any:
        self.calls.append("encoder")
        assert tuple(game.stage.shape) == (1, 1)
        assert tuple(controller.shoulder.shape) == (1, 1)
        return self.torch.ones((1, 1, 4))


def test_inference_order_uses_private_rng_and_no_delay_fifo() -> None:
    torch, modules = _runtime_modules()
    batch = frisson._tensorize_parsed_game(
        _parsed_game(),
        tensor_batch=modules["tensor_batch"],
        torch=torch,
        device=torch.device("cpu"),
    )
    calls: list[str] = []
    policy = _FakePolicy(torch, calls)
    cache = policy.backbone.init_cache(1, device="cpu")
    private_generator = torch.Generator(device="cpu").manual_seed(0)
    global_rng_before = torch.random.get_rng_state().clone()

    command, returned_cache, _ = frisson._sample_policy_frame(
        policy=policy,
        cache=cache,
        game_batch=batch,
        reset=True,
        generator=private_generator,
        torch=torch,
    )

    assert calls == ["init_cache", "encoder", "backbone.step", "controller_head.sample"]
    assert policy.controller_head.generators == [private_generator]
    assert torch.equal(torch.random.get_rng_state(), global_rng_before)
    assert returned_cache is cache
    assert command.buttons == ("A", "L")
    assert command.analog_r == 0.0


class _FakeParser:
    creations: ClassVar[list[tuple[int, int]]] = []

    def __init__(self, ports: tuple[int, int]) -> None:
        self.creations.append(ports)

    def get_game(self, gamestate: Any) -> Any:
        return _parsed_game()


def test_native_runtime_recreates_parser_and_cache_only_on_frame_minus_123() -> None:
    torch, modules = _runtime_modules()
    calls: list[str] = []
    policy = _FakePolicy(torch, calls)
    config = _workspace_config()
    runtime = frisson._NativeFrissonRuntime(config)
    runtime._started = True
    runtime._torch = torch
    runtime._modules = {
        "parser": SimpleNamespace(Parser=_FakeParser),
        "tensor_batch": modules["tensor_batch"],
    }
    runtime._policy = policy
    runtime._generator = torch.Generator(device="cpu").manual_seed(0)
    _FakeParser.creations = []

    first = runtime.step(SimpleNamespace(frame=-123))
    second = runtime.step(SimpleNamespace(frame=-122))
    third = runtime.step(SimpleNamespace(frame=-123))

    assert all(isinstance(command, CanonicalControllerCommand) for command in (first, second, third))
    assert _FakeParser.creations == [(1, 2), (1, 2)]
    assert policy.backbone.init_count == 2
    assert calls.count("encoder") == 3
    assert calls.count("controller_head.sample") == 3
    assert runtime.diagnostics()["parser_resets"] == 2
    assert runtime.diagnostics()["cache_resets"] == 2


class _FakeRuntime:
    def __init__(self) -> None:
        self.started = False
        self.closed = False
        self.frames: list[int] = []

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "checkpoint": {
                "format": FRISSON_CHECKPOINT_FORMAT,
                "step": 123,
                "sha256": "a" * 64,
            }
        }

    def start(self) -> None:
        self.started = True

    def step(self, gamestate: Any) -> CanonicalControllerCommand:
        self.frames.append(gamestate.frame)
        return CanonicalControllerCommand.neutral()

    def diagnostics(self) -> dict[str, Any]:
        return {"frames": len(self.frames)}

    def close(self) -> None:
        self.closed = True


def test_session_requires_exact_frames_and_reports_synchronous_barriers(tmp_path: Path) -> None:
    config = FrissonPolicyConfig(
        model_source_directory=tmp_path / "source",
        slippi_ai_source_directory=tmp_path / "slippi",
        checkpoint_path=tmp_path / "model.pt",
        port=2,
        opponent_port=1,
        evaluation_seed=123,
    )
    runtime = _FakeRuntime()
    session = FrissonPolicySession(config, runtime_factory=lambda ignored: runtime)

    assert session.step(SimpleNamespace(frame=-123)) == CanonicalControllerCommand.neutral()
    assert session.step(SimpleNamespace(frame=-122)) == CanonicalControllerCommand.neutral()
    with pytest.raises(ValueError, match="expected -121"):
        session.step(SimpleNamespace(frame=-120))

    diagnostics = session.diagnostics()
    assert runtime.frames == [-123, -122]
    assert diagnostics["frames_total"] == 2
    assert diagnostics["current_frame_inference_barriers"] == 2
    assert diagnostics["current_frame_inference_barrier_every_frame"] is True
    metadata = session.metadata()
    assert metadata["identity"]["policy"] == FRISSON_POLICY_IDENTITY
    assert metadata["identity"]["checkpoint_step"] == 123
    assert metadata["identity"]["checkpoint_sha256"] == "a" * 64
    assert metadata["identity"]["policy_instance"].endswith(f"/step-123/sha256-{'a' * 64}")
    assert metadata["runtime_contract"]["periodic_128_frame_reset"] is False
    assert metadata["runtime_contract"]["evaluation_seed"] == 123
    assert metadata["runtime_contract"]["private_torch_generator_seed"] == 123
    assert metadata["runtime_contract"]["kv_cache_capacity_frames"] == 256
    assert metadata["action_contract"]["slippi_ai_21_frame_fifo"] is False
    assert metadata["action_contract"]["expected_replay_audit_lag_frames"] == 1
    session.close()
    assert runtime.closed is True


def test_session_reset_requires_new_minus_123_frame(tmp_path: Path) -> None:
    config = FrissonPolicyConfig(
        model_source_directory=tmp_path / "source",
        slippi_ai_source_directory=tmp_path / "slippi",
        checkpoint_path=tmp_path / "model.pt",
        port=1,
        opponent_port=2,
    )
    runtime = _FakeRuntime()
    session = FrissonPolicySession(config, runtime_factory=lambda ignored: runtime)
    session.step(SimpleNamespace(frame=-123))
    session.reset("next game")
    with pytest.raises(ValueError, match="must be -123"):
        session.step(SimpleNamespace(frame=-122))
    session.step(SimpleNamespace(frame=-123))
    assert session.diagnostics()["resets"] == 1


def test_real_checkpoint_safe_strict_load_canary(monkeypatch: pytest.MonkeyPatch) -> None:
    torch, modules = _runtime_modules()
    config = _workspace_config()
    if not config.checkpoint_path.is_file():
        pytest.skip("downloaded Frisson checkpoint is unavailable")
    original_load = torch.load
    observed_kwargs: dict[str, Any] = {}

    def recording_load(*args: Any, **kwargs: Any) -> Any:
        observed_kwargs.update(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    policy, identity = frisson._load_checkpoint(
        config.checkpoint_path,
        torch=torch,
        model_module=modules["model"],
    )

    assert observed_kwargs == {"map_location": "cpu", "weights_only": True, "mmap": True}
    assert identity["format"] == FRISSON_CHECKPOINT_FORMAT
    assert identity["parameter_count"] == EXPECTED_PARAMETER_COUNT
    assert identity["state_tensor_count"] == 125
    assert identity["strict_state_dict"] is True
    assert identity["all_state_tensors_finite"] is True
    assert identity["model_config"]["context_length"] == 256
    assert identity["model_config"]["compute_dtype"] == "float32"
    assert identity["model_config"]["cache_dtype"] == "float32"
    assert identity["deployed_actor"]["actor"]["context_frames"] == 128
    assert identity["deployed_actor"]["policy"]["fast_step"] is True
    assert policy.parameter_counts()["total"] == EXPECTED_PARAMETER_COUNT


@pytest.mark.parametrize(
    ("filename", "expected_step", "expected_sha256", "expected_bytes"),
    [
        (
            "melee-rl-mixed-50-50-step123.pt",
            123,
            "45ae503fe422b4e7a0f281ed4c7b887d89d8a4692b539fa99f598f4db8b2e920",
            80_210_061,
        ),
        (
            "melee-rl-ko-weighted-step127.pt",
            127,
            "c924418e721bdf3b64f0a54507450930678c7785fd264fdd223a797bb038045c",
            80_210_125,
        ),
    ],
)
def test_requested_bc_checkpoint_safe_strict_load_canary(
    filename: str,
    expected_step: int,
    expected_sha256: str,
    expected_bytes: int,
) -> None:
    torch, modules = _runtime_modules()
    checkpoint = _project_root() / ".e010-cache" / "requested-checkpoints" / filename
    if not checkpoint.is_file():
        pytest.skip(f"requested checkpoint is unavailable: {filename}")

    policy, identity = frisson._load_checkpoint(
        checkpoint,
        torch=torch,
        model_module=modules["model"],
    )

    assert identity["format"] == FRISSON_BC_CHECKPOINT_FORMAT
    assert identity["step"] == expected_step
    assert identity["sha256"] == expected_sha256
    assert identity["byte_length"] == expected_bytes
    assert identity["config_yaml_sha256"] is None
    assert identity["training"]["step"] == expected_step
    assert identity["deployed_actor"]["checkpoint_embedded"] is False
    assert identity["deployed_actor"]["source"] == "launcher-pinned-frisson-evaluation.v1"
    assert identity["strict_state_dict"] is True
    assert identity["all_state_tensors_finite"] is True
    assert policy.parameter_counts()["total"] == EXPECTED_PARAMETER_COUNT


@pytest.mark.parametrize(
    (
        "run_id",
        "filename",
        "profile",
        "trial_id",
        "optimizer_steps",
        "processed_target_frames",
        "state_tensor_count",
        "parameter_count",
        "expected_sha256",
        "expected_bytes",
    ),
    [
        (
            "mpr-7310252a57a246be08f1",
            "frames-3499819008.pt",
            "75m",
            "75m-muon-low",
            53_403,
            3_499_819_008,
            205,
            75_305_709,
            "59542f7fd82d8e9c076e5cfbca8f8ac4724b2f30727048f5be8ee8b0a7ee4299",
            644_290_579,
        ),
        (
            "mpr-5b2bfa51c64dd715f8e2",
            "frames-3499819008.pt",
            "10m",
            "10m-wd-low",
            53_403,
            3_499_819_008,
            109,
            10_163_629,
            "d1771d01bcfc506b51ec89ff4937f4258b8db20a255c156b6f0135579e70eb6a",
            96_451_691,
        ),
        (
            "mpr-00d5f6f00b122510125b",
            "frames-7999586304.pt",
            "10m",
            "10m-muon-low",
            122_064,
            7_999_586_304,
            109,
            10_163_629,
            "5a54ea4ecfa150198180dff4d06ac4fe6ecec5d9433803cc13b527e41f41d14e",
            96_451_691,
        ),
    ],
)
def test_family_checkpoint_safe_strict_load_canary(
    run_id: str,
    filename: str,
    profile: str,
    trial_id: str,
    optimizer_steps: int,
    processed_target_frames: int,
    state_tensor_count: int,
    parameter_count: int,
    expected_sha256: str,
    expected_bytes: int,
) -> None:
    torch, modules = _runtime_modules()
    checkpoint = (
        _project_root()
        / ".e010-cache"
        / "requested-checkpoints"
        / "wandb-best"
        / run_id
        / "checkpoint"
        / filename
    )
    if not checkpoint.is_file():
        pytest.skip(f"family checkpoint is unavailable: {run_id}")

    inspected = frisson.inspect_frisson_checkpoint(checkpoint)
    policy, identity = frisson._load_checkpoint(
        checkpoint,
        torch=torch,
        model_module=modules["model"],
    )

    assert inspected["format"] == frisson.FRISSON_FAMILY_CHECKPOINT_FORMAT
    assert inspected["profile"] == profile
    assert inspected["step"] == optimizer_steps
    assert inspected["processed_target_frames"] == processed_target_frames
    assert inspected["state_tensor_count"] == state_tensor_count
    assert inspected["parameter_count"] == parameter_count
    assert inspected["sha256"] == expected_sha256
    assert inspected["byte_length"] == expected_bytes
    assert identity["sha256"] == inspected["sha256"]
    assert identity["family_pretraining"]["wandb_run_id"] == run_id
    assert identity["family_pretraining"]["trial_id"] == trial_id
    assert identity["codec"] == {"name": "custom_v1", "vocab_sizes": {"buttons": 728, "main_stick": 85}}
    assert identity["runtime_model_config"]["action_offset_frames"] == 1
    assert identity["runtime_model_config"]["compute_dtype"] == "float32"
    assert identity["runtime_model_config"]["cache_dtype"] == "float32"
    assert identity["model_config_compatibility"]["checkpoint_prevalidated_inputs"] is True
    assert identity["model_config_compatibility"]["checked_runtime_inputs"] is True
    assert identity["producing_source_attestation"]["cryptographic_producing_source_verification"] is False
    assert identity["strict_state_dict"] is True
    assert policy.parameter_counts()["total"] == parameter_count
    assert all(parameter.dtype == torch.float32 for parameter in policy.parameters())


@pytest.mark.parametrize(
    (
        "filename",
        "profile",
        "step",
        "processed_target_frames",
        "source_checkpoint_sha256",
        "expected_sha256",
        "expected_bytes",
    ),
    [
        (
            "frisson-melee-10m-posttrained-best-val.pt",
            "10m",
            195_248,
            12_795_772_928,
            "f5c35e51beadadb41a297d77a882b5a8957a34e4c8cd4ddba17cd51170442f31",
            "63b5ff05ef30476c4f41590c478f4a7218f3b72a8b5ef24a0e336eeb2b7c287b",
            96_452_075,
        ),
        (
            "frisson-melee-75m-posttrained-best-val.pt",
            "75m",
            127_214,
            8_337_096_704,
            "855deb6416f3d0333c55cde921aa5476e5b9a02873a6dea4e513d89bdf09c8eb",
            "8211f1832198646e9f4e3bacde26f181614f93320e68bd326dd5942c4e0f4077",
            644_291_987,
        ),
    ],
)
def test_posttraining_winner_safe_strict_load_canary(
    filename: str,
    profile: str,
    step: int,
    processed_target_frames: int,
    source_checkpoint_sha256: str,
    expected_sha256: str,
    expected_bytes: int,
) -> None:
    torch, modules = _runtime_modules()
    checkpoint = _project_root() / ".e013-cache" / "posttraining-winners" / filename
    if not checkpoint.is_file():
        pytest.skip(f"post-training checkpoint is unavailable: {filename}")

    inspected = frisson.inspect_frisson_checkpoint(checkpoint)
    policy, identity = frisson._load_checkpoint(
        checkpoint,
        torch=torch,
        model_module=modules["model"],
    )

    assert inspected["format"] == frisson.FRISSON_POSTTRAINING_CHECKPOINT_FORMAT
    assert inspected["profile"] == profile
    assert inspected["step"] == step
    assert inspected["processed_target_frames"] == processed_target_frames
    assert inspected["sha256"] == expected_sha256
    assert inspected["byte_length"] == expected_bytes
    assert inspected["safe_load"]["unsafe_globals"] == ["builtins.frozenset"]
    assert identity["sha256"] == inspected["sha256"]
    assert identity["posttraining"]["source_checkpoint_sha256"] == source_checkpoint_sha256
    assert identity["posttraining"]["dataset_identity"]["dataset_manifest_id"] == (
        "e018460bb8b2d5398b9005217a8cc0ba1ff0f93fb3fe8fb26b7e5f18e03d4ee3"
    )
    assert identity["posttraining"]["curriculum"]["arm"] == "B-mix10"
    assert identity["deployed_actor"]["policy"]["profile"] == profile
    assert identity["deployed_actor"]["actor"]["delay_frames"] == 0
    assert identity["strict_state_dict"] is True
    assert policy.parameter_counts()["total"] == identity["parameter_count"]
    assert all(parameter.dtype == torch.float32 for parameter in policy.parameters())


def test_safe_checkpoint_loader_allows_only_frozenset_for_trusted_posttraining(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "trusted-posttraining.pt"
    checkpoint.write_bytes(b"synthetic trusted checkpoint")
    observed_safe_globals: list[list[Any]] = []
    observed_load_kwargs: dict[str, Any] = {}

    class SafeGlobalsContext:
        def __init__(self, values: list[Any]) -> None:
            observed_safe_globals.append(values)

        def __enter__(self) -> None:
            return None

        def __exit__(self, *ignored: Any) -> None:
            return None

    def fake_load(*args: Any, **kwargs: Any) -> dict[str, Any]:
        observed_load_kwargs.update(kwargs)
        return {}

    trusted_sha256 = next(iter(frisson._TRUSTED_POSTTRAINING_SHA256))
    monkeypatch.setattr(
        torch.serialization,
        "get_unsafe_globals_in_checkpoint",
        lambda path: ["builtins.frozenset"],
    )
    monkeypatch.setattr(torch.serialization, "safe_globals", SafeGlobalsContext)
    monkeypatch.setattr(torch, "load", fake_load)
    monkeypatch.setattr(frisson, "_sha256_file", lambda path: trusted_sha256)

    assert frisson._safe_load_checkpoint_payload(checkpoint, torch=torch) == {}
    assert observed_safe_globals == [[frozenset]]
    assert observed_load_kwargs == {"map_location": "cpu", "weights_only": True, "mmap": True}


def test_safe_checkpoint_loader_rejects_frozenset_for_untrusted_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "untrusted-frozenset.pt"
    checkpoint.write_bytes(b"synthetic untrusted checkpoint")
    monkeypatch.setattr(
        torch.serialization,
        "get_unsafe_globals_in_checkpoint",
        lambda path: ["builtins.frozenset"],
    )
    monkeypatch.setattr(frisson, "_sha256_file", lambda path: "0" * 64)

    with pytest.raises(RuntimeError, match="content-addressed trusted post-training checkpoint"):
        frisson._safe_load_checkpoint_payload(checkpoint, torch=torch)


@pytest.mark.parametrize(
    "unsafe_globals",
    [
        ["collections.OrderedDict"],
        ["builtins.frozenset", "collections.OrderedDict"],
    ],
)
def test_safe_checkpoint_loader_rejects_every_other_global(
    unsafe_globals: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "unsafe.pt"
    checkpoint.write_bytes(b"synthetic unsafe checkpoint")
    monkeypatch.setattr(
        torch.serialization,
        "get_unsafe_globals_in_checkpoint",
        lambda path: unsafe_globals,
    )

    with pytest.raises(RuntimeError, match="unsafe globals"):
        frisson._safe_load_checkpoint_payload(checkpoint, torch=torch)


def test_posttraining_contract_rejects_crossed_curriculum_source_hash() -> None:
    torch = pytest.importorskip("torch")
    checkpoint = (
        _project_root() / ".e013-cache" / "posttraining-winners" / "frisson-melee-10m-posttrained-best-val.pt"
    )
    if not checkpoint.is_file():
        pytest.skip("post-training checkpoint is unavailable")
    with torch.serialization.safe_globals([frozenset]):
        payload = dict(torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True))
    resolved = dict(payload["resolved_config"])
    curriculum = dict(resolved["curriculum"])
    curriculum["source_checkpoint_sha256"] = "0" * 64
    resolved["curriculum"] = curriculum
    payload["resolved_config"] = resolved

    with pytest.raises(RuntimeError, match="curriculum metadata is not self-consistent"):
        frisson._posttraining_checkpoint_contract(payload)


def test_family_checkpoint_rejects_crossed_wandb_metadata_identity() -> None:
    torch = pytest.importorskip("torch")
    checkpoint = (
        _project_root()
        / ".e010-cache"
        / "requested-checkpoints"
        / "wandb-best"
        / "mpr-00d5f6f00b122510125b"
        / "checkpoint"
        / "frames-7999586304.pt"
    )
    if not checkpoint.is_file():
        pytest.skip("family checkpoint is unavailable")
    payload = dict(torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True))
    metadata = dict(payload["metadata"])
    metadata["wandb_run_id"] = "mpr-11111111111111111111"
    payload["metadata"] = metadata

    with pytest.raises(RuntimeError, match="metadata identity mismatch"):
        frisson._family_checkpoint_contract(payload)


def _synthetic_checkpoint_payload(torch: Any, *, step: Any) -> dict[str, Any]:
    return {
        "format": FRISSON_CHECKPOINT_FORMAT,
        "state_dict": {"weight": torch.ones(1)},
        "model_config": {
            "compute_dtype": "float32",
            "cache_dtype": "float32",
            "action_offset_frames": 1,
            "context_length": 256,
            "prevalidated_inputs": False,
        },
        "codec": {
            "name": "custom_v1",
            "vocab_sizes": {"buttons": 728, "main_stick": 85},
        },
        "slippi_ai_commit": frisson.SLIPPI_AI_SOURCE_REVISION,
        "config_yaml_sha256": "1" * 64,
        "step": step,
        "rl_config": {
            "format": frisson.FRISSON_CONFIG_FORMAT,
            "runtime": {"seed": 0, "deterministic": True},
            "policy": {
                "profile": "20m",
                "seed": 0,
                "compute_dtype": "float32",
                "cache_dtype": "float32",
                "fast_step": True,
            },
            "actor": {
                "context_frames": 128,
                "delay_frames": 0,
                "context_mode": "ring",
                "batch_steps": 1,
                "temperature": 1.0,
                "seed": 0,
            },
        },
        "delay_frames": 0,
        "context_mode": "ring",
    }


def _synthetic_bc_checkpoint_payload(torch: Any, *, step: Any) -> dict[str, Any]:
    rl_payload = _synthetic_checkpoint_payload(torch, step=step)
    return {
        key: value
        for key, value in rl_payload.items()
        if key
        in {
            "state_dict",
            "model_config",
            "codec",
            "slippi_ai_commit",
            "config_yaml_sha256",
        }
    } | {
        "format": FRISSON_BC_CHECKPOINT_FORMAT,
        "config_yaml_sha256": None,
        "created": "2026-08-28T00:00:00Z",
        "training": {
            "dataset": "rl_post_training",
            "recipe": "test recipe",
            "run": "test-run",
            "source_checkpoint": f"melee-rl-runs:/runs/test/best_{step}.pt",
            "source_sha256": "2" * 64,
            "step": step,
        },
    }


@pytest.mark.parametrize("step", [123, 127])
def test_bc_checkpoint_envelope_uses_training_step_and_pinned_evaluation_actor(
    step: int,
) -> None:
    torch = pytest.importorskip("torch")
    payload = _synthetic_bc_checkpoint_payload(torch, step=step)

    checkpoint_format, observed_step, config_sha, actor, training = frisson._checkpoint_envelope(payload)

    assert checkpoint_format == FRISSON_BC_CHECKPOINT_FORMAT
    assert observed_step == step
    assert config_sha is None
    assert actor["source"] == "launcher-pinned-frisson-evaluation.v1"
    assert actor["checkpoint_embedded"] is False
    assert actor["actor"]["delay_frames"] == 0
    assert actor["actor"]["context_mode"] == "ring"
    assert actor["actor"]["temperature"] == 1.0
    assert training == payload["training"]


@pytest.mark.parametrize("step", [123, 127, 304])
def test_checkpoint_loader_accepts_and_records_compatible_training_steps(
    step: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")

    @dataclass(frozen=True)
    class ModelConfig:
        compute_dtype: str = "float32"
        cache_dtype: str = "float32"
        action_offset_frames: int = 1
        context_length: int = 256

    class Policy(torch.nn.Module):
        def __init__(self, config: ModelConfig) -> None:
            super().__init__()
            assert asdict(config) == asdict(ModelConfig())
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.codec = SimpleNamespace(vocabulary_sizes=(728, 85))

        def parameter_counts(self) -> dict[str, int]:
            return {"total": 1}

    checkpoint = tmp_path / f"step-{step}.pt"
    checkpoint.write_bytes(b"synthetic-safe-checkpoint")
    payload = _synthetic_checkpoint_payload(torch, step=step)
    monkeypatch.setattr(torch.serialization, "get_unsafe_globals_in_checkpoint", lambda path: [])
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: payload)
    monkeypatch.setattr(frisson, "EXPECTED_STATE_TENSOR_COUNT", 1)
    monkeypatch.setattr(frisson, "EXPECTED_PARAMETER_COUNT", 1)

    policy, identity = frisson._load_checkpoint(
        checkpoint,
        torch=torch,
        model_module=SimpleNamespace(ModelConfig=ModelConfig, MeleePolicy=Policy),
    )

    assert identity["step"] == step
    assert identity["model_config"]["prevalidated_inputs"] is False
    assert identity["runtime_model_config"] == asdict(ModelConfig())
    assert identity["model_config_compatibility"]["checked_runtime_inputs"] is True
    assert policy.weight.item() == 1.0


@pytest.mark.parametrize("step", [-1, True, 1.5, "123"])
def test_checkpoint_loader_rejects_invalid_training_steps(
    step: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "invalid-step.pt"
    checkpoint.write_bytes(b"synthetic-safe-checkpoint")
    monkeypatch.setattr(torch.serialization, "get_unsafe_globals_in_checkpoint", lambda path: [])
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: _synthetic_checkpoint_payload(torch, step=step),
    )

    with pytest.raises(RuntimeError, match="training step"):
        frisson._load_checkpoint(
            checkpoint,
            torch=torch,
            model_module=SimpleNamespace(),
        )


def test_checkpoint_loader_rejects_unpinned_prevalidated_input_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "prevalidated.pt"
    checkpoint.write_bytes(b"synthetic-safe-checkpoint")
    payload = _synthetic_checkpoint_payload(torch, step=123)
    payload["model_config"]["prevalidated_inputs"] = True
    monkeypatch.setattr(torch.serialization, "get_unsafe_globals_in_checkpoint", lambda path: [])
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: payload)

    with pytest.raises(RuntimeError, match="unpinned prevalidated-input runtime"):
        frisson._load_checkpoint(
            checkpoint,
            torch=torch,
            model_module=SimpleNamespace(),
        )


def test_checkpoint_loader_rejects_unknown_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "unknown.pt"
    checkpoint.write_bytes(b"not read because torch.load is patched")
    payload = {
        "format": "unknown",
        "state_dict": {},
        "model_config": {},
        "codec": {},
        "slippi_ai_commit": "unknown",
        "config_yaml_sha256": "0" * 64,
        "step": 0,
        "rl_config": {},
        "delay_frames": 0,
        "context_mode": "ring",
    }
    monkeypatch.setattr(torch.serialization, "get_unsafe_globals_in_checkpoint", lambda path: [])
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: payload)
    with pytest.raises(RuntimeError, match="format mismatch"):
        frisson._load_checkpoint(
            checkpoint,
            torch=torch,
            model_module=SimpleNamespace(),
        )
