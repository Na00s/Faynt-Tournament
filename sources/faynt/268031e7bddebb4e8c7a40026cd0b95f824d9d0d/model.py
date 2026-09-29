"""Slippi-compatible PyTorch behavior-cloning policy.

The representation contract is pinned to vladfi1/slippi-ai commit
577965a7731dc53e3472ea63d9e9853a4e9d65fa.  A training position contains the
canonical post-frame state and recorded pre-frame controller at frame ``t``;
the caller supplies the already aligned controller from frame ``t + 1`` to
``MeleePolicy.loss``.  This module never shifts targets internally.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from controller_codec import ControllerLabels, ControllerState, CustomV1Codec

TensorTree = Mapping[str, Any]
ResidualMode = Literal["standard", "full_attnres"]

_SIZE_FIELDS = frozenset({"d_model", "n_layers", "n_heads", "n_kv_heads", "head_dim", "d_ff"})
_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


@dataclass(frozen=True)
class ModelConfig:
    """Complete, immutable policy configuration resolved from ``config.yaml``."""

    profile: str = "20m"
    framework: str = "pytorch"
    slippi_ai_commit: str = "577965a7731dc53e3472ea63d9e9853a4e9d65fa"

    encoder_type: str = "slippi_ai_enhanced"
    encoder_hidden_size: int = 128
    item_mlp_layers: int = 2
    controller_rnn_cell: str = "lstm"
    use_self_nana: bool = True
    use_controller_rnn: bool = False
    use_learned_character: bool = True
    use_learned_action: bool = True
    use_character_action_joint: bool = True
    use_item_sum: bool = True
    use_items: bool = True
    hybrid_embed: bool = False
    use_randall: bool = True
    use_fod_platforms: bool = True
    condition_on_player_name: bool = False
    player_name_vocab_size: int = 0

    action_offset_frames: int = 1
    action_codec: str = "custom_v1"

    backbone_type: str = "causal_transformer"
    context_length: int = 256
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 2
    head_dim: int = 64
    d_ff: int = 1216
    attention_type: str = "full"
    attention_output_gate: str = "elementwise_sigmoid"
    attention_gate_bias: bool = False
    attention_gate_compute_dtype: str = "float32"
    residual_mode: ResidualMode = "full_attnres"
    norm: str = "rmsnorm"
    norm_eps: float = 1.0e-6
    qk_norm: bool = True
    activation: str = "swiglu"
    position_encoding: str = "rope"
    rope_theta: float = 10_000.0
    bias: bool = False
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    attention_implementation: str = "sdpa"
    use_kv_cache: bool = True
    gradient_checkpointing: bool = True
    initializer_std: float = 0.02

    controller_head_type: str = "slippi_ai_autoregressive"
    component_depth: int = 2
    controller_residual_size: int = 128

    parameter_dtype: str = "float32"
    compute_dtype: str = "bfloat16"
    cache_dtype: str = "bfloat16"
    softmax_dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.n_heads <= 0:
            raise ValueError("d_model and n_heads must be positive")
        if self.d_model != self.n_heads * self.head_dim:
            raise ValueError("d_model must equal n_heads * head_dim")
        if self.n_kv_heads <= 0 or self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by a positive n_kv_heads")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.head_dim <= 0 or self.head_dim % 2:
            raise ValueError("head_dim must be a positive even number for RoPE")
        if self.rope_theta <= 0.0 or self.norm_eps <= 0.0:
            raise ValueError("rope_theta and norm_eps must be positive")
        if self.encoder_hidden_size <= 0 or self.item_mlp_layers < 0:
            raise ValueError("encoder hidden size must be positive and item_mlp_layers non-negative")
        if self.component_depth < 0 or self.controller_residual_size <= 0:
            raise ValueError("invalid autoregressive controller-head dimensions")
        if self.action_offset_frames != 1:
            raise ValueError("this policy contract requires action_offset_frames == 1")
        if self.action_codec != "custom_v1":
            raise ValueError("only the pinned custom_v1 controller codec is supported")
        if self.residual_mode not in ("standard", "full_attnres"):
            raise ValueError("residual_mode must be 'standard' or 'full_attnres'")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1)")
        if not 0.0 <= self.residual_dropout < 1.0:
            raise ValueError("residual_dropout must be in [0, 1)")
        if self.player_name_vocab_size < 0:
            raise ValueError("player_name_vocab_size must be non-negative")
        if self.condition_on_player_name and self.player_name_vocab_size == 0:
            raise ValueError("condition_on_player_name requires a positive vocabulary size")
        for name in (self.parameter_dtype, self.compute_dtype, self.cache_dtype, self.softmax_dtype):
            if name not in _DTYPES:
                raise ValueError(f"unsupported precision dtype: {name}")
        if self.attention_gate_compute_dtype != "float32":
            raise ValueError("attention_gate_compute_dtype must be 'float32'")
        if self.softmax_dtype != "float32":
            raise ValueError("softmax_dtype must be 'float32'")
        expected_options = {
            "framework": (self.framework, "pytorch"),
            "encoder_type": (self.encoder_type, "slippi_ai_enhanced"),
            "backbone_type": (self.backbone_type, "causal_transformer"),
            "attention_type": (self.attention_type, "full"),
            "attention_output_gate": (self.attention_output_gate, "elementwise_sigmoid"),
            "norm": (self.norm, "rmsnorm"),
            "activation": (self.activation, "swiglu"),
            "position_encoding": (self.position_encoding, "rope"),
            "attention_implementation": (self.attention_implementation, "sdpa"),
            "controller_head_type": (
                self.controller_head_type,
                "slippi_ai_autoregressive",
            ),
        }
        for option, (actual, expected) in expected_options.items():
            if actual != expected:
                raise ValueError(f"{option} must be {expected!r}, got {actual!r}")
        if self.bias:
            raise ValueError("the Transformer projections must be bias-free")
        if self.attention_gate_bias:
            raise ValueError("the full-rank attention gate must be bias-free")

    @property
    def torch_parameter_dtype(self) -> torch.dtype:
        return _DTYPES[self.parameter_dtype]

    @property
    def torch_compute_dtype(self) -> torch.dtype:
        return _DTYPES[self.compute_dtype]

    @property
    def torch_cache_dtype(self) -> torch.dtype:
        return _DTYPES[self.cache_dtype]

    @classmethod
    def from_yaml(cls, path: str | Path, profile: str | None = None) -> ModelConfig:
        """Load YAML and override exactly the six scaling dimensions."""

        import yaml

        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise TypeError("config root must be a mapping")
        return cls.from_mapping(cast(Mapping[str, Any], value), profile=profile)

    @classmethod
    def from_mapping(
        cls,
        root: Mapping[str, Any],
        profile: str | None = None,
    ) -> ModelConfig:
        metadata = _mapping(root, "metadata")
        model = _mapping(root, "model")
        encoder = _mapping(model, "encoder")
        prediction = _mapping(model, "prediction")
        backbone = dict(_mapping(model, "backbone"))
        controller_head = _mapping(model, "controller_head")
        precision = _mapping(model, "precision")
        scaling = _mapping(root, "scaling")
        profile_name = str(profile or scaling["active_profile"])
        profiles = _mapping(scaling, "profiles")
        if profile_name not in profiles:
            raise KeyError(f"unknown scaling profile: {profile_name}")
        selected = profiles[profile_name]
        if not isinstance(selected, Mapping):
            raise TypeError(f"profile {profile_name!r} must be a mapping")
        if set(selected) != _SIZE_FIELDS:
            raise ValueError(f"profile {profile_name!r} must override exactly {sorted(_SIZE_FIELDS)}")
        backbone.update(selected)
        kwargs: dict[str, Any] = {
            "profile": profile_name,
            "framework": metadata["framework"],
            "slippi_ai_commit": metadata["slippi_ai_commit"],
            "encoder_type": encoder["type"],
            "encoder_hidden_size": encoder["hidden_size"],
            "item_mlp_layers": encoder["item_mlp_layers"],
            "controller_rnn_cell": encoder["controller_rnn_cell"],
            "use_self_nana": encoder["use_self_nana"],
            "use_controller_rnn": encoder["use_controller_rnn"],
            "use_learned_character": encoder["use_learned_character"],
            "use_learned_action": encoder["use_learned_action"],
            "use_character_action_joint": encoder["use_character_action_joint"],
            "use_item_sum": encoder["use_item_sum"],
            "use_items": encoder["use_items"],
            "hybrid_embed": encoder["hybrid_embed"],
            "use_randall": encoder["use_randall"],
            "use_fod_platforms": encoder["use_fod_platforms"],
            "condition_on_player_name": encoder["condition_on_player_name"],
            "player_name_vocab_size": encoder["player_name_vocab_size"],
            "action_offset_frames": prediction["action_offset_frames"],
            "action_codec": prediction["action_codec"],
            "controller_head_type": controller_head["type"],
            "component_depth": controller_head["component_depth"],
            "controller_residual_size": controller_head["residual_size"],
            "parameter_dtype": precision["parameter_dtype"],
            "compute_dtype": precision["compute_dtype"],
            "cache_dtype": precision["cache_dtype"],
            "softmax_dtype": precision["softmax_dtype"],
        }
        backbone_names = {
            "type": "backbone_type",
            "attention_gate_compute_dtype": "attention_gate_compute_dtype",
        }
        for name, value in backbone.items():
            kwargs[backbone_names.get(name, name)] = value
        return cls(**kwargs)


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent[key]
    if not isinstance(value, Mapping):
        raise TypeError(f"{key!r} must be a mapping")
    return cast(Mapping[str, Any], value)


@dataclass
class KVCache:
    """Bounded GQA ring cache.

    ``keys`` and ``values`` have logical shape
    ``[n_layers, B, context_length, n_kv_heads, head_dim]``.  ``write_position``
    identifies the next slot (and therefore the oldest slot when full), while
    ``next_position`` is the RoPE position for the next valid frame.
    """

    keys: torch.Tensor
    values: torch.Tensor
    valid_length: torch.Tensor
    write_position: torch.Tensor
    next_position: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.keys.shape[1])

    @property
    def capacity(self) -> int:
        return int(self.keys.shape[2])

    @property
    def n_layers(self) -> int:
        return int(self.keys.shape[0])

    def reset_slots(self, reset_mask: torch.Tensor) -> None:
        reset = reset_mask.to(device=self.keys.device, dtype=torch.bool)
        if reset.shape != (self.batch_size,):
            raise ValueError(f"reset_mask must have shape {(self.batch_size,)}")
        if not bool(reset.any()):
            return
        self.keys[:, reset] = 0
        self.values[:, reset] = 0
        self.valid_length[reset] = 0
        self.write_position[reset] = 0
        self.next_position[reset] = 0

    def chronological(self, layer_index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return chronological ``[B, C, K, Dh]`` tensors and a valid mask."""

        return _chronological_cache_layer(
            self,
            layer_index,
            self.valid_length,
            self.write_position,
        )

    def storage_report(self) -> dict[str, int]:
        kv_tensors = (self.keys, self.values)
        metadata_tensors = (
            self.valid_length,
            self.write_position,
            self.next_position,
        )
        kv_elements = sum(item.numel() for item in kv_tensors)
        kv_bytes = sum(item.numel() * item.element_size() for item in kv_tensors)
        metadata_elements = sum(item.numel() for item in metadata_tensors)
        metadata_bytes = sum(item.numel() * item.element_size() for item in metadata_tensors)
        return {
            "kv_elements": kv_elements,
            "kv_bytes": kv_bytes,
            "metadata_elements": metadata_elements,
            "metadata_bytes": metadata_bytes,
            "elements": kv_elements + metadata_elements,
            "bytes": kv_bytes + metadata_bytes,
        }


def _chronological_cache_layer(
    cache: KVCache,
    layer_index: int,
    lengths: torch.Tensor,
    next_write: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not 0 <= layer_index < cache.n_layers:
        raise IndexError("cache layer index out of range")
    batch = cache.batch_size
    capacity = cache.capacity
    keys = cache.keys.new_zeros((batch, capacity, cache.keys.shape[3], cache.keys.shape[4]))
    values = cache.values.new_zeros(keys.shape)
    valid = torch.zeros((batch, capacity), dtype=torch.bool, device=cache.keys.device)
    for batch_index in range(batch):
        length = int(lengths[batch_index].item())
        if length == 0:
            continue
        start = (int(next_write[batch_index].item()) - length) % capacity
        indices = (torch.arange(length, device=cache.keys.device) + start) % capacity
        keys[batch_index, :length] = cache.keys[layer_index, batch_index, indices]
        values[batch_index, :length] = cache.values[layer_index, batch_index, indices]
        valid[batch_index, :length] = True
    return keys, values, valid


class RMSNorm(nn.Module):
    """RMSNorm with FP32 statistics and optional distinct head weights."""

    def __init__(self, dim: int, eps: float = 1.0e-6, n_heads: int | None = None):
        super().__init__()
        if dim <= 0:
            raise ValueError("RMSNorm dimension must be positive")
        shape = (dim,) if n_heads is None else (n_heads, dim)
        self.weight = nn.Parameter(torch.ones(shape, dtype=torch.float32))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        variance = inputs.float().square().mean(dim=-1, keepdim=True)
        normalized = (inputs.float() * torch.rsqrt(variance + self.eps)).to(inputs.dtype)
        weight = self.weight.to(inputs.dtype)
        if weight.ndim == 2:
            if inputs.ndim < 3 or inputs.shape[1] != weight.shape[0]:
                raise ValueError("head-wise RMSNorm expects shape [B, H, ..., Dh]")
            shape = (1, weight.shape[0], *([1] * (inputs.ndim - 3)), weight.shape[1])
            weight = weight.view(shape)
        return normalized * weight


class RotaryEmbedding(nn.Module):
    """Standard half-rotation RoPE for batch-specific positions."""

    def __init__(self, head_dim: int, theta: float = 10_000.0):
        super().__init__()
        if head_dim <= 0 or head_dim % 2:
            raise ValueError("RoPE head_dim must be a positive even number")
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.inv_freq: torch.Tensor
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def _rotate_half(inputs: torch.Tensor) -> torch.Tensor:
        first, second = inputs.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def apply_rotary(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(queries.shape[0], -1)
        if positions.shape != (queries.shape[0], queries.shape[2]):
            raise ValueError("RoPE positions must have shape [B, T] or [T]")
        frequencies = positions.float().unsqueeze(-1) * self.inv_freq.float()
        embedding = torch.cat((frequencies, frequencies), dim=-1).unsqueeze(1)
        cosine = embedding.cos().to(queries.dtype)
        sine = embedding.sin().to(queries.dtype)
        queries = queries * cosine + self._rotate_half(queries) * sine
        keys = keys * cosine + self._rotate_half(keys) * sine
        return queries, keys

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.apply_rotary(queries, keys, positions)


def _autocast_context(device: torch.device, dtype: torch.dtype) -> contextlib.AbstractContextManager[Any]:
    if dtype == torch.bfloat16 and device.type in ("cpu", "cuda"):
        return torch.autocast(device_type=device.type, dtype=dtype)
    if dtype == torch.float16 and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


_MISSING = object()
_RAW_STAGE_TO_LIBMELEE = {2: 8, 3: 18, 8: 6, 28: 26, 31: 24, 32: 25}
_BUTTON_NAMES = ("A", "B", "X", "Y", "Z", "L", "R", "D_UP")
_RANDALL_CORNERS = {
    416: (-33.184478759765625, 89.75263977050781),
    417: (-33.04470443725586, 90.07878112792969),
    418: (-32.904930114746094, 90.40492248535156),
    419: (-32.76515197753906, 90.73107147216797),
    420: (-32.49260711669922, 90.92455291748047),
    421: (-32.16635513305664, 91.06437683105469),
    422: (-31.840103149414062, 91.20419311523438),
    423: (-31.513851165771484, 91.3440170288086),
    469: (-15.1948881149292, 91.3371353149414),
    470: (-14.868742942810059, 91.1973648071289),
    471: (-14.542601585388184, 91.05758666992188),
    472: (-14.216456413269043, 90.91781616210938),
    473: (-13.967143058776855, 90.71036529541016),
    474: (-13.869664192199707, 90.36917877197266),
    475: (-13.772183418273926, 90.02799224853516),
    476: (-13.674698829650879, 89.68680572509766),
    1016: (-13.679760932922363, -101.919677734375),
    1017: (-13.819535255432129, -102.24581909179688),
    1018: (-13.959305763244629, -102.57196044921875),
    1019: (-14.099089622497559, -102.89810180664062),
    1020: (-14.320136070251465, -103.14761352539062),
    1021: (-14.6375150680542, -103.30630493164062),
    1022: (-14.954894065856934, -103.46499633789062),
    1069: (-31.590042114257812, -103.554931640625),
    1070: (-31.907413482666016, -103.39625549316406),
    1071: (-32.22478485107422, -103.23756408691406),
    1072: (-32.54215621948242, -103.07887268066406),
    1073: (-32.7216796875, -102.77439880371094),
    1074: (-32.89775085449219, -102.46626281738281),
    1075: (-33.07382583618164, -102.15814208984375),
}


def _get(value: Any, *names: str, default: Any = _MISSING) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    if default is _MISSING:
        joined = ", ".join(repr(name) for name in names)
        raise KeyError(f"none of {joined} are present")
    return default


def _has(value: Any, name: str) -> bool:
    return (isinstance(value, Mapping) and name in value) or hasattr(value, name)


def _as_prefix_tensor(
    value: Any,
    prefix_shape: torch.Size,
    device: torch.device,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.to(device=device, dtype=dtype)
    if tensor.shape != prefix_shape:
        try:
            tensor = torch.broadcast_to(tensor, prefix_shape)
        except RuntimeError as error:
            raise ValueError(
                f"field shape {tuple(tensor.shape)} is not broadcastable to {tuple(prefix_shape)}"
            ) from error
    return tensor


def _one_hot_empty_invalid(values: torch.Tensor, size: int) -> torch.Tensor:
    valid = (values >= 0) & (values < size)
    safe = values.clamp(0, size - 1)
    return F.one_hot(safe.long(), size).float() * valid.unsqueeze(-1)


def _controller_labels(codec: CustomV1Codec, value: Any) -> ControllerLabels:
    if not isinstance(value, ControllerLabels):
        return codec.encode(value)
    if value.buttons.shape != value.main_stick.shape:
        raise ValueError("controller label component shapes must match")
    for name, labels, size in zip(
        codec.component_order,
        value.values(),
        codec.vocabulary_sizes,
        strict=True,
    ):
        if labels.is_floating_point() or labels.is_complex():
            raise TypeError(f"{name} labels must have integer dtype")
        if bool(((labels < 0) | (labels >= size)).any()):
            raise ValueError(f"{name} labels must be in [0, {size})")
    return ControllerLabels(value.buttons.long(), value.main_stick.long())


def _derive_randall(
    stage: torch.Tensor,
    raw_frame_id: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Port libmelee's deterministic Randall approximation without a runtime import."""

    frame = torch.remainder(raw_frame_id.long() + 1200, 1200)
    x = torch.zeros_like(frame, dtype=torch.float32)
    y = torch.zeros_like(frame, dtype=torch.float32)
    width = 11.9

    top = (frame > 476) & (frame < 1016)
    top_center = 101.235443115234 + (-0.35484) * (frame.float() - 477.0) - width / 2.0
    x = torch.where(top, top_center, x)
    y = torch.where(top, torch.full_like(y, -13.64989), y)

    left = (frame > 1022) & (frame < 1069)
    x = torch.where(left, torch.full_like(x, (-103.6 - 91.7) / 2.0), x)
    y = torch.where(left, -15.2778692245483 + (-0.354839325) * (frame.float() - 1023), y)

    bottom = (frame > 1075) | (frame < 416)
    bottom_frames = torch.where(frame < 416, frame + 125, frame - 1076).float()
    bottom_left = -101.850006103516 + 0.35484 * bottom_frames
    x = torch.where(bottom, bottom_left + width / 2.0, x)
    y = torch.where(bottom, torch.full_like(y, -33.2489), y)

    right = (frame > 423) & (frame < 469)
    x = torch.where(right, torch.full_like(x, (91.35 + 103.25) / 2.0), x)
    y = torch.where(right, -31.160232543945312 + 0.354839325 * (frame.float() - 424), y)

    for index, (corner_y, corner_left) in _RANDALL_CORNERS.items():
        corner = frame == index
        x = torch.where(corner, torch.full_like(x, corner_left + width / 2.0), x)
        y = torch.where(corner, torch.full_like(y, corner_y), y)

    is_yoshis = stage == 6
    return torch.where(is_yoshis, x, 0.0), torch.where(is_yoshis, y, 0.0)


class _ControllerComponentRNN(nn.Module):
    def __init__(self, vocabulary_sizes: Sequence[int], hidden_size: int, cell: str):
        super().__init__()
        constructors: dict[str, type[nn.LSTMCell] | type[nn.GRUCell]] = {
            "lstm": nn.LSTMCell,
            "gru": nn.GRUCell,
        }
        if cell not in constructors:
            raise ValueError("controller_rnn_cell must be 'lstm' or 'gru'")
        self.cell_type = cell
        self.cells = nn.ModuleList([constructors[cell](size, hidden_size) for size in vocabulary_sizes])
        self.hidden_size = hidden_size

    def forward(self, labels: ControllerLabels, sizes: Sequence[int]) -> torch.Tensor:
        components = (labels.buttons, labels.main_stick)
        shape = components[0].shape
        hidden = torch.zeros((*shape, self.hidden_size), device=components[0].device)
        cell_state = torch.zeros_like(hidden)
        flat_hidden = hidden.reshape(-1, self.hidden_size)
        flat_cell = cell_state.reshape(-1, self.hidden_size)
        for module, values, size in zip(self.cells, components, sizes, strict=True):
            inputs = F.one_hot(values.long(), size).to(flat_hidden.dtype).reshape(-1, size)
            if self.cell_type == "lstm":
                flat_hidden, flat_cell = cast(
                    tuple[torch.Tensor, torch.Tensor],
                    module(inputs, (flat_hidden, flat_cell)),
                )
            else:
                flat_hidden = cast(torch.Tensor, module(inputs, flat_hidden))
        return flat_hidden.reshape(*shape, self.hidden_size)


class SlippiEncoder(nn.Module):
    """PyTorch port of the pinned JAX ``EnhancedEmbedModule``.

    The accepted tensor tree uses self/opponent (or p0/p1) perspective keys and
    the upstream field names.  E000 aliases such as ``x_position`` and
    ``action_state_id`` are accepted.  Missing Nana/items/platform tensors use
    the same all-zero raw records as upstream; preprocessing should enrich them
    when the source replay retains those fields.
    """

    ACTION_SIZE = 0x18F
    CHARACTER_SIZE = 0x21
    STAGE_SIZE = 64
    JUMPS_SIZE = 7
    ITEM_COUNT = 15
    ITEM_TYPE_INPUT_SIZE = 0xEC + 1
    ITEM_TYPE_SIZE = ITEM_TYPE_INPUT_SIZE + 1
    ITEM_STATE_INPUT_SIZE = 12
    ITEM_STATE_SIZE = ITEM_STATE_INPUT_SIZE + 1
    ITEM_SIZE = 1 + ITEM_TYPE_SIZE + ITEM_STATE_SIZE + 2

    def __init__(self, config: ModelConfig, codec: CustomV1Codec | None = None):
        super().__init__()
        self.config = config
        self.codec = codec or CustomV1Codec()
        hidden = config.encoder_hidden_size
        self.character_embedding = nn.Embedding(self.CHARACTER_SIZE, hidden)
        self.action_embedding = nn.Embedding(self.ACTION_SIZE, hidden)
        self.character_action_embedding = nn.Embedding(
            self.CHARACTER_SIZE * self.ACTION_SIZE,
            hidden,
        )
        nn.init.zeros_(self.character_action_embedding.weight)

        item_layers: list[nn.Module] = []
        input_size = self.ITEM_SIZE
        for layer_index in range(config.item_mlp_layers):
            if layer_index:
                item_layers.append(nn.ReLU())
            linear = nn.Linear(input_size, hidden, bias=True)
            nn.init.normal_(linear.weight, mean=0.0, std=1.0 / math.sqrt(input_size))
            nn.init.zeros_(linear.bias)
            item_layers.append(linear)
            input_size = hidden
        self.item_mlp = nn.Sequential(*item_layers) if item_layers else nn.Identity()
        self.item_output_size = input_size

        self.controller_rnn: _ControllerComponentRNN | None = None
        if config.use_controller_rnn:
            self.controller_rnn = _ControllerComponentRNN(
                self.codec.vocabulary_sizes,
                hidden,
                config.controller_rnn_cell,
            )
        self._output_dim = self._calculate_output_dim()

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def _calculate_output_dim(self) -> int:
        action_size = self.config.encoder_hidden_size if self.config.use_learned_action else self.ACTION_SIZE
        character_size = (
            self.config.encoder_hidden_size if self.config.use_learned_character else self.CHARACTER_SIZE
        )
        if self.config.hybrid_embed:
            if self.config.use_learned_action:
                action_size += self.ACTION_SIZE
            if self.config.use_learned_character:
                character_size += self.CHARACTER_SIZE
        player_core = 14 + action_size + character_size
        nana = player_core + 1
        players = player_core + (nana if self.config.use_self_nana else 0) + player_core + nana
        output = players + self.STAGE_SIZE
        output += 2 if self.config.use_randall else 0
        output += 2 if self.config.use_fod_platforms else 0
        if self.config.use_items:
            output += self.item_output_size if self.config.use_item_sum else self.ITEM_COUNT * self.ITEM_SIZE
        if self.config.condition_on_player_name:
            output += self.config.player_name_vocab_size
        output += (
            self.config.encoder_hidden_size
            if self.config.use_controller_rnn
            else sum(self.codec.vocabulary_sizes)
        )
        return output

    def _player_from_game(self, game: Any, self_player: bool) -> Any:
        names = ("p0", "self", "self_player") if self_player else ("p1", "opponent", "opponent_player")
        for name in names:
            if _has(game, name):
                return _get(game, name)
        players = _get(game, "players", default=None)
        if players is not None:
            for name in names:
                if _has(players, name):
                    return _get(players, name)
        raise KeyError(f"game state has no {'self' if self_player else 'opponent'} player tensor tree")

    def _field(
        self,
        value: Any,
        prefix: torch.Size,
        device: torch.device,
        *names: str,
        default: Any = _MISSING,
    ) -> torch.Tensor:
        raw = _get(value, *names, default=default)
        return _as_prefix_tensor(raw, prefix, device)

    @staticmethod
    def _scaled_float(value: torch.Tensor, scale: float) -> torch.Tensor:
        return (value.float() * scale).clamp(-10.0, 10.0).unsqueeze(-1)

    def _embed_player_or_nana(
        self,
        player: Any,
        prefix: torch.Size,
        device: torch.device,
        *,
        nana: bool,
    ) -> torch.Tensor:
        default = 0
        percent = self._field(player, prefix, device, "percent", default=default).long()
        percent = torch.remainder(percent, 1 << 16)
        facing_raw = self._field(
            player,
            prefix,
            device,
            "facing",
            "facing_direction",
            default=default,
        )
        facing = torch.where(
            facing_raw.bool() if facing_raw.dtype == torch.bool else facing_raw > 0, 1.0, -1.0
        )
        x = self._field(player, prefix, device, "x", "x_position", default=default)
        y = self._field(player, prefix, device, "y", "y_position", default=default)
        action = (
            self._field(
                player,
                prefix,
                device,
                "action",
                "action_state_id",
                default=default,
            )
            .long()
            .clamp(0, self.ACTION_SIZE - 1)
        )
        invulnerable = self._field(player, prefix, device, "invulnerable", default=default).bool()
        character = self._field(
            player,
            prefix,
            device,
            "character",
            "character_id",
            default=default,
        ).long()
        if bool(((character < 0) | (character >= self.CHARACTER_SIZE)).any()):
            raise ValueError("character index is outside the upstream [0, 33) vocabulary")
        jumps = self._field(
            player,
            prefix,
            device,
            "jumps_left",
            "jumps_remaining",
            default=default,
        ).long()
        if bool(((jumps < 0) | (jumps >= self.JUMPS_SIZE)).any()):
            raise ValueError("jumps_left is outside the upstream [0, 7) vocabulary")
        shield = self._field(
            player,
            prefix,
            device,
            "shield_strength",
            "shield_value",
            default=default,
        )
        on_ground = self._field(player, prefix, device, "on_ground", default=default).bool()

        action_one_hot = F.one_hot(action, self.ACTION_SIZE).float()
        if self.config.use_learned_action:
            action_embed = self.action_embedding(action)
            if self.config.use_character_action_joint:
                joint_index = character * self.ACTION_SIZE + action
                action_embed = action_embed + self.character_action_embedding(joint_index)
            if self.config.hybrid_embed:
                action_embed = torch.cat((action_embed, action_one_hot), dim=-1)
        else:
            action_embed = action_one_hot

        character_one_hot = F.one_hot(character, self.CHARACTER_SIZE).float()
        if self.config.use_learned_character:
            character_embed = self.character_embedding(character)
            if self.config.hybrid_embed:
                character_embed = torch.cat((character_embed, character_one_hot), dim=-1)
        else:
            character_embed = character_one_hot

        parts = [
            self._scaled_float(percent, 0.01),
            facing.unsqueeze(-1),
            self._scaled_float(x, 0.05),
            self._scaled_float(y, 0.05),
            action_embed,
            invulnerable.float().unsqueeze(-1),
            character_embed,
            F.one_hot(jumps, self.JUMPS_SIZE).float(),
            self._scaled_float(shield, 0.01),
            on_ground.float().unsqueeze(-1),
        ]
        if nana:
            exists = self._field(player, prefix, device, "exists", default=False).bool()
            parts.append(exists.float().unsqueeze(-1))
        return torch.cat([part.float() for part in parts], dim=-1)

    def _embed_player(
        self,
        player: Any,
        prefix: torch.Size,
        device: torch.device,
        *,
        with_nana: bool,
    ) -> torch.Tensor:
        parts = [self._embed_player_or_nana(player, prefix, device, nana=False)]
        if with_nana:
            nana = _get(player, "nana", "follower", default={})
            parts.append(self._embed_player_or_nana(nana, prefix, device, nana=True))
        return torch.cat(parts, dim=-1)

    def _item_fields(
        self,
        items: Any,
        prefix: torch.Size,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        item_shape = torch.Size((*prefix, self.ITEM_COUNT))
        if items is None:
            zeros = torch.zeros(item_shape, device=device)
            return zeros.bool(), zeros.long(), zeros.long(), zeros, zeros
        if isinstance(items, Sequence) and not isinstance(items, (str, bytes, torch.Tensor)):
            if len(items) != self.ITEM_COUNT:
                raise ValueError("items sequence must contain exactly 15 slots")
            fields: list[torch.Tensor] = []
            for names in (("exists",), ("type", "item_type"), ("state", "item_state"), ("x",), ("y",)):
                fields.append(
                    torch.stack(
                        [_as_prefix_tensor(_get(item, *names, default=0), prefix, device) for item in items],
                        dim=-1,
                    )
                )
            return fields[0].bool(), fields[1].long(), fields[2].long(), fields[3], fields[4]
        return (
            _as_prefix_tensor(_get(items, "exists", default=0), item_shape, device).bool(),
            _as_prefix_tensor(_get(items, "type", "item_type", default=0), item_shape, device).long(),
            _as_prefix_tensor(_get(items, "state", "item_state", default=0), item_shape, device).long(),
            _as_prefix_tensor(_get(items, "x", default=0), item_shape, device),
            _as_prefix_tensor(_get(items, "y", default=0), item_shape, device),
        )

    def _embed_items(
        self,
        game: Any,
        prefix: torch.Size,
        device: torch.device,
    ) -> torch.Tensor:
        exists, item_type, item_state, x, y = self._item_fields(
            _get(game, "items", default=None),
            prefix,
            device,
        )
        type_valid = (item_type >= 0) & (item_type < self.ITEM_TYPE_INPUT_SIZE)
        state_valid = (item_state >= 0) & (item_state < self.ITEM_STATE_INPUT_SIZE)
        item_type = torch.where(type_valid, item_type, self.ITEM_TYPE_INPUT_SIZE)
        item_state = torch.where(state_valid, item_state, self.ITEM_STATE_INPUT_SIZE)
        embedded = torch.cat(
            (
                exists.float().unsqueeze(-1),
                F.one_hot(item_type, self.ITEM_TYPE_SIZE).float(),
                F.one_hot(item_state, self.ITEM_STATE_SIZE).float(),
                self._scaled_float(x, 0.05),
                self._scaled_float(y, 0.05),
            ),
            dim=-1,
        )
        if embedded.shape[-2:] != (self.ITEM_COUNT, self.ITEM_SIZE):
            raise RuntimeError("internal item representation shape mismatch")
        if not self.config.use_item_sum:
            return embedded.flatten(start_dim=-2)
        transformed = self.item_mlp(embedded)
        transformed = torch.where(exists.unsqueeze(-1), transformed, 0.0)
        return transformed.sum(dim=-2).float()

    def _stage(
        self,
        game: Any,
        prefix: torch.Size,
        device: torch.device,
    ) -> torch.Tensor:
        if _has(game, "stage_id"):
            raw = _as_prefix_tensor(_get(game, "stage_id"), prefix, device).long()
            stage = torch.full_like(raw, -1)
            for raw_value, mapped in _RAW_STAGE_TO_LIBMELEE.items():
                stage = torch.where(raw == raw_value, mapped, stage)
        else:
            stage = _as_prefix_tensor(_get(game, "stage"), prefix, device).long()
        if bool(((stage < 0) | (stage >= self.STAGE_SIZE)).any()):
            raise ValueError("stage is invalid for the upstream 64-way vocabulary")
        return stage

    def _controller_embedding(self, labels: ControllerLabels) -> torch.Tensor:
        if self.controller_rnn is not None:
            return self.controller_rnn(labels, self.codec.vocabulary_sizes).float()
        return torch.cat(
            (
                F.one_hot(labels.buttons.long(), self.codec.vocabulary_sizes[0]),
                F.one_hot(labels.main_stick.long(), self.codec.vocabulary_sizes[1]),
            ),
            dim=-1,
        ).float()

    def forward(
        self,
        game_state_t: Any,
        controller_t: Any,
        player_name: Any | None = None,
    ) -> torch.Tensor:
        labels = _controller_labels(self.codec, controller_t)
        prefix = labels.buttons.shape
        if labels.main_stick.shape != prefix:
            raise ValueError("controller component shapes differ")
        device = labels.buttons.device
        self_player = self._player_from_game(game_state_t, self_player=True)
        opponent = self._player_from_game(game_state_t, self_player=False)
        stage = self._stage(game_state_t, prefix, device)
        parts = [
            self._embed_player(
                self_player,
                prefix,
                device,
                with_nana=self.config.use_self_nana,
            ),
            self._embed_player(opponent, prefix, device, with_nana=True),
            F.one_hot(stage, self.STAGE_SIZE).float(),
        ]

        if self.config.use_randall:
            randall = _get(game_state_t, "randall", default=None)
            if randall is None:
                frame = _as_prefix_tensor(
                    _get(game_state_t, "raw_frame_id", "frame", default=0),
                    prefix,
                    device,
                )
                randall_x, randall_y = _derive_randall(stage, frame)
            else:
                randall_x = _as_prefix_tensor(_get(randall, "x"), prefix, device)
                randall_y = _as_prefix_tensor(_get(randall, "y"), prefix, device)
            parts.extend((self._scaled_float(randall_x, 0.05), self._scaled_float(randall_y, 0.05)))

        if self.config.use_fod_platforms:
            fod = _get(game_state_t, "fod_platforms", "fod", default={})
            left = _as_prefix_tensor(_get(fod, "left", default=0), prefix, device)
            right = _as_prefix_tensor(_get(fod, "right", default=0), prefix, device)
            parts.extend((self._scaled_float(left, 0.05), self._scaled_float(right, 0.05)))

        if self.config.use_items:
            parts.append(self._embed_items(game_state_t, prefix, device))
        if self.config.condition_on_player_name:
            name_source = (
                player_name
                if player_name is not None
                else _get(game_state_t, "name", "player_name", default=-1)
            )
            if name_source is None:
                name_source = -1
            name = _as_prefix_tensor(
                name_source,
                prefix,
                device,
            ).long()
            parts.append(_one_hot_empty_invalid(name, self.config.player_name_vocab_size))
        parts.append(self._controller_embedding(labels))
        result = torch.cat([part.float() for part in parts], dim=-1)
        if result.shape[-1] != self.output_dim:
            raise RuntimeError(f"encoder produced {result.shape[-1]} features, expected {self.output_dim}")
        return result


class GroupedQueryCausalAttention(nn.Module):
    """Bias-free RoPE GQA using PyTorch SDPA and a full-rank output gate."""

    def __init__(self, config: ModelConfig, layer_index: int = 0):
        super().__init__()
        self.config = config
        self.layer_index = layer_index
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.groups = config.n_heads // config.n_kv_heads
        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.output_projection = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)
        self.attention_gate = nn.Linear(
            config.d_model,
            config.n_heads * config.head_dim,
            bias=False,
        )
        self.q_norm = (
            RMSNorm(config.head_dim, config.norm_eps, n_heads=config.n_heads)
            if config.qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            RMSNorm(config.head_dim, config.norm_eps, n_heads=config.n_kv_heads)
            if config.qk_norm
            else nn.Identity()
        )
        self.rope = RotaryEmbedding(config.head_dim, config.rope_theta)
        self.force_expanded_kv = False
        self.last_gate_product_dtype: torch.dtype | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for projection in (
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.output_projection,
        ):
            nn.init.normal_(projection.weight, mean=0.0, std=self.config.initializer_std)
        # The official qiuzh20/gated_attention modeling_qwen3.py initializes
        # every Linear N(0, initializer_range^2); its released default is .02.
        nn.init.normal_(self.attention_gate.weight, mean=0.0, std=0.02)

    def _project(
        self,
        inputs: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, time, _ = inputs.shape
        queries = self.q_proj(inputs).view(batch, time, self.n_heads, self.head_dim).transpose(1, 2)
        keys = self.k_proj(inputs).view(batch, time, self.n_kv_heads, self.head_dim).transpose(1, 2)
        values = self.v_proj(inputs).view(batch, time, self.n_kv_heads, self.head_dim).transpose(1, 2)
        queries = self.q_norm(queries)
        keys = self.k_norm(keys)
        queries, keys = self.rope(queries, keys, positions)
        return queries, keys, values

    def _sdpa(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        is_causal: bool,
    ) -> torch.Tensor:
        dropout = self.config.attention_dropout if self.training else 0.0
        if self.groups == 1:
            return F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=attention_mask,
                dropout_p=dropout,
                is_causal=is_causal,
            )
        if not self.force_expanded_kv:
            try:
                return F.scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    attn_mask=attention_mask,
                    dropout_p=dropout,
                    is_causal=is_causal,
                    enable_gqa=True,
                )
            except (TypeError, RuntimeError) as error:
                # Older PyTorch builds and a few device backends do not expose
                # native GQA. Only fall back for a head-count/backend complaint.
                message = str(error).lower()
                if not any(word in message for word in ("gqa", "head", "enable_gqa")):
                    raise
        keys = keys.repeat_interleave(self.groups, dim=1)
        values = values.repeat_interleave(self.groups, dim=1)
        return F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=attention_mask,
            dropout_p=dropout,
            is_causal=is_causal,
        )

    def _apply_output_gate(
        self,
        head_output: torch.Tensor,
        normalized_inputs: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, time, _ = head_output.shape
        gate = torch.sigmoid(self.attention_gate(normalized_inputs).float())
        gate = gate.view(batch, time, self.n_heads, self.head_dim).transpose(1, 2)
        gated = head_output.float() * gate
        self.last_gate_product_dtype = gated.dtype
        flattened = gated.transpose(1, 2).reshape(batch, time, -1).to(normalized_inputs.dtype)
        return self.output_projection(flattened)

    def forward(
        self,
        normalized_inputs: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Parallel attention over a complete training sequence."""

        queries, keys, values = self._project(normalized_inputs, positions)
        head_output = self._sdpa(
            queries,
            keys,
            values,
            attention_mask=attention_mask,
            is_causal=attention_mask is None,
        )
        return self._apply_output_gate(head_output, normalized_inputs)

    def forward_cached(
        self,
        normalized_inputs: torch.Tensor,
        positions: torch.Tensor,
        cache: KVCache,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Attend one frame against this layer's chronological ring history."""

        if normalized_inputs.shape[1] != 1:
            raise ValueError("cached attention accepts exactly one frame")
        queries, keys, values = self._project(normalized_inputs, positions)
        valid = valid_mask.to(device=cache.keys.device, dtype=torch.bool)
        for batch_index in range(cache.batch_size):
            if not bool(valid[batch_index]):
                continue
            write = int(cache.write_position[batch_index].item())
            cache.keys[self.layer_index, batch_index, write] = (
                keys[batch_index, :, 0].detach().to(cache.keys.dtype)
            )
            cache.values[self.layer_index, batch_index, write] = (
                values[batch_index, :, 0].detach().to(cache.values.dtype)
            )
        lengths = torch.minimum(
            cache.valid_length + valid.long(),
            torch.full_like(cache.valid_length, cache.capacity),
        )
        next_write = torch.where(
            valid,
            torch.remainder(cache.write_position + 1, cache.capacity),
            cache.write_position,
        )
        chronological_k, chronological_v, key_valid = _chronological_cache_layer(
            cache,
            self.layer_index,
            lengths,
            next_write,
        )
        chronological_k = chronological_k.permute(0, 2, 1, 3).to(queries.dtype)
        chronological_v = chronological_v.permute(0, 2, 1, 3).to(queries.dtype)
        mask = key_valid[:, None, None, :] & valid[:, None, None, None]
        head_output = self._sdpa(
            queries,
            chronological_k,
            chronological_v,
            attention_mask=mask,
            is_causal=False,
        )
        head_output = torch.where(valid[:, None, None, None], head_output, 0.0)
        return self._apply_output_gate(head_output, normalized_inputs)


class SwiGLU(nn.Module):
    """Bias-free ``down(silu(gate(x)) * up(x))`` feed-forward layer."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_projection = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_projection = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_projection = nn.Linear(config.d_ff, config.d_model, bias=False)
        for projection in (self.gate_projection, self.up_projection, self.down_projection):
            nn.init.normal_(projection.weight, mean=0.0, std=config.initializer_std)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.down_projection(F.silu(self.gate_projection(inputs)) * self.up_projection(inputs))


class FullAttentionResidual(nn.Module):
    """One exact Full Attention Residuals depth read.

    Each read has one affine RMSNorm and one zero-initialized learned pseudo-query.
    Source normalization affects routing scores only; raw source values are mixed.
    """

    def __init__(self, d_model: int, eps: float = 1.0e-6):
        super().__init__()
        self.routing_norm = RMSNorm(d_model, eps)
        # Attention Residuals requires zero pseudo-queries so every depth read
        # begins as a uniform mixture and avoids early routing volatility.
        self.depth_query = nn.Parameter(torch.zeros(d_model, dtype=torch.float32))

    def forward(self, values: Sequence[torch.Tensor]) -> torch.Tensor:
        if not values:
            raise ValueError("FullAttentionResidual requires at least one source")
        stacked = torch.stack(tuple(values), dim=0)  # [S, B, T, D]
        # The official Kimi AttnRes path keeps normalization, routing scores,
        # softmax, and value aggregation in FP32 before restoring activation dtype.
        normalized_sources = self.routing_norm(stacked.float())
        scores = torch.einsum(
            "d,sbtd->sbt",
            self.depth_query.float(),
            normalized_sources,
        )
        weights = torch.softmax(scores.float(), dim=0)
        # Kimi's official AttnRes path also keeps routing aggregation in FP32.
        mixed = torch.einsum("sbt,sbtd->btd", weights, stacked.float())
        return mixed.to(stacked.dtype)


class TransformerBlock(nn.Module):
    """One attention + SwiGLU block with both residual reference modes."""

    def __init__(self, config: ModelConfig, layer_index: int):
        super().__init__()
        self.config = config
        self.attention_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention = GroupedQueryCausalAttention(config, layer_index)
        self.feed_forward = SwiGLU(config)
        self.attention_dropout = nn.Dropout(config.residual_dropout)
        self.ffn_dropout = nn.Dropout(config.residual_dropout)
        if config.residual_mode == "full_attnres":
            self.attention_residual: FullAttentionResidual | None = FullAttentionResidual(
                config.d_model,
                config.norm_eps,
            )
            self.ffn_residual: FullAttentionResidual | None = FullAttentionResidual(
                config.d_model,
                config.norm_eps,
            )
        else:
            self.attention_residual = None
            self.ffn_residual = None

    def forward_standard(
        self,
        inputs: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        inputs = inputs + self.attention_dropout(
            self.attention(self.attention_norm(inputs), positions, attention_mask)
        )
        return inputs + self.ffn_dropout(self.feed_forward(self.ffn_norm(inputs)))

    def forward(
        self,
        inputs: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.config.residual_mode != "standard":
            raise RuntimeError(
                "a full_attnres TransformerBlock requires the backbone's shared "
                "depth-value list; call CausalTransformer instead"
            )
        return self.forward_standard(inputs, positions, attention_mask)


class CausalTransformer(nn.Module):
    """Batch-major causal temporal backbone with segmented masking and KV cache."""

    def __init__(self, config: ModelConfig, input_dim: int):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.config = config
        self.input_dim = input_dim
        self.input_projection = nn.Linear(input_dim, config.d_model, bias=False)
        # Documented PreNorm baseline: normal_(0, .02) for Transformer
        # projections; standard residual branches receive the scale below.
        nn.init.normal_(self.input_projection.weight, mean=0.0, std=config.initializer_std)
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_index) for layer_index in range(config.n_layers)]
        )
        self.output_residual = (
            FullAttentionResidual(config.d_model, config.norm_eps)
            if config.residual_mode == "full_attnres"
            else None
        )
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        if config.residual_mode == "standard":
            residual_scale = 1.0 / math.sqrt(2.0 * config.n_layers)
            for layer in self.layers:
                nn.init.normal_(
                    layer.attention.output_projection.weight,
                    mean=0.0,
                    std=config.initializer_std * residual_scale,
                )
                nn.init.normal_(
                    layer.feed_forward.down_projection.weight,
                    mean=0.0,
                    std=config.initializer_std * residual_scale,
                )

    def init_cache(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> KVCache:
        if batch_size <= 0:
            raise ValueError("cache batch_size must be positive")
        parameter = self.input_projection.weight
        cache_device = parameter.device if device is None else torch.device(device)
        cache_dtype = self.config.torch_cache_dtype if dtype is None else dtype
        shape = (
            self.config.n_layers,
            batch_size,
            self.config.context_length,
            self.config.n_kv_heads,
            self.config.head_dim,
        )
        return KVCache(
            keys=torch.zeros(shape, device=cache_device, dtype=cache_dtype),
            values=torch.zeros(shape, device=cache_device, dtype=cache_dtype),
            valid_length=torch.zeros(batch_size, device=cache_device, dtype=torch.long),
            write_position=torch.zeros(batch_size, device=cache_device, dtype=torch.long),
            next_position=torch.zeros(batch_size, device=cache_device, dtype=torch.long),
        )

    def _validate_cache(self, cache: KVCache, batch_size: int, device: torch.device) -> None:
        expected = (
            self.config.n_layers,
            batch_size,
            self.config.context_length,
            self.config.n_kv_heads,
            self.config.head_dim,
        )
        if tuple(cache.keys.shape) != expected or tuple(cache.values.shape) != expected:
            raise ValueError(f"cache K/V shapes must both equal {expected}")
        if cache.keys.device != device or cache.values.device != device:
            raise ValueError("cache and encoded inputs must be on the same device")
        if not cache.keys.is_floating_point() or not cache.values.is_floating_point():
            raise TypeError("cache keys and values must have floating dtype")
        for name in ("valid_length", "write_position", "next_position"):
            tensor = cast(torch.Tensor, getattr(cache, name))
            if tensor.shape != (batch_size,) or tensor.device != device:
                raise ValueError(f"cache {name} must have shape [B] on the input device")
            if tensor.dtype != torch.long:
                raise TypeError(f"cache {name} must have torch.long dtype")
        if bool(((cache.valid_length < 0) | (cache.valid_length > cache.capacity)).any()):
            raise ValueError("cache valid_length is outside [0, capacity]")
        if bool(((cache.write_position < 0) | (cache.write_position >= cache.capacity)).any()):
            raise ValueError("cache write_position is outside [0, capacity)")
        if bool((cache.next_position < 0).any()):
            raise ValueError("cache next_position must be non-negative")

    @staticmethod
    def _positions(
        batch: int,
        time: int,
        device: torch.device,
        reset_mask: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if reset_mask is None and padding_mask is None:
            return torch.arange(time, device=device).unsqueeze(0).expand(batch, -1)
        reset = (
            torch.zeros((batch, time), dtype=torch.bool, device=device)
            if reset_mask is None
            else reset_mask.to(device=device, dtype=torch.bool)
        )
        valid = (
            torch.ones((batch, time), dtype=torch.bool, device=device)
            if padding_mask is None
            else padding_mask.to(device=device, dtype=torch.bool)
        )
        positions = torch.zeros((batch, time), dtype=torch.long, device=device)
        next_position = torch.zeros(batch, dtype=torch.long, device=device)
        for index in range(time):
            next_position = torch.where(reset[:, index], 0, next_position)
            positions[:, index] = next_position
            next_position = next_position + valid[:, index].long()
        return positions

    @staticmethod
    def _segmented_attention_mask(
        batch: int,
        time: int,
        device: torch.device,
        reset_mask: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if reset_mask is None and padding_mask is None:
            return None
        reset = (
            torch.zeros((batch, time), dtype=torch.bool, device=device)
            if reset_mask is None
            else reset_mask.to(device=device, dtype=torch.bool)
        )
        valid = (
            torch.ones((batch, time), dtype=torch.bool, device=device)
            if padding_mask is None
            else padding_mask.to(device=device, dtype=torch.bool)
        )
        segment = torch.cumsum(reset.long(), dim=1)
        same_segment = segment[:, :, None] == segment[:, None, :]
        causal = torch.ones((time, time), dtype=torch.bool, device=device).tril()
        allowed = same_segment & causal.unsqueeze(0) & valid[:, :, None] & valid[:, None, :]
        return allowed.unsqueeze(1)

    def _attention_transform(
        self,
        layer: TransformerBlock,
        inputs: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        return layer.attention_dropout(
            layer.attention(layer.attention_norm(inputs), positions, attention_mask)
        )

    @staticmethod
    def _ffn_transform(layer: TransformerBlock, inputs: torch.Tensor) -> torch.Tensor:
        return layer.ffn_dropout(layer.feed_forward(layer.ffn_norm(inputs)))

    def _parallel(
        self,
        projected: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor | None,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        use_checkpoint = self.config.gradient_checkpointing and self.training and torch.is_grad_enabled()
        if self.config.residual_mode == "standard":
            hidden = projected
            for layer in self.layers:
                if use_checkpoint:
                    hidden = checkpoint(
                        lambda value, current=layer: current.forward_standard(
                            value,
                            positions,
                            attention_mask,
                        ),
                        hidden,
                        use_reentrant=False,
                    )
                else:
                    hidden = layer.forward_standard(hidden, positions, attention_mask)
        else:
            values: list[torch.Tensor] = [projected]
            for layer in self.layers:
                if layer.attention_residual is None or layer.ffn_residual is None:
                    raise RuntimeError("full_attnres layer is missing its depth reads")
                attention_input = layer.attention_residual(values)
                if use_checkpoint:
                    attention_output = checkpoint(
                        lambda value, current=layer: self._attention_transform(
                            current,
                            value,
                            positions,
                            attention_mask,
                        ),
                        attention_input,
                        use_reentrant=False,
                    )
                else:
                    attention_output = self._attention_transform(
                        layer,
                        attention_input,
                        positions,
                        attention_mask,
                    )
                values.append(attention_output)
                ffn_input = layer.ffn_residual(values)
                if use_checkpoint:
                    ffn_output = checkpoint(
                        lambda value, current=layer: self._ffn_transform(current, value),
                        ffn_input,
                        use_reentrant=False,
                    )
                else:
                    ffn_output = self._ffn_transform(layer, ffn_input)
                values.append(ffn_output)
            if self.output_residual is None:
                raise RuntimeError("full_attnres backbone is missing its final depth read")
            hidden = self.output_residual(values)
        hidden = self.final_norm(hidden)
        return torch.where(valid.unsqueeze(-1), hidden, 0.0)

    def _cached_frame(
        self,
        encoded_frame: torch.Tensor,
        cache: KVCache,
        reset_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, KVCache]:
        cache.reset_slots(reset_mask)
        positions = cache.next_position[:, None]
        projected = self.input_projection(encoded_frame).unsqueeze(1)
        valid = valid_mask.to(device=encoded_frame.device, dtype=torch.bool)
        if self.config.residual_mode == "standard":
            hidden = projected
            for layer in self.layers:
                attention_output = layer.attention.forward_cached(
                    layer.attention_norm(hidden),
                    positions,
                    cache,
                    valid,
                )
                hidden = hidden + layer.attention_dropout(attention_output)
                hidden = hidden + self._ffn_transform(layer, hidden)
        else:
            values: list[torch.Tensor] = [projected]
            for layer in self.layers:
                if layer.attention_residual is None or layer.ffn_residual is None:
                    raise RuntimeError("full_attnres layer is missing its depth reads")
                attention_input = layer.attention_residual(values)
                attention_output = layer.attention.forward_cached(
                    layer.attention_norm(attention_input),
                    positions,
                    cache,
                    valid,
                )
                values.append(layer.attention_dropout(attention_output))
                ffn_input = layer.ffn_residual(values)
                values.append(self._ffn_transform(layer, ffn_input))
            if self.output_residual is None:
                raise RuntimeError("full_attnres backbone is missing its final depth read")
            hidden = self.output_residual(values)
        hidden = self.final_norm(hidden).squeeze(1)
        hidden = torch.where(valid.unsqueeze(-1), hidden, 0.0)
        cache.valid_length.copy_(
            torch.minimum(
                cache.valid_length + valid.long(),
                torch.full_like(cache.valid_length, cache.capacity),
            )
        )
        cache.write_position.copy_(
            torch.where(
                valid,
                torch.remainder(cache.write_position + 1, cache.capacity),
                cache.write_position,
            )
        )
        cache.next_position.add_(valid.long())
        return hidden, cache

    def forward(
        self,
        encoded_frames: torch.Tensor,
        reset_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        cache: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        """Transform ``[B,T,input_dim]`` frames, optionally updating a ring cache."""

        if encoded_frames.ndim != 3 or encoded_frames.shape[-1] != self.input_dim:
            raise ValueError(f"encoded_frames must have shape [B, T, {self.input_dim}]")
        batch, time, _ = encoded_frames.shape
        if time <= 0:
            raise ValueError("encoded sequence must contain at least one frame")
        if reset_mask is not None and reset_mask.shape != (batch, time):
            raise ValueError("reset_mask must have shape [B, T]")
        if padding_mask is not None and padding_mask.shape != (batch, time):
            raise ValueError("padding_mask must have shape [B, T]")
        device = encoded_frames.device
        valid = (
            torch.ones((batch, time), dtype=torch.bool, device=device)
            if padding_mask is None
            else padding_mask.to(device=device, dtype=torch.bool)
        )
        reset = (
            torch.zeros((batch, time), dtype=torch.bool, device=device)
            if reset_mask is None
            else reset_mask.to(device=device, dtype=torch.bool)
        )
        with _autocast_context(device, self.config.torch_compute_dtype):
            if use_cache:
                if not self.config.use_kv_cache:
                    raise ValueError("use_cache requested while KV caching is disabled")
                if self.training and torch.is_grad_enabled():
                    raise RuntimeError("the rolling KV cache is inference-only; call eval() or no_grad()")
                if cache is None:
                    cache = self.init_cache(batch, device=device)
                self._validate_cache(cache, batch, device)
                outputs: list[torch.Tensor] = []
                for index in range(time):
                    output, cache = self._cached_frame(
                        encoded_frames[:, index],
                        cache,
                        reset[:, index],
                        valid[:, index],
                    )
                    outputs.append(output)
                return torch.stack(outputs, dim=1), cache

            if cache is not None:
                raise ValueError("a cache was supplied but use_cache is false")
            if time > self.config.context_length:
                raise ValueError(
                    f"parallel sequence length {time} exceeds context_length {self.config.context_length}"
                )
            pure_causal = not bool(reset.any()) and bool(valid.all())
            if pure_causal:
                positions = torch.arange(time, device=device).unsqueeze(0).expand(batch, -1)
                attention_mask = None
            else:
                positions = self._positions(batch, time, device, reset, valid)
                attention_mask = self._segmented_attention_mask(
                    batch,
                    time,
                    device,
                    reset,
                    valid,
                )
            projected = self.input_projection(encoded_frames)
            return self._parallel(projected, positions, attention_mask, valid), None

    def step(
        self,
        encoded_frame: torch.Tensor,
        cache: KVCache,
        reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        """Transform one ``[B,input_dim]`` frame and update ``cache`` in place."""

        if not self.config.use_kv_cache:
            raise ValueError("step requested while KV caching is disabled")
        if encoded_frame.ndim != 2 or encoded_frame.shape[-1] != self.input_dim:
            raise ValueError(f"encoded_frame must have shape [B, {self.input_dim}]")
        batch = encoded_frame.shape[0]
        self._validate_cache(cache, batch, encoded_frame.device)
        if self.training and torch.is_grad_enabled():
            raise RuntimeError("the rolling KV cache is inference-only; call eval() or no_grad()")
        reset = (
            torch.zeros(batch, dtype=torch.bool, device=encoded_frame.device)
            if reset_mask is None
            else reset_mask.to(device=encoded_frame.device, dtype=torch.bool)
        )
        if reset.shape != (batch,):
            raise ValueError("reset_mask must have shape [B]")
        valid = torch.ones(batch, dtype=torch.bool, device=encoded_frame.device)
        with _autocast_context(encoded_frame.device, self.config.torch_compute_dtype):
            return self._cached_frame(encoded_frame, cache, reset, valid)


class _AutoregressiveComponent(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        residual_size: int,
        depth: int,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        input_size = residual_size + vocabulary_size
        features = [residual_size] * depth + [vocabulary_size]
        for index, output_size in enumerate(features):
            if index:
                layers.append(nn.ReLU())
            linear = nn.Linear(input_size, output_size, bias=True)
            nn.init.normal_(linear.weight, mean=0.0, std=1.0 / math.sqrt(input_size))
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            input_size = output_size
        self.encoder = nn.Sequential(*layers)
        self.decoder = nn.Linear(vocabulary_size, residual_size, bias=True)
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)
        self.vocabulary_size = vocabulary_size

    def logits(self, residual: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        previous_embedding = F.one_hot(previous.long(), self.vocabulary_size).to(residual.dtype)
        return self.encoder(torch.cat((residual, previous_embedding), dim=-1))

    def update(self, residual: torch.Tensor, component: torch.Tensor) -> torch.Tensor:
        embedding = F.one_hot(component.long(), self.vocabulary_size).to(residual.dtype)
        return residual + self.decoder(embedding)


@dataclass
class ControllerHeadOutput:
    """Autoregressive controller distributions and their conditioning state."""

    logits: dict[str, torch.Tensor]
    labels: ControllerLabels
    controller_state: ControllerState
    hidden: torch.Tensor
    previous_controller: ControllerLabels
    teacher_forced: bool


class AutoregressiveControllerHead(nn.Module):
    """Pinned slippi-ai custom_v1 autoregressive controller head."""

    def __init__(self, config: ModelConfig, codec: CustomV1Codec | None = None):
        super().__init__()
        self.config = config
        self.codec = codec or CustomV1Codec()
        residual_size = config.controller_residual_size
        self.to_residual = nn.Linear(config.d_model, residual_size, bias=True)
        nn.init.normal_(
            self.to_residual.weight,
            mean=0.0,
            std=1.0 / math.sqrt(config.d_model),
        )
        nn.init.zeros_(self.to_residual.bias)
        self.components = nn.ModuleDict(
            {
                name: _AutoregressiveComponent(
                    vocabulary_size,
                    residual_size,
                    config.component_depth,
                )
                for name, vocabulary_size in zip(
                    self.codec.component_order,
                    self.codec.vocabulary_sizes,
                    strict=True,
                )
            }
        )

    @staticmethod
    def _component_values(labels: ControllerLabels) -> dict[str, torch.Tensor]:
        return {"buttons": labels.buttons, "main_stick": labels.main_stick}

    @staticmethod
    def _labels(values: Mapping[str, torch.Tensor]) -> ControllerLabels:
        return ControllerLabels(
            buttons=values["buttons"],
            main_stick=values["main_stick"],
        )

    def _predict(
        self,
        hidden: torch.Tensor,
        previous: ControllerLabels,
        *,
        teacher: ControllerLabels | None = None,
        sample: bool = False,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> tuple[dict[str, torch.Tensor], ControllerLabels]:
        if hidden.ndim < 2 or hidden.shape[-1] != self.config.d_model:
            raise ValueError(
                f"hidden must have shape [..., {self.config.d_model}] with at least one batch axis"
            )
        prefix = hidden.shape[:-1]
        if previous.buttons.shape != prefix or previous.main_stick.shape != prefix:
            raise ValueError(f"previous controller labels must have shape {tuple(prefix)}")
        if previous.buttons.device != hidden.device or previous.main_stick.device != hidden.device:
            raise ValueError("previous controller labels and hidden state must share a device")
        if teacher is not None and (teacher.buttons.shape != prefix or teacher.main_stick.shape != prefix):
            raise ValueError(f"teacher-forcing labels must have shape {tuple(prefix)}")
        if teacher is not None and (
            teacher.buttons.device != hidden.device or teacher.main_stick.device != hidden.device
        ):
            raise ValueError("teacher-forcing labels and hidden state must share a device")
        with _autocast_context(hidden.device, self.config.torch_compute_dtype):
            residual = self.to_residual(hidden)
            previous_values = self._component_values(previous)
            teacher_values = self._component_values(teacher) if teacher is not None else None
            logits: dict[str, torch.Tensor] = {}
            selected: dict[str, torch.Tensor] = {}
            for name in self.codec.component_order:
                component = self.components[name]
                component_logits = component.logits(residual, previous_values[name])
                logits[name] = component_logits
                if teacher_values is not None:
                    choice = teacher_values[name]
                elif sample and temperature > 0.0:
                    probabilities = torch.softmax(component_logits.float() / temperature, dim=-1)
                    flat = probabilities.reshape(-1, probabilities.shape[-1])
                    choice = torch.multinomial(flat, 1, generator=generator).reshape(
                        component_logits.shape[:-1]
                    )
                else:
                    choice = component_logits.float().argmax(dim=-1)
                selected[name] = choice
                residual = component.update(residual, choice)
        return logits, self._labels(selected)

    def forward(
        self,
        hidden: torch.Tensor,
        previous_controller: Any,
        teacher_forcing_targets: Any | None = None,
    ) -> ControllerHeadOutput:
        previous = _controller_labels(self.codec, previous_controller)
        teacher = (
            None
            if teacher_forcing_targets is None
            else _controller_labels(self.codec, teacher_forcing_targets)
        )
        logits, labels = self._predict(hidden, previous, teacher=teacher)
        return ControllerHeadOutput(
            logits=logits,
            labels=labels,
            controller_state=self.codec.decode(labels),
            hidden=hidden,
            previous_controller=previous,
            teacher_forced=teacher is not None,
        )

    def sample(
        self,
        hidden: torch.Tensor,
        previous_controller: Any,
        *,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> ControllerHeadOutput:
        previous = _controller_labels(self.codec, previous_controller)
        logits, labels = self._predict(
            hidden,
            previous,
            sample=True,
            temperature=temperature,
            generator=generator,
        )
        return ControllerHeadOutput(
            logits=logits,
            labels=labels,
            controller_state=self.codec.decode(labels),
            hidden=hidden,
            previous_controller=previous,
            teacher_forced=False,
        )

    def loss(
        self,
        outputs: ControllerHeadOutput,
        controller_t_plus_1: Any,
        valid_position_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Teacher-forced summed component NLL for already aligned targets."""

        target = _controller_labels(self.codec, controller_t_plus_1)
        # ``MeleePolicy.forward`` is given the already-aligned target during
        # training, so its autoregressive logits are teacher forced against
        # exactly these labels.  Reuse them instead of running the decoder a
        # second time.  The fallback keeps ``loss`` useful for inference-style
        # outputs produced without teacher forcing.
        logits = outputs.logits
        if not outputs.teacher_forced:
            logits, _ = self._predict(
                outputs.hidden,
                outputs.previous_controller,
                teacher=target,
            )
        targets = self._component_values(target)
        prefix = outputs.hidden.shape[:-1]
        valid = valid_position_mask.to(device=outputs.hidden.device, dtype=torch.bool)
        if valid.shape != prefix:
            raise ValueError(f"valid_position_mask must have shape {tuple(prefix)}")
        denominator = valid.float().sum().clamp_min(1.0)
        per_component: dict[str, torch.Tensor] = {}
        per_position: list[torch.Tensor] = []
        for name in self.codec.component_order:
            component_nll = F.cross_entropy(
                logits[name].float().reshape(-1, logits[name].shape[-1]),
                targets[name].long().reshape(-1),
                reduction="none",
            ).reshape(prefix)
            per_position.append(component_nll)
            per_component[name] = (component_nll * valid.float()).sum() / denominator
        total_per_position = torch.stack(per_position, dim=0).sum(dim=0)
        loss = (total_per_position * valid.float()).sum() / denominator
        metrics = {
            "loss": loss.detach(),
            "nll/buttons": per_component["buttons"].detach(),
            "nll/main_stick": per_component["main_stick"].detach(),
            "valid_positions": valid.float().sum().detach(),
        }
        return loss.float(), metrics


class MeleePolicy(nn.Module):
    """End-to-end frame encoder, causal Transformer, and controller policy."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.codec = CustomV1Codec()
        self.encoder = SlippiEncoder(config, self.codec)
        self.backbone = CausalTransformer(config, input_dim=self.encoder.output_dim)
        self.controller_head = AutoregressiveControllerHead(config, self.codec)
        if config.torch_parameter_dtype != torch.float32:
            self.to(dtype=config.torch_parameter_dtype)

    @classmethod
    def from_yaml(
        cls,
        path: str | Path = "config.yaml",
        profile: str | None = None,
    ) -> MeleePolicy:
        return cls(ModelConfig.from_yaml(path, profile=profile))

    def forward(
        self,
        game_state_t: Any,
        controller_t: Any,
        reset_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        cache: KVCache | None = None,
        use_cache: bool = False,
        controller_t_plus_1: Any | None = None,
        player_name: Any | None = None,
    ) -> tuple[ControllerHeadOutput, KVCache | None]:
        # controller_t_plus_1 is optional solely to expose teacher-forced logits;
        # it is never shifted, sliced, or otherwise aligned in this module.
        controller_labels = _controller_labels(self.codec, controller_t)
        device = controller_labels.buttons.device
        with _autocast_context(device, self.config.torch_compute_dtype):
            encoded = self.encoder(
                game_state_t,
                controller_labels,
                player_name=player_name,
            )
            hidden, cache = self.backbone(
                encoded,
                reset_mask=reset_mask,
                padding_mask=padding_mask,
                cache=cache,
                use_cache=use_cache,
            )
            controller_outputs = self.controller_head(
                hidden,
                previous_controller=controller_labels,
                teacher_forcing_targets=controller_t_plus_1,
            )
        return controller_outputs, cache

    def loss(
        self,
        controller_outputs: ControllerHeadOutput,
        controller_t_plus_1: Any,
        valid_position_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.controller_head.loss(
            controller_outputs,
            controller_t_plus_1,
            valid_position_mask,
        )

    def parameter_counts(self) -> dict[str, int]:
        counts = {
            "encoder": _trainable_parameters(self.encoder),
            "backbone": _trainable_parameters(self.backbone),
            "controller_head": _trainable_parameters(self.controller_head),
        }
        counts["total"] = sum(counts.values())
        return counts

    def buffer_report(self) -> dict[str, dict[str, int]]:
        return {
            "encoder": _buffer_size(self.encoder),
            "backbone": _buffer_size(self.backbone),
            "controller_head": _buffer_size(self.controller_head),
        }

    def cache_storage_report(self, batch_size: int) -> dict[str, int]:
        return self.backbone.init_cache(batch_size).storage_report()


def _trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _buffer_size(module: nn.Module) -> dict[str, int]:
    buffers = tuple(module.buffers())
    return {
        "elements": sum(buffer.numel() for buffer in buffers),
        "bytes": sum(buffer.numel() * buffer.element_size() for buffer in buffers),
    }


__all__ = [
    "AutoregressiveControllerHead",
    "CausalTransformer",
    "ControllerHeadOutput",
    "FullAttentionResidual",
    "GroupedQueryCausalAttention",
    "KVCache",
    "MeleePolicy",
    "ModelConfig",
    "RMSNorm",
    "RotaryEmbedding",
    "SlippiEncoder",
    "SwiGLU",
    "TransformerBlock",
]
