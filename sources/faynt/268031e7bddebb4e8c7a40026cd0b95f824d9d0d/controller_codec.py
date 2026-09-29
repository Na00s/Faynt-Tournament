"""PyTorch port of slippi-ai's ``custom_v1`` controller codec.

The representation in this file follows vladfi1/slippi-ai commit
``577965a7731dc53e3472ea63d9e9853a4e9d65fa``.  It intentionally has no
dependency on NumPy, JAX, Flax, libmelee, or peppi.  The two categorical
components and their autoregressive order are derived from the configured
Cartesian bucket structure rather than duplicated as unrelated constants.

Native slippi-ai controller sticks use the libmelee convention ``[0, 1]``.
The earlier melee-policy E000 audit stores logical stick aliases in ``[-1, 1]``;
those aliases are accepted only when processed buttons are supplied explicitly.
E000's physical button bitset is not interchangeable with the processed button
state used by slippi-ai and is therefore rejected when it is the only button
source.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from types import MappingProxyType
from typing import Any, Final

import torch
import torch.nn.functional as F
from torch import nn

Tensor = torch.Tensor

SLIPPI_AI_COMMIT: Final = "577965a7731dc53e3472ea63d9e9853a4e9d65fa"

BUTTON_ORDER: Final[tuple[str, ...]] = (
    "A",
    "B",
    "X",
    "Y",
    "Z",
    "L",
    "R",
    "D_UP",
)
COMPONENT_ORDER: Final[tuple[str, ...]] = ("buttons", "main_stick")

# Packed tensors are an explicit convenience representation.  Stick values in
# this form already use slippi-ai's [0, 1] convention.
PACKED_CONTROLLER_ORDER: Final[tuple[str, ...]] = (
    "main_x",
    "main_y",
    "c_x",
    "c_y",
    "shoulder",
    *BUTTON_ORDER,
)

_LIGHT_SHOULDER_THRESHOLD: Final = 0.3
_FULL_SHOULDER_THRESHOLD: Final = 0.9
_SHOULDER_VALUES: Final[tuple[float, ...]] = (0.0, 0.35, 1.0)
_MIN_NONZERO_RADIUS: Final = 23
_MAX_RADIUS: Final = 80
_RAW_AXIS_SPACING: Final = 160
_RAW_AXIS_RADIUS: Final = 80


@dataclass(frozen=True)
class StickState:
    """One batched stick in upstream normalized ``[0, 1]`` coordinates."""

    x: Tensor
    y: Tensor


@dataclass(frozen=True)
class ButtonState:
    """Processed digital buttons in slippi-ai's fixed legal-button order."""

    A: Tensor
    B: Tensor
    X: Tensor
    Y: Tensor
    Z: Tensor
    L: Tensor
    R: Tensor
    D_UP: Tensor

    def values(self) -> tuple[Tensor, ...]:
        return tuple(getattr(self, name) for name in BUTTON_ORDER)

    def as_dict(self) -> dict[str, Tensor]:
        return {name: getattr(self, name) for name in BUTTON_ORDER}


@dataclass(frozen=True)
class ControllerState:
    """A typed, tensor-batched equivalent of ``slippi_ai.types.Controller``."""

    main_stick: StickState
    c_stick: StickState
    shoulder: Tensor
    buttons: ButtonState

    @property
    def batch_shape(self) -> torch.Size:
        return self.shoulder.shape

    def as_packed_tensor(self, *, dtype: torch.dtype = torch.float32) -> Tensor:
        """Return ``[..., 13]`` values in :data:`PACKED_CONTROLLER_ORDER`."""

        values = (
            self.main_stick.x,
            self.main_stick.y,
            self.c_stick.x,
            self.c_stick.y,
            self.shoulder,
            *self.buttons.values(),
        )
        return torch.stack(tuple(value.to(dtype=dtype) for value in values), dim=-1)


@dataclass(frozen=True)
class ControllerLabels:
    """The two ``custom_v1`` integer components, each shaped ``[...]``."""

    buttons: Tensor
    main_stick: Tensor

    def values(self) -> tuple[Tensor, Tensor]:
        return self.buttons, self.main_stick

    def as_dict(self) -> dict[str, Tensor]:
        return {"buttons": self.buttons, "main_stick": self.main_stick}

    def as_tensor(self) -> Tensor:
        """Stack labels as ``[..., 2]`` in autoregressive component order."""

        return torch.stack(self.values(), dim=-1)


def _mapping_value(value: Any, names: Sequence[str]) -> tuple[Any, bool]:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name], True
        return None, False

    for name in names:
        if hasattr(value, name):
            return getattr(value, name), True
    return None, False


def _require_value(value: Any, names: Sequence[str], description: str) -> Any:
    result, found = _mapping_value(value, names)
    if not found:
        joined = ", ".join(repr(name) for name in names)
        raise ValueError(f"controller is missing {description}; expected one of {joined}")
    return result


def _button_name(value: Any) -> str:
    candidates = [value]
    if hasattr(value, "value"):
        candidates.insert(0, value.value)
    if hasattr(value, "name"):
        candidates.insert(0, value.name)

    for candidate in candidates:
        text = str(candidate).upper().split(".")[-1]
        if text.startswith("BUTTON_"):
            text = text[len("BUTTON_") :]
        if text in BUTTON_ORDER:
            return text
    raise ValueError(f"unsupported processed button name: {value!r}")


def _infer_device(*values: Any) -> torch.device:
    for value in values:
        if isinstance(value, Tensor):
            return value.device
        if is_dataclass(value) and not isinstance(value, type):
            device = _infer_device(*(getattr(value, field.name) for field in fields(value)))
            if device.type != "cpu":
                return device
        if isinstance(value, Mapping):
            device = _infer_device(*value.values())
            if device.type != "cpu":
                return device
        elif isinstance(value, (tuple, list)):
            device = _infer_device(*value)
            if device.type != "cpu":
                return device
        elif hasattr(value, "__dict__"):
            device = _infer_device(*vars(value).values())
            if device.type != "cpu":
                return device
    return torch.device("cpu")


def _as_float_tensor(value: Any, *, device: torch.device) -> Tensor:
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _as_button_tensor(value: Any, *, device: torch.device) -> Tensor:
    return torch.as_tensor(value, device=device).to(dtype=torch.bool)


def _coerce_stick(value: Any, *, device: torch.device, description: str) -> StickState:
    if isinstance(value, StickState):
        return StickState(
            _as_float_tensor(value.x, device=device),
            _as_float_tensor(value.y, device=device),
        )

    if isinstance(value, Tensor):
        if value.ndim == 0 or value.shape[-1] != 2:
            raise ValueError(f"{description} tensor must have shape [..., 2], got {tuple(value.shape)}")
        return StickState(
            _as_float_tensor(value[..., 0], device=device),
            _as_float_tensor(value[..., 1], device=device),
        )

    if isinstance(value, Mapping) or hasattr(value, "x"):
        x = _require_value(value, ("x",), f"{description}.x")
        y = _require_value(value, ("y",), f"{description}.y")
        return StickState(_as_float_tensor(x, device=device), _as_float_tensor(y, device=device))

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        return StickState(
            _as_float_tensor(value[0], device=device),
            _as_float_tensor(value[1], device=device),
        )

    raise TypeError(f"unsupported {description} representation: {type(value).__name__}")


def _buttons_from_pressed_names(values: Iterable[Any], *, device: torch.device) -> ButtonState:
    pressed = {_button_name(value) for value in values}
    return ButtonState(
        **{name: torch.tensor(name in pressed, dtype=torch.bool, device=device) for name in BUTTON_ORDER}
    )


def _coerce_buttons(value: Any, *, device: torch.device) -> ButtonState:
    if isinstance(value, ButtonState):
        return ButtonState(
            **{name: _as_button_tensor(getattr(value, name), device=device) for name in BUTTON_ORDER}
        )

    if isinstance(value, Tensor):
        if value.ndim == 0 or value.shape[-1] != len(BUTTON_ORDER):
            raise ValueError(
                "processed-button tensor must have shape [..., 8] in "
                f"{BUTTON_ORDER!r} order, got {tuple(value.shape)}"
            )
        return ButtonState(
            **{
                name: value[..., index].to(device=device, dtype=torch.bool)
                for index, name in enumerate(BUTTON_ORDER)
            }
        )

    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, pressed in value.items():
            try:
                name = _button_name(key)
            except ValueError:
                # Upstream processed-button maps can contain START and d-pad
                # directions not represented by custom_v1. START is not a legal
                # model action; left/right/down are deliberately ignored.
                text = str(getattr(key, "value", key)).upper().split(".")[-1]
                if text in {
                    "START",
                    "BUTTON_START",
                    "D_LEFT",
                    "D_RIGHT",
                    "D_DOWN",
                    "BUTTON_D_LEFT",
                    "BUTTON_D_RIGHT",
                    "BUTTON_D_DOWN",
                }:
                    continue
                raise
            normalized[name] = pressed
        return ButtonState(
            **{name: _as_button_tensor(normalized.get(name, False), device=device) for name in BUTTON_ORDER}
        )

    if all(hasattr(value, name) for name in BUTTON_ORDER):
        return ButtonState(
            **{name: _as_button_tensor(getattr(value, name), device=device) for name in BUTTON_ORDER}
        )

    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        values = tuple(value)
        if len(values) == len(BUTTON_ORDER) and not any(isinstance(item, str) for item in values):
            tensor = torch.as_tensor(values, device=device)
            return _coerce_buttons(tensor, device=device)
        return _buttons_from_pressed_names(values, device=device)

    # A scalar integer is normally E000's physical bitset.  Decoding it here
    # would be wrong when processed button state differs (notably Z -> A+Z).
    raise ValueError(
        "custom_v1 requires processed buttons as named booleans, pressed names, "
        "or an [..., 8] tensor; a physical-only button bitset is not accepted"
    )


def _broadcast_controller(state: ControllerState) -> ControllerState:
    values = (
        state.main_stick.x,
        state.main_stick.y,
        state.c_stick.x,
        state.c_stick.y,
        state.shoulder,
        *state.buttons.values(),
    )
    try:
        broadcast = torch.broadcast_tensors(*values)
    except RuntimeError as error:
        shapes = [tuple(value.shape) for value in values]
        raise ValueError(f"controller component shapes are not broadcastable: {shapes!r}") from error

    return ControllerState(
        main_stick=StickState(broadcast[0].float(), broadcast[1].float()),
        c_stick=StickState(broadcast[2].float(), broadcast[3].float()),
        shoulder=broadcast[4].float(),
        buttons=ButtonState(**{name: broadcast[5 + index].bool() for index, name in enumerate(BUTTON_ORDER)}),
    )


def _coerce_controller(
    value: Any,
    *,
    processed_buttons: Any | None = None,
) -> ControllerState:
    if isinstance(value, ControllerState):
        if processed_buttons is not None:
            raise ValueError("processed_buttons must not be supplied with a ControllerState")
        return _broadcast_controller(value)

    if isinstance(value, Tensor):
        if processed_buttons is not None:
            raise ValueError("processed_buttons must not be supplied with a packed controller tensor")
        if value.ndim == 0 or value.shape[-1] != len(PACKED_CONTROLLER_ORDER):
            raise ValueError(
                f"packed controller tensor must have shape [..., {len(PACKED_CONTROLLER_ORDER)}] "
                f"in {PACKED_CONTROLLER_ORDER!r} order, got {tuple(value.shape)}"
            )
        state = ControllerState(
            main_stick=StickState(value[..., 0].float(), value[..., 1].float()),
            c_stick=StickState(value[..., 2].float(), value[..., 3].float()),
            shoulder=value[..., 4].float(),
            buttons=_coerce_buttons(value[..., 5:], device=value.device),
        )
        return _broadcast_controller(state)

    device = _infer_device(value, processed_buttons)
    nested_main, has_nested_main = _mapping_value(value, ("main_stick",))
    nested_c, has_nested_c = _mapping_value(value, ("c_stick",))

    if has_nested_main or has_nested_c:
        if not (has_nested_main and has_nested_c):
            raise ValueError("controller must supply both main_stick and c_stick")
        main_stick = _coerce_stick(nested_main, device=device, description="main_stick")
        c_stick = _coerce_stick(nested_c, device=device, description="c_stick")
        shoulder = _require_value(
            value,
            ("shoulder", "l_shoulder", "analog_l", "trigger_logical"),
            "single logical shoulder",
        )

        nested_buttons, has_nested_buttons = _mapping_value(value, ("buttons", "processed_button"))
        supplied_alias, has_supplied_alias = _mapping_value(value, ("buttons_processed",))
        if processed_buttons is not None:
            if has_nested_buttons or has_supplied_alias:
                raise ValueError("processed buttons were supplied more than once")
            button_source = processed_buttons
        elif has_nested_buttons:
            button_source = nested_buttons
        elif has_supplied_alias:
            button_source = supplied_alias
        else:
            _, has_physical = _mapping_value(value, ("buttons_physical", "button"))
            if has_physical:
                raise ValueError(
                    "controller supplies only physical buttons; pass processed_buttons or "
                    "buttons_processed because slippi-ai consumes processed button state"
                )
            raise ValueError("controller is missing processed buttons")

        state = ControllerState(
            main_stick=main_stick,
            c_stick=c_stick,
            shoulder=_as_float_tensor(shoulder, device=device),
            buttons=_coerce_buttons(button_source, device=device),
        )
        return _broadcast_controller(state)

    # E000 aliases: sticks are stored in [-1, 1], unlike slippi-ai's [0, 1].
    main_x = _require_value(value, ("main_x",), "E000 main_x")
    main_y = _require_value(value, ("main_y",), "E000 main_y")
    c_x = _require_value(value, ("c_x",), "E000 c_x")
    c_y = _require_value(value, ("c_y",), "E000 c_y")

    supplied_alias, has_supplied_alias = _mapping_value(value, ("buttons_processed",))
    if processed_buttons is not None:
        if has_supplied_alias:
            raise ValueError("processed buttons were supplied more than once")
        button_source = processed_buttons
    elif has_supplied_alias:
        button_source = supplied_alias
    else:
        _, has_physical = _mapping_value(value, ("buttons_physical", "button"))
        suffix = " (physical-only buttons are not equivalent)" if has_physical else ""
        raise ValueError(f"E000 controller aliases require buttons_processed{suffix}")

    logical_shoulder, has_logical_shoulder = _mapping_value(
        value, ("trigger_logical", "shoulder", "l_shoulder", "analog_l")
    )
    if not has_logical_shoulder:
        shoulder_l, has_l = _mapping_value(value, ("shoulder_l",))
        shoulder_r, has_r = _mapping_value(value, ("shoulder_r",))
        if not (has_l and has_r):
            raise ValueError(
                "E000 controller aliases require trigger_logical or both shoulder_l and shoulder_r"
            )
        # E000 stores per-shoulder physical triggers, whereas upstream consumes
        # the game's combined logical trigger.  max is the only faithful
        # information-preserving fallback available without the logical column.
        logical_shoulder = torch.maximum(
            _as_float_tensor(shoulder_l, device=device),
            _as_float_tensor(shoulder_r, device=device),
        )

    def e000_axis(axis: Any) -> Tensor:
        return (_as_float_tensor(axis, device=device) + 1.0) * 0.5

    state = ControllerState(
        main_stick=StickState(e000_axis(main_x), e000_axis(main_y)),
        c_stick=StickState(e000_axis(c_x), e000_axis(c_y)),
        shoulder=_as_float_tensor(logical_shoulder, device=device),
        buttons=_coerce_buttons(button_source, device=device),
    )
    return _broadcast_controller(state)


class _PolarStickCodec(nn.Module):
    """Torch implementation of upstream ``PolarStickBucketer``."""

    radius_table: Tensor
    count_table: Tensor
    offset_table: Tensor
    label_to_radius: Tensor
    label_to_inner: Tensor
    angle_table: Tensor

    def __init__(self, angle_buckets_by_nonzero_radius: Sequence[int]) -> None:
        super().__init__()
        angle_buckets = tuple(int(value) for value in angle_buckets_by_nonzero_radius)
        if not angle_buckets or any(value <= 0 for value in angle_buckets):
            raise ValueError("each nonzero radius must have a positive angle-bucket count")

        self.angle_buckets_by_nonzero_radius = angle_buckets
        counts = (1, *angle_buckets)
        offsets: list[int] = []
        offset = 0
        for count in counts:
            offsets.append(offset)
            offset += count
        self.num_labels = offset

        radius_values = torch.exp(
            torch.linspace(
                math.log(_MIN_NONZERO_RADIUS),
                math.log(_MAX_RADIUS),
                len(angle_buckets),
                dtype=torch.float64,
            )
        )
        radius_table = torch.cat((torch.zeros(1, dtype=torch.float64), radius_values)).float()
        count_table = torch.tensor(counts, dtype=torch.long)
        offset_table = torch.tensor(offsets, dtype=torch.long)
        label_to_radius = torch.repeat_interleave(torch.arange(len(counts), dtype=torch.long), count_table)
        label_to_inner = torch.cat(tuple(torch.arange(count, dtype=torch.long) for count in counts))
        angle_table = torch.cat(
            tuple(
                torch.arange(count, dtype=torch.float64) / count * (2.0 * math.pi) - math.pi
                for count in counts
            )
        ).float()

        self.register_buffer("radius_table", radius_table, persistent=False)
        self.register_buffer("count_table", count_table, persistent=False)
        self.register_buffer("offset_table", offset_table, persistent=False)
        self.register_buffer("label_to_radius", label_to_radius, persistent=False)
        self.register_buffer("label_to_inner", label_to_inner, persistent=False)
        self.register_buffer("angle_table", angle_table, persistent=False)

    @staticmethod
    def _at(buffer: Tensor, reference: Tensor) -> Tensor:
        return buffer if buffer.device == reference.device else buffer.to(reference.device)

    def bucket(self, stick: StickState) -> Tensor:
        x = stick.x.float()
        y = stick.y.float()
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError("stick coordinates must be finite")

        raw_x = torch.round(x * _RAW_AXIS_SPACING - _RAW_AXIS_RADIUS).long()
        raw_y = torch.round(y * _RAW_AXIS_SPACING - _RAW_AXIS_RADIUS).long()
        radius = torch.sqrt((raw_x.square() + raw_y.square()).float())
        is_origin = radius <= (_MIN_NONZERO_RADIUS - 1)

        min_log_radius = math.log(_MIN_NONZERO_RADIUS)
        max_log_radius = math.log(_MAX_RADIUS)
        normalized_radius = (torch.log(radius + 1.0e-3) - min_log_radius) / (max_log_radius - min_log_radius)
        nonzero_radius = torch.round(
            normalized_radius * (len(self.angle_buckets_by_nonzero_radius) - 1)
        ).long()
        radius_bucket = torch.where(is_origin, torch.zeros_like(nonzero_radius), nonzero_radius + 1)

        invalid = (radius_bucket < 0) | (radius_bucket >= self.count_table.numel())
        if invalid.any():
            bad_radius = radius[invalid].max().item()
            raise ValueError(
                "stick lies outside custom_v1's legal radial domain; "
                f"largest invalid raw radius was {bad_radius:g}"
            )

        counts = self._at(self.count_table, radius_bucket)[radius_bucket]
        angle = torch.atan2(raw_y.float(), raw_x.float())
        normalized_angle = (angle + math.pi) / (2.0 * math.pi)
        angle_bucket = torch.remainder(torch.round(normalized_angle * counts).long(), counts)
        offsets = self._at(self.offset_table, radius_bucket)
        return offsets[radius_bucket] + angle_bucket

    def decode(self, labels: Tensor) -> StickState:
        labels = _integer_labels(labels, self.num_labels, "stick")
        label_to_radius = self._at(self.label_to_radius, labels)
        radius_table = self._at(self.radius_table, labels)
        angle_table = self._at(self.angle_table, labels)
        radius_bucket = label_to_radius[labels]
        raw_radius = radius_table[radius_bucket]
        angle = angle_table[labels]
        raw_x = raw_radius * torch.cos(angle)
        raw_y = raw_radius * torch.sin(angle)
        return StickState(
            x=((raw_x + _RAW_AXIS_RADIUS) / _RAW_AXIS_SPACING).float(),
            y=((raw_y + _RAW_AXIS_RADIUS) / _RAW_AXIS_SPACING).float(),
        )


def _integer_labels(labels: Any, vocab_size: int, component: str) -> Tensor:
    tensor = torch.as_tensor(labels)
    if tensor.is_floating_point() or tensor.is_complex():
        if tensor.is_floating_point() and torch.equal(tensor, tensor.round()):
            tensor = tensor.long()
        else:
            raise TypeError(f"{component} labels must be integer-valued")
    else:
        tensor = tensor.long()
    if ((tensor < 0) | (tensor >= vocab_size)).any():
        minimum = tensor.min().item() if tensor.numel() else None
        maximum = tensor.max().item() if tensor.numel() else None
        raise ValueError(f"{component} labels must be in [0, {vocab_size}), got range [{minimum}, {maximum}]")
    return tensor


class CustomV1Codec(nn.Module):
    """Faithful, batched PyTorch implementation of slippi-ai ``custom_v1``.

    Args:
        c_stick_angle_buckets: Angle counts for each nonzero C-stick radius.
        main_stick_angle_buckets: Angle counts for each nonzero main-stick radius.

    The default configuration derives component vocabularies ``(728, 85)``.
    The stale prose comment in upstream says 416 button labels; the executable
    upstream Cartesian axes are ``2 * 2 * 2 * 7 * 13 == 728``.
    """

    component_order: Final[tuple[str, ...]] = COMPONENT_ORDER
    button_order: Final[tuple[str, ...]] = BUTTON_ORDER

    def __init__(
        self,
        *,
        c_stick_angle_buckets: Sequence[int] = (4, 8),
        main_stick_angle_buckets: Sequence[int] = (4, 16, 64),
    ) -> None:
        super().__init__()
        self.c_stick = _PolarStickCodec(c_stick_angle_buckets)
        self.main_stick = _PolarStickCodec(main_stick_angle_buckets)

        # B, X|Y, L|R, Z/A/shoulder, C-stick.
        self._button_axis_sizes = (2, 2, 2, 7, self.c_stick.num_labels)
        button_vocab = math.prod(self._button_axis_sizes)
        self._vocab_sizes = MappingProxyType(
            {"buttons": button_vocab, "main_stick": self.main_stick.num_labels}
        )

    @property
    def vocab_sizes(self) -> Mapping[str, int]:
        """Read-only component vocabulary mapping in autoregressive order."""

        return self._vocab_sizes

    @property
    def axis_sizes(self) -> tuple[int, int]:
        """Upstream-compatible ``(buttons, main_stick)`` vocabulary tuple."""

        return tuple(self._vocab_sizes[name] for name in COMPONENT_ORDER)  # type: ignore[return-value]

    @property
    def vocabulary_sizes(self) -> tuple[int, int]:
        """Alias used by model modules that consume component sizes positionally."""

        return self.axis_sizes

    @property
    def encoded_size(self) -> int:
        """Size of the concatenated previous-controller one-hot representation."""

        return sum(self.axis_sizes)

    def coerce_controller(
        self,
        controller: Any,
        *,
        processed_buttons: Any | None = None,
    ) -> ControllerState:
        """Normalize typed, upstream nested, E000, or packed tensor inputs."""

        return _coerce_controller(controller, processed_buttons=processed_buttons)

    def encode(
        self,
        controller: Any,
        *,
        processed_buttons: Any | None = None,
    ) -> ControllerLabels:
        """Bucket a controller, preserving all leading batch dimensions."""

        state = self.coerce_controller(controller, processed_buttons=processed_buttons)
        c_stick = self.c_stick.bucket(state.c_stick)

        shoulder_bucket = torch.zeros_like(state.shoulder, dtype=torch.long)
        shoulder_bucket = torch.where(
            state.shoulder > _LIGHT_SHOULDER_THRESHOLD,
            torch.ones_like(shoulder_bucket),
            shoulder_bucket,
        )
        shoulder_bucket = torch.where(
            state.shoulder > _FULL_SHOULDER_THRESHOLD,
            torch.full_like(shoulder_bucket, 2),
            shoulder_bucket,
        )

        buttons = state.buttons
        xy = buttons.X | buttons.Y
        lr = buttons.L | buttons.R
        z_a_shoulder = buttons.A.long() * 3 + shoulder_bucket
        z_a_shoulder = torch.where(buttons.Z, torch.full_like(z_a_shoulder, 6), z_a_shoulder)

        label = buttons.B.long()
        for size, component in zip(
            self._button_axis_sizes[1:],
            (xy.long(), lr.long(), z_a_shoulder, c_stick),
            strict=True,
        ):
            label = label * size + component

        main_stick = self.main_stick.bucket(state.main_stick)
        return ControllerLabels(buttons=label.long(), main_stick=main_stick.long())

    def _coerce_labels(self, labels: Any) -> ControllerLabels:
        if isinstance(labels, ControllerLabels):
            buttons, main_stick = labels.values()
        elif isinstance(labels, Mapping):
            buttons = labels["buttons"]
            main_stick = labels["main_stick"]
        elif isinstance(labels, Tensor):
            if labels.ndim == 0 or labels.shape[-1] != len(COMPONENT_ORDER):
                raise ValueError(f"label tensor must have shape [..., 2], got {tuple(labels.shape)}")
            buttons, main_stick = labels.unbind(dim=-1)
        elif isinstance(labels, Sequence) and len(labels) == 2:
            buttons, main_stick = labels
        else:
            raise TypeError(f"unsupported controller-label representation: {type(labels).__name__}")

        buttons = _integer_labels(buttons, self.vocab_sizes["buttons"], "buttons")
        main_stick = _integer_labels(main_stick, self.vocab_sizes["main_stick"], "main_stick")
        if buttons.shape != main_stick.shape:
            raise ValueError(
                f"controller label shapes must match, got {tuple(buttons.shape)} and "
                f"{tuple(main_stick.shape)}"
            )
        if buttons.device != main_stick.device:
            raise ValueError("controller labels must be on the same device")
        return ControllerLabels(buttons=buttons, main_stick=main_stick)

    def decode(self, labels: Any) -> ControllerState:
        """Decode category labels to valid canonical GameCube controller states."""

        labels = self._coerce_labels(labels)
        button_label = labels.buttons

        components: list[Tensor] = []
        quotient = button_label
        for size in reversed(self._button_axis_sizes):
            quotient, component = (
                torch.div(quotient, size, rounding_mode="floor"),
                torch.remainder(quotient, size),
            )
            components.append(component)
        components.reverse()
        b, xy, lr, z_a_shoulder, c_stick_label = components

        z = z_a_shoulder == 6
        a = torch.where(z, torch.ones_like(z), torch.div(z_a_shoulder, 3, rounding_mode="floor").bool())
        shoulder_bucket = torch.where(z, torch.ones_like(z_a_shoulder), torch.remainder(z_a_shoulder, 3))
        shoulder_values = torch.tensor(_SHOULDER_VALUES, dtype=torch.float32, device=button_label.device)

        return ControllerState(
            main_stick=self.main_stick.decode(labels.main_stick),
            c_stick=self.c_stick.decode(c_stick_label),
            shoulder=shoulder_values[shoulder_bucket],
            buttons=ButtonState(
                A=a.bool(),
                B=b.bool(),
                X=torch.zeros_like(xy, dtype=torch.bool),
                Y=xy.bool(),
                Z=z.bool(),
                L=lr.bool(),
                R=torch.zeros_like(lr, dtype=torch.bool),
                D_UP=torch.zeros_like(b, dtype=torch.bool),
            ),
        )

    def one_hot_component(
        self,
        component: str,
        labels: Any,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """One-hot encode one component using its codec-derived vocabulary."""

        if component not in self.vocab_sizes:
            raise KeyError(f"unknown controller component {component!r}; expected {COMPONENT_ORDER!r}")
        integer = _integer_labels(labels, self.vocab_sizes[component], component)
        return F.one_hot(integer, num_classes=self.vocab_sizes[component]).to(dtype=dtype)

    def component_one_hots(
        self,
        labels: Any,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> dict[str, Tensor]:
        """Return ordered per-component one-hots without concatenating them."""

        labels = self._coerce_labels(labels)
        return {
            name: self.one_hot_component(name, getattr(labels, name), dtype=dtype) for name in COMPONENT_ORDER
        }

    def one_hot(
        self,
        labels: Any,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Return upstream's concatenated previous-controller representation."""

        components = self.component_one_hots(labels, dtype=dtype)
        return torch.cat(tuple(components[name] for name in COMPONENT_ORDER), dim=-1)

    def sample(
        self,
        logits: ControllerLabels | Mapping[str, Tensor] | Sequence[Tensor],
        *,
        temperature: float | None = None,
        generator: torch.Generator | None = None,
        deterministic: bool = False,
    ) -> ControllerLabels:
        """Sample valid component labels from logits in FP32.

        Autoregressive conditioning remains the controller head's responsibility;
        this method samples whichever two finalized component distributions it is
        given and guarantees labels that :meth:`decode` accepts.
        """

        if isinstance(logits, ControllerLabels):
            logits_by_name = logits.as_dict()
        elif isinstance(logits, Mapping):
            logits_by_name = {name: logits[name] for name in COMPONENT_ORDER}
        elif isinstance(logits, Sequence) and len(logits) == len(COMPONENT_ORDER):
            logits_by_name = dict(zip(COMPONENT_ORDER, logits, strict=True))
        else:
            raise TypeError(f"unsupported controller-logit representation: {type(logits).__name__}")

        if temperature is not None and (not math.isfinite(temperature) or temperature <= 0.0):
            raise ValueError("temperature must be finite and positive")

        samples: dict[str, Tensor] = {}
        prefix_shape: torch.Size | None = None
        device: torch.device | None = None
        for name in COMPONENT_ORDER:
            component_logits = torch.as_tensor(logits_by_name[name])
            vocab_size = self.vocab_sizes[name]
            if component_logits.ndim == 0 or component_logits.shape[-1] != vocab_size:
                raise ValueError(
                    f"{name} logits must have shape [..., {vocab_size}], got {tuple(component_logits.shape)}"
                )
            if prefix_shape is None:
                prefix_shape = component_logits.shape[:-1]
                device = component_logits.device
            elif component_logits.shape[:-1] != prefix_shape or component_logits.device != device:
                raise ValueError("all component logits must have the same prefix shape and device")

            fp32_logits = component_logits.float()
            if temperature is not None:
                fp32_logits = fp32_logits / temperature
            if deterministic:
                sample = fp32_logits.argmax(dim=-1)
            else:
                probabilities = torch.softmax(fp32_logits, dim=-1)
                if not torch.isfinite(probabilities).all() or (probabilities.sum(dim=-1) <= 0).any():
                    raise ValueError(f"{name} logits do not define a finite categorical distribution")
                flat = probabilities.reshape(-1, vocab_size)
                sample = torch.multinomial(flat, 1, generator=generator).reshape(prefix_shape)
            samples[name] = sample.long()

        return ControllerLabels(
            buttons=samples["buttons"],
            main_stick=samples["main_stick"],
        )


__all__ = [
    "BUTTON_ORDER",
    "COMPONENT_ORDER",
    "PACKED_CONTROLLER_ORDER",
    "SLIPPI_AI_COMMIT",
    "ButtonState",
    "ControllerLabels",
    "ControllerState",
    "CustomV1Codec",
    "StickState",
]
