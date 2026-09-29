"""Parser-free PyTorch tensor boundary for canonical Melee policy batches.

The nested records mirror ``slippi_ai.types`` at the repository-pinned
slippi-ai revision.  Every leaf is a :class:`torch.Tensor`; this module does
not import libmelee, Peppi, slippi-ai, or the earlier replay repository.

Component records use a shared leading shape ``S``.  ``S`` is normally
``[B, T]`` for training and full-sequence inference, and may be ``[B]`` for a
single cached inference step.  The one exception is :class:`ItemsBatch`,
whose leaves have shape ``[*S, 15]`` for upstream slots ``item_0`` through
``item_14``.

The earlier E000 representation is not sufficient to construct this boundary
without enrichment.  Preprocessing must retain the processed controller
button mask (not only ``buttons_physical``), follower/Nana state, item slots,
Randall and FoD inputs, and the configured player-name code.  It must also
resolve missing values into the exact slippi-ai categorical sentinels before
creating these tensors.  Replay parsing and enrichment belong upstream, not in
the model repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch

MAX_ITEMS: Final[int] = 15
"""Number of ordered item slots in the pinned slippi-ai ``Items`` schema."""

BUTTON_ORDER: Final[tuple[str, ...]] = ("A", "B", "X", "Y", "Z", "L", "R", "D_UP")
"""Exact upstream controller-button field order."""


@dataclass(frozen=True, slots=True)
class ButtonsBatch:
    """Named digital controller inputs; every leaf is ``bool [*S]``.

    These values must be derived with the pinned slippi-ai controller
    semantics from the processed Slippi button mask.  E000's physical button
    bitset alone is not a lossless substitute.
    """

    A: torch.Tensor
    B: torch.Tensor
    X: torch.Tensor
    Y: torch.Tensor
    Z: torch.Tensor
    L: torch.Tensor
    R: torch.Tensor
    D_UP: torch.Tensor


@dataclass(frozen=True, slots=True)
class StickBatch:
    """One analog stick with ``float32 [*S]`` x and y axes."""

    x: torch.Tensor
    y: torch.Tensor


@dataclass(frozen=True, slots=True)
class ControllerBatch:
    """Raw slippi-ai controller representation over leading shape ``S``.

    ``main_stick`` and ``c_stick`` use the upstream logical coordinate range.
    ``shoulder`` is the shared logical shoulder value used by slippi-ai, with
    shape ``[*S]`` and floating dtype.  The custom_v1 codec consumes this raw
    structure and owns all bucketing and decoding.
    """

    main_stick: StickBatch
    c_stick: StickBatch
    shoulder: torch.Tensor
    buttons: ButtonsBatch


@dataclass(frozen=True, slots=True)
class NanaBatch:
    """Ice Climbers follower state, matching upstream field order.

    All leaves have shape ``[*S]``.  Boolean fields use ``torch.bool``;
    categorical/count fields use an integer dtype; positions and shield use a
    floating dtype.  When Nana is absent, ``exists`` is false and all other
    leaves still contain the exact upstream missing/default representation.
    """

    exists: torch.Tensor
    percent: torch.Tensor
    facing: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    action: torch.Tensor
    invulnerable: torch.Tensor
    character: torch.Tensor
    jumps_left: torch.Tensor
    shield_strength: torch.Tensor
    on_ground: torch.Tensor


@dataclass(frozen=True, slots=True)
class PlayerBatch:
    """One player slot, matching pinned ``slippi_ai.types.Player``.

    Scalar leaves have shape ``[*S]``.  ``controller`` is retained for exact
    game-schema parity even though the policy's controlled previous/current
    action is supplied separately as :attr:`PolicyBatch.controller_t`.
    """

    percent: torch.Tensor
    facing: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    action: torch.Tensor
    invulnerable: torch.Tensor
    character: torch.Tensor
    jumps_left: torch.Tensor
    shield_strength: torch.Tensor
    on_ground: torch.Tensor
    controller: ControllerBatch
    nana: NanaBatch


@dataclass(frozen=True, slots=True)
class RandallBatch:
    """Yoshi's Story Randall position, ``float32 [*S]`` per coordinate."""

    x: torch.Tensor
    y: torch.Tensor


@dataclass(frozen=True, slots=True)
class FoDPlatformsBatch:
    """Fountain of Dreams platform heights, ``float32 [*S]``."""

    left: torch.Tensor
    right: torch.Tensor


@dataclass(frozen=True, slots=True)
class ItemsBatch:
    """The 15 ordered upstream item slots in a stacked tensor layout.

    Each leaf has shape ``[*S, MAX_ITEMS]``.  The final dimension maps directly
    to ``item_0`` through ``item_14``.  ``exists`` uses ``torch.bool``;
    ``type`` and ``state`` use integer dtypes; x and y use floating dtypes.
    Values in unused slots must match the upstream defaults and are ignored
    when ``exists`` is false.
    """

    exists: torch.Tensor
    type: torch.Tensor
    state: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor


@dataclass(frozen=True, slots=True)
class GameStateBatch:
    """Full slippi-ai game state over leading shape ``S``.

    Perspective preprocessing makes ``p0`` the controlled/self player and
    ``p1`` the opponent.  ``stage`` is an integer categorical tensor with
    shape ``[*S]``.  The remaining fields preserve the pinned upstream nesting.
    """

    p0: PlayerBatch
    p1: PlayerBatch
    stage: torch.Tensor
    randall: RandallBatch
    fod_platforms: FoDPlatformsBatch
    items: ItemsBatch


@dataclass(frozen=True, slots=True)
class PolicyBatch:
    """Aligned behavior-cloning batch consumed by the policy and its loss.

    Every temporal leaf has batch-major shape ``[B, T, ...]``.

    ``game_state_t`` and ``controller_t`` are inputs at frame ``t``.
    ``controller_t_plus_1`` is already aligned by preprocessing and is the
    target for that same tensor position.  Model code must not shift it again.

    ``reset_mask[b, t]`` means a new replay segment begins before position t.
    ``padding_mask[b, t]`` is true for a real, attention-valid frame.
    ``valid_position_mask[b, t]`` is true only when the aligned next-frame
    controller target is valid for loss.  It must be false at replay ends,
    across raw-frame gaps, and wherever required source or target fields are
    missing.  All three masks use ``torch.bool [B, T]``.

    ``player_name`` is the optional upstream categorical name code with shape
    ``[B, T]``.  It may be ``None`` when player-name conditioning is disabled.
    """

    game_state_t: GameStateBatch
    controller_t: ControllerBatch
    controller_t_plus_1: ControllerBatch
    reset_mask: torch.Tensor
    padding_mask: torch.Tensor
    valid_position_mask: torch.Tensor
    player_name: torch.Tensor | None = None


__all__ = [
    "BUTTON_ORDER",
    "MAX_ITEMS",
    "ButtonsBatch",
    "ControllerBatch",
    "FoDPlatformsBatch",
    "GameStateBatch",
    "ItemsBatch",
    "NanaBatch",
    "PlayerBatch",
    "PolicyBatch",
    "RandallBatch",
    "StickBatch",
]
