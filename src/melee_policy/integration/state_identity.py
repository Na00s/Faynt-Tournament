"""Stable content identities for ignored local policy state files."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping

import torch


def state_dictionary_sha256(state_dictionary: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, dtypes, shapes, and canonical CPU values.

    ``torch.save`` bytes are not a stable identity because archive metadata can
    differ between otherwise identical saves.  This digest deliberately ignores
    storage layout and serialization details while retaining every value needed
    to distinguish model states.
    """

    digest = hashlib.sha256()
    for name in sorted(state_dictionary):
        tensor = state_dictionary[name]
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError("state dictionaries must map string names to tensors")
        if tensor.is_sparse or tensor.is_quantized:
            raise TypeError(f"unsupported tensor layout for {name!r}")

        canonical = tensor.detach().to(device="cpu").contiguous()
        encoded_name = name.encode("utf-8")
        encoded_dtype = str(canonical.dtype).encode("ascii")
        raw = canonical.numpy().tobytes(order="C")

        digest.update(struct.pack(">I", len(encoded_name)))
        digest.update(encoded_name)
        digest.update(struct.pack(">I", len(encoded_dtype)))
        digest.update(encoded_dtype)
        digest.update(struct.pack(">I", canonical.ndim))
        for dimension in canonical.shape:
            digest.update(struct.pack(">Q", int(dimension)))
        digest.update(struct.pack(">Q", len(raw)))
        digest.update(raw)
    return digest.hexdigest()


__all__ = ["state_dictionary_sha256"]
