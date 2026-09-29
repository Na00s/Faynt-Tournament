"""Versioned protocol mappings used by E000."""

from __future__ import annotations

import hashlib
import json

from melee_policy.e000 import ENUM_MAP_VERSION

# Slippi CSS-order IDs to the internal character IDs emitted in post-frame events.
EXTERNAL_CHARACTER_TO_INTERNAL: dict[int, int] = {
    0: 2,
    1: 3,
    2: 1,
    3: 24,
    4: 4,
    5: 5,
    6: 6,
    7: 17,
    8: 0,
    9: 18,
    10: 16,
    11: 8,
    12: 9,
    13: 12,
    14: 10,
    15: 15,
    16: 13,
    17: 14,
    18: 19,
    19: 7,
    20: 22,
    21: 20,
    22: 21,
    23: 26,
    24: 23,
    25: 25,
}

CHARACTER_FOLDER_TO_INTERNAL: dict[str, tuple[int, ...]] = {
    "BOWSER": (5,),
    "CPTFALCON": (2,),
    "DK": (3,),
    "DOC": (21,),
    "FALCO": (22,),
    "FOX": (1,),
    "GAMEANDWATCH": (24,),
    "GANONDORF": (25,),
    "ICE_CLIMBERS": (10, 11),
    "JIGGLYPUFF": (15,),
    "KIRBY": (4,),
    "LINK": (6,),
    "LUIGI": (17,),
    "MARIO": (0,),
    "MARTH": (18,),
    "MEWTWO": (16,),
    "NESS": (8,),
    "PEACH": (9,),
    "PICHU": (23,),
    "PIKACHU": (12,),
    "ROY": (26,),
    "SAMUS": (13,),
    "SHEIK": (7,),
    "YLINK": (20,),
    "YOSHI": (14,),
    "ZELDA": (19,),
}

# Raw Slippi stage ID to libmelee Stage value. The raw ID remains canonical.
SLIPPI_STAGE_TO_LIBMELEE: dict[int, int] = {
    2: 8,
    3: 18,
    8: 6,
    28: 26,
    31: 24,
    32: 25,
}

STAGE_EDGE_GROUND_X: dict[int, float] = {
    2: 63.3475494385,
    3: 87.75,
    8: 56.0,
    28: 77.2713012695,
    31: 68.4000015259,
    32: 85.5656967163,
}

BUTTON_MASKS: dict[str, int] = {
    "DPAD_LEFT": 1 << 0,
    "DPAD_RIGHT": 1 << 1,
    "DPAD_DOWN": 1 << 2,
    "DPAD_UP": 1 << 3,
    "Z": 1 << 4,
    "R": 1 << 5,
    "L": 1 << 6,
    "A": 1 << 8,
    "B": 1 << 9,
    "X": 1 << 10,
    "Y": 1 << 11,
    "START": 1 << 12,
}


def enum_map_digest() -> str:
    payload = {
        "version": ENUM_MAP_VERSION,
        "external_character_to_internal": EXTERNAL_CHARACTER_TO_INTERNAL,
        "character_folder_to_internal": CHARACTER_FOLDER_TO_INTERNAL,
        "slippi_stage_to_libmelee": SLIPPI_STAGE_TO_LIBMELEE,
        "button_masks": BUTTON_MASKS,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
