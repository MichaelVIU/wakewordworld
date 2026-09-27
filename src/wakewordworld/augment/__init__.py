"""Augmentation lane: real noise and measured room impulse responses at controlled levels.

Augmented results live in their own manifests (``<version>-aug-<condition>``) and are
never pooled with raw results. See :mod:`wakewordworld.augment.lane`.
"""

from wakewordworld.augment.lane import (
    DEFAULT_CONDITIONS,
    Condition,
    build_lane,
    data_root_for_condition,
    parse_condition,
)
from wakewordworld.augment.mix import MixResult, mix_at_snr, speed_perturb
from wakewordworld.augment.noise import NoiseBank, NoiseClip, NoiseSet, load_noise_sets
from wakewordworld.augment.rir import RirBank, apply_rir, normalise_rir

__all__ = [
    "DEFAULT_CONDITIONS",
    "Condition",
    "MixResult",
    "NoiseBank",
    "NoiseClip",
    "NoiseSet",
    "RirBank",
    "apply_rir",
    "build_lane",
    "data_root_for_condition",
    "load_noise_sets",
    "mix_at_snr",
    "normalise_rir",
    "parse_condition",
    "speed_perturb",
]
