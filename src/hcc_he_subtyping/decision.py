from __future__ import annotations


def fuse_binary_predictions(m1_vs_m23: int, m12_vs_m3: int) -> str:
    """Fuse binary outputs: 0/0 -> M1, 1/1 -> M3, discordant -> M2."""
    if m1_vs_m23 not in (0, 1) or m12_vs_m3 not in (0, 1):
        raise ValueError("Binary predictions must be 0 or 1")
    if m1_vs_m23 == 0 and m12_vs_m3 == 0:
        return "M1"
    if m1_vs_m23 == 1 and m12_vs_m3 == 1:
        return "M3"
    return "M2"

