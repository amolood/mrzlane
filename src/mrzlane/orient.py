"""Orientation resolution driven by the MRZ — the only reliable compass.

Face detection alone is not enough: an upside-down face still scores highly on
YuNet, so a 180°-rotated passport passes face orientation yet reads mirrored.
The MRZ, by contrast, is self-validating — the orientation whose MRZ passes its
check digits is unambiguously correct.

Strategy (cheap-first): try the current orientation; if its MRZ does not
validate, rotate through the other three and keep the first that yields a valid
MRZ. Only pays the extra OCR cost on images that actually need it.
"""
from __future__ import annotations

import cv2

from .mrz import read_mrz

_ROTATIONS = [
    (0, None),
    (180, cv2.ROTATE_180),
    (90, cv2.ROTATE_90_CLOCKWISE),
    (270, cv2.ROTATE_90_COUNTERCLOCKWISE),
]


def resolve_by_mrz(img, read_lines_fn):
    """Return (upright_image, mrz, lines) using the MRZ as the orientation oracle.

    ``read_lines_fn(image) -> list[str]`` runs OCR and returns text lines; it is
    injected so this module stays independent of the OCR engine. Falls back to
    the best-effort original orientation when no rotation validates.
    """
    best_invalid = None  # (image, mrz, lines) for a parsed-but-unvalidated MRZ
    for _angle, rot in _ROTATIONS:
        cand = img if rot is None else cv2.rotate(img, rot)
        lines = read_lines_fn(cand)
        mrz = read_mrz(lines)
        if mrz is not None and mrz.valid:
            return cand, mrz, lines                 # validated -> definitely correct
        if mrz is not None and best_invalid is None:
            best_invalid = (cand, mrz, lines)
    if best_invalid is not None:
        return best_invalid
    # no MRZ anywhere: return original with a fresh read
    lines = read_lines_fn(img)
    return img, read_mrz(lines), lines
