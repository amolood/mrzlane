"""End-to-end pipeline: image -> clean v1 result (and legacy-API compat).

Loads OCR engines once (module-level cache in `visual`) and runs:
  preprocess -> visual OCR tokens -> MRZ read -> fuse+sanity -> portrait -> output
"""
from __future__ import annotations

from . import (preprocess, visual, mrz as mrz_mod, fuse as fuse_mod,
               portrait as portrait_mod, schema, orient as orient_mod)


def process(path_or_bytes, *, with_portrait: bool = True, compat: bool = False) -> dict:
    prep = preprocess.prepare(path_or_bytes)
    ocr_img = prep["ocr_image"]

    # Cheap path first: full en+ar read on the image as-is. The vast majority of
    # scans are already upright, so they pay nothing extra.
    tokens = visual.read_tokens(ocr_img, langs=("en", "ar"))
    mrz = mrz_mod.read_mrz([t.text for t in tokens])
    upright = ocr_img

    # Only if the MRZ did not validate do we try other orientations — using an
    # English-only probe (it reads the MRZ) to find the correct rotation, then a
    # full en+ar read on it. This keeps the 4x cost confined to images that need it.
    if not (mrz and mrz.valid):
        def read_en(image):
            return [t.text for t in visual.read_tokens(image, langs=("en",))]

        rotated, rmrz, _ = orient_mod.resolve_by_mrz(ocr_img, read_en)
        if rmrz and rmrz.valid and rotated is not ocr_img:
            upright = rotated
            tokens = visual.read_tokens(upright, langs=("en", "ar"))
            mrz = mrz_mod.read_mrz([t.text for t in tokens]) or rmrz

    visual_fields = visual.extract_visual_fields(tokens)
    fields = fuse_mod.fuse(mrz, visual_fields)

    portrait_b64 = portrait_mod.portrait_base64(upright) if with_portrait else None
    result = schema.build_output(fields, mrz, portrait_b64, detected=True)

    if compat:
        from . import compat as compat_mod
        return compat_mod.to_legacy(result)
    return result
