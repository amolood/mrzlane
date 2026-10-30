"""Output assembly: clean JSON v1 for the engine's native response.

Keeps the per-field {value, confidence, source, needs_review, flags} shape from
fusion and adds document-level metadata plus an overall `needs_review` that is
True if *any* critical field needs review.
"""
from __future__ import annotations

ENGINE_NAME = "mrzlane"
ENGINE_VERSION = "0.1.0"
SCHEMA_VERSION = "1.0"

# Fields a human must trust completely before the record is auto-accepted.
CRITICAL_FIELDS = (
    "passport_number", "full_name_latin", "date_of_birth",
    "date_of_expiry", "nationality", "sex",
)


def build_output(fields: dict, mrz, portrait: str | None,
                 detected: bool = True) -> dict:
    critical_review = [k for k in CRITICAL_FIELDS
                       if fields.get(k, {}).get("needs_review", True)]
    warnings = []
    for k, f in fields.items():
        for fl in f.get("flags", []):
            warnings.append(f"{k}: {fl}")

    mrz_block = None
    if mrz is not None:
        mrz_block = {
            "line1": mrz.line1,
            "line2": mrz.line2,
            "valid": mrz.valid,
            "check_digits": mrz.check_digits,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "document_type": "passport",
        "country": "SDN",
        "detected": detected,
        "needs_review": bool(critical_review),
        "review_fields": critical_review,
        "fields": fields,
        "mrz": mrz_block,
        "portrait": portrait,
        "warnings": warnings,
        "engine": {"name": ENGINE_NAME, "version": ENGINE_VERSION},
    }
