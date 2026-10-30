"""Legacy API compatibility layer.

Translates the clean v1 output into the response shape used by common
commercial ID-OCR services (a ``verify_document`` style endpoint), so the engine
can be dropped in behind an existing integration without changing the caller:

    { "status": "ok",
      "data": { "errorCode": 0, "ocr": {...}, "mrz": {...} } }

A field flagged needs_review is still emitted; review state is surfaced through
``errorCode``/``message`` only when a *critical* field is missing entirely,
matching the failure expectations of those services.
"""
from __future__ import annotations


def _val(fields: dict, key: str):
    f = fields.get(key) or {}
    return f.get("value")


def to_legacy(result: dict) -> dict:
    """Map clean v1 output -> legacy verify_document-style payload."""
    fields = result.get("fields", {})
    mrz = result.get("mrz") or {}

    ocr = {
        "name": _val(fields, "full_name_latin"),
        "nameArabic": _val(fields, "full_name_arabic"),
        "documentNumber": _val(fields, "passport_number"),
        "nationality": _val(fields, "nationality"),
        "issuingCountry": _val(fields, "issuing_country"),
        "dateOfBirth": _val(fields, "date_of_birth"),
        "sex": _val(fields, "sex"),
        "dateOfIssue": _val(fields, "date_of_issue"),
        "dateOfExpiry": _val(fields, "date_of_expiry"),
        "placeOfBirth": _val(fields, "place_of_birth"),
        "personalNumber": _val(fields, "national_number"),
    }

    # Split the Latin full name into surname/given for the provider's MRZ path.
    line1 = mrz.get("line1", "")
    line2 = mrz.get("line2", "")
    mrz_block = {
        "surname": "",
        "givenNames": _val(fields, "full_name_latin") or "",
        "firstName": "",
        "lastName": "",
        "documentNumber": _val(fields, "passport_number"),
        "issuingStateCode": _val(fields, "issuing_country") or _val(fields, "nationality"),
        "nationality": _val(fields, "nationality"),
        "dateOfBirth": _val(fields, "date_of_birth"),
        "dateOfExpiry": _val(fields, "date_of_expiry"),
        "sex": _val(fields, "sex"),
        "line1": line1,
        "line2": line2,
        "mrzCode": f"{line1}\n{line2}" if line1 or line2 else "",
        # provider treats validState == 0 as valid
        "validState": 0 if mrz.get("valid") else 1,
    }

    # A usable record needs at least a passport number or a name.
    has_core = bool(ocr["documentNumber"] or ocr["name"])
    error_code = 0 if has_core else 6

    payload = {
        "status": "ok" if has_core else "error",
        "data": {
            "errorCode": error_code,
            "ocr": ocr,
            "mrz": mrz_block,
            # extra, ignored by the current provider but useful for a smarter one
            "needs_review": result.get("needs_review", False),
            "review_fields": result.get("review_fields", []),
            "portrait": result.get("portrait"),
        },
    }
    if not has_core:
        payload["data"]["message"] = "Passport OCR could not extract core fields; manual entry required."
    return payload
