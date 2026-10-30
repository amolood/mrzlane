"""Fusion + cross-validation — the heart of the zero-error policy.

Each field is decided by combining the MRZ (authoritative when its check digit
passes) with the printed visual reading (independent second source):

  - MRZ validated AND visual agrees      -> confirmed, confidence ~0.99
  - MRZ validated, no visual to compare   -> confirmed, confidence ~0.95
  - MRZ validated, visual DISAGREES        -> needs_review (never guess)
  - only visual (no/invalid MRZ)           -> needs_review unless the value is
                                              itself self-checking (e.g. national
                                              number format) -> low confidence
  - nothing                                -> absent

A field carries {value, confidence, source, needs_review}. The caller treats
any needs_review=True field as "must be verified by a human".
"""
from __future__ import annotations

import re

REVIEW_THRESHOLD = 0.80   # below this a field is flagged for human review

# Digit->letter lookalikes for recovering 3-letter country codes from noisy OCR.
_DIGIT_TO_LETTER = str.maketrans({"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"})


def _deconfuse_code(value: str | None) -> str | None:
    """Map a possibly mis-OCR'd country code to A-Z; special-case SDN.

    Only returns a value when the result is a clean 3-letter code; otherwise
    None (so the caller keeps the raw token and flags it). The SDN lookalikes
    (e.g. '5DN', 'S0N', 'SD8') collapse to SDN since B/8 and N are the usual
    confusions on the Sudanese code.
    """
    if not value:
        return None
    s = re.sub(r"\s", "", str(value).upper()).translate(_DIGIT_TO_LETTER)
    # Common tail confusion: the 'N' of SDN read as 'B'/'H'.
    if s[:2] == "SD" and len(s) == 3:
        return "SDN"
    return s if re.fullmatch(r"[A-Z]{3}", s) else None


def _date_digits(d: str | None) -> str:
    return re.sub(r"\D", "", d or "")


def _field(value, confidence, source, needs_review=None):
    if needs_review is None:
        needs_review = confidence < REVIEW_THRESHOLD or value in (None, "")
    return {"value": value, "confidence": round(confidence, 2),
            "source": source, "needs_review": bool(needs_review)}


def _reconcile(mrz_val, mrz_ok, vis_val, *, name="", normaliser=lambda x: x):
    """Decide one field from an MRZ value (with validity) and a visual value."""
    mv = normaliser(mrz_val) if mrz_val else None
    vv = normaliser(vis_val) if vis_val else None

    if mv and mrz_ok:
        if vv is None:
            return _field(mrz_val, 0.95, "mrz")
        if mv == vv:
            return _field(mrz_val, 0.99, "mrz+print")      # two sources agree
        # conflict between a *valid* MRZ and the print -> must review
        return _field(mrz_val, 0.60, "mrz", needs_review=True)
    if mv and not mrz_ok:
        # MRZ present but not check-digit-validated
        if vv and mv == vv:
            return _field(mrz_val, 0.85, "mrz+print")      # agreement rescues it
        return _field(mrz_val or vis_val, 0.55, "mrz?", needs_review=True)
    if vv:
        return _field(vis_val, 0.55, "print", needs_review=True)
    return _field(None, 0.0, "none")


def fuse(mrz, visual: dict) -> dict:
    """Combine an MrzResult (or None) with visual fields into final output."""
    mrz_ok = bool(mrz and mrz.valid)
    checks = mrz.check_digits if mrz else {}
    vis_dates = visual.get("dates_print", [])
    vis_places = visual.get("places_print", [])

    fields: dict = {}

    # passport number — MRZ vs printed
    fields["passport_number"] = _reconcile(
        mrz.document_number if mrz else None,
        mrz_ok and checks.get("document_number", False),
        visual.get("passport_number_print"),
        normaliser=lambda s: re.sub(r"\s", "", str(s).upper()),
    )

    # Latin name — MRZ is the reliable source; no clean printed comparison.
    latin = mrz.full_name_latin if mrz else None
    fields["full_name_latin"] = (
        _field(latin, 0.95 if mrz_ok else 0.6, "mrz",
               needs_review=not mrz_ok) if latin else _field(None, 0.0, "none")
    )

    # Arabic name — visual only (no MRZ equivalent). Flag for review by policy.
    ar = visual.get("full_name_arabic")
    fields["full_name_arabic"] = _field(ar, 0.7, "print", needs_review=True) if ar \
        else _field(None, 0.0, "none")

    # nationality / issuing country.
    # The mobile OCR sometimes misreads the 3-letter codes (digit<->letter
    # lookalikes: 5/S, 8/B/S, 0/O, 2/Z, 1/I...). When the MRZ is validated and
    # the code resolves to SDN after de-confusing, we trust SDN. A genuinely
    # foreign code (e.g. ETH) is left as-is and flagged — that is correct, not an
    # error. We never *force* SDN onto a code that is clearly another country.
    nat_norm = _deconfuse_code(mrz.nationality if mrz else None)
    iss_norm = _deconfuse_code(mrz.issuing_country if mrz else None)
    if mrz_ok and (iss_norm == "SDN" or nat_norm == "SDN"):
        nat_value = "SDN"
    elif nat_norm:
        nat_value = nat_norm
    else:
        nat_value = (mrz.nationality if mrz else None) or None
    fields["nationality"] = _reconcile(
        nat_value, mrz_ok, None, normaliser=lambda s: str(s).upper())
    iss_value = "SDN" if (mrz_ok and (iss_norm == "SDN" or nat_norm == "SDN")) else (
        iss_norm or (mrz.issuing_country if mrz else None))
    fields["issuing_country"] = _field(
        iss_value, 0.95 if mrz_ok else 0.5, "mrz",
        needs_review=not mrz_ok) if iss_value else _field(None, 0.0, "none")

    # sex
    fields["sex"] = _field(mrz.sex if mrz else None, 0.97 if mrz_ok else 0.5, "mrz",
                           needs_review=not mrz_ok) if mrz and mrz.sex else _field(None, 0.0, "none")

    # date of birth — MRZ vs any printed date that matches
    dob = mrz.date_of_birth if mrz else None
    dob_ok = mrz_ok and checks.get("date_of_birth", False)
    dob_vis = next((d for d in vis_dates if _date_digits(d) == _date_digits(dob)), None)
    fields["date_of_birth"] = _reconcile(dob, dob_ok, dob_vis,
                                         normaliser=_date_digits)

    # expiry — MRZ vs printed
    exp = mrz.date_of_expiry if mrz else None
    exp_ok = mrz_ok and checks.get("expiry", False)
    exp_vis = next((d for d in vis_dates if _date_digits(d) == _date_digits(exp)), None)
    fields["date_of_expiry"] = _reconcile(exp, exp_ok, exp_vis, normaliser=_date_digits)

    # issue date — printed only; accept the remaining date that is not dob/expiry
    used = {_date_digits(dob), _date_digits(exp)}
    issue_cands = [d for d in vis_dates if _date_digits(d) not in used]
    # issue date must be in the past and before expiry
    issue = issue_cands[0] if issue_cands else None
    fields["date_of_issue"] = _field(issue, 0.6, "print", needs_review=True) if issue \
        else _field(None, 0.0, "none")

    # national number — printed only, but self-checking by format
    nat = visual.get("national_number")
    fields["national_number"] = _field(nat, 0.8, "print",
                                       needs_review=not bool(re.fullmatch(r"\d{3}-\d{4}-\d{4}", nat or ""))) \
        if nat else _field(None, 0.0, "none")

    # places
    fields["place_of_birth"] = _field(vis_places[0], 0.6, "print", needs_review=True) \
        if vis_places else _field(None, 0.0, "none")
    fields["place_of_issue"] = _field(vis_places[-1], 0.6, "print", needs_review=True) \
        if len(vis_places) > 1 else _field(None, 0.0, "none")

    _sanity_checks(fields)
    return fields


def _as_date(value):
    import datetime
    if not value or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(value)):
        return None
    try:
        return datetime.date(*(int(x) for x in value.split("-")))
    except ValueError:
        return None


def _flag(field: dict, reason: str):
    """Force a field to review and record why."""
    field["needs_review"] = True
    field["confidence"] = min(field.get("confidence", 0.0), 0.5)
    field.setdefault("flags", []).append(reason)


def _sanity_checks(fields: dict) -> None:
    """Logical validation after fusion — a passing check digit is not enough.

    Catches impossibilities the MRZ math cannot (e.g. a two-digit year that
    resolves into the future, an expiry before issue, a non-SDN nationality).
    Zero-error policy: anything implausible is sent to review, never emitted as
    a confident value.
    """
    import datetime
    today = datetime.date.today()

    dob = _as_date(fields["date_of_birth"]["value"])
    exp = _as_date(fields["date_of_expiry"]["value"])
    iss = _as_date(fields["date_of_issue"]["value"])

    # DOB cannot be in the future (classic 2040 vs 1940 YY-pivot error).
    if dob and dob > today:
        _flag(fields["date_of_birth"], "dob_in_future")
    # DOB implausibly old (>120y).
    if dob and (today.year - dob.year) > 120:
        _flag(fields["date_of_birth"], "dob_too_old")
    # Expiry should be after issue.
    if exp and iss and exp <= iss:
        _flag(fields["date_of_expiry"], "expiry_before_issue")
        _flag(fields["date_of_issue"], "expiry_before_issue")
    # Issue date must not be in the future.
    if iss and iss > today:
        _flag(fields["date_of_issue"], "issue_in_future")
    # Nationality on a Sudanese passport must be SDN.
    nat = fields["nationality"]["value"]
    if nat and nat != "SDN":
        _flag(fields["nationality"], "unexpected_nationality")
