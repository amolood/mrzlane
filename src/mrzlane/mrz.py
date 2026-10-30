"""MRZ (Machine Readable Zone) reading and validation for TD3 passports.

The Sudanese passport carries a TD3 zone: two lines of 44 characters in the
OCR-B font. Because the font is standardised and the zone embeds check digits,
this is the single most reliable source on the page — we read it first and
treat validated fields as authoritative.

This module is deliberately pure (no OCR engine dependency): it takes already
extracted text lines and parses/validates them. The visual OCR layer feeds it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re

# TD3: 2 lines, 44 chars each. Filler is '<'.
TD3_LINE_LEN = 44
_WEIGHTS = (7, 3, 1)
_VALUE = {c: i for i, c in enumerate("0123456789")}
_VALUE.update({c: 10 + i for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")})
_VALUE["<"] = 0


def check_digit(data: str) -> int:
    """Compute the standard ICAO 9303 check digit over *data*."""
    total = 0
    for i, ch in enumerate(data):
        total += _VALUE.get(ch, 0) * _WEIGHTS[i % 3]
    return total % 10


def _to_int_digit(ch: str) -> int | None:
    return int(ch) if ch.isdigit() else None


def _parse_date(yymmdd: str, kind: str = "generic") -> str | None:
    """YYMMDD -> ISO YYYY-MM-DD with a century pivot.

    ``kind`` refines the two-digit-year century:
      - "birth":  must be in the past -> a future result is pushed back 100y
      - "expiry": must be in the future -> a past result is pushed forward 100y
      - "generic": pivot at +20y into the future (ICAO-style default)
    """
    if not re.fullmatch(r"\d{6}", yymmdd):
        return None
    yy, mm, dd = int(yymmdd[0:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
    import datetime

    today = datetime.date.today()
    pivot = (today.year % 100) + 20
    year = 1900 + yy if yy > pivot else 2000 + yy

    def valid(y):
        try:
            datetime.date(y, mm, dd)
            return True
        except ValueError:
            return False

    if kind == "birth" and valid(year) and datetime.date(year, mm, dd) > today:
        year -= 100                       # a birth date cannot be in the future
    elif kind == "expiry" and valid(year) and datetime.date(year, mm, dd) < today:
        year += 100                       # an expiry date cannot be in the past
    try:
        datetime.date(year, mm, dd)
    except ValueError:
        return None
    return f"{year:04d}-{mm:02d}-{dd:02d}"


@dataclass
class MrzResult:
    line1: str
    line2: str
    document_type: str = ""
    issuing_country: str = ""
    surname: str = ""
    given_names: str = ""
    document_number: str = ""
    nationality: str = ""
    date_of_birth: str | None = None
    sex: str = ""
    date_of_expiry: str | None = None
    personal_number: str = ""
    valid: bool = False
    check_digits: dict = field(default_factory=dict)

    @property
    def full_name_latin(self) -> str:
        return " ".join(p for p in (self.given_names, self.surname) if p).strip()

    def to_dict(self) -> dict:
        return {
            "line1": self.line1,
            "line2": self.line2,
            "document_type": self.document_type,
            "issuing_country": self.issuing_country,
            "surname": self.surname,
            "given_names": self.given_names,
            "full_name_latin": self.full_name_latin,
            "document_number": self.document_number,
            "nationality": self.nationality,
            "date_of_birth": self.date_of_birth,
            "sex": self.sex,
            "date_of_expiry": self.date_of_expiry,
            "personal_number": self.personal_number,
            "valid": self.valid,
            "check_digits": self.check_digits,
        }


def _normalise_line(line: str) -> str:
    """Uppercase, strip spaces, map common OCR confusions to the MRZ charset."""
    s = re.sub(r"\s+", "", line.upper())
    # MRZ alphabet is A-Z 0-9 <. Map frequent lookalikes to filler.
    s = s.replace("«", "<").replace("»", "<").replace("≤", "<").replace("K<", "K<")
    s = re.sub(r"[^A-Z0-9<]", "<", s)
    return s


def find_mrz_lines(text_lines: list[str]) -> tuple[str, str] | None:
    """Pick the two TD3 lines out of arbitrary OCR output."""
    candidates = []
    for raw in text_lines:
        s = _normalise_line(raw)
        # A TD3 line is long and dominated by the MRZ alphabet.
        if len(s) >= 30 and (s.count("<") + sum(c.isalnum() for c in s)) >= len(s) - 2:
            candidates.append(s)
    # Prefer a line that starts with 'P' (passport) for line1 and a digit-heavy line2.
    if len(candidates) < 2:
        return None
    # Pad/trim to 44 and take the best two consecutive long lines.
    candidates = [c[:TD3_LINE_LEN].ljust(TD3_LINE_LEN, "<") for c in candidates]
    line1 = next((c for c in candidates if c.startswith("P")), candidates[0])
    rest = [c for c in candidates if c is not line1]
    # line2 is the one richest in digits.
    line2 = max(rest, key=lambda c: sum(ch.isdigit() for ch in c)) if rest else candidates[1]
    return line1, line2


def parse_td3(line1: str, line2: str) -> MrzResult:
    """Parse and validate a TD3 passport MRZ (ICAO 9303)."""
    l1 = _normalise_line(line1)[:TD3_LINE_LEN].ljust(TD3_LINE_LEN, "<")
    l2 = _normalise_line(line2)[:TD3_LINE_LEN].ljust(TD3_LINE_LEN, "<")

    res = MrzResult(line1=l1, line2=l2)
    res.document_type = l1[0:2].replace("<", "")
    res.issuing_country = l1[2:5].replace("<", "")

    # Standard TD3 separates surname from given names with '<<'. Some Sudanese
    # passports pack the whole name with single '<' and no '<<' separator — in
    # that case the printed "Full Name" has no surname/given split, so we keep
    # the entire string as the full name and leave surname/given empty.
    name_field = l1[5:].rstrip("<")
    if "<<" in name_field:
        surname, given = name_field.split("<<", 1)
        res.surname = surname.replace("<", " ").strip()
        res.given_names = given.replace("<", " ").strip()
    else:
        # No explicit split: treat the whole thing as given names (full name).
        res.given_names = name_field.replace("<", " ").strip()
        res.surname = ""

    doc_num = l2[0:9]
    doc_cd = _to_int_digit(l2[9])
    res.document_number = doc_num.replace("<", "")
    res.nationality = l2[10:13].replace("<", "")

    dob = l2[13:19]
    dob_cd = _to_int_digit(l2[19])
    res.date_of_birth = _parse_date(dob, "birth")
    res.sex = {"M": "M", "F": "F"}.get(l2[20], "X" if l2[20] == "<" else l2[20])

    exp = l2[21:27]
    exp_cd = _to_int_digit(l2[27])
    res.date_of_expiry = _parse_date(exp, "expiry")

    personal = l2[28:42]
    personal_cd = _to_int_digit(l2[42])
    res.personal_number = personal.replace("<", "")

    composite = l2[0:10] + l2[13:20] + l2[21:28] + l2[28:43]
    composite_cd = _to_int_digit(l2[43])

    checks = {
        "document_number": doc_cd is not None and check_digit(doc_num) == doc_cd,
        "date_of_birth": dob_cd is not None and check_digit(dob) == dob_cd,
        "expiry": exp_cd is not None and check_digit(exp) == exp_cd,
        # personal number check is '<' (0) when unused — accept either.
        "personal_number": (personal_cd is None) or check_digit(personal) == personal_cd
        or (personal.strip("<") == ""),
        "composite": composite_cd is not None and check_digit(composite) == composite_cd,
    }
    res.check_digits = checks
    # Valid when the three core identity check digits pass.
    res.valid = checks["document_number"] and checks["date_of_birth"] and checks["expiry"]
    return res


def read_mrz(text_lines: list[str]) -> MrzResult | None:
    """End-to-end: locate TD3 lines in OCR output and parse them."""
    found = find_mrz_lines(text_lines)
    if not found:
        return None
    return parse_td3(*found)
