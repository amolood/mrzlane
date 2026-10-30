"""Visual-zone OCR: read the printed fields (Arabic + English) with positions.

The MRZ gives us the authoritative identity fields. This layer reads the rest of
the page — National No., place of birth/issue, date of issue, and the Arabic
full name — and provides a second, independent reading of MRZ fields so the
fusion step can *cross-check* (agreement => high confidence; conflict => review).

PaddleOCR is loaded once per language and cached. We keep every token with its
box and score so field binding can use geometry (label-to-value proximity).
"""
from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

_ENGINES: dict[str, object] = {}

# Lightweight mobile models: ~2x faster than the medium defaults with no loss on
# the clear printed text of a passport data page. Orientation/unwarp models are
# disabled — we handle rotation in preprocess/portrait, so they only cost time.
_MODELS = {
    "en": ("PP-OCRv5_mobile_det", "en_PP-OCRv5_mobile_rec"),
    "ar": ("PP-OCRv5_mobile_det", "arabic_PP-OCRv5_mobile_rec"),
}


def _engine(lang: str):
    if lang not in _ENGINES:
        from paddleocr import PaddleOCR
        det, rec = _MODELS.get(lang, _MODELS["en"])
        _ENGINES[lang] = PaddleOCR(
            use_textline_orientation=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            text_detection_model_name=det,
            text_recognition_model_name=rec,
        )
    return _ENGINES[lang]


@dataclass
class Token:
    text: str
    box: tuple          # (x1, y1, x2, y2)
    score: float
    lang: str

    @property
    def cx(self) -> float:
        return (self.box[0] + self.box[2]) / 2

    @property
    def cy(self) -> float:
        return (self.box[1] + self.box[3]) / 2


def _poly_to_box(poly) -> tuple:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))


def read_tokens(img: np.ndarray, langs=("en", "ar")) -> list[Token]:
    """Run OCR for each language and return all tokens with geometry."""
    tokens: list[Token] = []
    for lang in langs:
        for page in _engine(lang).predict(img):
            texts = page.get("rec_texts") or []
            scores = page.get("rec_scores") or []
            polys = page.get("rec_polys") or page.get("dt_polys") or []
            for t, s, poly in zip(texts, scores, polys):
                if not t or not str(t).strip():
                    continue
                tokens.append(Token(str(t).strip(), _poly_to_box(poly), float(s), lang))
    return tokens


# ── Field extractors (pattern + geometry) ─────────────────────────────────

_NAT_ID_RE = re.compile(r"\b(\d{3})[\s\-]?(\d{4})[\s\-]?(\d{4})\b")       # e.g. 123-4567-8901
_PASSPORT_RE = re.compile(r"\b([A-Z]\d{7,8})\b")                           # e.g. A1234567
_DATE_RE = re.compile(r"\b(\d{1,2})[\-/\. ](\d{1,2})[\-/\. ](\d{2,4})\b")


def _norm_date(m: re.Match) -> str | None:
    d, mo, y = int(m.group(1)), int(m.group(2)), m.group(3)
    year = int(y) if len(y) == 4 else (2000 + int(y) if int(y) < 50 else 1900 + int(y))
    import datetime
    try:
        datetime.date(year, mo, d)
    except ValueError:
        return None
    return f"{year:04d}-{mo:02d}-{d:02d}"


# Arabic place names seen on Sudanese passports -> canonical English.
_PLACES = {
    "امدرمان": "OMDURMAN", "أمدرمان": "OMDURMAN", "ام درمان": "OMDURMAN",
    "الخرطوم": "KHARTOUM", "بحري": "BAHRI", "الخرطوم بحري": "KHARTOUM BAHRI",
    "بورتسودان": "PORT SUDAN", "بورت سودان": "PORT SUDAN", "كسلا": "KASSALA",
    "القضارف": "GEDAREF", "القضارف": "GEDAREF", "ودمدني": "WAD MADANI",
    "الأبيض": "EL OBEID", "نيالا": "NYALA", "الفاشر": "EL FASHER",
    "سنار": "SENNAR", "عطبرة": "ATBARA", "دنقلا": "DONGOLA",
}


def extract_visual_fields(tokens: list[Token]) -> dict:
    """Pull printed fields that complement or cross-check the MRZ."""
    full = " ".join(t.text for t in tokens)
    out: dict = {}

    m = _NAT_ID_RE.search(full)
    if m:
        out["national_number"] = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    m = _PASSPORT_RE.search(full.replace(" ", ""))
    if m:
        out["passport_number_print"] = m.group(1)

    # Dates: collect all, let fusion map them against MRZ (dob/expiry) +
    # the remaining one as issue date.
    dates = []
    for t in tokens:
        for m in _DATE_RE.finditer(t.text):
            d = _norm_date(m)
            if d:
                dates.append(d)
    if dates:
        out["dates_print"] = sorted(set(dates))

    # Arabic place names (birth/issue).
    places = []
    for t in tokens:
        if t.lang != "ar":
            continue
        key = re.sub(r"\s+", " ", t.text.strip())
        for ar, en in _PLACES.items():
            if ar in key:
                places.append(en)
    if places:
        out["places_print"] = list(dict.fromkeys(places))

    # Arabic full name: longest Arabic token that looks like a name (3+ words,
    # letters only), excluding header lines.
    HEADERS = ("جمهورية", "السودان", "جواز", "وزارة", "الداخلية", "الجنسية",
               "الاسم", "تاريخ", "مكان", "الرقم", "الوطني", "النوع")
    name_cands = []
    for t in tokens:
        if t.lang != "ar":
            continue
        words = t.text.split()
        letters_only = all(re.fullmatch(r"[؀-ۿ]+", w) for w in words) if words else False
        if len(words) >= 3 and letters_only and not any(h in t.text for h in HEADERS):
            name_cands.append((t.score, t.text))
    if name_cands:
        out["full_name_arabic"] = max(name_cands)[1]

    return out
