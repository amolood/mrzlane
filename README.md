# mrzlane

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Status](https://img.shields.io/badge/status-beta-orange.svg)](#)
[![Standard](https://img.shields.io/badge/MRZ-ICAO%209303-informational.svg)](https://www.icao.int/publications/pages/publication.aspx?docnum=9303)
[![Offline](https://img.shields.io/badge/runtime-offline%20CPU-success.svg)](#)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#contributing)

MRZ-first OCR for passports and ID documents. It reads the Machine Readable
Zone, validates it against its own check digits, and cross-checks the printed
fields — so every value it returns as confirmed is backed by at least one
verifiable source. Anything it cannot confirm is flagged for review instead of
being guessed.

Built on open components (PaddleOCR, OpenCV). No proprietary engines, no network
calls, no telemetry. Runs fully offline on CPU.

## Why MRZ-first

A passport's MRZ is printed in a fixed OCR-B layout and carries ICAO 9303 check
digits. That makes it the most reliable region on the page: a reading either
satisfies its check digits or it does not. mrzlane reads the MRZ first and treats
validated fields as authoritative, then uses the printed (visual) zone to fill in
fields the MRZ does not contain and to cross-check the ones it does. When two
independent sources agree, confidence is high; when they conflict, the field is
sent to review.

## Design goals

- **No wrong confident output.** A field is returned as confirmed only when it is
  check-digit valid, cross-checked, and logically consistent. Everything else is
  marked `needs_review`.
- **Robust to real-world scans.** Orientation is resolved using the MRZ as the
  compass — the engine tries rotations until the check digits pass — so sideways
  and upside-down photographs are read correctly.
- **Drop-in friendly.** A compatibility layer emits the response shape used by
  common commercial `verify_document` services, so mrzlane can replace one with
  no changes to the caller.

## Pipeline

```
image
  -> preprocess        document detection + perspective correction + enhancement
  -> visual OCR        PaddleOCR (English + Arabic), tokens with positions
  -> MRZ               TD3 parse + ICAO 9303 check-digit validation
  -> orientation       MRZ-driven rotation when the first read does not validate
  -> fuse              cross-validate MRZ vs print + logical sanity checks
  -> portrait          face detection (YuNet) + head-and-shoulders crop
  -> output            clean JSON with per-field confidence and review flags
```

## Install

```bash
pip install -r requirements.txt
# or, as a package:
pip install .
```

Models are downloaded automatically on first run and cached locally.

## Usage

### Library

```python
from mrzlane.pipeline import process

result = process("passport.jpg", with_portrait=True)

if result["needs_review"]:
    print("Review:", result["review_fields"])
else:
    print(result["fields"]["passport_number"]["value"])
```

### HTTP service

```bash
PYTHONPATH=src uvicorn server.app:app --host 0.0.0.0 --port 8080
```

```
POST /api/v1/passport        multipart 'image'  -> clean v1 JSON
POST /api/verify_document    multipart 'image'  -> legacy verify_document JSON
GET  /health
```

Set `MRZLANE_TOKEN` to require an `X-API-Token` header.

## Output

Each field carries its value, a confidence score, the source it came from
(`mrz`, `print`, or `mrz+print`), and a review flag. Values below are synthetic.

```json
{
  "schema_version": "1.0",
  "document_type": "passport",
  "country": "SDN",
  "needs_review": false,
  "fields": {
    "passport_number":  { "value": "A1234567", "confidence": 0.98, "source": "mrz" },
    "full_name_latin":  { "value": "GIVEN MIDDLE SURNAME", "confidence": 0.95, "source": "mrz" },
    "date_of_birth":    { "value": "1990-01-01", "confidence": 0.99, "source": "mrz+print" },
    "date_of_expiry":   { "value": "2030-01-01", "confidence": 0.99, "source": "mrz+print" },
    "sex":              { "value": "M", "confidence": 0.97, "source": "mrz" },
    "nationality":      { "value": "SDN", "confidence": 0.95, "source": "mrz" },
    "national_number":  { "value": "123-4567-8901", "confidence": 0.80, "source": "print" }
  },
  "mrz": { "line1": "...", "line2": "...", "valid": true, "check_digits": { } },
  "portrait": "data:image/jpeg;base64,...",
  "engine": { "name": "mrzlane", "version": "0.1.0" }
}
```

A field with `needs_review: true` should be verified by a human before use.

## Fields

| Field | Primary source | Notes |
| --- | --- | --- |
| `passport_number` | MRZ | cross-checked against the printed number |
| `full_name_latin` | MRZ | reconstructed from surname / given names |
| `full_name_arabic` | print | right-to-left script, review by default |
| `date_of_birth` | MRZ | century pivot forces a past date |
| `date_of_expiry` | MRZ | century pivot forces a future date |
| `sex` | MRZ | |
| `nationality` | MRZ | 3-letter code, de-confused from OCR lookalikes |
| `issuing_country` | MRZ | |
| `national_number` | print | format-validated |
| `place_of_birth` / `place_of_issue` | print | |

## Document support

The current release targets TD3 passports. The architecture generalises to other
document types (national ID cards, driving licences) through the same
MRZ/visual/fusion layering; those are planned.

## Privacy

mrzlane processes images in-memory and returns results to the caller. It performs
no network requests beyond the one-time model download and stores nothing. It is
the integrator's responsibility to handle extracted personal data in line with
applicable law. The repository ships no document images or extracted data.

## Testing

```bash
python -m pytest tests/
```

Tests use the public ICAO 9303 specimen data only.

## Contributing

Contributions are welcome — bug reports, new document types, accuracy
improvements, and additional language support. Please:

- Open an issue to discuss substantial changes before a pull request.
- Keep tests passing (`python -m pytest tests/`) and add tests for new behaviour.
- Never include real document images or personal data in issues, tests, or
  commits. Use synthetic or ICAO specimen data only.

## License

MIT. See [LICENSE](LICENSE).
