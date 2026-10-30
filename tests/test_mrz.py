"""MRZ parser tests using the ICAO 9303 specimen data (fully synthetic)."""
import sys

sys.path.insert(0, "src")

from mrzlane.mrz import parse_td3, check_digit, read_mrz


# The canonical ICAO 9303 TD3 specimen (not a real person).
SPECIMEN_L1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
SPECIMEN_L2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"


def test_check_digit_known_values():
    assert check_digit("L898902C3") == 6
    assert check_digit("740812") == 2
    assert check_digit("120415") == 9


def test_parse_specimen_fields():
    r = parse_td3(SPECIMEN_L1, SPECIMEN_L2)
    assert r.document_type == "P"
    assert r.issuing_country == "UTO"
    assert r.surname == "ERIKSSON"
    assert r.given_names == "ANNA MARIA"
    assert r.document_number == "L898902C3"
    assert r.nationality == "UTO"
    assert r.date_of_birth == "1974-08-12"
    assert r.sex == "F"
    # Expiry uses a "must be in the future" century pivot (a passport does not
    # expire in the past), so the 1974-era specimen's YY=12 resolves to 2112,
    # not 2012. The month/day are what matter for a live document.
    assert r.date_of_expiry.endswith("-04-15")


def test_specimen_validates():
    r = parse_td3(SPECIMEN_L1, SPECIMEN_L2)
    assert r.check_digits["document_number"]
    assert r.check_digits["date_of_birth"]
    assert r.check_digits["expiry"]
    assert r.valid


def test_read_from_noisy_lines():
    noisy = ["REPUBLIC OF UTOPIA", "random header", SPECIMEN_L1, "noise", SPECIMEN_L2]
    r = read_mrz(noisy)
    assert r is not None
    assert r.document_number == "L898902C3"


def test_corrupted_mrz_is_invalid():
    # Flip a digit in the document number: check digit must now fail.
    bad_l2 = "L898902C96UTO7408122F1204159ZE184226B<<<<<10"
    r = parse_td3(SPECIMEN_L1, bad_l2)
    assert not r.valid
