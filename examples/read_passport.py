"""Minimal usage example.

    python examples/read_passport.py path/to/passport.jpg

Prints the clean v1 result (portrait base64 is truncated for readability).
"""
import json
import sys

sys.path.insert(0, "src")

from mrzlane.pipeline import process


def main():
    if len(sys.argv) < 2:
        print("usage: python examples/read_passport.py <image>")
        raise SystemExit(1)

    result = process(sys.argv[1], with_portrait=True)

    # Truncate the portrait so the console stays readable.
    if result.get("portrait"):
        result["portrait"] = result["portrait"][:48] + "...(truncated)"

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if result["needs_review"]:
        print("\nReview required for:", ", ".join(result["review_fields"]))
    else:
        print("\nAll critical fields confirmed.")


if __name__ == "__main__":
    main()
