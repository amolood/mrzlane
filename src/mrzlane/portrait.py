"""Portrait extraction: locate the holder's photo and return it as base64 JPEG.

On a passport data page the portrait sits on the left. We detect the face,
expand the box to a portrait crop (head + shoulders), and encode it. Fails soft
to ``None`` when no face is found, so the pipeline never breaks on this step.

Uses OpenCV's bundled YuNet DNN detector (Apache-2.0, ~230 KB) — far more
reliable on faint/ID-sized passport photos than the old Haar cascade.
"""
from __future__ import annotations

import base64
import os

import cv2
import numpy as np

# YuNet face detector (Apache-2.0, ~230 KB). Downloaded once to a user cache on
# first use, mirroring how PaddleOCR fetches its models — the repository ships no
# model weights.
_MODEL_NAME = "face_detection_yunet_2023mar.onnx"
_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/" + _MODEL_NAME
)
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "mrzlane")
_DETECTOR = None


def _ensure_model() -> str:
    # Allow an explicit override, else use the per-user cache.
    override = os.environ.get("MRZLANE_YUNET_PATH")
    if override and os.path.exists(override):
        return override
    path = os.path.join(_CACHE_DIR, _MODEL_NAME)
    if not os.path.exists(path):
        import urllib.request
        os.makedirs(_CACHE_DIR, exist_ok=True)
        tmp = path + ".part"
        urllib.request.urlretrieve(_MODEL_URL, tmp)
        os.replace(tmp, path)
    return path


def _detector() -> "cv2.FaceDetectorYN":
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = cv2.FaceDetectorYN.create(
            _ensure_model(), "", (320, 320),
            score_threshold=0.6, nms_threshold=0.3, top_k=50,
        )
    return _DETECTOR


def detect_face(img: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the best face box (x, y, w, h) or None.

    Passports put the portrait on the left half, so we bias toward faces there
    to avoid picking up a tiny ghost/hologram face on the right.
    """
    H, W = img.shape[:2]
    det = _detector()
    det.setInputSize((W, H))
    _, faces = det.detect(img)
    if faces is None or len(faces) == 0:
        return None

    # A real passport portrait occupies a meaningful fraction of the page; tiny
    # high-confidence boxes are watermark/pattern false positives — drop them.
    min_side = 0.08 * min(W, H)
    real = [f for f in faces if f[2] >= min_side and f[3] >= min_side]
    if not real:
        return None

    def score(f):
        x, y, w, h = f[:4]
        cx = x + w / 2
        left_bias = 1.0 if cx < W * 0.55 else 0.4   # prefer left-side portrait
        conf = f[14]
        return w * h * left_bias * conf

    best = max(real, key=score)
    x, y, w, h = (int(max(0, v)) for v in best[:4])
    return x, y, w, h


_ROTATIONS = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def detect_face_any_orientation(img: np.ndarray):
    """Try all 4 orientations; return (upright_image, box) for the best face.

    Passport scans are often rotated, and the face detector only finds upright
    faces — so we rotate the page until a face appears. This doubles as
    orientation correction for the whole crop.
    """
    best = None  # (area, angle, box, image)
    for angle, rot in _ROTATIONS.items():
        cand = img if rot is None else cv2.rotate(img, rot)
        box = detect_face(cand)
        if box is None:
            continue
        x, y, w, h = box
        area = w * h
        if best is None or area > best[0]:
            best = (area, angle, box, cand)
    if best is None:
        return None, None
    _, _, box, upright = best
    return upright, box


def crop_portrait(img: np.ndarray, pad_top=0.45, pad_side=0.25, pad_bottom=0.55
                  ) -> np.ndarray | None:
    """Detect the face (in any orientation) and expand to a portrait crop."""
    img, box = detect_face_any_orientation(img)
    if box is None:
        return None
    x, y, w, h = box
    H, W = img.shape[:2]
    x1 = max(0, int(x - pad_side * w))
    y1 = max(0, int(y - pad_top * h))
    x2 = min(W, int(x + w + pad_side * w))
    y2 = min(H, int(y + h + pad_bottom * h))
    crop = img[y1:y2, x1:x2]
    return crop if crop.size else None


def portrait_base64(img: np.ndarray, quality: int = 85, as_data_uri: bool = True
                    ) -> str | None:
    """Full step: crop the portrait and return it as (data-URI) base64 JPEG."""
    crop = crop_portrait(img)
    if crop is None:
        return None
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}" if as_data_uri else b64
