"""Image preprocessing: load, orient, detect the document, and enhance.

Goal of this layer is to hand the OCR stages a clean, upright, tightly-cropped
page. It fails soft: if document detection is not confident, it returns the
(oriented) full image rather than a bad crop.
"""
from __future__ import annotations

import cv2
import numpy as np


# Upper bound on decoded pixels (~48 MP). A decoded image beyond this is
# downscaled to fit before any processing, so a crafted tiny-but-huge-resolution
# file cannot exhaust memory downstream.
MAX_PIXELS = 48_000_000
MAX_ASPECT_RATIO = 10.0


def _clamp_decoded(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if min(h, w) == 0:
        raise ValueError("Degenerate image dimensions.")
    if max(h, w) / min(h, w) > MAX_ASPECT_RATIO:
        raise ValueError("Extreme aspect ratio.")
    if h * w > MAX_PIXELS:
        scale = (MAX_PIXELS / (h * w)) ** 0.5
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    return img


def load_bgr(path_or_bytes) -> np.ndarray:
    """Load an image (path or raw bytes) as a BGR numpy array, size-clamped."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        arr = np.frombuffer(path_or_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(path_or_bytes), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image.")
    return _clamp_decoded(img)


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]       # top-left
    rect[2] = pts[np.argmax(s)]       # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]    # top-right
    rect[3] = pts[np.argmax(diff)]    # bottom-left
    return rect


def four_point_transform(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    tl, tr, br, bl = rect
    maxW = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    maxH = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
    if maxW < 50 or maxH < 50:
        return img
    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (maxW, maxH))


def detect_document(img: np.ndarray) -> np.ndarray:
    """Find the largest 4-corner quad and deskew it. Falls back to the input."""
    h, w = img.shape[:2]
    scale = 1000.0 / max(h, w) if max(h, w) > 1000 else 1.0
    small = cv2.resize(img, (int(w * scale), int(h * scale))) if scale != 1.0 else img

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 50, 150)
    edged = cv2.dilate(edged, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    page_area = small.shape[0] * small.shape[1]

    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.2 * page_area:        # ignore small quads
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype("float32") / scale
            return four_point_transform(img, quad)
    return img


def auto_orient(img: np.ndarray) -> np.ndarray:
    """Make the image landscape (passports are wider than tall)."""
    h, w = img.shape[:2]
    if h > w:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def enhance_for_ocr(img: np.ndarray) -> np.ndarray:
    """Upscale small images and boost contrast; returns BGR for PaddleOCR."""
    h, w = img.shape[:2]
    if w < 1200:
        # Bound the upscale so the result stays within the pixel budget — a very
        # tall, thin image must not be amplified without limit.
        scale = min(1200.0 / w, (MAX_PIXELS / max(1, h * w)) ** 0.5)
        if scale > 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def orient_by_face(img: np.ndarray) -> np.ndarray:
    """Rotate the page so the holder's face is upright.

    A face is only detected reliably when upright, so it is a dependable compass
    for page orientation — far more robust than the width>height heuristic, which
    fails on the many passport scans photographed sideways (90°/270°). Falls back
    to the input when no face is found in any orientation.
    """
    # Imported here to avoid a circular import at module load.
    from .portrait import detect_face_any_orientation
    upright, box = detect_face_any_orientation(img)
    return upright if box is not None else img


def prepare(path_or_bytes) -> dict:
    """Full preprocess. Returns the cropped page and an enhanced OCR image.

    Orientation is NOT forced here — the pipeline resolves it with the MRZ (the
    only reliable compass) and may re-crop the portrait from the corrected page.
    """
    original = load_bgr(path_or_bytes)
    doc = detect_document(original)
    return {"original": original, "document": doc, "ocr_image": enhance_for_ocr(doc)}
