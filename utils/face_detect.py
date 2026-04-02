from __future__ import annotations

import cv2


def load_face_cascade() -> cv2.CascadeClassifier:
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade: {path}")
    return cascade


def detect_faces_bgr(
    frame_bgr,
    cascade: cv2.CascadeClassifier,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size=(48, 48),
):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # cheap; helps uneven lighting
    rects = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )
    return [tuple(int(v) for v in r) for r in rects]


def pick_largest_face(rects: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    if not rects:
        return None
    return max(rects, key=lambda r: r[2] * r[3])


def detect_faces_bgr_relaxed(
    frame_bgr,
    cascade: cv2.CascadeClassifier,
) -> list[tuple[int, int, int, int]]:
    """Second pass: smaller min face + fewer neighbors — reduces misses on some portraits."""
    return detect_faces_bgr(
        frame_bgr,
        cascade,
        scale_factor=1.05,
        min_neighbors=3,
        min_size=(32, 32),
    )
