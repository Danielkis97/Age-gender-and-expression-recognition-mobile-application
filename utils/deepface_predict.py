from __future__ import annotations

import os
import sys
from typing import Any

import cv2

_said_deepface_error = False


def clahe_face_crop_bgr(bgr_face):
    # webcam-only trick; keeps chroma, just evens out luminance
    if bgr_face is None or bgr_face.size == 0:
        return bgr_face
    h, w = bgr_face.shape[:2]
    if h < 8 or w < 8:
        return bgr_face
    lab = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l2 = clahe.apply(l_ch)
    merged = cv2.merge([l2, a_ch, b_ch])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def predict_face_region(bgr_face, enforce_detection: bool = False) -> dict[str, Any] | None:
    global _said_deepface_error

    # DeepFace currently relies on legacy Keras APIs on many setups (notably Colab).
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

    try:
        from deepface import DeepFace
    except ImportError as e:
        if not _said_deepface_error:
            print(
                f"[deepface] import failed once: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            _said_deepface_error = True
        return None

    # First attempt: regular detector path.
    try:
        result = DeepFace.analyze(
            img_path=bgr_face,
            actions=("age", "gender", "emotion"),
            enforce_detection=enforce_detection,
            silent=True,
        )
    except Exception as first_error:
        # Colab can fail in detector backends even when model inference itself would work.
        # Fallback to "skip" backend: no detection stage, analyze full input directly.
        try:
            result = DeepFace.analyze(
                img_path=bgr_face,
                actions=("age", "gender", "emotion"),
                enforce_detection=False,
                detector_backend="skip",
                silent=True,
            )
        except Exception as second_error:
            if not _said_deepface_error:
                print(
                    "[deepface] analyze failed once: "
                    f"{type(first_error).__name__}: {first_error} "
                    f"| fallback(skip) {type(second_error).__name__}: {second_error}",
                    file=sys.stderr,
                )
                _said_deepface_error = True
            # bad crop / backend issue / model issue — caller records missing prediction.
            return None

    if isinstance(result, list):
        if not result:
            return None
        result = result[0]

    age = result.get("age")
    gender_raw = result.get("gender")
    emotion_dict = result.get("emotion") or {}

    gender_scores = None
    if isinstance(gender_raw, dict) and gender_raw:
        gender_scores = {str(k): float(v) for k, v in gender_raw.items()}
        gender = max(gender_raw, key=gender_raw.get)
    else:
        gender = str(gender_raw) if gender_raw is not None else "Unknown"

    if emotion_dict:
        emotion = max(emotion_dict, key=emotion_dict.get)
    else:
        emotion = "Unknown"

    emotion_scores = None
    if emotion_dict:
        emotion_scores = {str(k): float(v) for k, v in emotion_dict.items()}

    try:
        age_int = int(round(float(age))) if age is not None else -1
    except (TypeError, ValueError):
        age_int = -1

    return {
        "age": age_int,
        "gender": str(gender),
        "emotion": str(emotion),
        "gender_scores": gender_scores,
        "emotion_scores": emotion_scores,
    }


def age_to_group(age: int, elderly_threshold: int | None = None) -> str:
    from utils.label_mapping import ELDERLY_THRESHOLD

    t = ELDERLY_THRESHOLD if elderly_threshold is None else elderly_threshold
    if age < 0:
        return "unknown"
    return "Elderly" if age >= t else "Adult"
