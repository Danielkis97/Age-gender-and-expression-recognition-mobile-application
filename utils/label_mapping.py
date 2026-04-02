"""
Squash model output + CSV labels into fixed buckets for UI and evaluation.
Thresholds below are a bit arbitrary — tweak if your data looks off.
"""
from __future__ import annotations

ELDERLY_THRESHOLD = 50
GENDER_CONF_MIN = 0.55
EMOTION_CONF_MIN = 0.35


def age_to_group_label(age: int) -> str:
    if age < 0:
        return "unknown"
    return "Elderly" if age >= ELDERLY_THRESHOLD else "Adult"


def map_gender_male_female(
    raw: str | None, gender_scores: dict | None = None
) -> str:
    if gender_scores:
        try:
            if not gender_scores:
                return "—"
            k = max(gender_scores, key=gender_scores.get)
            raw = k
        except (TypeError, ValueError):
            raw = raw or ""
    s = str(raw or "").strip().lower()
    if not s:
        return "—"
    if s in ("woman", "female", "f"):
        return "Female"
    if s in ("man", "male", "m"):
        return "Male"
    if "female" in s:
        return "Female"
    if "male" in s:
        return "Male"
    return "—"


def map_emotion_happy_sad(
    raw: str | None, emotion_scores: dict | None = None
) -> str:
    if emotion_scores:
        try:
            if not emotion_scores:
                return "—"
            top = max(emotion_scores, key=emotion_scores.get)
            base = str(top).strip().lower()
        except (TypeError, ValueError):
            base = str(raw or "").strip().lower()
    else:
        base = str(raw or "").strip().lower()
    # Clean mapping rule: happy -> Happy, all others -> Sad
    return "Happy" if base == "happy" else "Sad"


def format_age_display(age: int) -> str:
    if age < 0:
        return "—"
    return str(age)


def format_prediction_display(pred: dict | None) -> tuple[str, str, str]:
    if pred is None:
        return ("—", "—", "—")
    age_s = format_age_display(int(pred.get("age", -1)))
    g = map_gender_male_female(pred.get("gender"), pred.get("gender_scores"))
    e = map_emotion_happy_sad(pred.get("emotion"), pred.get("emotion_scores"))
    return (age_s, g, e)


def canonical_gender_from_label(s: str | None) -> str:
    if not s or not str(s).strip():
        return ""
    s = str(s).strip().lower()
    if s in ("woman", "female", "f"):
        return "Female"
    if s in ("man", "male", "m"):
        return "Male"
    if "female" in s:
        return "Female"
    if "male" in s:
        return "Male"
    return ""


def canonical_age_group_from_label(s: str | None) -> str:
    if not s or not str(s).strip():
        return ""
    s = str(s).strip().lower()
    if s in ("elderly", "old", "senior"):
        return "Elderly"
    if s in ("adult", "young", "middle"):
        return "Adult"
    return ""


def canonical_emotion_from_label(s: str | None) -> str:
    if not s or not str(s).strip():
        return ""
    s = str(s).strip().lower().replace(" ", "")
    if s == "happy":
        return "Happy"
    return "Sad"
