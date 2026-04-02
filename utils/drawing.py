from __future__ import annotations

import cv2


def draw_face_labels(
    frame_bgr,
    x: int,
    y: int,
    w: int,
    h: int,
    age_text: str,
    gender_text: str,
    emotion_text: str,
    font_scale: float = 0.55,
    stack_offset: int = 0,
) -> None:
    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 200, 0), 2)
    lines = [
        f"Age: {age_text}",
        f"Gender: {gender_text}",
        f"Emotion: {emotion_text}",
    ]
    line_h = int(22 * font_scale * 2)
    # stack_offset bumps the block up when drawing multiple faces
    ty = max(y - 8 - line_h * len(lines) - stack_offset, 4)
    for i, line in enumerate(lines):
        y0 = ty + i * line_h
        cv2.putText(
            frame_bgr,
            line,
            (x, y0 + line_h - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
