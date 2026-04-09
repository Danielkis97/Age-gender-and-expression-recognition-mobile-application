"""
TFLite inference timing on local images.

This is a deployment/performance demonstration only (NOT used for accuracy metrics).
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeRemainingColumn

from utils.face_detect import detect_faces_bgr, load_face_cascade


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_RESULTS_CSV_ENCODING = "utf-8-sig"
_RESULTS_CSV_DELIMITER = ";"


def _collect_images(images_dir: Path, max_images: int) -> list[Path]:
    paths: list[Path] = []
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(p)
    return paths[:max_images]


def _prep_face_crop(frame_bgr, cascade) -> np.ndarray | None:
    faces = detect_faces_bgr(frame_bgr, cascade)
    if not faces:
        return None
    x, y, w, h = faces[0]
    y0 = max(0, y - int(h * 0.15))
    y1 = min(frame_bgr.shape[0], y + h + int(h * 0.15))
    x0 = max(0, x - int(w * 0.15))
    x1 = min(frame_bgr.shape[1], x + w + int(w * 0.15))
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return crop


def measure_tflite_performance(
    images_dir: Path,
    iterations: int,
    out_csv: Path = Path("results/tflite_performance.csv"),
    model_path: Path = Path("models/model.tflite"),
    console: Console | None = None,
    max_images: int = 20,
) -> dict:
    con = console or Console()

    if not images_dir.is_dir():
        con.print(f"[red]Not a directory:[/red] {images_dir}")
        return {"mean_s": 0.0, "std_s": 0.0, "n_images_used": 0}

    if not model_path.is_file():
        con.print(f"[yellow]Model not found:[/yellow] {model_path}")
        con.print("[dim]Exporting demo model...[/dim]")
        from tflite_export import export_tflite

        export_tflite(out_path=model_path)

    img_paths = _collect_images(images_dir, max_images=max_images)
    if not img_paths:
        con.print(f"[red]No images found in:[/red] {images_dir}")
        return {"mean_s": 0.0, "std_s": 0.0, "n_images_used": 0}

    # Load interpreter.
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]["index"]
    input_shape = input_details[0]["shape"]  # [1, h, w, c]
    input_dtype = input_details[0]["dtype"]
    _, in_h, in_w, in_c = input_shape

    cascade = load_face_cascade()

    con.print(Panel.fit(f"[bold]TFLite input:[/bold] {in_w}x{in_h}x{in_c}"))

    # Precompute face crops (so timing focuses on interpreter.invoke()).
    crops: list[np.ndarray] = []
    for p in img_paths:
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        crop = _prep_face_crop(frame, cascade)
        if crop is None:
            continue
        crops.append(crop)

    if not crops:
        con.print("[red]No faces found in provided images.[/red]")
        return {"mean_s": 0.0, "std_s": 0.0, "n_images_used": 0}

    # Warm-up (not timed).
    for _ in range(2):
        for crop in crops[: min(2, len(crops))]:
            img_resized = cv2.resize(crop, (in_w, in_h))
            img_resized = img_resized.astype(np.float32)
            if input_dtype != np.float32:
                img_resized = img_resized.astype(input_dtype)
            interpreter.set_tensor(input_index, img_resized[np.newaxis, ...])
            interpreter.invoke()

    samples: list[float] = []
    for _ in range(iterations):
        for crop in crops:
            img_resized = cv2.resize(crop, (in_w, in_h))
            img_resized = img_resized.astype(np.float32)
            if input_dtype != np.float32:
                img_resized = img_resized.astype(input_dtype)

            t0 = time.perf_counter()
            interpreter.set_tensor(input_index, img_resized[np.newaxis, ...])
            interpreter.invoke()
            _ = interpreter.get_tensor(output_details[0]["index"])
            samples.append(time.perf_counter() - t0)

    mean_s = statistics.mean(samples)
    std_s = statistics.stdev(samples) if len(samples) > 1 else 0.0
    total_s = sum(samples)
    min_s = min(samples)
    max_s = max(samples)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "device",
        "model_path",
        "time_per_image_s",
        "std_s",
        "min_time_s",
        "max_time_s",
        "total_runtime_s",
        "n_runs",
        "n_images_used",
        "notes",
    ]
    row = {
        "device": "TFLite",
        "model_path": str(model_path),
        "time_per_image_s": round(mean_s, 6),
        "std_s": round(std_s, 6),
        "min_time_s": round(min_s, 6),
        "max_time_s": round(max_s, 6),
        "total_runtime_s": round(total_s, 6),
        "n_runs": iterations,
        "n_images_used": len(crops),
        "notes": "demo model (mobile deployment compatibility)",
    }

    with out_csv.open("w", newline="", encoding=_RESULTS_CSV_ENCODING) as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=_RESULTS_CSV_DELIMITER)
        w.writeheader()
        w.writerow(row)

    con.print(f"[dim]Saved[/dim] [cyan]{out_csv.resolve()}[/cyan]")
    return {
        "mean_s": mean_s,
        "std_s": std_s,
        "total_s": total_s,
        "min_s": min_s,
        "max_s": max_s,
        "n_images_used": len(crops),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=Path("results/tflite_performance.csv"),
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("models/model.tflite"),
    )
    parser.add_argument("--max_images", type=int, default=20)
    args = parser.parse_args()

    console = Console()
    measure_tflite_performance(
        images_dir=args.images_dir,
        iterations=max(1, args.iterations),
        out_csv=args.out_csv,
        model_path=args.model_path,
        console=console,
        max_images=args.max_images,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

