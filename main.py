# Webcam and image demo with a Rich menu. All inference stays local (DeepFace on disk).
from __future__ import annotations

import argparse
import sys
import time

_WIN32 = sys.platform == "win32"
from collections import deque
from pathlib import Path

import cv2
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule

from utils.deepface_predict import clahe_face_crop_bgr, predict_face_region
from utils.drawing import draw_face_labels
from utils.face_detect import detect_faces_bgr, load_face_cascade
from utils.label_mapping import format_prediction_display

WINDOW_TITLE = "Age / Gender / Emotion (local)"
# DeepFace is heavy — don't call it every single frame.
PREDICT_EVERY_N_FRAMES = 10
FPS_SMOOTHING = 30
# Stagger labels when there are 2+ faces so text doesn't sit on top of itself.
LABEL_STACK_STEP = 58

_said_slow = False


def maybe_warn_first_model_load():
    global _said_slow
    if not _said_slow:
        print("First prediction can take a bit — models load/download on first use.")
        _said_slow = True


def crop_face_bgr(frame_bgr, x: int, y: int, w: int, h: int, pad: float = 0.15):
    # pad a little past the haar box; models usually like a bit of context
    fh, fw = frame_bgr.shape[:2]
    px = int(w * pad)
    py = int(h * pad)
    x0 = max(0, x - px)
    y0 = max(0, y - py)
    x1 = min(fw, x + w + px)
    y1 = min(fh, y + h + py)
    return frame_bgr[y0:y1, x0:x1]


def pick_image_path_tk() -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except tk.TclError:
            pass
        path = filedialog.askopenfilename(
            title="Select an image to analyze",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        return path or None
    except Exception:
        return None


def pick_image_interactive_rich(console: Console) -> str | None:
    console.print(
        Panel(
            "[bold]Image source[/bold]\n"
            "[1] File picker (tkinter)\n"
            "[2] Type path manually",
            title="Choose",
            border_style="cyan",
        )
    )
    mode = Prompt.ask("Option", choices=["1", "2"], default="1")
    if mode == "1":
        console.print("[dim]Opening file dialog…[/dim]")
        p = pick_image_path_tk()
        if p:
            return p
        console.print("[yellow]Dialog cancelled or failed — try manual path.[/yellow]")
    raw = Prompt.ask("Full path to image", default="").strip()
    return raw or None


def run_static_image_analysis(image_path: str, cascade) -> int:
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not read image: {image_path}")
        return 1

    maybe_warn_first_model_load()
    faces = detect_faces_bgr(frame, cascade)
    out = frame.copy()

    if not faces:
        cv2.putText(
            out,
            "No face detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        print("No face detected in image.")
    else:
        t0 = time.perf_counter()
        for fi, (x, y, w, h) in enumerate(faces):
            crop = crop_face_bgr(frame, x, y, w, h)
            pred = predict_face_region(crop, enforce_detection=False)
            age_s, g, e = format_prediction_display(pred)
            draw_face_labels(
                out, x, y, w, h, age_s, g, e, stack_offset=fi * LABEL_STACK_STEP
            )
            print(f"  Face @({x},{y}): Age={age_s}  Gender={g}  Emotion={e}")
        dt = time.perf_counter() - t0
        print(f"Inference (all faces): {dt:.3f} s")

    max_side = 960
    h, w = out.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        out = cv2.resize(out, (int(w * scale), int(h * scale)))

    cv2.imshow(WINDOW_TITLE, out)
    print("Window open — press q to close.")
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    return 0


def run_webcam_loop(cascade) -> int:
    # DirectShow on Windows avoids many "black window / no preview" issues with MSMF.
    if _WIN32:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_ANY)
    if not cap.isOpened():
        print("Could not open webcam (device 0). Close Zoom/Teams/other camera apps and retry.")
        return 1
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    frame_index = 0
    predictions_on = True
    face_cache: list[dict] = []
    times = deque(maxlen=FPS_SMOOTHING)

    maybe_warn_first_model_load()
    print("Webcam — q quit, p toggles predictions. If you see no window, check the taskbar behind PyCharm.")

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

    try:
        while True:
            t_loop = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed; exiting.")
                break

            faces = detect_faces_bgr(frame, cascade)

            if len(faces) != len(face_cache):
                face_cache = [
                    {
                        "rect": tuple(f),
                        "age": None,
                        "gender": None,
                        "emotion": None,
                        "gender_scores": None,
                        "emotion_scores": None,
                    }
                    for f in faces
                ]
            else:
                for i, f in enumerate(faces):
                    face_cache[i]["rect"] = tuple(f)

            run_models = (
                predictions_on and faces and (frame_index % PREDICT_EVERY_N_FRAMES == 0)
            )
            if run_models:
                for i, (x, y, w, h) in enumerate(faces):
                    crop = crop_face_bgr(frame, x, y, w, h)
                    crop = clahe_face_crop_bgr(crop)  # helps a lot in dim rooms
                    pred = predict_face_region(crop, enforce_detection=False)
                    if pred:
                        face_cache[i]["age"] = pred["age"]
                        face_cache[i]["gender"] = pred["gender"]
                        face_cache[i]["emotion"] = pred["emotion"]
                        face_cache[i]["gender_scores"] = pred.get("gender_scores")
                        face_cache[i]["emotion_scores"] = pred.get("emotion_scores")

            display = frame.copy()
            for fi, entry in enumerate(face_cache):
                x, y, w, h = entry["rect"]
                if not predictions_on:
                    age_s, g, e = "(off)", "(off)", "(off)"
                elif (
                    entry["age"] is None
                    and not entry.get("gender")
                    and not entry.get("emotion")
                ):
                    age_s, g, e = "—", "—", "—"
                else:
                    pred_dict = {
                        "age": entry["age"] if entry["age"] is not None else -1,
                        "gender": entry.get("gender") or "",
                        "emotion": entry.get("emotion") or "",
                        "gender_scores": entry.get("gender_scores"),
                        "emotion_scores": entry.get("emotion_scores"),
                    }
                    age_s, g, e = format_prediction_display(pred_dict)
                draw_face_labels(
                    display, x, y, w, h, age_s, g, e, stack_offset=fi * LABEL_STACK_STEP
                )

            if not faces:
                cv2.putText(
                    display,
                    "No face detected",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )

            times.append(time.perf_counter() - t_loop)
            fps = len(times) / sum(times) if times else 0.0
            cv2.putText(
                display,
                f"FPS: {fps:.1f}  |  predictions: {'on' if predictions_on else 'off'}",
                (10, display.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_TITLE, display)
            frame_index += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                predictions_on = not predictions_on
                print("Predictions", "on" if predictions_on else "off")
    finally:
        cap.release()
        try:
            cv2.destroyWindow(WINDOW_TITLE)
        except cv2.error:
            pass
        cv2.destroyAllWindows()
    return 0


def _evaluation_path_defaults(project_root: Path) -> tuple[Path, Path]:
    """Prefer dataset/images + dataset/labels.csv; else data/."""
    dataset_images = project_root / "dataset" / "images"
    data_dir = project_root / "data"
    if dataset_images.is_dir():
        images_dir = dataset_images
    elif data_dir.is_dir():
        images_dir = data_dir
    else:
        images_dir = dataset_images

    for candidate in (
        project_root / "dataset" / "labels.csv",
        data_dir / "labels.csv",
        data_dir / "labels_example.csv",
    ):
        if candidate.is_file():
            return images_dir, candidate
    return images_dir, project_root / "dataset" / "labels.csv"


def _log_path_check(console: Console, label: str, path: Path, must_be: str) -> None:
    """must_be: 'dir' | 'file'."""
    if must_be == "dir":
        ok = path.is_dir()
        hint = "folder with image files"
    else:
        ok = path.is_file()
        hint = "file"
    if ok:
        console.print(f"  [green]OK[/green] {label}: [cyan]{path}[/cyan]")
    else:
        console.print(
            f"  [yellow]Warning[/yellow] {label} not found as {hint}: [cyan]{path}[/cyan] "
            "(evaluation will fail until this path is valid)."
        )


def load_cascade_safe(console: Console):
    try:
        with console.status("[bold yellow]Loading face cascade…[/bold yellow]"):
            return load_face_cascade()
    except RuntimeError as e:
        console.print(f"[red]Cascade failed:[/red] {e}")
        return None


def run_rich_menu_loop(console: Console) -> int:
    root = Path(__file__).resolve().parent

    while True:
        console.print()
        console.print(Rule("[bold cyan]Local edge · Age / Gender / Emotion[/bold cyan]"))
        console.print(
            Panel.fit(
                "[bold]1.[/bold]  Webcam (real-time)\n"
                "[bold]2.[/bold]  Single image prediction\n"
                "[bold]3.[/bold]  Evaluation (folder + labels CSV, up to 20 imgs)\n"
                "[bold]4.[/bold]  Run Performance Benchmark\n"
                "[bold]5.[/bold]  Exit",
                title="[bold cyan]Main menu[/bold cyan]",
                border_style="cyan",
            )
        )
        # No default: Enter must not silently start webcam (was default="1").
        choice = Prompt.ask(
            "Select [1-5]",
            choices=["1", "2", "3", "4", "5"],
        )

        if choice == "5":
            console.print(Panel("[green]Goodbye.[/green]", border_style="green"))
            return 0

        if choice == "1":
            cascade = load_cascade_safe(console)
            if cascade is None:
                continue
            console.print("[green]Starting webcam — q to quit, p toggles predictions.[/green]")
            run_webcam_loop(cascade)
            console.print("[dim]Webcam closed.[/dim]")
            continue

        if choice == "2":
            cascade = load_cascade_safe(console)
            if cascade is None:
                continue
            path = pick_image_interactive_rich(console)
            if not path:
                console.print("[yellow]No image selected.[/yellow]")
                continue
            console.print("[dim]Running predictions (OpenCV window opens when ready)…[/dim]")
            rc = run_static_image_analysis(path, cascade)
            if rc == 0:
                console.print("[green]Done.[/green]")
            else:
                console.print("[red]Image run failed.[/red]")
            continue

        if choice == "3":
            # import here so starting the app doesn't pull sklearn etc. unless needed
            from evaluate import run_evaluation

            img_def, lab_def = _evaluation_path_defaults(root)
            results_dir = root / "results"
            out_eval = results_dir / "evaluation_results.csv"
            out_pred = results_dir / "predictions.csv"

            console.print()
            console.print(
                Panel.fit(
                    "[bold]You only choose two things:[/bold]\n\n"
                    "[bold]1[/bold]  [cyan]Images folder[/cyan] — where the photos are (e.g. dataset\\images).\n"
                    "[bold]2[/bold]  [cyan]Labels CSV[/cyan] — your ground-truth file ([dim]input only[/dim]; "
                    "it is read, not changed).\n\n"
                    "[bold]Results[/bold] land automatically in [cyan]results\\[/cyan]: "
                    "[bold]evaluation_results.csv[/bold] (full table), [bold]metrics.csv[/bold] (Acc/Prec/Rec/F1), "
                    "[bold]confusion_*.csv[/bold], and [bold]predictions.csv[/bold] (same run, compact text per row). "
                    "[italic]labels.csv[/italic] is only read (ground truth); it is not overwritten.",
                    title="[bold]Evaluation[/bold]",
                    border_style="blue",
                )
            )

            console.print("[bold]Step 1/2[/bold] — Images folder")
            d = Prompt.ask("Path", default=str(img_def)).strip()
            _log_path_check(console, "Images folder", Path(d), "dir")

            console.print("[bold]Step 2/2[/bold] — Labels CSV file")
            lab = Prompt.ask("Path", default=str(lab_def)).strip()
            _log_path_check(console, "Labels CSV", Path(lab), "file")

            console.print(
                f"[dim]Writing outputs under[/dim] [cyan]{results_dir}[/cyan] "
                f"[dim](created if missing).[/dim]"
            )
            console.print("[dim]Running evaluation…[/dim]")
            run_evaluation(
                Path(d),
                Path(lab),
                out_csv=out_eval,
                predictions_csv=out_pred,
                confusion_gender=root / "results" / "confusion_gender.csv",
                confusion_age=root / "results" / "confusion_age.csv",
                confusion_emotion=root / "results" / "confusion_emotion.csv",
                max_images=20,
                console=console,
                use_rich=True,
            )
            continue

        if choice == "4":
            from performance import run_benchmark  # pulls tf path; only on demand

            default_images = root / "dataset" / "images"
            d = Prompt.ask(
                "Images folder for benchmarking",
                default=str(default_images),
            ).strip()

            if not d:
                d = str(default_images)

            iters_s = Prompt.ask("Iterations (1-5)", default="3").strip() or "3"
            try:
                n_iters = max(1, min(5, int(iters_s)))
            except ValueError:
                n_iters = 3

            run_benchmark(
                Path(d),
                iterations=n_iters,
                out_csv=root / "results" / "performance.csv",
                console=console,
            )
            continue


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--webcam", action="store_true")
    parser.add_argument("--image", type=str, default=None, metavar="PATH")
    args = parser.parse_args(argv)

    console = Console()

    if args.image:
        try:
            with console.status("[bold yellow]Loading cascade…[/bold yellow]"):
                cascade = load_face_cascade()
        except RuntimeError as e:
            console.print(f"[red]{e}[/red]")
            return 1
        return run_static_image_analysis(args.image, cascade)

    if args.webcam:
        try:
            with console.status("[bold yellow]Loading cascade…[/bold yellow]"):
                cascade = load_face_cascade()
        except RuntimeError as e:
            console.print(f"[red]{e}[/red]")
            return 1
        return run_webcam_loop(cascade)

    return run_rich_menu_loop(console)


if __name__ == "__main__":
    sys.exit(main())
