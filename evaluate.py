"""
Batch evaluation + confusion matrices. This script does not send data externally.
"""
from __future__ import annotations

import os

# Before TensorFlow loads (first DeepFace call): less C++ log spam on stderr so Rich progress bar doesn't look "stuck at 0%".
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# Ensure local imports (utils.*) work even when launched outside project root (e.g. Colab cells).
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.deepface_predict import clahe_face_crop_bgr, predict_face_region
from utils.face_detect import (
    detect_faces_bgr,
    detect_faces_bgr_relaxed,
    load_face_cascade,
    pick_largest_face,
)
from utils.label_mapping import (
    age_to_group_label,
    canonical_age_group_from_label,
    canonical_emotion_from_label,
    canonical_gender_from_label,
    map_emotion_happy_sad,
    map_gender_male_female,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

GENDER_LABELS = ["Male", "Female"]
AGE_LABELS = ["Adult", "Elderly"]
EMOTION_LABELS = ["Happy", "Sad"]

# Semicolon + UTF-8 BOM: double-click in Excel (DE) splits columns; BOM fixes "—" mojibake.
# pandas / Colab: pd.read_csv(..., sep=";")
_RESULTS_CSV_ENCODING = "utf-8-sig"
_RESULTS_CSV_DELIMITER = ";"


def _sanitize_csv_cell(v):
    if isinstance(v, str):
        return v.replace("\u2014", "-")  # em dash → ASCII for universal CSV readability
    return v


def _sanitize_row_dict(d: dict) -> dict:
    return {k: _sanitize_csv_cell(v) for k, v in d.items()}


def _fmt_metrics_csv_row(row: list) -> list:
    """Scope string + 4 metrics + 3 times + n_images — 4 decimal places for floats."""
    out: list = [row[0]]
    for i in range(1, 8):
        v = row[i]
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append(f"{float(v):.4f}")
        else:
            out.append(v)
    out.append(int(row[8]) if len(row) > 8 else row[-1])
    return out


def match_gender(tg: str, pg: str) -> bool:
    tc = canonical_gender_from_label(tg)
    return bool(tc) and pg not in ("", "—") and tc == pg


def match_emotion(te: str, pe: str) -> bool:
    tc = canonical_emotion_from_label(te)
    return bool(tc) and pe not in ("", "—") and tc == pe


def match_age_group(tag: str, pa: str) -> bool:
    tc = canonical_age_group_from_label(tag)
    return bool(tc) and pa not in ("", "unknown") and tc == pa


def load_labels_csv(path: Path) -> dict[str, dict[str, str]]:
    by_file: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = (row.get("filename") or "").strip()
            if fn:
                by_file[fn] = row
    return by_file


def validate_labels_header(con: Console, labels_csv: Path) -> bool:
    required = {"filename", "true_gender", "true_emotion", "true_age_group"}
    try:
        with labels_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
    except OSError as e:
        con.print(f"[red]Could not read labels CSV:[/red] {e}")
        return False

    missing = sorted(required - fieldnames)
    if missing:
        con.print()
        con.print(
            Panel(
                "\n".join(f"[yellow]•[/yellow] Missing column: {c}" for c in missing),
                title="[bold red]labels.csv header validation failed[/bold red]",
                border_style="red",
            )
        )
        return False
    return True


def collect_images(images_dir: Path, limit: int | None) -> list[Path]:
    paths = []
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(p)
    if limit is not None:
        paths = paths[:limit]
    return paths


def validate_dataset(
    con: Console,
    images: list[Path],
    label_map: dict[str, dict[str, str]],
    expect_count: int,
) -> bool:
    msgs: list[str] = []
    ok = True
    if len(images) != expect_count:
        msgs.append(
            f"Expected exactly {expect_count} images in run, found {len(images)}."
        )
        ok = False
    names = {p.name for p in images}
    for p in images:
        row = label_map.get(p.name)
        if row is None:
            msgs.append(f"No CSV row for file: {p.name}")
            ok = False
            continue
        if not (row.get("true_gender") or "").strip():
            msgs.append(f"{p.name}: missing true_gender")
            ok = False
        if not (row.get("true_emotion") or "").strip():
            msgs.append(f"{p.name}: missing true_emotion")
            ok = False
        if not (row.get("true_age_group") or "").strip():
            msgs.append(f"{p.name}: missing true_age_group")
            ok = False
    for fn in label_map:
        if fn not in names:
            msgs.append(f"CSV references {fn} but file not in image set")
    if msgs:
        # Dataset checks are strict, but we still show all warnings first.
        con.print()
        con.print(
            Panel(
                "\n".join(f"[yellow]•[/yellow] {m}" for m in msgs[:25])
                + ("\n[yellow]…[/yellow]" if len(msgs) > 25 else ""),
                title="[bold yellow]Dataset validation warnings[/bold yellow]",
                border_style="yellow",
            )
        )
        if not ok:
            con.print("[red]Fix dataset issues first.[/red]\n")
        else:
            con.print("[dim]Continuing with evaluation.[/dim]\n")
    return ok


def run_one_image(
    image_path: Path,
    cascade,
    labels_row: dict[str, str] | None,
):
    frame = cv2.imread(str(image_path))
    if frame is None:
        elapsed = 0.0
        tg = labels_row.get("true_gender", "") if labels_row else ""
        te = labels_row.get("true_emotion", "") if labels_row else ""
        tag = labels_row.get("true_age_group", "") if labels_row else ""
        return {
            "filename": image_path.name,
            "inference_seconds": elapsed,
            "pred_age": -1,
            "pred_gender": "—",
            "pred_emotion": "—",
            "pred_age_group": "unknown",
            "true_gender": tg,
            "true_emotion": te,
            "true_age_group": tag,
            "face_found": False,
        }

    t0 = time.perf_counter()
    faces = detect_faces_bgr(frame, cascade)
    if not faces:
        faces = detect_faces_bgr_relaxed(frame, cascade)
    face_rect = pick_largest_face(faces) if faces else None

    pred = None
    if face_rect is not None:
        x, y, w, h = face_rect
        y0 = max(0, y - int(h * 0.15))
        y1 = min(frame.shape[0], y + h + int(h * 0.15))
        x0 = max(0, x - int(w * 0.15))
        x1 = min(frame.shape[1], x + w + int(w * 0.15))
        pad_crop = frame[y0:y1, x0:x1]
        pad_crop = clahe_face_crop_bgr(pad_crop)
        pred = predict_face_region(pad_crop, enforce_detection=False)
        # If cropped path fails, try full frame without strict detection first.
        if pred is None:
            pred = predict_face_region(frame, enforce_detection=False)
        # Final fallback: strict internal detection on full frame.
        if pred is None:
            pred = predict_face_region(frame, enforce_detection=True)
    else:
        # No Haar face found: allow relaxed DeepFace run first, then strict.
        pred = predict_face_region(frame, enforce_detection=False)
        if pred is None:
            pred = predict_face_region(frame, enforce_detection=True)

    elapsed = time.perf_counter() - t0

    tg = labels_row.get("true_gender", "") if labels_row else ""
    te = labels_row.get("true_emotion", "") if labels_row else ""
    tag = labels_row.get("true_age_group", "") if labels_row else ""

    if pred is None:
        return {
            "filename": image_path.name,
            "inference_seconds": elapsed,
            "pred_age": -1,
            "pred_gender": "—",
            "pred_emotion": "—",
            "pred_age_group": "unknown",
            "true_gender": tg,
            "true_emotion": te,
            "true_age_group": tag,
            "face_found": False,
        }

    age = pred["age"]
    pg = map_gender_male_female(pred["gender"], pred.get("gender_scores"))
    pe = map_emotion_happy_sad(pred["emotion"], pred.get("emotion_scores"))
    pa = age_to_group_label(age) if age >= 0 else "unknown"

    return {
        "filename": image_path.name,
        "inference_seconds": elapsed,
        "pred_age": age,
        "pred_gender": pg,
        "pred_emotion": pe,
        "pred_age_group": pa,
        "true_gender": tg,
        "true_emotion": te,
        "true_age_group": tag,
        "face_found": True,
    }


def _fmt_triplet(g: str, e: str, a: str) -> str:
    g, e, a = (g or "—").strip(), (e or "—").strip(), (a or "—").strip()
    return f"{g} / {e} / {a}"


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def build_metric_table(rows_out: list[dict]) -> tuple[Table, dict]:
    n = len(rows_out)
    times = [r["inference_seconds"] for r in rows_out]
    mean_t = statistics.mean(times) if times else 0.0
    std_t = statistics.stdev(times) if len(times) > 1 else 0.0
    total_t = sum(times)

    # Compute accuracy/precision/recall/f1 per task over the full 20-image set.
    # Missing/invalid predictions are treated as incorrect by using a sentinel label.
    missing_pred = "__MISSING__"

    def _task_metrics(task: str, labels: list[str]) -> dict:
        y_true: list[str] = []
        y_pred: list[str] = []
        for r in rows_out:
            if task == "gender":
                yt = canonical_gender_from_label(r["true_gender"])
                yp = r["pred_gender"]
            elif task == "age":
                yt = canonical_age_group_from_label(r["true_age_group"])
                yp = r["pred_age_group"]
            else:
                yt = canonical_emotion_from_label(r["true_emotion"])
                yp = r["pred_emotion"]

            if yp not in labels:
                yp = missing_pred
            if yt not in labels:
                yt = missing_pred

            y_true.append(yt)
            y_pred.append(yp)

        acc = float(accuracy_score(y_true, y_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
        return {
            "accuracy": acc,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        }

    m_gender = _task_metrics("gender", GENDER_LABELS)
    m_age = _task_metrics("age", AGE_LABELS)
    m_emotion = _task_metrics("emotion", EMOTION_LABELS)

    overall_accuracy = statistics.mean(
        [m_gender["accuracy"], m_emotion["accuracy"], m_age["accuracy"]]
    )
    overall_precision = statistics.mean(
        [m_gender["precision"], m_emotion["precision"], m_age["precision"]]
    )
    overall_recall = statistics.mean(
        [m_gender["recall"], m_emotion["recall"], m_age["recall"]]
    )
    overall_f1 = statistics.mean([m_gender["f1"], m_emotion["f1"], m_age["f1"]])

    tbl = Table(
        title="[bold]Classification-style summary[/bold] [dim](G / E / age-group)[/dim]",
        show_header=True,
        header_style="bold cyan",
    )
    tbl.add_column("Metric", style="dim")
    tbl.add_column("Value", justify="right")

    tbl.add_row("Accuracy", f"{overall_accuracy:.4f}")
    tbl.add_row("Precision", f"{overall_precision:.4f}")
    tbl.add_row("Recall", f"{overall_recall:.4f}")
    tbl.add_row("F1 score", f"{overall_f1:.4f}")
    tbl.add_row("—", "—")
    tbl.add_row(
        "Gender accuracy",
        f"{m_gender['accuracy']:.4f}  (P/R/F1: {m_gender['precision']:.3f}/{m_gender['recall']:.3f}/{m_gender['f1']:.3f})",
    )
    tbl.add_row(
        "Emotion accuracy",
        f"{m_emotion['accuracy']:.4f}  (P/R/F1: {m_emotion['precision']:.3f}/{m_emotion['recall']:.3f}/{m_emotion['f1']:.3f})",
    )
    tbl.add_row(
        "Age-group accuracy",
        f"{m_age['accuracy']:.4f}  (P/R/F1: {m_age['precision']:.3f}/{m_age['recall']:.3f}/{m_age['f1']:.3f})",
    )
    tbl.add_row("Mean time / image (s)", f"{mean_t:.4f}")
    tbl.add_row("Std dev time (s)", f"{std_t:.4f}")
    tbl.add_row("Total runtime (s)", f"{total_t:.4f}")
    tbl.add_row("Images", str(n))

    meta = {
        "overall_accuracy": overall_accuracy,
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_f1": overall_f1,
        "gender": m_gender,
        "emotion": m_emotion,
        "age": m_age,
        "mean_t": mean_t,
        "std_t": std_t,
        "total_t": total_t,
        "n": n,
    }
    return tbl, meta


def build_predictions_table(rows_out: list[dict]) -> Table:
    t = Table(title="[bold]Per-image summary[/bold]", show_header=True, header_style="bold magenta")
    t.add_column("Image", style="cyan", no_wrap=True)
    t.add_column("True (G / E / age)", overflow="fold")
    t.add_column("Predicted", overflow="fold")
    t.add_column("OK", justify="center")

    for r in rows_out:
        tr = _fmt_triplet(r["true_gender"], r["true_emotion"], r["true_age_group"])
        pr = _fmt_triplet(
            r["pred_gender"],
            r["pred_emotion"],
            r["pred_age_group"],
        )
        ok = "[green]Y[/green]" if r.get("all_correct") else "[red]N[/red]"
        if not r["face_found"]:
            ok = "[dim]-[/dim]"
        t.add_row(r["filename"][:32], tr[:52], pr[:52], ok)
    return t


def confusion_pairs(rows: list[dict], task: str) -> tuple[list[str], list[str]]:
    yt: list[str] = []
    yp: list[str] = []
    for r in rows:
        if not r["face_found"]:
            continue
        if task == "gender":
            tc = canonical_gender_from_label(r["true_gender"])
            p = r["pred_gender"]
            labs = GENDER_LABELS
        elif task == "age":
            tc = canonical_age_group_from_label(r["true_age_group"])
            p = r["pred_age_group"]
            labs = AGE_LABELS
        else:
            tc = canonical_emotion_from_label(r["true_emotion"])
            p = r["pred_emotion"]
            labs = EMOTION_LABELS
        if not tc or p in ("", "—") or p == "unknown":
            continue
        if tc in labs and p in labs:
            yt.append(tc)
            yp.append(p)
    return yt, yp


def save_confusion_csv(path: Path, cm, labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding=_RESULTS_CSV_ENCODING) as f:
        w = csv.writer(f, delimiter=_RESULTS_CSV_DELIMITER)
        w.writerow([""] + [f"pred_{x}" for x in labels])
        for i, lab in enumerate(labels):
            w.writerow([f"true_{lab}"] + [str(int(cm[i, j])) for j in range(len(labels))])


def rich_confusion_table(title: str, cm, labels: list[str]) -> Table:
    t = Table(title=title, header_style="bold white")
    t.add_column("True \\ Pred", style="dim")
    for lb in labels:
        t.add_column(lb, justify="right")
    for i, lab in enumerate(labels):
        t.add_row(lab, *[str(int(cm[i, j])) for j in range(len(labels))])
    return t


def _save_evaluation_plots(
    out_dir: Path,
    meta: dict,
    confusion_payload: list[tuple[str, np.ndarray, list[str]]],
    con: Console,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        con.print("[yellow]matplotlib not available — skipped evaluation plots.[/yellow]")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    # 1) Grouped bars for overall/gender/age/emotion metrics.
    scopes = ["overall", "gender", "age", "emotion"]
    metric_defs = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
    ]
    x = np.arange(len(scopes))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (metric_key, metric_label) in enumerate(metric_defs):
        vals = []
        for scope in scopes:
            if scope == "overall":
                vals.append(float(meta.get(f"overall_{metric_key}", 0.0)))
            else:
                vals.append(float(meta.get(scope, {}).get(metric_key, 0.0)))
        bar_pos = x + (i - (len(metric_defs) - 1) / 2) * width
        ax.bar(bar_pos, vals, width=width, label=metric_label)

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Evaluation metrics overview")
    ax.set_xticks(x)
    ax.set_xticklabels(["Overall", "Gender", "Age", "Emotion"])
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    metrics_plot_path = out_dir / "evaluation_metrics_plot.png"
    fig.savefig(metrics_plot_path, dpi=140)
    plt.close(fig)
    saved_paths.append(metrics_plot_path)

    # 2) One confusion heatmap per task for quick visual inspection.
    for task, cm_raw, labels in confusion_payload:
        cm = np.asarray(cm_raw, dtype=int)
        fig, ax = plt.subplots(figsize=(5.3, 4.6))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{task.capitalize()} confusion matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                val = int(cm[r, c])
                txt_color = "white" if val > (cm.max() / 2 if cm.size else 0) else "black"
                ax.text(c, r, str(val), ha="center", va="center", color=txt_color)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        cm_plot_path = out_dir / f"confusion_{task}_plot.png"
        fig.savefig(cm_plot_path, dpi=140)
        plt.close(fig)
        saved_paths.append(cm_plot_path)

    return saved_paths


def run_evaluation(
    images_dir: Path,
    labels_csv: Path,
    out_csv: Path = Path("results/evaluation_results.csv"),
    predictions_csv: Path = Path("results/predictions.csv"),
    confusion_gender: Path = Path("results/confusion_gender.csv"),
    confusion_age: Path = Path("results/confusion_age.csv"),
    confusion_emotion: Path = Path("results/confusion_emotion.csv"),
    max_images: int = 20,
    console: Console | None = None,
    use_rich: bool = True,
    require_gpu: bool = False,
) -> int:
    con = console or Console()

    if require_gpu:
        try:
            import tensorflow as tf
        except Exception as e:
            con.print(f"[red]GPU required, but TensorFlow failed to import:[/red] {e}")
            con.print(
                "[yellow]Hint:[/yellow] enable a TensorFlow-compatible GPU runtime/environment."
            )
            return 1

        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            con.print("[red]GPU required, but no TensorFlow GPU device is available.[/red]")
            con.print(
                "[yellow]Hint:[/yellow] enable GPU in your runtime and restart the process."
            )
            return 1
        con.print(f"[green]GPU check passed.[/green] Found {len(gpus)} GPU device(s).")

    if not images_dir.is_dir():
        con.print(f"[red]Not a directory:[/red] {images_dir}")
        return 1
    if not labels_csv.is_file():
        con.print(f"[red]Labels file not found:[/red] {labels_csv}")
        return 1

    if not validate_labels_header(con, labels_csv):
        return 1

    try:
        with con.status("[bold yellow]Loading cascade…[/bold yellow]"):
            cascade = load_face_cascade()
    except RuntimeError as e:
        con.print(f"[red]{e}[/red]")
        return 1
    con.print("[green]Cascade ready.[/green]")

    label_map = load_labels_csv(labels_csv)

    all_images = collect_images(images_dir, limit=None)
    if len(all_images) < max_images:
        con.print()
        con.print(
            Panel(
                f"[bold]Expected {max_images} images but found {len(all_images)}.[/bold]\n"
                "Provide a folder with exactly 20 labeled images (or set --max_images).",
                title="[bold red]Dataset validation failed[/bold red]",
                border_style="red",
            )
        )
        return 1
    if len(all_images) != max_images:
        con.print()
        con.print(
            Panel(
                f"[yellow]Found {len(all_images)} images.[/yellow] "
                f"Using only the first {max_images} for evaluation.",
                title="[bold yellow]Dataset size warning[/bold yellow]",
                border_style="yellow",
            )
        )

    images = all_images[:max_images]
    if not images:
        con.print(f"[red]No images in[/red] {images_dir}")
        return 1

    if not validate_dataset(con, images, label_map, max_images):
        return 1

    rows_out: list[dict] = []

    progress_cols = (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    )

    with Progress(*progress_cols, console=con) as prog:
        task = prog.add_task("[cyan]Running predictions…[/cyan]", total=len(images))
        for img_path in images:
            row_labels = label_map.get(img_path.name)
            r = run_one_image(img_path, cascade, row_labels)
            prog.advance(task)
            if r is None:
                continue

            tg = r["true_gender"]
            te = r["true_emotion"]
            tag = r["true_age_group"]

            g_ok = match_gender(tg, r["pred_gender"]) if tg and r["face_found"] else False
            e_ok = match_emotion(te, r["pred_emotion"]) if te and r["face_found"] else False
            a_ok = (
                match_age_group(tag, r["pred_age_group"])
                if tag and r["face_found"]
                else False
            )

            r["correct_gender"] = int(g_ok)
            r["correct_emotion"] = int(e_ok)
            r["correct_age_group"] = int(a_ok)
            # bool(...) required: chained `and` with strings returns last string, not True (e.g. 'Elderly').
            labels_present = all(str(x or "").strip() for x in (tg, te, tag))
            r["all_correct"] = int(bool(g_ok and e_ok and a_ok and labels_present))
            rows_out.append(r)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "filename",
        "true_gender",
        "true_emotion",
        "true_age_group",
        "pred_age",
        "pred_gender",
        "pred_emotion",
        "pred_age_group",
        "correct_gender",
        "correct_emotion",
        "correct_age_group",
        "all_correct",
        "inference_seconds",
        "face_found",
    ]
    with out_csv.open("w", newline="", encoding=_RESULTS_CSV_ENCODING) as f:
        w = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
            delimiter=_RESULTS_CSV_DELIMITER,
        )
        w.writeheader()
        for r in rows_out:
            w.writerow(_sanitize_row_dict({k: r.get(k) for k in fieldnames}))

    pred_fields = [
        "filename",
        "true_gender_emotion_agegroup",
        "pred_gender_emotion_agegroup",
        "pred_age_numeric",
        "all_correct",
        "correct_gender",
        "correct_emotion",
        "correct_age_group",
        "face_found",
        "inference_seconds",
    ]
    with predictions_csv.open("w", newline="", encoding=_RESULTS_CSV_ENCODING) as f:
        w = csv.DictWriter(
            f,
            pred_fields,
            extrasaction="ignore",
            delimiter=_RESULTS_CSV_DELIMITER,
        )
        w.writeheader()
        for r in rows_out:
            w.writerow(
                _sanitize_row_dict(
                    {
                        "filename": r["filename"],
                        "true_gender_emotion_agegroup": _fmt_triplet(
                            r["true_gender"], r["true_emotion"], r["true_age_group"]
                        ),
                        "pred_gender_emotion_agegroup": _fmt_triplet(
                            r["pred_gender"],
                            r["pred_emotion"],
                            r["pred_age_group"],
                        ),
                        "pred_age_numeric": r["pred_age"],
                        "all_correct": r["all_correct"],
                        "correct_gender": r["correct_gender"],
                        "correct_emotion": r["correct_emotion"],
                        "correct_age_group": r["correct_age_group"],
                        "face_found": int(r["face_found"]),
                        "inference_seconds": f"{r['inference_seconds']:.4f}",
                    }
                )
            )

    n = len(rows_out)
    if n == 0:
        con.print("[yellow]No rows to display.[/yellow]")
        return 0

    tasks = [
        ("gender", GENDER_LABELS, confusion_gender, "[bold]Gender (Male / Female)[/bold]"),
        ("age", AGE_LABELS, confusion_age, "[bold]Age group (Adult / Elderly)[/bold]"),
        ("emotion", EMOTION_LABELS, confusion_emotion, "[bold]Emotion (Happy / Sad)[/bold]"),
    ]
    confusion_payload: list[tuple[str, np.ndarray, list[str]]] = []

    metric_table, meta = build_metric_table(rows_out)
    # Always save metrics (even in --plain mode); same folder as evaluation_results.csv.
    metrics_csv = out_csv.parent / "metrics.csv"
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    with metrics_csv.open("w", newline="", encoding=_RESULTS_CSV_ENCODING) as f:
        w = csv.writer(f, delimiter=_RESULTS_CSV_DELIMITER)
        hdr = [
            "scope",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "mean_inference_time_s",
            "std_inference_time_s",
            "total_runtime_s",
            "n_images",
        ]
        w.writerow(hdr)
        w.writerow(
            _fmt_metrics_csv_row(
                [
                    "overall",
                    meta["overall_accuracy"],
                    meta["overall_precision"],
                    meta["overall_recall"],
                    meta["overall_f1"],
                    meta["mean_t"],
                    meta["std_t"],
                    meta["total_t"],
                    meta["n"],
                ]
            )
        )
        w.writerow(
            _fmt_metrics_csv_row(
                [
                    "gender",
                    meta["gender"]["accuracy"],
                    meta["gender"]["precision"],
                    meta["gender"]["recall"],
                    meta["gender"]["f1"],
                    meta["mean_t"],
                    meta["std_t"],
                    meta["total_t"],
                    meta["n"],
                ]
            )
        )
        w.writerow(
            _fmt_metrics_csv_row(
                [
                    "emotion",
                    meta["emotion"]["accuracy"],
                    meta["emotion"]["precision"],
                    meta["emotion"]["recall"],
                    meta["emotion"]["f1"],
                    meta["mean_t"],
                    meta["std_t"],
                    meta["total_t"],
                    meta["n"],
                ]
            )
        )
        w.writerow(
            _fmt_metrics_csv_row(
                [
                    "age",
                    meta["age"]["accuracy"],
                    meta["age"]["precision"],
                    meta["age"]["recall"],
                    meta["age"]["f1"],
                    meta["mean_t"],
                    meta["std_t"],
                    meta["total_t"],
                    meta["n"],
                ]
            )
        )

    if use_rich:
        con.print()
        con.print(
            Panel.fit("[bold white]Evaluation complete[/bold white]", style="green")
        )
        con.print(metric_table)
        con.print()
        con.print(build_predictions_table(rows_out))

    for key, labs, cpath, title in tasks:
        yt, yp = confusion_pairs(rows_out, key)
        if len(yt) == 0:
            empty_cm = np.zeros((len(labs), len(labs)), dtype=int)
            save_confusion_csv(
                cpath,
                empty_cm,
                labs,
            )
            confusion_payload.append((key, empty_cm, labs))
            if use_rich:
                con.print()
                con.print(
                    f"[yellow]No valid pairs for {key} confusion — empty {cpath.name}[/yellow]"
                )
            continue
        cm = confusion_matrix(yt, yp, labels=labs)
        save_confusion_csv(cpath, cm, labs)
        confusion_payload.append((key, cm, labs))
        if use_rich:
            con.print()
            con.print(rich_confusion_table(title, cm, labs))
            rep = classification_report(
                yt, yp, labels=labs, zero_division=0, digits=3
            )
            con.print(
                Panel(
                    rep,
                    title=f"[cyan]{key} · classification_report[/cyan]",
                    border_style="cyan",
                )
            )

    plot_paths = _save_evaluation_plots(out_csv.parent, meta, confusion_payload, con)

    results_folder = out_csv.parent.resolve()
    written_lines = [
        f"[green]✓[/green] [bold]{out_csv.name}[/bold] — full table per image "
        "(true vs predicted gender, emotion, age group, correct flags, timing).",
        f"[green]✓[/green] [bold]{predictions_csv.name}[/bold] — compact file: one row per image with "
        "[cyan]true[/cyan] and [cyan]pred[/cyan] as combined text (e.g. Male | Happy | Adult); "
        "handy for quick review.",
        f"[green]✓[/green] [bold]{metrics_csv.name}[/bold] — accuracy / precision / recall / F1 summary.",
    ]
    for _, _, cpath, _ in tasks:
        written_lines.append(f"[green]✓[/green] [bold]{cpath.name}[/bold] — confusion matrix counts.")
    for p in plot_paths:
        written_lines.append(
            f"[green]✓[/green] [bold]{p.name}[/bold] — chart image."
        )

    if use_rich:
        con.print()
        con.print(
            Panel.fit(
                "\n".join(written_lines)
                + "\n\n[dim]CSV: semicolon (;) separator + UTF-8 with BOM — opens in columns in Excel (DE). "
                "In Python/pandas:[/dim] [cyan]pd.read_csv(..., sep=';')[/cyan]",
                title=f"[bold green]Saved under {results_folder}[/bold green]",
                border_style="green",
            )
        )
        con.print(
            "[dim]Tip:[/dim] [bold]evaluation_results.csv[/bold] + [bold]metrics.csv[/bold] "
            "are usually enough for most checks; [bold]predictions.csv[/bold] is the same run in a shorter layout."
        )
    else:
        times = [r["inference_seconds"] for r in rows_out]
        mean_t = statistics.mean(times)
        std_t = statistics.stdev(times) if n > 1 else 0.0
        print(f"Images: {n}  mean time: {mean_t:.4f}s  std: {std_t:.4f}s")
        print(f"Saved under: {results_folder}")
        print(f"  {out_csv.name}")
        print(f"  {predictions_csv.name}")
        print(f"  {metrics_csv.name}")
        for _, _, cpath, _ in tasks:
            print(f"  {cpath.name}")
        for p in plot_paths:
            print(f"  {p.name}")

    con.print("[green]Done.[/green]")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=Path, required=True)
    parser.add_argument("--labels_csv", type=Path, required=True)
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=Path("results/evaluation_results.csv"),
    )
    parser.add_argument(
        "--predictions_csv",
        type=Path,
        default=Path("results/predictions.csv"),
    )
    parser.add_argument(
        "--confusion_gender",
        type=Path,
        default=Path("results/confusion_gender.csv"),
    )
    parser.add_argument(
        "--confusion_age",
        type=Path,
        default=Path("results/confusion_age.csv"),
    )
    parser.add_argument(
        "--confusion_emotion",
        type=Path,
        default=Path("results/confusion_emotion.csv"),
    )
    parser.add_argument(
        "--require_gpu",
        action="store_true",
        help="Fail fast if no TensorFlow GPU is available.",
    )
    parser.add_argument("--max_images", type=int, default=20)
    parser.add_argument("--plain", action="store_true")
    args = parser.parse_args()
    return run_evaluation(
        args.images_dir,
        args.labels_csv,
        out_csv=args.out_csv,
        predictions_csv=args.predictions_csv,
        confusion_gender=args.confusion_gender,
        confusion_age=args.confusion_age,
        confusion_emotion=args.confusion_emotion,
        max_images=args.max_images,
        use_rich=not args.plain,
        require_gpu=args.require_gpu,
    )


if __name__ == "__main__":
    raise SystemExit(main())
