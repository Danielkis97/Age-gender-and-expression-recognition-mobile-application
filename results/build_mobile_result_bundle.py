from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = ROOT / "dataset" / "labels.csv"
DEFAULT_OUT_DIR = ROOT / "results" / "Results mobile metrics"
DEFAULT_MOBILE_CSV_CANDIDATES = [
    ROOT / "results" / "Results mobile metrics" / "mobile_browser_metrics.csv",
    ROOT / "results" / "mobile_browser_metrics.csv",
]


@dataclass
class LabelRow:
    filename: str
    true_gender: str
    true_emotion: str
    true_age_group: str


def _read_labels(path: Path) -> list[LabelRow]:
    rows: list[LabelRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                LabelRow(
                    filename=(row.get("filename") or "").strip(),
                    true_gender=(row.get("true_gender") or "").strip(),
                    true_emotion=(row.get("true_emotion") or "").strip(),
                    true_age_group=(row.get("true_age_group") or "").strip(),
                )
            )
    return [r for r in rows if r.filename]


def _extract_image_name(rec: dict) -> str:
    direct = (rec.get("image_name") or "").strip()
    if direct:
        return direct
    notes = (rec.get("notes") or "").strip()
    marker = "image_name="
    if marker in notes:
        tail = notes.split(marker, 1)[1].strip()
        return tail.split(";", 1)[0].strip()
    return ""


def _read_mobile_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        return list(reader)


def _pick_run_id(rows: list[dict], run_id: str | None) -> str:
    candidates = [r for r in rows if (r.get("input_source") or "").strip() == "dataset_images"]
    if run_id:
        matches = [r for r in candidates if (r.get("run_id") or "").strip() == run_id]
        if not matches:
            raise ValueError(f"Requested run_id not found in dataset_images rows: {run_id}")
        return run_id
    if not candidates:
        raise ValueError("No dataset_images rows found in mobile CSV.")
    # Pick latest by timestamp string (ISO format).
    candidates.sort(key=lambda r: (r.get("timestamp_iso") or "", r.get("run_id") or ""))
    return (candidates[-1].get("run_id") or "").strip()


def _resolve_mobile_csv(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    for candidate in DEFAULT_MOBILE_CSV_CANDIDATES:
        if candidate.is_file():
            return candidate
    return DEFAULT_MOBILE_CSV_CANDIDATES[0]


def build_bundle(
    mobile_csv: Path,
    labels_csv: Path,
    out_dir: Path,
    run_id: str | None = None,
) -> dict[str, str]:
    if not labels_csv.is_file():
        raise FileNotFoundError(f"labels.csv not found: {labels_csv}")
    if not mobile_csv.is_file():
        raise FileNotFoundError(f"mobile csv not found: {mobile_csv}")

    labels = _read_labels(labels_csv)
    if not labels:
        raise ValueError(f"No label rows in: {labels_csv}")
    labels_by_name = {r.filename: r for r in labels}

    all_rows = _read_mobile_rows(mobile_csv)
    chosen_run = _pick_run_id(all_rows, run_id=run_id)
    run_rows = [r for r in all_rows if (r.get("run_id") or "").strip() == chosen_run]
    if not run_rows:
        raise ValueError(f"No rows found for run_id: {chosen_run}")

    normalized: list[dict] = []
    for i, rec in enumerate(run_rows):
        try:
            latency_ms = float((rec.get("latency_ms") or "").strip())
        except ValueError:
            continue
        image_name = _extract_image_name(rec)
        if not image_name:
            try:
                sample_idx = int((rec.get("sample_idx") or "").strip())
                if 1 <= sample_idx <= len(labels):
                    image_name = labels[sample_idx - 1].filename
            except ValueError:
                pass
        if not image_name and i < len(labels):
            image_name = labels[i].filename
        normalized.append(
            {
                "filename": image_name,
                "latency_ms": latency_ms,
                "device_label": (rec.get("device_label") or "").strip(),
                "run_id": (rec.get("run_id") or "").strip(),
            }
        )

    if not normalized:
        raise ValueError("Run contains no valid latency rows.")

    out_dir.mkdir(parents=True, exist_ok=True)
    eval_path = out_dir / "evaluation_results.csv"
    pred_path = out_dir / "predictions.csv"
    metrics_path = out_dir / "metrics.csv"
    note_path = out_dir / "mobile_quality_note.txt"

    eval_fields = [
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
    with eval_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=eval_fields, delimiter=";")
        w.writeheader()
        for row in normalized:
            label = labels_by_name.get(row["filename"], LabelRow(row["filename"], "", "", ""))
            w.writerow(
                {
                    "filename": row["filename"],
                    "true_gender": label.true_gender,
                    "true_emotion": label.true_emotion,
                    "true_age_group": label.true_age_group,
                    "pred_age": "",
                    "pred_gender": "",
                    "pred_emotion": "",
                    "pred_age_group": "",
                    "correct_gender": "",
                    "correct_emotion": "",
                    "correct_age_group": "",
                    "all_correct": "",
                    "inference_seconds": f"{row['latency_ms'] / 1000.0:.6f}",
                    "face_found": "True",
                }
            )

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
    with pred_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pred_fields, delimiter=";")
        w.writeheader()
        for row in normalized:
            label = labels_by_name.get(row["filename"], LabelRow(row["filename"], "", "", ""))
            truth = " / ".join(x for x in [label.true_gender, label.true_emotion, label.true_age_group] if x)
            w.writerow(
                {
                    "filename": row["filename"],
                    "true_gender_emotion_agegroup": truth,
                    "pred_gender_emotion_agegroup": "",
                    "pred_age_numeric": "",
                    "all_correct": "",
                    "correct_gender": "",
                    "correct_emotion": "",
                    "correct_age_group": "",
                    "face_found": "1",
                    "inference_seconds": f"{row['latency_ms'] / 1000.0:.4f}",
                }
            )

    times_s = [r["latency_ms"] / 1000.0 for r in normalized]
    mean_s = statistics.mean(times_s)
    std_s = statistics.stdev(times_s) if len(times_s) > 1 else 0.0
    total_s = sum(times_s)

    metric_fields = [
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
    with metrics_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=metric_fields, delimiter=";")
        w.writeheader()
        for scope in ("overall", "gender", "emotion", "age"):
            w.writerow(
                {
                    "scope": scope,
                    "accuracy": "",
                    "precision": "",
                    "recall": "",
                    "f1": "",
                    "mean_inference_time_s": f"{mean_s:.4f}",
                    "std_inference_time_s": f"{std_s:.4f}",
                    "total_runtime_s": f"{total_s:.4f}",
                    "n_images": str(len(times_s)),
                }
            )

    note_path.write_text(
        "\n".join(
            [
                "Mobile result bundle generated from browser on-device timing run.",
                f"Source run_id: {chosen_run}",
                "",
                "Timing metrics are directly measured on iPhone (latency_ms).",
                "Quality metrics (accuracy/precision/recall/f1, predictions) are not available",
                "for this mobile TFLite demo model and are intentionally left blank.",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "run_id": chosen_run,
        "n_rows": str(len(times_s)),
        "metrics": str(metrics_path),
        "evaluation_results": str(eval_path),
        "predictions": str(pred_path),
        "note": str(note_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build CPU/GPU-style result files from mobile browser timing CSV.")
    parser.add_argument("--mobile_csv", type=Path, default=None)
    parser.add_argument("--labels_csv", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--run_id", type=str, default=None, help="Optional run_id to export. Default: latest dataset_images run")
    args = parser.parse_args()

    mobile_csv = _resolve_mobile_csv(args.mobile_csv)
    result = build_bundle(
        mobile_csv=mobile_csv,
        labels_csv=args.labels_csv,
        out_dir=args.out_dir,
        run_id=args.run_id,
    )
    print("Built mobile result bundle:")
    for k, v in result.items():
        print(f"- {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
