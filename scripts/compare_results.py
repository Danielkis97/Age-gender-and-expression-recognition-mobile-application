from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MetricsRow:
    scope: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    mean_inference_time_s: float
    std_inference_time_s: float
    total_runtime_s: float
    n_images: int


@dataclass
class TimeStats:
    mean_all: float
    median: float
    mean_excl_first: float
    trimmed_mean: float
    p90: float
    min_v: float
    max_v: float
    n: int


def _read_metrics(path: Path) -> dict[str, MetricsRow]:
    rows: dict[str, MetricsRow] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            m = MetricsRow(
                scope=row["scope"],
                accuracy=float(row["accuracy"]),
                precision=float(row["precision"]),
                recall=float(row["recall"]),
                f1=float(row["f1"]),
                mean_inference_time_s=float(row["mean_inference_time_s"]),
                std_inference_time_s=float(row["std_inference_time_s"]),
                total_runtime_s=float(row["total_runtime_s"]),
                n_images=int(float(row["n_images"])),
            )
            rows[m.scope] = m
    return rows


def _read_eval(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            out[row["filename"]] = row
    return out


def _time_stats(eval_rows: dict[str, dict]) -> TimeStats:
    times = [float(r["inference_seconds"]) for r in eval_rows.values()]
    if not times:
        return TimeStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    s = sorted(times)
    mean_all = statistics.mean(times)
    median = statistics.median(times)
    mean_excl_first = statistics.mean(times[1:]) if len(times) > 1 else mean_all
    trimmed_mean = statistics.mean(s[1:-1]) if len(s) > 2 else mean_all
    p90_idx = int(0.9 * (len(s) - 1))
    p90 = s[p90_idx]
    return TimeStats(
        mean_all=mean_all,
        median=median,
        mean_excl_first=mean_excl_first,
        trimmed_mean=trimmed_mean,
        p90=p90,
        min_v=min(s),
        max_v=max(s),
        n=len(s),
    )


def _pct(a: float, b: float) -> float:
    if a == 0:
        return 0.0
    return (b - a) / a * 100.0


def _f(x: float) -> str:
    return f"{x:.4f}"


def build_markdown(
    cpu_name: str,
    gpu_name: str,
    cpu_metrics: dict[str, MetricsRow],
    gpu_metrics: dict[str, MetricsRow],
    cpu_times: TimeStats,
    gpu_times: TimeStats,
    n_label_diffs: int,
    n_images: int,
) -> str:
    scopes = ["overall", "gender", "emotion", "age"]
    lines: list[str] = []
    lines.append("# CPU vs Colab-GPU comparison")
    lines.append("")
    lines.append(f"- CPU source: `{cpu_name}`")
    lines.append(f"- GPU source: `{gpu_name}`")
    lines.append("")
    lines.append("## Metrics comparison (line-by-line)")
    lines.append("")
    for s in scopes:
        c = cpu_metrics[s]
        g = gpu_metrics[s]
        lines.append(f"### {s}")
        lines.append(f"- accuracy: CPU `{_f(c.accuracy)}` | GPU `{_f(g.accuracy)}` | delta `{_f(g.accuracy - c.accuracy)}`")
        lines.append(f"- precision: CPU `{_f(c.precision)}` | GPU `{_f(g.precision)}`")
        lines.append(f"- recall: CPU `{_f(c.recall)}` | GPU `{_f(g.recall)}`")
        lines.append(f"- f1: CPU `{_f(c.f1)}` | GPU `{_f(g.f1)}`")
        lines.append("")
    lines.append("")
    lines.append("## Serving-time comparison")
    lines.append("")
    c0 = cpu_metrics["overall"]
    g0 = gpu_metrics["overall"]
    lines.append(
        f"- mean inference time / image (s): CPU `{_f(c0.mean_inference_time_s)}` | GPU `{_f(g0.mean_inference_time_s)}` | "
        f"delta `{_f(g0.mean_inference_time_s - c0.mean_inference_time_s)}` ({_f(_pct(c0.mean_inference_time_s, g0.mean_inference_time_s))}%)"
    )
    lines.append(
        f"- total runtime (s, {c0.n_images} images): CPU `{_f(c0.total_runtime_s)}` | GPU `{_f(g0.total_runtime_s)}` | "
        f"delta `{_f(g0.total_runtime_s - c0.total_runtime_s)}` ({_f(_pct(c0.total_runtime_s, g0.total_runtime_s))}%)"
    )
    lines.append("")
    lines.append("### Warm-up aware timing (from `evaluation_results.csv`)")
    lines.append("")
    lines.append(
        f"- median inference time (s): CPU `{_f(cpu_times.median)}` | GPU `{_f(gpu_times.median)}` | "
        f"delta `{_f(gpu_times.median - cpu_times.median)}` ({_f(_pct(cpu_times.median, gpu_times.median))}%)"
    )
    lines.append(
        f"- mean excluding first image (s): CPU `{_f(cpu_times.mean_excl_first)}` | GPU `{_f(gpu_times.mean_excl_first)}` | "
        f"delta `{_f(gpu_times.mean_excl_first - cpu_times.mean_excl_first)}` ({_f(_pct(cpu_times.mean_excl_first, gpu_times.mean_excl_first))}%)"
    )
    lines.append(
        f"- trimmed mean (drop min/max) (s): CPU `{_f(cpu_times.trimmed_mean)}` | GPU `{_f(gpu_times.trimmed_mean)}` | "
        f"delta `{_f(gpu_times.trimmed_mean - cpu_times.trimmed_mean)}` ({_f(_pct(cpu_times.trimmed_mean, gpu_times.trimmed_mean))}%)"
    )
    lines.append(
        f"- p90 inference time (s): CPU `{_f(cpu_times.p90)}` | GPU `{_f(gpu_times.p90)}` | "
        f"delta `{_f(gpu_times.p90 - cpu_times.p90)}` ({_f(_pct(cpu_times.p90, gpu_times.p90))}%)"
    )
    lines.append(
        f"- min / max (s): CPU `{_f(cpu_times.min_v)} / {_f(cpu_times.max_v)}` | "
        f"GPU `{_f(gpu_times.min_v)} / {_f(gpu_times.max_v)}`"
    )
    lines.append("")
    lines.append("## Prediction-level consistency")
    lines.append("")
    lines.append(
        f"- Compared `{n_images}` images by `(pred_gender, pred_emotion, pred_age_group)`."
    )
    lines.append(f"- Label-level differences between runs: **{n_label_diffs}**.")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- In this run, classification metrics are identical across CPU and Colab-GPU."
    )
    lines.append(
        "- The first-image warm-up is very large in both runs, so 'mean over all images' can hide steady-state behavior."
    )
    lines.append(
        "- For the report: mention both views: (1) all-image mean (end-to-end) and (2) warm-up-aware metrics (median or mean excluding first image)."
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-dir", type=Path, required=True)
    ap.add_argument("--gpu-dir", type=Path, required=True)
    ap.add_argument(
        "--out-md",
        type=Path,
        default=Path("results/comparison_cpu_vs_colab.md"),
    )
    args = ap.parse_args()

    cpu_metrics = _read_metrics(args.cpu_dir / "metrics.csv")
    gpu_metrics = _read_metrics(args.gpu_dir / "metrics.csv")

    cpu_eval = _read_eval(args.cpu_dir / "evaluation_results.csv")
    gpu_eval = _read_eval(args.gpu_dir / "evaluation_results.csv")
    cpu_times = _time_stats(cpu_eval)
    gpu_times = _time_stats(gpu_eval)
    all_names = sorted(set(cpu_eval) | set(gpu_eval))

    label_diffs = 0
    for name in all_names:
        c = cpu_eval.get(name, {})
        g = gpu_eval.get(name, {})
        ct = (c.get("pred_gender"), c.get("pred_emotion"), c.get("pred_age_group"))
        gt = (g.get("pred_gender"), g.get("pred_emotion"), g.get("pred_age_group"))
        if ct != gt:
            label_diffs += 1

    md = build_markdown(
        cpu_name=str(args.cpu_dir),
        gpu_name=str(args.gpu_dir),
        cpu_metrics=cpu_metrics,
        gpu_metrics=gpu_metrics,
        cpu_times=cpu_times,
        gpu_times=gpu_times,
        n_label_diffs=label_diffs,
        n_images=len(all_names),
    )

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")
    print(f"Wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
