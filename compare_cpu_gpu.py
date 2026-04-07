from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _detect_delimiter(header_line: str) -> str:
    return ";" if header_line.count(";") >= header_line.count(",") else ","


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        first = f.readline()
        if not first:
            return []
        delim = _detect_delimiter(first)
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delim)
        return [dict(r) for r in reader]


def _to_float(v: str | None, default: float = 0.0) -> float:
    try:
        return float(str(v).strip().replace(",", "."))
    except Exception:
        return default


def _index_by(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        k = (r.get(key) or "").strip()
        if k:
            out[k] = r
    return out


def _load_eval(metrics_csv: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv_rows(metrics_csv)
    return _index_by(rows, "scope")


def _load_perf(perf_csv: Path) -> dict[str, dict[str, str]]:
    if not perf_csv.is_file():
        return {}
    rows = _read_csv_rows(perf_csv)
    return _index_by(rows, "device")


def _perf_time_with_fallback(
    perf_rows: dict[str, dict[str, str]],
    device_key: str,
    eval_time_s: float,
) -> float:
    """
    Prefer explicit benchmark timing from performance.csv.
    If unavailable (or zero), fall back to evaluation mean timing.
    """
    t = _to_float(perf_rows.get(device_key, {}).get("time_per_image_s"))
    if t > 0:
        return t
    return eval_time_s


def _save_summary_csv(
    out_path: Path,
    cpu_eval: dict[str, dict[str, str]],
    gpu_eval: dict[str, dict[str, str]],
    cpu_perf: dict[str, dict[str, str]],
    gpu_perf: dict[str, dict[str, str]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for scope in ("overall", "gender", "age", "emotion"):
        c = cpu_eval.get(scope, {})
        g = gpu_eval.get(scope, {})
        for metric in ("accuracy", "precision", "recall", "f1"):
            c_v = _to_float(c.get(metric))
            g_v = _to_float(g.get(metric))
            rows.append(
                {
                    "section": "evaluation",
                    "item": f"{scope}.{metric}",
                    "local_cpu": f"{c_v:.6f}",
                    "colab_gpu": f"{g_v:.6f}",
                    "delta_gpu_minus_cpu": f"{(g_v - c_v):.6f}",
                }
            )

    cpu_time = _to_float(cpu_perf.get("CPU", {}).get("time_per_image_s"))
    gpu_time = _to_float(gpu_perf.get("GPU", {}).get("time_per_image_s"))
    speedup = (cpu_time / gpu_time) if gpu_time > 0 else 0.0
    rows.append(
        {
            "section": "performance",
            "item": "time_per_image_s",
            "local_cpu": f"{cpu_time:.6f}",
            "colab_gpu": f"{gpu_time:.6f}",
            "delta_gpu_minus_cpu": f"{(gpu_time - cpu_time):.6f}",
        }
    )
    rows.append(
        {
            "section": "performance",
            "item": "speedup_cpu_over_gpu",
            "local_cpu": "1.000000",
            "colab_gpu": f"{speedup:.6f}",
            "delta_gpu_minus_cpu": f"{(speedup - 1.0):.6f}",
        }
    )

    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "section",
                "item",
                "local_cpu",
                "colab_gpu",
                "delta_gpu_minus_cpu",
            ],
            delimiter=";",
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _plot_eval_comparison(
    out_path: Path,
    cpu_eval: dict[str, dict[str, str]],
    gpu_eval: dict[str, dict[str, str]],
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    scopes = ["overall", "gender", "age", "emotion"]
    metrics = ["accuracy", "precision", "recall", "f1"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes_flat = axes.flatten()

    for i, scope in enumerate(scopes):
        ax = axes_flat[i]
        cpu_vals = [_to_float(cpu_eval.get(scope, {}).get(m)) for m in metrics]
        gpu_vals = [_to_float(gpu_eval.get(scope, {}).get(m)) for m in metrics]
        x = np.arange(len(metrics))
        w = 0.36
        ax.bar(x - w / 2, cpu_vals, width=w, label="Local CPU", color="#4C78A8")
        ax.bar(x + w / 2, gpu_vals, width=w, label="Colab GPU", color="#54A24B")
        ax.set_title(scope.capitalize())
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in metrics], rotation=20, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.25)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Evaluation metrics: Local CPU vs Colab GPU", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_perf_comparison(
    out_path: Path,
    cpu_eval: dict[str, dict[str, str]],
    gpu_eval: dict[str, dict[str, str]],
    cpu_perf: dict[str, dict[str, str]],
    gpu_perf: dict[str, dict[str, str]],
) -> None:
    import matplotlib.pyplot as plt

    cpu_eval_t = _to_float(cpu_eval.get("overall", {}).get("mean_inference_time_s"))
    gpu_eval_t = _to_float(gpu_eval.get("overall", {}).get("mean_inference_time_s"))
    cpu_bench_t = _perf_time_with_fallback(cpu_perf, "CPU", cpu_eval_t)
    gpu_bench_t = _perf_time_with_fallback(gpu_perf, "GPU", gpu_eval_t)

    labels = [
        "Eval local CPU",
        "Eval colab GPU",
        "Bench local CPU",
        "Bench colab GPU",
    ]
    values = [cpu_eval_t, gpu_eval_t, cpu_bench_t, gpu_bench_t]
    colors = ["#4C78A8", "#54A24B", "#72B7B2", "#F58518"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Seconds per image")
    ax.set_title("Runtime comparison: Local CPU vs Colab GPU")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=15, ha="right")
    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{v:.3f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def run(cpu_dir: Path, gpu_dir: Path, out_dir: Path) -> int:
    cpu_eval = _load_eval(cpu_dir / "metrics.csv")
    gpu_eval = _load_eval(gpu_dir / "metrics.csv")
    cpu_perf = _load_perf(cpu_dir / "performance.csv")
    gpu_perf = _load_perf(gpu_dir / "performance.csv")

    out_dir.mkdir(parents=True, exist_ok=True)
    eval_plot = out_dir / "cpu_vs_gpu_eval_metrics.png"
    perf_plot = out_dir / "cpu_vs_gpu_runtime.png"
    summary_csv = out_dir / "cpu_vs_gpu_summary.csv"

    _plot_eval_comparison(eval_plot, cpu_eval, gpu_eval)
    _plot_perf_comparison(perf_plot, cpu_eval, gpu_eval, cpu_perf, gpu_perf)
    _save_summary_csv(summary_csv, cpu_eval, gpu_eval, cpu_perf, gpu_perf)

    print("Created:")
    print(f"  {eval_plot}")
    print(f"  {perf_plot}")
    print(f"  {summary_csv}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create Local CPU vs Colab GPU comparison plots from result CSVs."
    )
    parser.add_argument("--cpu_dir", type=Path, required=True)
    parser.add_argument("--gpu_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("results/comparison"))
    args = parser.parse_args()
    return run(args.cpu_dir, args.gpu_dir, args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
