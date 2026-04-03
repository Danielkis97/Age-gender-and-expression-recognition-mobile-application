from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCOPE_ORDER = ["overall", "gender", "emotion", "age"]
METRIC_ORDER = ["accuracy", "precision", "recall", "f1"]


def load_data(repo_root: Path) -> dict[str, pd.DataFrame]:
    cpu_dir = repo_root / "results" / "Results CPU PYCHARM"
    gpu_dir = repo_root / "results" / "RESULTS GPU TF Google Collab"
    mobile_dir = repo_root / "results" / "Results mobile metrics"
    return {
        "cpu_metrics": pd.read_csv(cpu_dir / "metrics.csv", sep=";"),
        "gpu_metrics": pd.read_csv(gpu_dir / "metrics.csv", sep=";"),
        "mobile_metrics": pd.read_csv(mobile_dir / "metrics.csv", sep=";"),
        "cpu_eval": pd.read_csv(cpu_dir / "evaluation_results.csv", sep=";"),
        "gpu_eval": pd.read_csv(gpu_dir / "evaluation_results.csv", sep=";"),
        "mobile_eval": pd.read_csv(mobile_dir / "evaluation_results.csv", sep=";"),
    }


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _timing_summary(series: pd.Series) -> dict[str, float]:
    s = _to_num(series).dropna().reset_index(drop=True)
    if len(s) == 0:
        return {
            "Median": float("nan"),
            "Mean excl. first image": float("nan"),
            "Trimmed mean (drop min/max)": float("nan"),
            "P90": float("nan"),
        }
    sorted_values = s.sort_values().reset_index(drop=True)
    steady = s.iloc[1:] if len(s) > 1 else s
    trimmed = sorted_values.iloc[1:-1] if len(sorted_values) > 2 else sorted_values
    return {
        "Median": float(s.median()),
        "Mean excl. first image": float(steady.mean()),
        "Trimmed mean (drop min/max)": float(trimmed.mean()),
        "P90": float(s.quantile(0.90)),
    }


def _extract_image_num(filename: pd.Series) -> pd.Series:
    out = filename.str.extract(r"Image_(\d+)", expand=False)
    return pd.to_numeric(out, errors="coerce")


def _aligned_table(cpu_eval: pd.DataFrame, gpu_eval: pd.DataFrame, mobile_eval: pd.DataFrame) -> pd.DataFrame:
    cpu = cpu_eval[["filename", "inference_seconds"]].copy().rename(columns={"inference_seconds": "cpu_s"})
    gpu = gpu_eval[["filename", "inference_seconds"]].copy().rename(columns={"inference_seconds": "gpu_s"})
    mob = mobile_eval[["filename", "inference_seconds"]].copy().rename(columns={"inference_seconds": "mobile_s"})

    merged = cpu.merge(gpu, on="filename", how="inner").merge(mob, on="filename", how="inner")
    merged["cpu_s"] = _to_num(merged["cpu_s"])
    merged["gpu_s"] = _to_num(merged["gpu_s"])
    merged["mobile_s"] = _to_num(merged["mobile_s"])
    merged = merged.dropna(subset=["cpu_s", "gpu_s", "mobile_s"]).copy()
    merged["img_num"] = _extract_image_num(merged["filename"])
    merged = merged.sort_values(["img_num", "filename"], na_position="last").reset_index(drop=True)
    merged["image_id"] = merged["filename"].str.extract(r"(Image_\d+)", expand=False).fillna(merged["filename"])
    return merged


def save_quality_parity_panel(cpu_metrics: pd.DataFrame, gpu_metrics: pd.DataFrame, mobile_metrics: pd.DataFrame, output_path: Path) -> None:
    cpu = cpu_metrics.set_index("scope").loc[SCOPE_ORDER, METRIC_ORDER]
    gpu = gpu_metrics.set_index("scope").loc[SCOPE_ORDER, METRIC_ORDER]

    delta = gpu - cpu
    abs_delta = np.abs(delta.to_numpy())
    max_abs = float(abs_delta.max())
    mean_abs = float(abs_delta.mean())
    same_cells = int(np.sum(abs_delta < 1e-12))
    total_cells = int(abs_delta.size)

    values = cpu if np.allclose(cpu.to_numpy(), gpu.to_numpy(), atol=1e-12) else gpu

    fig = plt.figure(figsize=(11.2, 5.0), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.9, 1.05], wspace=0.15)
    ax_matrix = fig.add_subplot(grid[0, 0])
    ax_info = fig.add_subplot(grid[0, 1])

    image = ax_matrix.imshow(values.to_numpy(), cmap="YlGnBu", aspect="auto", vmin=0.55, vmax=0.90)
    ax_matrix.set_title("Quality metrics (CPU vs Colab-GPU)", fontsize=12.5, weight="bold")
    ax_matrix.set_xticks(np.arange(len(values.columns)))
    ax_matrix.set_xticklabels([column.title() for column in values.columns], fontsize=10)
    ax_matrix.set_yticks(np.arange(len(values.index)))
    ax_matrix.set_yticklabels([index.title() for index in values.index], fontsize=10)

    vmin, vmax = 0.55, 0.90
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = float(values.iloc[row, col])
            normalized = (value - vmin) / (vmax - vmin)
            text_color = "#ffffff" if normalized >= 0.58 else "#1f2933"
            text_weight = "semibold" if normalized >= 0.58 else "normal"
            ax_matrix.text(
                col,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
                fontweight=text_weight,
            )

    fig.colorbar(image, ax=ax_matrix, fraction=0.046, pad=0.03, label="Metric value")

    ax_info.axis("off")
    summary_text = (
        "Delta summary (GPU - CPU)\n"
        f"- max |delta|: {max_abs:.4f}\n"
        f"- mean |delta|: {mean_abs:.4f}\n"
        f"- identical cells: {same_cells}/{total_cells}\n\n"
        "Mobile Edge quality is intentionally\n"
        "excluded here because the browser run\n"
        "is timing-only in this demo path."
    )
    ax_info.text(
        0.02,
        0.96,
        summary_text,
        va="top",
        ha="left",
        fontsize=10.2,
        color="#1f2933",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f3f6fa", "edgecolor": "#d0d7de"},
    )

    fig.suptitle("Quality overview", fontsize=15, weight="bold")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_timing_dumbbell(
    cpu_metrics: pd.DataFrame,
    gpu_metrics: pd.DataFrame,
    mobile_metrics: pd.DataFrame,
    cpu_eval: pd.DataFrame,
    gpu_eval: pd.DataFrame,
    mobile_eval: pd.DataFrame,
    output_path: Path,
) -> None:
    cpu_time = _to_num(cpu_eval["inference_seconds"]).dropna().reset_index(drop=True)
    gpu_time = _to_num(gpu_eval["inference_seconds"]).dropna().reset_index(drop=True)
    mob_time = _to_num(mobile_eval["inference_seconds"]).dropna().reset_index(drop=True)

    timing_rows = {
        "Mean (all images)": float(_to_num(cpu_metrics.loc[cpu_metrics["scope"] == "overall", "mean_inference_time_s"]).iloc[0]),
        "Median": _timing_summary(cpu_time)["Median"],
        "Mean excl. first image": _timing_summary(cpu_time)["Mean excl. first image"],
        "Trimmed mean (drop min/max)": _timing_summary(cpu_time)["Trimmed mean (drop min/max)"],
        "P90": _timing_summary(cpu_time)["P90"],
    }
    timing_rows_gpu = {
        "Mean (all images)": float(_to_num(gpu_metrics.loc[gpu_metrics["scope"] == "overall", "mean_inference_time_s"]).iloc[0]),
        "Median": _timing_summary(gpu_time)["Median"],
        "Mean excl. first image": _timing_summary(gpu_time)["Mean excl. first image"],
        "Trimmed mean (drop min/max)": _timing_summary(gpu_time)["Trimmed mean (drop min/max)"],
        "P90": _timing_summary(gpu_time)["P90"],
    }
    timing_rows_mobile = {
        "Mean (all images)": float(_to_num(mobile_metrics.loc[mobile_metrics["scope"] == "overall", "mean_inference_time_s"]).iloc[0]),
        "Median": _timing_summary(mob_time)["Median"],
        "Mean excl. first image": _timing_summary(mob_time)["Mean excl. first image"],
        "Trimmed mean (drop min/max)": _timing_summary(mob_time)["Trimmed mean (drop min/max)"],
        "P90": _timing_summary(mob_time)["P90"],
    }
    df = pd.DataFrame(
        {
            "CPU": timing_rows,
            "Colab-GPU": timing_rows_gpu,
            "Mobile Edge": timing_rows_mobile,
        }
    )

    fig, ax = plt.subplots(figsize=(12.0, 6.2))
    y = np.arange(len(df))
    for i, (_, row) in enumerate(df.iterrows()):
        low = float(np.nanmin(row.to_numpy(dtype=float)))
        high = float(np.nanmax(row.to_numpy(dtype=float)))
        ax.hlines(i, low, high, color="#9ca3af", linewidth=2.0, zorder=1)

    ax.scatter(df["CPU"], y, color="#1f77b4", s=78, label="CPU", zorder=3)
    ax.scatter(df["Colab-GPU"], y, color="#ff7f0e", marker="s", s=78, label="Colab-GPU", zorder=3)
    ax.scatter(df["Mobile Edge"], y, color="#2ca02c", marker="^", s=88, label="Mobile Edge", zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(df.index, fontsize=10.4)
    ax.set_xlabel("Seconds (log scale)")
    ax.set_xscale("log")
    ax.set_title("Timing KPI comparison (CPU vs Colab-GPU vs Mobile Edge)", fontsize=14, weight="bold")
    ax.grid(axis="x", alpha=0.28, which="both")
    ax.legend(loc="upper right", frameon=True)
    ax.invert_yaxis()

    fig.subplots_adjust(bottom=0.20)
    fig.text(
        0.01,
        0.03,
        (
            "How to read: farther right means slower. "
            "Lines show KPI range from fastest to slowest across the three systems."
        ),
        ha="left",
        va="bottom",
        fontsize=9.2,
        color="#23303d",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f7f9fc", "edgecolor": "#d0d7de"},
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_latency_boxstrip(cpu_eval: pd.DataFrame, gpu_eval: pd.DataFrame, mobile_eval: pd.DataFrame, output_path: Path) -> None:
    cpu_time = _to_num(cpu_eval["inference_seconds"]).dropna().to_numpy()
    gpu_time = _to_num(gpu_eval["inference_seconds"]).dropna().to_numpy()
    mob_time = _to_num(mobile_eval["inference_seconds"]).dropna().to_numpy()
    datasets = [cpu_time, gpu_time, mob_time]
    labels = ["CPU", "Colab-GPU", "Mobile Edge"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(10.8, 5.7))
    box = ax.boxplot(datasets, widths=0.55, patch_artist=True, showfliers=False, medianprops={"color": "#222222"})
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.22)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)

    rng = np.random.default_rng(7)
    for i, (series, color) in enumerate(zip(datasets, colors), start=1):
        x = i + rng.normal(0.0, 0.05, size=len(series))
        ax.scatter(x, series, s=28, alpha=0.72, color=color, edgecolors="white", linewidth=0.35, zorder=3)

    warmups = [cpu_time[0], gpu_time[0], mob_time[0]]
    ax.scatter([1, 2, 3], warmups, marker="*", s=170, color="#111827", zorder=4, label="Warm-up image")
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels, fontsize=10.6)
    ax.set_ylabel("Inference time in seconds (log scale)")
    ax.set_title("Per-image latency distribution (CPU / Colab-GPU / Mobile Edge)", fontsize=14, weight="bold")
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.legend(loc="upper right")

    fig.subplots_adjust(bottom=0.25)
    fig.text(
        0.01,
        0.03,
        (
            "How to read: box = middle 50%, line in box = median.\n"
            "Lower is faster. Mobile Edge points should appear at the lowest latency range."
        ),
        ha="left",
        va="bottom",
        fontsize=9.2,
        color="#23303d",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f7f9fc", "edgecolor": "#d0d7de"},
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_latency_sequence(cpu_eval: pd.DataFrame, gpu_eval: pd.DataFrame, mobile_eval: pd.DataFrame, output_path: Path) -> None:
    merged = _aligned_table(cpu_eval, gpu_eval, mobile_eval)
    image_idx = np.arange(1, len(merged) + 1)
    cpu = merged["cpu_s"].to_numpy()
    gpu = merged["gpu_s"].to_numpy()
    mobile = merged["mobile_s"].to_numpy()

    fig, ax = plt.subplots(figsize=(12.0, 5.2))
    ax.plot(image_idx, cpu, color="#1f77b4", marker="o", linewidth=1.8, markersize=4.3, label="CPU")
    ax.plot(image_idx, gpu, color="#ff7f0e", marker="s", linewidth=1.8, markersize=4.3, label="Colab-GPU")
    ax.plot(image_idx, mobile, color="#2ca02c", marker="^", linewidth=1.8, markersize=4.3, label="Mobile Edge")

    ax.set_yscale("log")
    ax.set_xticks(image_idx)
    ax.set_xlabel("Image order")
    ax.set_ylabel("Inference time (seconds, log scale)")
    ax.set_title("Latency by image order (CPU vs Colab-GPU vs Mobile Edge)", fontsize=14, weight="bold")
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="upper right")
    ax.set_xlim(0.6, len(image_idx) + 0.4)

    steady_cpu = float(np.median(cpu[1:])) if len(cpu) > 1 else float(np.median(cpu))
    steady_gpu = float(np.median(gpu[1:])) if len(gpu) > 1 else float(np.median(gpu))
    steady_mob = float(np.median(mobile[1:])) if len(mobile) > 1 else float(np.median(mobile))
    summary = (
        f"Warm-up (img1): CPU {cpu[0]:.2f}s | GPU {gpu[0]:.2f}s | Mobile {mobile[0]:.4f}s\n"
        f"Steady median (img2-20): CPU {steady_cpu:.2f}s | GPU {steady_gpu:.2f}s | Mobile {steady_mob:.4f}s"
    )

    fig.subplots_adjust(bottom=0.21)
    fig.text(
        0.012,
        0.025,
        "How to read: image 1 is cold-start, later points show steady inference behavior across all systems.",
        ha="left",
        va="bottom",
        fontsize=9.0,
        color="#23303d",
        bbox={"boxstyle": "round,pad=0.30", "facecolor": "#f7f9fc", "edgecolor": "#d0d7de"},
    )
    fig.text(0.012, 0.072, summary, ha="left", va="bottom", fontsize=8.6, color="#23303d")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_image_delta_lollipop(cpu_eval: pd.DataFrame, gpu_eval: pd.DataFrame, mobile_eval: pd.DataFrame, output_path: Path) -> None:
    merged = _aligned_table(cpu_eval, gpu_eval, mobile_eval)
    merged["delta_cpu_minus_gpu"] = merged["cpu_s"] - merged["gpu_s"]
    merged["delta_cpu_minus_mobile"] = merged["cpu_s"] - merged["mobile_s"]
    merged = merged.sort_values("delta_cpu_minus_mobile", ascending=False).reset_index(drop=True)
    y = np.arange(len(merged))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 7.0), sharey=True)

    colors_gpu = np.where(merged["delta_cpu_minus_gpu"] >= 0, "#2ca02c", "#d62728")
    ax1.hlines(y, 0, merged["delta_cpu_minus_gpu"], color=colors_gpu, linewidth=2.0, alpha=0.85)
    ax1.scatter(merged["delta_cpu_minus_gpu"], y, color=colors_gpu, s=45, zorder=3)
    ax1.axvline(0, color="#4b5563", linewidth=1.1)
    ax1.set_title("CPU - Colab-GPU")
    ax1.set_xlabel("Delta seconds")
    ax1.grid(axis="x", alpha=0.25)

    colors_mobile = np.where(merged["delta_cpu_minus_mobile"] >= 0, "#2ca02c", "#d62728")
    ax2.hlines(y, 0, merged["delta_cpu_minus_mobile"], color=colors_mobile, linewidth=2.0, alpha=0.85)
    ax2.scatter(merged["delta_cpu_minus_mobile"], y, color=colors_mobile, s=45, zorder=3)
    ax2.axvline(0, color="#4b5563", linewidth=1.1)
    ax2.set_title("CPU - Mobile Edge")
    ax2.set_xlabel("Delta seconds")
    ax2.grid(axis="x", alpha=0.25)

    # Add small per-bar labels so each delta is readable.
    gpu_vals = merged["delta_cpu_minus_gpu"].to_numpy(dtype=float)
    mob_vals = merged["delta_cpu_minus_mobile"].to_numpy(dtype=float)
    xpad_gpu = (np.nanmax(gpu_vals) - np.nanmin(gpu_vals)) * 0.015 if len(gpu_vals) else 0.03
    xpad_mob = (np.nanmax(mob_vals) - np.nanmin(mob_vals)) * 0.010 if len(mob_vals) else 0.03

    for yi, v in enumerate(gpu_vals):
        ha = "left" if v >= 0 else "right"
        ax1.text(v + (xpad_gpu if v >= 0 else -xpad_gpu), yi, f"{v:+.2f}s", va="center", ha=ha, fontsize=7.2, color="#374151")

    for yi, v in enumerate(mob_vals):
        ha = "left" if v >= 0 else "right"
        ax2.text(v + (xpad_mob if v >= 0 else -xpad_mob), yi, f"{v:+.2f}s", va="center", ha=ha, fontsize=7.2, color="#374151")

    ax1.set_yticks(y)
    ax1.set_yticklabels(merged["image_id"], fontsize=8.7)
    ax1.invert_yaxis()
    fig.suptitle("Image-wise latency delta vs CPU baseline", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_three_way_speedup_panel(
    cpu_metrics: pd.DataFrame,
    gpu_metrics: pd.DataFrame,
    mobile_metrics: pd.DataFrame,
    cpu_eval: pd.DataFrame,
    gpu_eval: pd.DataFrame,
    mobile_eval: pd.DataFrame,
    output_path: Path,
) -> None:
    cpu_mean = float(_to_num(cpu_metrics.loc[cpu_metrics["scope"] == "overall", "mean_inference_time_s"]).iloc[0])
    gpu_mean = float(_to_num(gpu_metrics.loc[gpu_metrics["scope"] == "overall", "mean_inference_time_s"]).iloc[0])
    mob_mean = float(_to_num(mobile_metrics.loc[mobile_metrics["scope"] == "overall", "mean_inference_time_s"]).iloc[0])

    cpu_med = _timing_summary(cpu_eval["inference_seconds"])["Median"]
    gpu_med = _timing_summary(gpu_eval["inference_seconds"])["Median"]
    mob_med = _timing_summary(mobile_eval["inference_seconds"])["Median"]

    cpu_p90 = _timing_summary(cpu_eval["inference_seconds"])["P90"]
    gpu_p90 = _timing_summary(gpu_eval["inference_seconds"])["P90"]
    mob_p90 = _timing_summary(mobile_eval["inference_seconds"])["P90"]

    categories = ["Mean", "Median", "P90"]
    cpu_ms = np.array([cpu_mean * 1000.0, cpu_med * 1000.0, cpu_p90 * 1000.0], dtype=float)
    gpu_ms = np.array([gpu_mean * 1000.0, gpu_med * 1000.0, gpu_p90 * 1000.0], dtype=float)
    mob_ms = np.array([mob_mean * 1000.0, mob_med * 1000.0, mob_p90 * 1000.0], dtype=float)

    x = np.arange(len(categories))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.6, 5.4))
    bars_cpu = ax.bar(x - width, cpu_ms, width, label="CPU", color="#1f77b4", alpha=0.92)
    bars_gpu = ax.bar(x, gpu_ms, width, label="Colab-GPU", color="#ff7f0e", alpha=0.92)
    bars_mob = ax.bar(x + width, mob_ms, width, label="Mobile Edge", color="#2ca02c", alpha=0.92)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_yscale("log")
    ax.set_ylabel("Milliseconds (log scale)")
    ax.set_title("Three-way timing snapshot (values shown)", fontsize=13.5, weight="bold")
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.legend(loc="upper right")

    def _add_labels(bars):
        for b in bars:
            h = float(b.get_height())
            ax.text(
                b.get_x() + b.get_width() / 2,
                h * 1.08,
                f"{h:.1f} ms",
                ha="center",
                va="bottom",
                fontsize=8.0,
                color="#374151",
                rotation=0,
            )

    _add_labels(bars_cpu)
    _add_labels(bars_gpu)
    _add_labels(bars_mob)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _generate_suite(
    output_dir: Path,
    cpu_metrics: pd.DataFrame,
    gpu_metrics: pd.DataFrame,
    mobile_metrics: pd.DataFrame,
    cpu_eval: pd.DataFrame,
    gpu_eval: pd.DataFrame,
    mobile_eval: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_quality_parity_panel(cpu_metrics, gpu_metrics, mobile_metrics, output_dir / "01_quality_parity_panel.png")
    save_timing_dumbbell(
        cpu_metrics,
        gpu_metrics,
        mobile_metrics,
        cpu_eval,
        gpu_eval,
        mobile_eval,
        output_dir / "02_timing_dumbbell_clean.png",
    )
    save_latency_boxstrip(cpu_eval, gpu_eval, mobile_eval, output_dir / "03_latency_distribution_boxstrip.png")
    save_latency_sequence(cpu_eval, gpu_eval, mobile_eval, output_dir / "04_latency_by_image_order.png")
    save_image_delta_lollipop(cpu_eval, gpu_eval, mobile_eval, output_dir / "05_image_delta_lollipop.png")
    save_three_way_speedup_panel(
        cpu_metrics,
        gpu_metrics,
        mobile_metrics,
        cpu_eval,
        gpu_eval,
        mobile_eval,
        output_dir / "06_three_way_speedup_panel.png",
    )


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    repo_root = Path(__file__).resolve().parents[1]

    data = load_data(repo_root)
    cpu_metrics = data["cpu_metrics"]
    gpu_metrics = data["gpu_metrics"]
    mobile_metrics = data["mobile_metrics"]
    cpu_eval = data["cpu_eval"]
    gpu_eval = data["gpu_eval"]
    mobile_eval = data["mobile_eval"]

    out_cpu_gpu = repo_root / "results" / "figures_cpu_vs_colab"
    out_three_way = repo_root / "results" / "figures_three_way"
    _generate_suite(out_cpu_gpu, cpu_metrics, gpu_metrics, mobile_metrics, cpu_eval, gpu_eval, mobile_eval)
    _generate_suite(out_three_way, cpu_metrics, gpu_metrics, mobile_metrics, cpu_eval, gpu_eval, mobile_eval)
    print(f"Saved charts in: {out_cpu_gpu}")
    print(f"Saved charts in: {out_three_way}")


if __name__ == "__main__":
    main()
