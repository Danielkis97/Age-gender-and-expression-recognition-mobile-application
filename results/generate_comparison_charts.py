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

    return {
        "cpu_metrics": pd.read_csv(cpu_dir / "metrics.csv", sep=";"),
        "gpu_metrics": pd.read_csv(gpu_dir / "metrics.csv", sep=";"),
        "cpu_eval": pd.read_csv(cpu_dir / "evaluation_results.csv", sep=";"),
        "gpu_eval": pd.read_csv(gpu_dir / "evaluation_results.csv", sep=";"),
    }


def timing_summary(series: pd.Series) -> dict[str, float]:
    sorted_values = series.sort_values().reset_index(drop=True)
    return {
        "Median": float(series.median()),
        "Mean excl. first image": float(series.iloc[1:].mean()),
        "Trimmed mean (drop min/max)": float(sorted_values.iloc[1:-1].mean()),
        "P90": float(series.quantile(0.90)),
    }


def save_quality_parity_panel(cpu_metrics: pd.DataFrame, gpu_metrics: pd.DataFrame, output_path: Path) -> None:
    cpu = cpu_metrics.set_index("scope").loc[SCOPE_ORDER, METRIC_ORDER]
    gpu = gpu_metrics.set_index("scope").loc[SCOPE_ORDER, METRIC_ORDER]
    delta = gpu - cpu
    abs_delta = np.abs(delta.to_numpy())
    max_abs = float(abs_delta.max())
    mean_abs = float(abs_delta.mean())
    same_cells = int(np.sum(abs_delta < 1e-12))
    total_cells = int(abs_delta.size)

    # If both runs are identical, show only one matrix to reduce visual clutter.
    values = cpu if np.allclose(cpu.to_numpy(), gpu.to_numpy(), atol=1e-12) else gpu
    matrix_title = "Quality metrics (single matrix view)"

    fig = plt.figure(figsize=(11.2, 5.0), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.9, 1.05], wspace=0.15)
    ax_matrix = fig.add_subplot(grid[0, 0])
    ax_info = fig.add_subplot(grid[0, 1])

    image = ax_matrix.imshow(values.to_numpy(), cmap="YlGnBu", aspect="auto", vmin=0.55, vmax=0.90)
    ax_matrix.set_title(matrix_title, fontsize=12.5, weight="bold")
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
        "Interpretation:\n"
        "Quality metrics are identical in this run.\n"
        "The chart therefore shows one clean matrix\n"
        "instead of duplicated CPU/GPU panels."
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
    cpu_metrics: pd.DataFrame, gpu_metrics: pd.DataFrame, cpu_eval: pd.DataFrame, gpu_eval: pd.DataFrame, output_path: Path
) -> None:
    cpu_time = cpu_eval["inference_seconds"]
    gpu_time = gpu_eval["inference_seconds"]

    timing_rows = {
        "Mean (all images)": float(cpu_metrics.loc[cpu_metrics["scope"] == "overall", "mean_inference_time_s"].iloc[0]),
        "Median": timing_summary(cpu_time)["Median"],
        "Mean excl. first image": timing_summary(cpu_time)["Mean excl. first image"],
        "Trimmed mean (drop min/max)": timing_summary(cpu_time)["Trimmed mean (drop min/max)"],
        "P90": timing_summary(cpu_time)["P90"],
    }
    timing_rows_gpu = {
        "Mean (all images)": float(gpu_metrics.loc[gpu_metrics["scope"] == "overall", "mean_inference_time_s"].iloc[0]),
        "Median": timing_summary(gpu_time)["Median"],
        "Mean excl. first image": timing_summary(gpu_time)["Mean excl. first image"],
        "Trimmed mean (drop min/max)": timing_summary(gpu_time)["Trimmed mean (drop min/max)"],
        "P90": timing_summary(gpu_time)["P90"],
    }
    timing_df = pd.DataFrame({"CPU": timing_rows, "Colab-GPU": timing_rows_gpu})

    fig, ax = plt.subplots(figsize=(11.4, 5.8))
    y = np.arange(len(timing_df))
    for i, (_, row) in enumerate(timing_df.iterrows()):
        low = min(row["CPU"], row["Colab-GPU"])
        high = max(row["CPU"], row["Colab-GPU"])
        ax.hlines(i, low, high, color="#999999", linewidth=2.2, zorder=1)

    ax.scatter(timing_df["CPU"], y, color="#1f77b4", s=85, label="CPU", zorder=3)
    ax.scatter(timing_df["Colab-GPU"], y, color="#ff7f0e", marker="s", s=85, label="Colab-GPU", zorder=3)

    for i, (_, row) in enumerate(timing_df.iterrows()):
        delta = row["Colab-GPU"] - row["CPU"]
        midpoint = (row["CPU"] + row["Colab-GPU"]) / 2
        ax.text(midpoint, i - 0.22, f"Delta {delta:+.3f}s", color="#444444", fontsize=9, ha="center")

    ax.set_yticks(y)
    ax.set_yticklabels(timing_df.index, fontsize=10.5)
    ax.set_xlabel("Seconds")
    ax.set_title("Timing KPI comparison (clean dumbbell view)", fontsize=14, weight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="upper left", frameon=True)
    ax.invert_yaxis()

    xmin = float(min(timing_df["CPU"].min(), timing_df["Colab-GPU"].min()))
    xmax = float(max(timing_df["CPU"].max(), timing_df["Colab-GPU"].max()))
    padding = (xmax - xmin) * 0.08
    ax.set_xlim(max(0.0, xmin - padding), xmax + padding)

    fig.subplots_adjust(bottom=0.21)
    fig.text(
        0.01,
        0.03,
        (
            "How to read: farther right means slower. "
            "Distance between blue/orange markers shows the CPU-GPU gap per KPI."
        ),
        ha="left",
        va="bottom",
        fontsize=9.2,
        color="#23303d",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f7f9fc", "edgecolor": "#d0d7de"},
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_latency_boxstrip(cpu_eval: pd.DataFrame, gpu_eval: pd.DataFrame, output_path: Path) -> None:
    cpu_time = cpu_eval["inference_seconds"].to_numpy()
    gpu_time = gpu_eval["inference_seconds"].to_numpy()
    datasets = [cpu_time, gpu_time]
    colors = ["#1f77b4", "#ff7f0e"]

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    box = ax.boxplot(datasets, widths=0.52, patch_artist=True, showfliers=False, medianprops={"color": "#222222"})

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.22)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)

    rng = np.random.default_rng(7)
    for i, (series, color) in enumerate(zip(datasets, colors), start=1):
        x = i + rng.normal(0.0, 0.045, size=len(series))
        ax.scatter(x, series, s=30, alpha=0.70, color=color, edgecolors="white", linewidth=0.4, zorder=3)

    # Highlight warm-up sample (first image).
    ax.scatter([1, 2], [cpu_time[0], gpu_time[0]], marker="*", s=170, color="#222222", zorder=4, label="Warm-up image")
    ax.set_yscale("log")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["CPU", "Colab-GPU"], fontsize=11)
    ax.set_ylabel("Inference time in seconds (log scale)")
    ax.set_title("Per-image latency distribution (box + points)", fontsize=14, weight="bold")
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.legend(loc="upper right")

    fig.subplots_adjust(bottom=0.26)
    fig.text(
        0.01,
        0.03,
        (
            "How to read: box = typical range (middle 50%), line in box = median.\n"
            "Dots outside whiskers = outliers. Lower is faster. Left=CPU, right=Colab-GPU."
        ),
        ha="left",
        va="bottom",
        fontsize=9.2,
        color="#23303d",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f7f9fc", "edgecolor": "#d0d7de"},
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_latency_sequence(cpu_eval: pd.DataFrame, gpu_eval: pd.DataFrame, output_path: Path) -> None:
    cpu_time = cpu_eval["inference_seconds"].to_numpy()
    gpu_time = gpu_eval["inference_seconds"].to_numpy()
    image_idx = np.arange(1, len(cpu_time) + 1)

    fig, ax = plt.subplots(figsize=(11.0, 4.9))
    ax.plot(image_idx, cpu_time, color="#1f77b4", marker="o", linewidth=1.8, markersize=4.5, label="CPU")
    ax.plot(image_idx, gpu_time, color="#ff7f0e", marker="s", linewidth=1.8, markersize=4.5, label="Colab-GPU")

    ax.scatter([1], [cpu_time[0]], color="#1f77b4", edgecolor="black", s=140, zorder=4)
    ax.scatter([1], [gpu_time[0]], color="#ff7f0e", marker="s", edgecolor="black", s=140, zorder=4)

    ax.set_yscale("log")
    y_min = float(min(cpu_time.min(), gpu_time.min()))
    y_max = float(max(cpu_time.max(), gpu_time.max()))
    tick_candidates = np.array([0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 20, 30, 40, 50], dtype=float)
    ticks = tick_candidates[(tick_candidates >= y_min * 0.9) & (tick_candidates <= y_max * 1.1)]
    if len(ticks) >= 4:
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{tick:g}s" for tick in ticks], fontsize=9.5)

    ax.set_xticks(image_idx)
    ax.set_xlabel("Image order")
    ax.set_ylabel("Inference time (log scale, seconds shown)")
    ax.set_title("Latency by image order (warm-up spike at first image)", fontsize=14, weight="bold")
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="upper right")
    ax.set_xlim(0.6, len(image_idx) + 0.4)
    ax.annotate(
        f"CPU cold start: {cpu_time[0]:.2f}s\nGPU cold start: {gpu_time[0]:.2f}s",
        xy=(1, max(cpu_time[0], gpu_time[0])),
        xytext=(2.2, max(cpu_time[0], gpu_time[0]) * 0.55),
        textcoords="data",
        fontsize=8.8,
        color="#23303d",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#f7f9fc", "edgecolor": "#d0d7de"},
        arrowprops={"arrowstyle": "->", "color": "#6b7280", "lw": 1.0},
    )

    steady_cpu_median = float(np.median(cpu_time[1:])) if len(cpu_time) > 1 else float(np.median(cpu_time))
    steady_gpu_median = float(np.median(gpu_time[1:])) if len(gpu_time) > 1 else float(np.median(gpu_time))
    summary = (
        f"Warm-up (image 1): CPU {cpu_time[0]:.2f}s | GPU {gpu_time[0]:.2f}s\n"
        f"Steady median (images 2-20): CPU {steady_cpu_median:.2f}s | GPU {steady_gpu_median:.2f}s"
    )
    fig.subplots_adjust(bottom=0.20)
    fig.text(
        0.012,
        0.025,
        (
            "How to read: y-axis is log(seconds). "
            "Image 1 is a cold-start point (model initialization), later points show steady inference."
        ),
        ha="left",
        va="bottom",
        fontsize=9.0,
        color="#23303d",
        bbox={"boxstyle": "round,pad=0.30", "facecolor": "#f7f9fc", "edgecolor": "#d0d7de"},
    )
    fig.text(
        0.012,
        0.068,
        summary,
        ha="left",
        va="bottom",
        fontsize=8.6,
        color="#23303d",
    )

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_image_delta_lollipop(cpu_eval: pd.DataFrame, gpu_eval: pd.DataFrame, output_path: Path) -> None:
    merged = cpu_eval[["filename", "inference_seconds"]].rename(columns={"inference_seconds": "cpu_seconds"}).merge(
        gpu_eval[["filename", "inference_seconds"]].rename(columns={"inference_seconds": "gpu_seconds"}),
        on="filename",
        how="inner",
    )
    merged["delta_cpu_minus_gpu"] = merged["cpu_seconds"] - merged["gpu_seconds"]
    merged["image_id"] = merged["filename"].str.extract(r"(Image_\d+)", expand=False).fillna(merged["filename"])
    merged = merged.sort_values("delta_cpu_minus_gpu", ascending=False).reset_index(drop=True)

    y = np.arange(len(merged))
    colors = np.where(merged["delta_cpu_minus_gpu"] >= 0, "#2ca02c", "#d62728")
    max_abs = float(np.abs(merged["delta_cpu_minus_gpu"]).max())

    fig, ax = plt.subplots(figsize=(10.6, 6.8))
    ax.hlines(y, 0, merged["delta_cpu_minus_gpu"], color=colors, linewidth=2.2, alpha=0.85)
    ax.scatter(merged["delta_cpu_minus_gpu"], y, color=colors, s=55, zorder=3)
    ax.axvline(0, color="#555555", linewidth=1.2)

    ax.set_yticks(y)
    ax.set_yticklabels(merged["image_id"], fontsize=9)
    ax.set_xlabel("Delta seconds (CPU - GPU)")
    ax.set_title("Image-wise latency delta (positive = GPU faster)", fontsize=14, weight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    ax.set_xlim(-(max_abs * 1.15), max_abs * 1.15)

    top_positive = merged.nlargest(2, "delta_cpu_minus_gpu")
    top_negative = merged.nsmallest(2, "delta_cpu_minus_gpu")
    for _, row in pd.concat([top_positive, top_negative]).iterrows():
        idx = int(merged.index[merged["image_id"] == row["image_id"]][0])
        ax.text(
            row["delta_cpu_minus_gpu"] + (0.08 if row["delta_cpu_minus_gpu"] >= 0 else -0.08),
            idx - 0.15,
            f"{row['delta_cpu_minus_gpu']:+.2f}s",
            fontsize=8.5,
            color="#333333",
            ha="left" if row["delta_cpu_minus_gpu"] >= 0 else "right",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "results" / "figures_cpu_vs_colab"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(repo_root)
    cpu_metrics = data["cpu_metrics"]
    gpu_metrics = data["gpu_metrics"]
    cpu_eval = data["cpu_eval"]
    gpu_eval = data["gpu_eval"]

    save_quality_parity_panel(cpu_metrics, gpu_metrics, output_dir / "01_quality_parity_panel.png")
    save_timing_dumbbell(cpu_metrics, gpu_metrics, cpu_eval, gpu_eval, output_dir / "02_timing_dumbbell_clean.png")
    save_latency_boxstrip(cpu_eval, gpu_eval, output_dir / "03_latency_distribution_boxstrip.png")
    save_latency_sequence(cpu_eval, gpu_eval, output_dir / "04_latency_by_image_order.png")
    save_image_delta_lollipop(cpu_eval, gpu_eval, output_dir / "05_image_delta_lollipop.png")

    print(f"Saved charts in: {output_dir}")


if __name__ == "__main__":
    main()
