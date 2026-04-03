from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCOPE_ORDER = ["overall", "gender", "emotion", "age"]
QUALITY_METRICS = ["accuracy", "precision", "recall", "f1"]


def _safe_float(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.iloc[0]) if len(values) else float("nan")


def _time_kpis(eval_df: pd.DataFrame, mean_all_s: float, total_s: float) -> dict[str, float]:
    times = pd.to_numeric(eval_df["inference_seconds"], errors="coerce").dropna()
    if len(times) == 0:
        return {
            "Mean (all images)": mean_all_s,
            "Median": float("nan"),
            "Mean excl. first image": float("nan"),
            "P90": float("nan"),
            "Total runtime (20 images)": total_s,
        }
    steady = times.iloc[1:] if len(times) > 1 else times
    return {
        "Mean (all images)": mean_all_s,
        "Median": float(times.median()),
        "Mean excl. first image": float(steady.mean()),
        "P90": float(times.quantile(0.90)),
        "Total runtime (20 images)": total_s,
    }


def _load_inputs(repo_root: Path) -> dict[str, pd.DataFrame]:
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


def _build_timing_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cpu_metrics = data["cpu_metrics"]
    gpu_metrics = data["gpu_metrics"]
    mobile_metrics = data["mobile_metrics"]

    cpu_mean = _safe_float(cpu_metrics.loc[cpu_metrics["scope"] == "overall", "mean_inference_time_s"])
    gpu_mean = _safe_float(gpu_metrics.loc[gpu_metrics["scope"] == "overall", "mean_inference_time_s"])
    mob_mean = _safe_float(mobile_metrics.loc[mobile_metrics["scope"] == "overall", "mean_inference_time_s"])

    cpu_total = _safe_float(cpu_metrics.loc[cpu_metrics["scope"] == "overall", "total_runtime_s"])
    gpu_total = _safe_float(gpu_metrics.loc[gpu_metrics["scope"] == "overall", "total_runtime_s"])
    mob_total = _safe_float(mobile_metrics.loc[mobile_metrics["scope"] == "overall", "total_runtime_s"])

    rows = {
        "CPU (PyCharm)": _time_kpis(data["cpu_eval"], cpu_mean, cpu_total),
        "Colab-GPU": _time_kpis(data["gpu_eval"], gpu_mean, gpu_total),
        "iPhone Safari (on-device)": _time_kpis(data["mobile_eval"], mob_mean, mob_total),
    }
    df = pd.DataFrame(rows)
    return df


def _build_quality_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out_rows = []
    cpu = data["cpu_metrics"].set_index("scope")
    gpu = data["gpu_metrics"].set_index("scope")
    mob = data["mobile_metrics"].set_index("scope")

    for scope in SCOPE_ORDER:
        row = {"scope": scope}
        for metric in QUALITY_METRICS:
            row[f"CPU {metric}"] = _safe_float(pd.Series([cpu.loc[scope, metric]])) if scope in cpu.index else np.nan
            row[f"GPU {metric}"] = _safe_float(pd.Series([gpu.loc[scope, metric]])) if scope in gpu.index else np.nan
            row[f"Mobile {metric}"] = _safe_float(pd.Series([mob.loc[scope, metric]])) if scope in mob.index else np.nan
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def _save_timing_chart(timing_s: pd.DataFrame, output_path: Path) -> None:
    timing_ms = timing_s * 1000.0
    systems = list(timing_ms.columns)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    width = 0.24
    width_total = 0.16

    per_image_labels = ["Mean (all images)", "Median", "Mean excl. first image", "P90"]
    per_image_df = timing_ms.loc[per_image_labels]
    total_label = "Total runtime (20 images)"
    total_df = timing_ms.loc[[total_label]]

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.8),
        gridspec_kw={"width_ratios": [2.2, 1.1]},
        constrained_layout=True,
    )

    x1 = np.arange(len(per_image_df.index))
    for idx, system in enumerate(systems):
        vals = per_image_df[system].to_numpy(dtype=float)
        bars = ax1.bar(x1 + (idx - 1) * width, vals, width=width, label=system, color=colors[idx], alpha=0.92)
        for b in bars:
            h = float(b.get_height())
            ax1.text(
                b.get_x() + b.get_width() / 2,
                h * 1.04,
                f"{h:,.1f} ms",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#374151",
                rotation=90,
            )

    ax1.set_xticks(x1)
    ax1.set_xticklabels(per_image_labels, rotation=14, ha="right")
    ax1.set_yscale("log")
    ax1.set_ylabel("Milliseconds (log scale)")
    ax1.set_title("Per-image timing KPIs")
    ax1.grid(axis="y", alpha=0.25, which="both")

    x2 = np.arange(1)
    for idx, system in enumerate(systems):
        val = float(total_df[system].iloc[0])
        bars = ax2.bar(x2 + (idx - 1) * width_total, [val], width=width_total, color=colors[idx], alpha=0.92, label=system)
        b = bars[0]
        ax2.text(
            b.get_x() + b.get_width() / 2,
            val * 1.04,
            f"{val:,.1f} ms\n({val/1000.0:.2f} s)",
            ha="center",
            va="bottom",
            fontsize=7.4,
            color="#374151",
            rotation=0,
        )

    ax2.set_xticks(x2)
    ax2.set_xticklabels([total_label], rotation=10, ha="right")
    ax2.set_yscale("log")
    ax2.set_title("Total runtime")
    ax2.grid(axis="y", alpha=0.25, which="both")
    max_total = float(np.nanmax(total_df.to_numpy(dtype=float)))
    ax2.set_ylim(50.0, max_total * 2.2)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("CPU vs Colab-GPU vs iPhone timing KPIs (values annotated)", fontsize=14, weight="bold", y=1.07)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _to_markdown_value(v: float) -> str:
    if pd.isna(v):
        return "N/A"
    return f"{v:.4f}"


def _write_markdown(
    timing_s: pd.DataFrame,
    quality_df: pd.DataFrame,
    output_path: Path,
    chart_rel_path: str,
) -> None:
    timing_ms = timing_s * 1000.0
    cpu_mean = float(timing_ms.loc["Mean (all images)", "CPU (PyCharm)"])
    gpu_mean = float(timing_ms.loc["Mean (all images)", "Colab-GPU"])
    mobile_mean = float(timing_ms.loc["Mean (all images)", "iPhone Safari (on-device)"])

    lines: list[str] = []
    lines.append("# CPU vs Colab-GPU vs iPhone (On-Device) Comparison")
    lines.append("")
    lines.append("## Quick Takeaways")
    lines.append("")
    lines.append(f"- Mean latency: **CPU {cpu_mean:.2f} ms**, **Colab-GPU {gpu_mean:.2f} ms**, **iPhone {mobile_mean:.2f} ms**.")
    lines.append("- This three-way view is intended for deployment timing comparison across environments.")
    lines.append("- CPU and Colab-GPU keep full quality metrics from the DeepFace evaluation pipeline.")
    lines.append("- Mobile Edge run is timing-focused in this demo path, so quality fields are reported as `N/A`.")
    lines.append("")
    lines.append("## Runtime Metrics")
    lines.append("")
    lines.append("| Metric | CPU (ms) | Colab-GPU (ms) | iPhone Safari (ms) |")
    lines.append("|---|---:|---:|---:|")
    for metric in timing_ms.index:
        c = timing_ms.loc[metric, "CPU (PyCharm)"]
        g = timing_ms.loc[metric, "Colab-GPU"]
        m = timing_ms.loc[metric, "iPhone Safari (on-device)"]
        lines.append(f"| {metric} | {c:.2f} | {g:.2f} | {m:.2f} |")

    lines.append("")
    lines.append("## Quality Metrics (CPU/GPU comparable)")
    lines.append("")
    lines.append("| Scope | CPU Accuracy | GPU Accuracy | Delta | CPU F1 | GPU F1 | Delta |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, row in quality_df.iterrows():
        cpu_acc = row["CPU accuracy"]
        gpu_acc = row["GPU accuracy"]
        cpu_f1 = row["CPU f1"]
        gpu_f1 = row["GPU f1"]
        d_acc = (gpu_acc - cpu_acc) if (not pd.isna(cpu_acc) and not pd.isna(gpu_acc)) else np.nan
        d_f1 = (gpu_f1 - cpu_f1) if (not pd.isna(cpu_f1) and not pd.isna(gpu_f1)) else np.nan
        lines.append(
            "| {scope} | {ca} | {ga} | {da} | {cf} | {gf} | {df} |".format(
                scope=row["scope"],
                ca=_to_markdown_value(cpu_acc),
                ga=_to_markdown_value(gpu_acc),
                da=_to_markdown_value(d_acc),
                cf=_to_markdown_value(cpu_f1),
                gf=_to_markdown_value(gpu_f1),
                df=_to_markdown_value(d_f1),
            )
        )

    lines.append("")
    lines.append("## Figure")
    lines.append("")
    lines.append(f"![Three-way timing KPI chart]({chart_rel_path})")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Mobile timing is measured on iPhone Safari with on-device TFLite browser inference.")
    lines.append("- Quality metrics here are intentionally CPU/GPU-only, because the mobile browser run")
    lines.append("  in this demo path does not emit equivalent label predictions.")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    repo_root = Path(__file__).resolve().parents[1]
    out_fig_dir = repo_root / "results" / "figures_three_way"
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    data = _load_inputs(repo_root)
    timing_s = _build_timing_table(data)
    quality_df = _build_quality_table(data)

    chart_path = out_fig_dir / "01_timing_kpis_three_way.png"
    _save_timing_chart(timing_s, chart_path)

    md_path = repo_root / "results" / "comparison_cpu_gpu_mobile.md"
    _write_markdown(
        timing_s=timing_s,
        quality_df=quality_df,
        output_path=md_path,
        chart_rel_path="figures_three_way/01_timing_kpis_three_way.png",
    )
    print(f"Saved: {md_path}")
    print(f"Saved: {chart_path}")


if __name__ == "__main__":
    main()
