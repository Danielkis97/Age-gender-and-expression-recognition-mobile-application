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
    labels = list(timing_ms.index)
    systems = list(timing_ms.columns)
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11.8, 5.8))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for idx, system in enumerate(systems):
        vals = timing_ms[system].to_numpy(dtype=float)
        ax.bar(x + (idx - 1) * width, vals, width=width, label=system, color=colors[idx], alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=14, ha="right")
    ax.set_ylabel("Milliseconds (log scale)")
    ax.set_yscale("log")
    ax.set_title("CPU vs Colab-GPU vs iPhone timing KPIs", fontsize=14, weight="bold")
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.legend(loc="upper right")
    fig.tight_layout()
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
    lines: list[str] = []
    lines.append("# CPU vs Colab-GPU vs iPhone (On-Device) Comparison")
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
    lines.append("## Quality Metrics (same schema)")
    lines.append("")
    lines.append(
        "| Scope | CPU Acc | GPU Acc | Mobile Acc | CPU Prec | GPU Prec | Mobile Prec | CPU Recall | GPU Recall | Mobile Recall | CPU F1 | GPU F1 | Mobile F1 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, row in quality_df.iterrows():
        lines.append(
            "| {scope} | {ca} | {ga} | {ma} | {cp} | {gp} | {mp} | {cr} | {gr} | {mr} | {cf} | {gf} | {mf} |".format(
                scope=row["scope"],
                ca=_to_markdown_value(row["CPU accuracy"]),
                ga=_to_markdown_value(row["GPU accuracy"]),
                ma=_to_markdown_value(row["Mobile accuracy"]),
                cp=_to_markdown_value(row["CPU precision"]),
                gp=_to_markdown_value(row["GPU precision"]),
                mp=_to_markdown_value(row["Mobile precision"]),
                cr=_to_markdown_value(row["CPU recall"]),
                gr=_to_markdown_value(row["GPU recall"]),
                mr=_to_markdown_value(row["Mobile recall"]),
                cf=_to_markdown_value(row["CPU f1"]),
                gf=_to_markdown_value(row["GPU f1"]),
                mf=_to_markdown_value(row["Mobile f1"]),
            )
        )

    lines.append("")
    lines.append("## Figure")
    lines.append("")
    lines.append(f"![Three-way timing KPI chart]({chart_rel_path})")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- CPU and Colab-GPU include full quality metrics from the DeepFace evaluation pipeline.")
    lines.append("- Mobile timing is measured on iPhone Safari with on-device TFLite browser inference.")
    lines.append("- Mobile quality fields stay `N/A` for this demo TFLite deployment path.")

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
