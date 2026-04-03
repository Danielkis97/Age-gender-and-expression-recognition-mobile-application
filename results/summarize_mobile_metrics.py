from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def summarize(csv_path: Path) -> dict[str, dict[str, float | int | str]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    by_run: dict[str, list[float]] = defaultdict(list)
    run_device: dict[str, str] = {}
    run_input: dict[str, str] = {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            run_id = (row.get("run_id") or "").strip()
            if not run_id:
                continue
            try:
                latency = float(row.get("latency_ms", ""))
            except ValueError:
                continue
            by_run[run_id].append(latency)
            run_device[run_id] = (row.get("device_label") or "").strip()
            run_input[run_id] = (row.get("input_source") or "").strip()

    out: dict[str, dict[str, float | int | str]] = {}
    for run_id, values in by_run.items():
        if not values:
            continue
        sorted_vals = sorted(values)
        out[run_id] = {
            "device": run_device.get(run_id, ""),
            "input": run_input.get(run_id, ""),
            "samples": len(sorted_vals),
            "mean_ms": statistics.mean(sorted_vals),
            "median_ms": statistics.median(sorted_vals),
            "p90_ms": _percentile(sorted_vals, 0.9),
            "min_ms": sorted_vals[0],
            "max_ms": sorted_vals[-1],
            "std_ms": statistics.stdev(sorted_vals) if len(sorted_vals) > 1 else 0.0,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize iPhone browser latency metrics from CSV.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/Results mobile metrics/mobile_browser_metrics.csv"),
        help="Path to metrics CSV produced by mobile_eval_server.py",
    )
    args = parser.parse_args()

    stats = summarize(args.csv)
    if not stats:
        print(f"No valid run rows found in: {args.csv.resolve()}")
        return 0

    print(f"Source: {args.csv.resolve()}")
    print("-" * 92)
    print("run_id | device | input | n | mean_ms | median_ms | p90_ms | min_ms | max_ms | std_ms")
    print("-" * 92)
    for run_id in sorted(stats):
        s = stats[run_id]
        print(
            f"{run_id} | {s['device']} | {s['input']} | {s['samples']} | "
            f"{s['mean_ms']:.2f} | {s['median_ms']:.2f} | {s['p90_ms']:.2f} | "
            f"{s['min_ms']:.2f} | {s['max_ms']:.2f} | {s['std_ms']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
