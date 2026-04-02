# Timing harness — CPU run is a subprocess so TF really starts without CUDA.
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_RESULTS_CSV_ENCODING = "utf-8-sig"
_RESULTS_CSV_DELIMITER = ";"


def _sanitize_csv_cell(v):
    if isinstance(v, str):
        return v.replace("\u2014", "-")
    return v


def _sanitize_row_dict(d: dict) -> dict:
    return {k: _sanitize_csv_cell(v) for k, v in d.items()}


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _first_face_image(images_dir: Path) -> Path | None:
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            return p
    return None


def bench_once(image_path: Path, n_runs: int) -> dict:
    import cv2

    from utils.deepface_predict import predict_face_region
    from utils.face_detect import detect_faces_bgr, load_face_cascade

    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        device_label = "GPU" if gpus else "CPU"
    except Exception:
        device_label = "unknown"

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(str(image_path))

    cascade = load_face_cascade()
    faces = detect_faces_bgr(frame, cascade)
    if not faces:
        raise RuntimeError("No face in benchmark image — pick another photo.")

    x, y, w, h = faces[0]
    y0 = max(0, y - int(h * 0.15))
    y1 = min(frame.shape[0], y + h + int(h * 0.15))
    x0 = max(0, x - int(w * 0.15))
    x1 = min(frame.shape[1], x + w + int(w * 0.15))
    crop = frame[y0:y1, x0:x1]

    predict_face_region(crop, enforce_detection=False)
    predict_face_region(crop, enforce_detection=False)  # warm up — first calls are noisy

    samples = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        predict_face_region(crop, enforce_detection=False)
        samples.append(time.perf_counter() - t0)

    mean_s = statistics.mean(samples)
    std_s = statistics.stdev(samples) if len(samples) > 1 else 0.0
    total_s = sum(samples)
    min_s = min(samples)
    max_s = max(samples)
    return {
        "device_label": device_label,
        "mean_s": mean_s,
        "std_s": std_s,
        "total_s": total_s,
        "min_s": min_s,
        "max_s": max_s,
        "n_runs": n_runs,
    }


def _isolated_bench(mode: str, image_path: Path, n_runs: int) -> dict:
    root = _project_root()
    cfg = {
        "root": str(root.resolve()),
        "img": str(image_path.resolve()),
        "n": n_runs,
    }
    payload = json.dumps(cfg)
    # fresh interpreter + env so CUDA_VISIBLE_DEVICES actually sticks
    code = (
        "import json,sys; "
        f"c=json.loads({repr(payload)}); "
        "sys.path.insert(0,c['root']); "
        "from pathlib import Path; "
        "from performance import bench_once; "
        "print(json.dumps(bench_once(Path(c['img']), c['n'])))"
    )
    env = os.environ.copy()
    if mode == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(root),
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or "worker failed")
    for line in reversed(proc.stdout.splitlines()):
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                continue
    raise RuntimeError("no json in worker output: " + proc.stdout[:500])


def _collect_images(images_dir: Path, max_images: int | None) -> list[Path]:
    paths: list[Path] = []
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(p)
    if max_images is not None:
        paths = paths[:max_images]
    return paths


def _bench_images_worker(image_paths: list[str], iterations: int) -> dict:
    """
    Worker entrypoint for subprocess benchmarking.

    Notes:
    - Face detection happens inside the worker (cropping is required for DeepFace).
    - Only DeepFace inference is timed; face detection/cropping is done before timing.
    """
    import cv2

    from utils.deepface_predict import predict_face_region
    from utils.face_detect import detect_faces_bgr, load_face_cascade

    cascade = load_face_cascade()

    # Precompute face crops so the timing focuses on DeepFace inference.
    crops: list = []
    used_files: list[str] = []
    for p in image_paths:
        frame = cv2.imread(p)
        if frame is None:
            continue
        faces = detect_faces_bgr(frame, cascade)
        if not faces:
            continue
        x, y, w, h = faces[0]
        y0 = max(0, y - int(h * 0.15))
        y1 = min(frame.shape[0], y + h + int(h * 0.15))
        x0 = max(0, x - int(w * 0.15))
        x1 = min(frame.shape[1], x + w + int(w * 0.15))
        crop = frame[y0:y1, x0:x1]

        # Warm-up to reduce cold-start noise (not included in timing).
        predict_face_region(crop, enforce_detection=False)

        crops.append(crop)
        used_files.append(Path(p).name)

    samples: list[float] = []
    failures = 0
    for _ in range(iterations):
        for crop in crops:
            t0 = time.perf_counter()
            pred = predict_face_region(crop, enforce_detection=False)
            _ = pred  # prediction output is not needed for timing
            if pred is None:
                failures += 1
            samples.append(time.perf_counter() - t0)

    if not samples:
        return {
            "mean_s": 0.0,
            "std_s": 0.0,
            "total_s": 0.0,
            "min_s": 0.0,
            "max_s": 0.0,
            "iterations": iterations,
            "n_inferences": 0,
            "n_images_used": len(crops),
            "n_failures": failures,
        }

    mean_s = statistics.mean(samples)
    std_s = statistics.stdev(samples) if len(samples) > 1 else 0.0
    total_s = sum(samples)

    return {
        "mean_s": mean_s,
        "std_s": std_s,
        "total_s": total_s,
        "min_s": min(samples),
        "max_s": max(samples),
        "iterations": iterations,
        "n_inferences": len(samples),
        "n_images_used": len(crops),
        "n_failures": failures,
    }


def _isolated_bench_many(
    mode: str,
    image_paths: list[Path],
    iterations: int,
) -> dict:
    root = _project_root()
    payload = json.dumps(
        {
            "root": str(root.resolve()),
            "paths": [str(p.resolve()) for p in image_paths],
            "iterations": iterations,
        }
    )

    code = (
        "import json,sys; "
        f"c=json.loads({repr(payload)}); "
        "sys.path.insert(0,c['root']); "
        "from performance import _bench_images_worker; "
        "print(json.dumps(_bench_images_worker(c['paths'], c['iterations'])))"
    )

    env = os.environ.copy()
    if mode == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)

    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(root),
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or "worker failed")

    for line in reversed(proc.stdout.splitlines()):
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                continue

    raise RuntimeError("no json in worker output: " + proc.stdout[:500])


def run_benchmark(
    images_dir: Path,
    iterations: int,
    out_csv: Path = Path("results/performance.csv"),
    console: Console | None = None,
    max_images: int = 20,
    plot_path: Path = Path("results/performance_plot.png"),
    make_plot: bool = True,
    tflite_out_csv: Path = Path("results/tflite_performance.csv"),
    tflite_model_path: Path = Path("models/model.tflite"),
) -> int:
    """
    Interactive CLI benchmarking for local DeepFace inference.

    Output:
    - Rich summary (CPU/GPU/Edge)
    - CSV at `results/performance.csv`

    Notes:
    - "Edge" is treated as CPU-based edge execution (same machine), labeled separately.
    - GPU is real when TensorFlow detects one, otherwise simulated (cpu_time / 3).
    - TFLite is deployment/performance-only (demo model): measured if available, otherwise estimated.
    """
    con = console or Console()

    if not images_dir.is_dir():
        con.print(f"[red]Not a directory:[/red] {images_dir}")
        return 1

    image_paths = _collect_images(images_dir, max_images=max_images)
    if not image_paths:
        con.print(f"[red]No images found in:[/red] {images_dir}")
        return 1

    con.print(Panel.fit(f"[bold]Benchmark images:[/bold] {len(image_paths)}", style="blue"))

    cpu_stats: dict | None = None
    gpu_stats: dict | None = None
    has_gpu = False

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=con,
    ) as prog:
        t_cpu = prog.add_task("[yellow]Running CPU test...[/yellow]", total=1)
        con.print("[dim]Measuring DeepFace inference on CPU (CUDA hidden)...[/dim]")
        try:
            cpu_stats = _isolated_bench_many("cpu", image_paths, iterations)
        except Exception as e:
            con.print(f"[red]CPU benchmark failed:[/red] {e}")
            return 1
        prog.advance(t_cpu)

        # Determine GPU availability in the parent (for labeling & simulation decision).
        try:
            import tensorflow as tf

            has_gpu = bool(tf.config.list_physical_devices("GPU"))
        except Exception:
            has_gpu = False

        t_gpu = prog.add_task("[green]Running GPU test...[/green]", total=1)
        if has_gpu:
            con.print("[dim]Measuring DeepFace inference on the real GPU...[/dim]")
            try:
                gpu_stats = _isolated_bench_many("gpu", image_paths, iterations)
            except Exception as e:
                con.print(f"[yellow]GPU benchmark failed; simulating GPU time:[/yellow] {e}")
                gpu_stats = None
        prog.advance(t_gpu)

        t_edge = prog.add_task("[cyan]Preparing Edge results...[/cyan]", total=1)
        prog.advance(t_edge)

    if cpu_stats is None:
        return 1

    cpu_mean = float(cpu_stats.get("mean_s", 0.0))
    cpu_std = float(cpu_stats.get("std_s", 0.0))

    if has_gpu and gpu_stats is not None:
        gpu_mean = float(gpu_stats.get("mean_s", 0.0))
        gpu_std = float(gpu_stats.get("std_s", 0.0))
        gpu_notes = (
            f"Real GPU (measured, {gpu_stats.get('n_images_used', '?')} images used)"
        )
    else:
        # Required simulation: 3x faster than CPU.
        gpu_mean = cpu_mean / 3.0 if cpu_mean > 0 else 0.0
        gpu_std = cpu_std / 3.0 if cpu_std > 0 else 0.0
        gpu_notes = "GPU simulated (cpu/3)"

    edge_mean = cpu_mean
    edge_std = cpu_std
    edge_notes = "Edge device (same CPU, CPU-based edge execution)"

    # TFLite timing (deployment-only). If it fails, we keep the comparison usable by estimating.
    con.print("[magenta]Running TFLite deployment demo…[/magenta]")
    tflite_mean = cpu_mean / 2.5 if cpu_mean > 0 else 0.0
    tflite_std = cpu_std / 2.5 if cpu_std > 0 else 0.0
    tflite_total_s = float(cpu_stats.get("total_s", 0.0)) / 2.5
    tflite_notes = "Estimated mobile optimization (TFLite faster than CPU)"
    try:
        from tflite_inference import measure_tflite_performance

        tflite_stats = measure_tflite_performance(
            images_dir=images_dir,
            iterations=iterations,
            out_csv=tflite_out_csv,
            model_path=tflite_model_path,
            console=con,
            max_images=max_images,
        )
        tflite_mean = float(tflite_stats.get("mean_s", tflite_mean))
        tflite_std = float(tflite_stats.get("std_s", tflite_std))
        tflite_total_s = float(tflite_stats.get("total_s", tflite_total_s))
        used = tflite_stats.get("n_images_used", "?")
        tflite_notes = f"Measured mobile optimization (TFLite demo model, {used} images used)"
    except Exception as e:
        con.print(f"[yellow]TFLite timing failed; estimating instead:[/yellow] {e}")
        try:
            # Ensure the required output file exists even when timing fails.
            tflite_est = {
                "device": "TFLite",
                "model_path": str(tflite_model_path),
                "time_per_image_s": round(tflite_mean, 6),
                "std_s": round(tflite_std, 6),
                "min_time_s": round(tflite_mean, 6),
                "max_time_s": round(tflite_mean, 6),
                "total_runtime_s": round(float(tflite_total_s), 6),
                "n_runs": iterations,
                "n_images_used": len(image_paths),
                "notes": "estimated mobile optimization (TFLite timing failed)",
            }
            tflite_out_csv.parent.mkdir(parents=True, exist_ok=True)
            with tflite_out_csv.open("w", newline="", encoding=_RESULTS_CSV_ENCODING) as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "device",
                        "model_path",
                        "time_per_image_s",
                        "std_s",
                        "min_time_s",
                        "max_time_s",
                        "total_runtime_s",
                        "n_runs",
                        "n_images_used",
                        "notes",
                    ],
                    delimiter=_RESULTS_CSV_DELIMITER,
                )
                w.writeheader()
                w.writerow(_sanitize_row_dict(tflite_est))
        except Exception:
            pass

    def _speedup(mean_s: float) -> float:
        return cpu_mean / mean_s if mean_s > 0 else 0.0

    speed_cpu = 1.0 if cpu_mean > 0 else 0.0
    speed_gpu = _speedup(gpu_mean)
    speed_edge = 1.0 if cpu_mean > 0 else 0.0
    speed_tflite = _speedup(tflite_mean)

    t = Table(
        title="[bold]Benchmark: CPU vs GPU vs Edge vs TFLite[/bold]",
        header_style="bold cyan",
    )
    t.add_column("Device")
    t.add_column("Avg Time (s)", justify="right")
    t.add_column("Std Dev (s)", justify="right")
    t.add_column("Speedup", justify="right")
    t.add_column("Notes", overflow="fold")
    t.add_row(
        "CPU",
        f"{cpu_mean:.4f}",
        f"{cpu_std:.4f}",
        f"{speed_cpu:.2f}x",
        "Local execution (CPU)",
    )
    t.add_row(
        "GPU",
        f"{gpu_mean:.4f}",
        f"{gpu_std:.4f}",
        f"{speed_gpu:.2f}x",
        gpu_notes,
    )
    t.add_row(
        "Edge",
        f"{edge_mean:.4f}",
        f"{edge_std:.4f}",
        f"{speed_edge:.2f}x",
        edge_notes,
    )
    t.add_row(
        "TFLite",
        f"{tflite_mean:.4f}",
        f"{tflite_std:.4f}",
        f"{speed_tflite:.2f}x",
        tflite_notes,
    )

    con.print()
    con.print(t)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "device": "CPU",
            "time_per_image_s": round(cpu_mean, 6),
            "total_runtime_s": round(float(cpu_stats.get("total_s", 0.0)), 6),
            "std_s": round(cpu_std, 6),
            "speedup_vs_cpu": round(speed_cpu, 6),
            "notes": "Local execution (CPU)",
        },
        {
            "device": "GPU",
            "time_per_image_s": round(gpu_mean, 6),
            "total_runtime_s": round(float(cpu_stats.get("total_s", 0.0)) / 3.0, 6)
            if (not has_gpu or gpu_stats is None)
            else round(float(gpu_stats.get("total_s", 0.0)), 6),
            "std_s": round(gpu_std, 6),
            "speedup_vs_cpu": round(speed_gpu, 6),
            "notes": gpu_notes,
        },
        {
            "device": "Edge",
            "time_per_image_s": round(edge_mean, 6),
            "total_runtime_s": round(float(cpu_stats.get("total_s", 0.0)), 6),
            "std_s": round(edge_std, 6),
            "speedup_vs_cpu": round(speed_edge, 6),
            "notes": edge_notes,
        },
        {
            "device": "TFLite",
            "time_per_image_s": round(tflite_mean, 6),
            "total_runtime_s": round(tflite_total_s, 6),
            "std_s": round(tflite_std, 6),
            "speedup_vs_cpu": round(speed_tflite, 6),
            "notes": tflite_notes,
        },
    ]

    with out_csv.open("w", newline="", encoding=_RESULTS_CSV_ENCODING) as f:
        fieldnames = [
            "device",
            "time_per_image_s",
            "total_runtime_s",
            "std_s",
            "speedup_vs_cpu",
            "notes",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=_RESULTS_CSV_DELIMITER)
        w.writeheader()
        for r in rows:
            w.writerow(_sanitize_row_dict(r))

    con.print(f"\n[dim]Saved[/dim] [cyan]{out_csv.resolve()}[/cyan]")
    con.print(
        "[dim]CSV format:[/dim] semicolon (;) + UTF-8 BOM (Excel-friendly, DE locale)."
    )

    if make_plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            con.print("[yellow]matplotlib not available — skipped plot.[/yellow]")
        else:
            labels = ["CPU", "GPU", "Edge", "TFLite"]
            values = [cpu_mean, gpu_mean, edge_mean, tflite_mean]
            colors = ["#4C78A8", "#72B7B2", "#F58518", "#54A24B"]
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            bars = ax.bar(labels, values, color=colors[: len(labels)])
            ax.set_ylabel("Seconds per inference (mean)")
            ax.set_title("Inference time comparison")
            ax.grid(axis="y", alpha=0.25)
            for b, v in zip(bars, values):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height(),
                    f"{v:.3f}s",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            plt.xticks(rotation=12, ha="right")
            fig.tight_layout()
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_path, dpi=120)
            plt.close(fig)
            con.print(
                f"[dim]Saved plot[/dim] [cyan]{plot_path.resolve()}[/cyan]"
            )
    return 0


def save_performance_csv(
    path: Path,
    rows: list[dict],
):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "device",
        "time_per_image_s",
        "std_s",
        "min_time_s",
        "max_time_s",
        "total_runtime_s",
        "n_runs",
        "speedup_vs_cpu",
        "pct_faster_than_cpu",
        "notes",
    ]
    with path.open("w", newline="", encoding=_RESULTS_CSV_ENCODING) as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=_RESULTS_CSV_DELIMITER)
        w.writeheader()
        for r in rows:
            w.writerow(_sanitize_row_dict(r))


def build_perf_table(rows: list[dict]) -> Table:
    t = Table(title="[bold]Performance comparison[/bold]", header_style="bold green")
    t.add_column("Device")
    t.add_column("Mean ± std (s)", justify="right")
    t.add_column("Min / Max (s)", justify="right")
    t.add_column("Total (s)", justify="right")
    t.add_column("Speedup", justify="right")
    t.add_column("Notes", overflow="fold")
    for r in rows:
        t.add_row(
            r["device"],
            f"{r['time_per_image_s']:.4f} ± {r['std_s']:.4f}",
            f"{r.get('min_time_s', 0):.4f} / {r.get('max_time_s', 0):.4f}",
            f"{r.get('total_runtime_s', 0):.4f}",
            r.get("speedup_str", "—"),
            r.get("notes", ""),
        )
    return t


def try_plot(
    cpu_s: float,
    gpu_s: float | None,
    edge_s: float,
    out_path: Path,
    show_gpu_bar: bool,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    labels = ["CPU (forced)"]
    values = [cpu_s]
    if show_gpu_bar and gpu_s is not None:
        labels.append("GPU")
        values.append(gpu_s)
    labels.append("Edge (this PC)")
    values.append(edge_s)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#4472c4", "#ed7d31", "#70ad47", "#ffc000"][: len(values)]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Seconds per inference (mean)")
    ax.set_title("Inference time: CPU vs GPU vs Edge")
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def run_performance_benchmark(
    images_dir: Path,
    n_runs: int = 15,
    out_csv: Path = Path("results/performance.csv"),
    plot_path: Path = Path("results/performance_plot.png"),
    console: Console | None = None,
    make_plot: bool = True,
) -> int:
    con = console or Console()

    if not images_dir.is_dir():
        con.print(f"[red]Not a directory:[/red] {images_dir}")
        return 1

    img = _first_face_image(images_dir)
    if img is None:
        con.print(f"[red]No images in[/red] {images_dir}")
        return 1

    con.print(Panel.fit(f"[bold]Benchmark image:[/bold] {img.name}", style="blue"))

    rows_out: list[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=con,
    ) as prog:
        t_cpu = prog.add_task("[yellow]CPU (isolated subprocess)[/yellow]", total=1)
        try:
            cpu_stats = _isolated_bench("cpu", img, n_runs)
        except Exception as e:
            con.print(f"[red]CPU benchmark failed:[/red] {e}")
            return 1
        prog.advance(t_cpu)

        t_gpu = prog.add_task("[green]Default device (warmup + timed runs)[/green]", total=n_runs + 2)
        try:
            import cv2

            from utils.deepface_predict import predict_face_region
            from utils.face_detect import detect_faces_bgr, load_face_cascade

            try:
                import tensorflow as tf

                gpus = tf.config.list_physical_devices("GPU")
                device_label = "GPU" if gpus else "CPU"
            except Exception:
                device_label = "unknown"

            frame = cv2.imread(str(img))
            if frame is None:
                raise FileNotFoundError(str(img))
            cascade = load_face_cascade()
            faces = detect_faces_bgr(frame, cascade)
            if not faces:
                raise RuntimeError("No face in benchmark image.")
            x, y, w, h = faces[0]
            y0 = max(0, y - int(h * 0.15))
            y1 = min(frame.shape[0], y + h + int(h * 0.15))
            x0 = max(0, x - int(w * 0.15))
            x1 = min(frame.shape[1], x + w + int(w * 0.15))
            crop = frame[y0:y1, x0:x1]
            predict_face_region(crop, enforce_detection=False)
            prog.advance(t_gpu)
            predict_face_region(crop, enforce_detection=False)
            prog.advance(t_gpu)
            samples = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                predict_face_region(crop, enforce_detection=False)
                samples.append(time.perf_counter() - t0)
                prog.advance(t_gpu)
            gpu_stats = {
                "device_label": device_label,
                "mean_s": statistics.mean(samples),
                "std_s": statistics.stdev(samples) if len(samples) > 1 else 0.0,
                "total_s": sum(samples),
                "min_s": min(samples),
                "max_s": max(samples),
                "n_runs": n_runs,
            }
        except Exception as e:
            con.print(f"[red]Default-device benchmark failed:[/red] {e}")
            return 1

    cpu_mean = cpu_stats["mean_s"]
    gpu_mean = gpu_stats["mean_s"]
    has_gpu = gpu_stats["device_label"] == "GPU"

    speedup = cpu_mean / gpu_mean if gpu_mean > 0 else 0.0
    pct_faster = (1.0 - gpu_mean / cpu_mean) * 100.0 if cpu_mean > 0 else 0.0

    edge_mean = gpu_mean
    edge_note = "Same stack as live app on this machine"
    if not has_gpu:
        edge_note = "No GPU visible to TensorFlow — edge equals CPU path"

    row_cpu = {
        "device": "CPU (CUDA hidden)",
        "time_per_image_s": round(cpu_mean, 6),
        "std_s": round(cpu_stats["std_s"], 6),
        "min_time_s": round(cpu_stats["min_s"], 6),
        "max_time_s": round(cpu_stats["max_s"], 6),
        "total_runtime_s": round(cpu_stats["total_s"], 6),
        "n_runs": n_runs,
        "speedup_vs_cpu": 1.0,
        "pct_faster_than_cpu": 0.0,
        "notes": "Subprocess with CUDA_VISIBLE_DEVICES=-1",
    }
    row_gpu = {
        "device": "GPU" if has_gpu else "Default device (CPU)",
        "time_per_image_s": round(gpu_mean, 6),
        "std_s": round(gpu_stats["std_s"], 6),
        "min_time_s": round(gpu_stats["min_s"], 6),
        "max_time_s": round(gpu_stats["max_s"], 6),
        "total_runtime_s": round(gpu_stats["total_s"], 6),
        "n_runs": n_runs,
        "speedup_vs_cpu": round(speedup, 4) if has_gpu else 1.0,
        "pct_faster_than_cpu": round(pct_faster, 2) if has_gpu else 0.0,
        "notes": gpu_stats["device_label"],
    }
    row_edge = {
        "device": "Edge (this laptop)",
        "time_per_image_s": round(edge_mean, 6),
        "std_s": round(gpu_stats["std_s"], 6),
        "min_time_s": round(gpu_stats["min_s"], 6),
        "max_time_s": round(gpu_stats["max_s"], 6),
        "total_runtime_s": round(gpu_stats["total_s"], 6),
        "n_runs": n_runs,
        "speedup_vs_cpu": round(speedup, 4) if has_gpu else 1.0,
        "pct_faster_than_cpu": round(pct_faster, 2) if has_gpu else 0.0,
        "notes": edge_note,
    }

    display_rows = [
        {
            **row_cpu,
            "speedup_str": "1.00× (baseline)",
        },
        {
            **row_gpu,
            "speedup_str": f"{speedup:.2f}×" if has_gpu else "—",
        },
        {
            **row_edge,
            "speedup_str": f"{speedup:.2f}×" if has_gpu else "1.00×",
        },
    ]

    save_performance_csv(
        out_csv,
        [
            {k: v for k, v in r.items() if k != "speedup_str"}
            for r in [row_cpu, row_gpu, row_edge]
        ],
    )

    con.print()
    con.print(Panel.fit("[bold green]Benchmark done[/bold green]"))
    con.print(build_perf_table(display_rows))
    con.print(f"\n[dim]Saved[/dim] [cyan]{out_csv.resolve()}[/cyan]")

    if make_plot:
        ok = try_plot(
            cpu_mean,
            gpu_mean,
            edge_mean,
            plot_path,
            show_gpu_bar=has_gpu,
        )
        if ok:
            con.print(f"[dim]Saved plot[/dim] [cyan]{plot_path.resolve()}[/cyan]")
        else:
            con.print("[yellow]matplotlib not available — skipped plot.[/yellow]")

    con.print("[green]Done.[/green]")
    return 0


def prompt_benchmark_interactive(console: Console) -> int:
    root = _project_root()
    default_dir = root / "data"
    d = console.input(
        f"Image folder [{default_dir}]: "
    ).strip() or str(default_dir)
    n_s = console.input("Runs per device [15]: ").strip() or "15"
    try:
        n = max(3, int(n_s))
    except ValueError:
        n = 15
    return run_benchmark(
        Path(d),
        iterations=n,
        out_csv=root / "results" / "performance.csv",
        plot_path=root / "results" / "performance_plot.png",
        console=console,
        make_plot=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=Path, default=Path("data"))
    parser.add_argument("--n_runs", type=int, default=15)
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=Path("results/performance.csv"),
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("results/performance_plot.png"),
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    return run_benchmark(
        args.images_dir,
        iterations=args.n_runs,
        out_csv=args.out_csv,
        plot_path=args.plot,
        console=Console(),
        make_plot=not args.no_plot,
    )


if __name__ == "__main__":
    raise SystemExit(main())
