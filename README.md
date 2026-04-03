# Age, Gender, and Emotion Recognition (Local Edge Application)

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](requirements.txt)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-FF6F00?logo=tensorflow&logoColor=white)](requirements.txt)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white)](requirements.txt)
[![DeepFace](https://img.shields.io/badge/DeepFace-0.0.79%2B-7B1FA2)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/Danielkis97/Age-Gender-and-Emotion-Recognition-Local-Edge-Application)](https://github.com/Danielkis97/Age-Gender-and-Emotion-Recognition-Local-Edge-Application/commits/main)
[![Issues](https://img.shields.io/github/issues/Danielkis97/Age-Gender-and-Emotion-Recognition-Local-Edge-Application)](https://github.com/Danielkis97/Age-Gender-and-Emotion-Recognition-Local-Edge-Application/issues)

Local-first computer vision application for face-based age, gender, and emotion prediction.
Inference runs on the local machine (edge scenario) with no cloud API dependency.

## Architecture Preview

[![Architecture preview](docs/architecture_preview.png)](docs/architecture_preview.png)

PlantUML source for the same high-level structure is available in `docs/architecture.puml`.

The repository includes:

- Interactive app (`main.py`) for webcam and single-image analysis
- Batch evaluation (`evaluate.py`) with confusion matrices and CSV outputs
- Performance benchmark (`performance.py`) for CPU/GPU/Edge/TFLite comparison
- TFLite export and inference demo for mobile deployment compatibility

## Key Characteristics

- **Local execution:** no external inference service required
- **Reproducible setup:** pinned dependency list via `requirements.txt`
- **Evaluation outputs:** structured CSV artifacts in `results/`
- **Deployment perspective:** TFLite demo path separated from evaluation metrics

## System Requirements

- Python **3.12** (recommended) or **3.11**
- OS: Windows, Linux, or macOS
- Webcam only required for webcam mode

Why Python 3.12/3.11:
TensorFlow wheels are reliably available there. On newer Python versions (for example 3.13+), installation may fail with `No matching distribution found for tensorflow`.

## Quick Start

### Windows (PowerShell)

```powershell
git clone https://github.com/Danielkis97/Age-Gender-and-Emotion-Recognition-Local-Edge-Application.git
cd Age-Gender-and-Emotion-Recognition-Local-Edge-Application
.\setup.ps1
.\.venv\Scripts\python.exe main.py
```

### Linux / macOS

```bash
git clone https://github.com/Danielkis97/Age-Gender-and-Emotion-Recognition-Local-Edge-Application.git
cd Age-Gender-and-Emotion-Recognition-Local-Edge-Application
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

## Installation Verification (Smoke Test)

Run once after setup:

```bash
python -c "import cv2, tensorflow, deepface; print('imports ok')"
```

Optional TensorFlow device check:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
```

## Reproducible Setup (Fresh Clone)

### Windows (PowerShell)

```powershell
cd path\to\Age-Gender-and-Emotion-Recognition-Local-Edge-Application
Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If needed, replace `-3.12` with `-3.11`.

If script activation is blocked:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Linux / macOS

```bash
cd Age-Gender-and-Emotion-Recognition-Local-Edge-Application
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Application

Default interactive menu:

```bash
python main.py
```

Direct CLI modes:

```bash
python main.py --webcam
python main.py --image path/to/photo.jpg
```

Webcam controls:

- `q`: quit
- `p`: toggle predictions

## Evaluation (Quality Metrics)

Expected label CSV columns:

- `filename`
- `true_gender`
- `true_emotion`
- `true_age_group`

Template: [data/labels_example.csv](data/labels_example.csv)

Ground-truth note:

- The repository includes a manually labeled 20-image set in `dataset/labels.csv`
- If you use your own images or another dataset, create matching ground-truth labels first
- Without ground-truth labels, quality metrics (accuracy/precision/recall/F1) cannot be computed reliably

Run evaluation:

```bash
python evaluate.py --images_dir dataset/images --labels_csv dataset/labels.csv --out_csv results/evaluation_results.csv --predictions_csv results/predictions.csv
```

Main evaluation outputs:

- `results/evaluation_results.csv`
- `results/predictions.csv`
- `results/confusion_gender.csv`
- `results/confusion_age.csv`
- `results/confusion_emotion.csv`

## Performance Benchmark

Run benchmark:

```bash
python performance.py --images_dir dataset/images --n_runs 3 --out_csv results/performance.csv --plot results/performance_plot.png
```

Notes:

- `--no-plot` disables plot generation
- If no TensorFlow GPU is available, the GPU line is marked as simulation/fallback
- Edge in this project means local on-device execution (same host system)

## TFLite Deployment Demo

This is a deployment/performance demonstration and is **not** used for evaluation accuracy metrics.

Run export and inference:

```bash
python tflite_export.py
python tflite_inference.py --images_dir dataset/images --iterations 3 --out_csv results/tflite_performance.csv
```

Expected outputs:

- `models/model.tflite`
- `results/tflite_performance.csv`

Important separation:

- Evaluation metrics (accuracy/precision/recall/F1) are computed from the DeepFace pipeline in `evaluate.py`
- TFLite timings are deployment-oriented and independent from evaluation quality scores
- `utils/label_mapping.py` normalizes both model outputs and CSV labels into the canonical buckets used in this project:
  - gender: `Male` / `Female`
  - emotion: `Happy` / `Sad`
  - age group: `Adult` / `Elderly` (threshold at age `50`)

## Quick Validation Steps

To validate a fresh setup end-to-end:

1. Clone repository
2. Set up Python 3.12/3.11 virtual environment
3. Install dependencies from `requirements.txt`
4. Run smoke test import command
5. Run `python main.py`
6. Run evaluation command once
7. Run benchmark command once
8. Confirm output files exist under `results/`

## Directory Structure

```text
.
├── main.py                    # Interactive entry point
├── evaluate.py                # Batch evaluation and metrics export
├── performance.py             # CPU / GPU / Edge / TFLite benchmark
├── tflite_export.py           # Demo model export to TensorFlow Lite
├── tflite_inference.py        # TensorFlow Lite timing run
├── requirements.txt
├── setup.ps1
├── LICENSE
├── data/
│   └── labels_example.csv
├── dataset/
│   ├── images/
│   └── labels.csv
├── docs/
│   └── architecture.puml
├── models/
│   └── model.tflite
├── results/
│   ├── comparison_cpu_vs_colab.md
│   ├── figures_cpu_vs_colab/
│   ├── performance.csv
│   ├── performance_plot.png
│   └── tflite_performance.csv
└── utils/
    ├── deepface_predict.py
    ├── drawing.py
    ├── face_detect.py
    └── label_mapping.py
```

## Possible Bugs and Solutions

### `ModuleNotFoundError: No module named 'tensorflow'`

Likely cause: wrong interpreter. Use the project virtual environment interpreter explicitly.

Windows:

```powershell
.\.venv\Scripts\python.exe main.py
```

### `No matching distribution found for tensorflow`

Use Python 3.12 or 3.11, recreate `.venv`, then reinstall dependencies.

### TensorFlow does not detect GPU on native Windows

For TensorFlow >= 2.11, native Windows GPU support is limited.
Use CPU, WSL2, or TensorFlow-DirectML if GPU acceleration is required.

### `Activate.ps1` blocked by execution policy

Run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### No faces detected during evaluation or TFLite timing

- Use clear front-facing images
- Improve lighting and image quality
- Verify that `dataset/images` contains supported formats (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`)

### Webcam window does not open

- Check whether another application is already using the camera
- On Windows, verify camera permissions in system settings
- If the OpenCV window appears behind the IDE, look for it in the taskbar

### First run is much slower than later runs

- This is expected for model initialization and one-time setup work
- Use median or warm-up-aware timing values for a fairer comparison

### DeepFace downloads weights on first use

- The first execution may take longer because pretrained files are cached locally
- Re-running the same command should be noticeably faster after the initial download

## CPU vs Colab-GPU Comparison

The repository also includes a documented comparison between a local CPU run in PyCharm and a Google Colab GPU run.
All related assets are stored in `results/figures_cpu_vs_colab/`, with the written summary in [results/comparison_cpu_vs_colab.md](results/comparison_cpu_vs_colab.md).

High-level outcome:

- Quality metrics were identical across the reported scopes in this run
- Mean inference time per image was `3477.40 ms` on CPU and `3446.70 ms` on Colab-GPU
- Median inference time was `731.20 ms` on CPU and `551.70 ms` on Colab-GPU
- Prediction tuples were identical for all `20` images
- Most visible timing differences came from warm-up and image-level latency variation rather than from quality changes

The comparison charts can be regenerated with:

```bash
python results/generate_comparison_charts.py
```

### Comparison Figures

#### 1) Quality overview

![Quality parity panel](results/figures_cpu_vs_colab/01_quality_parity_panel.png)

#### 2) Timing KPI comparison

![Timing dumbbell clean](results/figures_cpu_vs_colab/02_timing_dumbbell_clean.png)

#### 3) Latency distribution

![Latency distribution boxstrip](results/figures_cpu_vs_colab/03_latency_distribution_boxstrip.png)

#### 4) Latency by image order

![Latency by image order](results/figures_cpu_vs_colab/04_latency_by_image_order.png)

#### 5) Image-wise latency delta

![Image delta lollipop](results/figures_cpu_vs_colab/05_image_delta_lollipop.png)

## License

This project is licensed under the [MIT License](LICENSE).

## Limitations

- Prediction quality depends on image quality and lighting
- Pretrained models can have demographic bias
- Small test datasets may not generalize

## Development Environment

The code for this project was developed using PyCharm, which offers a powerful IDE for Python development.
