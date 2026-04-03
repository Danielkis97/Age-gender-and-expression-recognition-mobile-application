# Age, Gender, and Emotion Recognition (Local Edge Application)

Local-first computer vision application for face-based age, gender, and emotion prediction.
Inference runs on the local machine (edge scenario) with no cloud API dependency.

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

## Troubleshooting

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

## Repository Structure

```text
main.py
evaluate.py
performance.py
tflite_export.py
tflite_inference.py
requirements.txt
setup.ps1
utils/
  face_detect.py
  deepface_predict.py
  drawing.py
  label_mapping.py
data/
  labels_example.csv
dataset/
  images/
results/
```

## CPU vs Colab-GPU Comparison

The repository also includes a documented comparison between a local CPU run in PyCharm and a Google Colab GPU run.
All related assets are stored in `results/figures_cpu_vs_colab/`, with the written summary in [results/comparison_cpu_vs_colab.md](results/comparison_cpu_vs_colab.md).

High-level outcome:

- Quality metrics were identical across the reported scopes in this run
- Mean inference time per image was `3.4774 s` (`3477.4 ms`) on CPU and `3.4467 s` (`3446.7 ms`) on Colab-GPU
- Median inference time was `0.7312 s` (`731.2 ms`) on CPU and `0.5517 s` (`551.7 ms`) on Colab-GPU
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

## Limitations

- Prediction quality depends on image quality and lighting
- Pretrained models can have demographic bias
- Small test datasets may not generalize
