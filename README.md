# Edge AI Face Recognition (Age, Gender, Emotion)

This app runs age, gender, and emotion recognition on a local machine (edge setup, no cloud API calls).
It supports webcam mode and single-image mode, using OpenCV + [DeepFace](https://github.com/serengil/deepface).

Dependencies are in [`requirements.txt`](requirements.txt) so anyone can recreate the same environment.

> Note: this README is local-first and focused on app setup/run.

## Python version (important)

Use **Python 3.12** (best) or **Python 3.11**.

Why: TensorFlow installation is stable there.  
On Windows, Python 3.13/3.14 often fails with:
`No matching distribution found for tensorflow`

### How to get Python 3.12 / 3.11

Option A (recommended): download from [python.org](https://www.python.org/downloads/)

- install **64-bit** Python 3.12 (or 3.11)
- enable:
  - `Add python.exe to PATH`
  - Python launcher (`py`)

Option B (Windows via winget):

```powershell
winget install Python.Python.3.12 --accept-package-agreements --accept-source-agreements
```

(or replace `3.12` with `3.11`)

## Quick Start (copy/paste)

**Windows (PowerShell)**

```powershell
git clone https://github.com/Danielkis97/Age-Gender-and-Emotion-Recognition-Local-Edge-Application.git
cd Age-Gender-and-Emotion-Recognition-Local-Edge-Application
.\setup.ps1
python main.py
```

**Linux / macOS**

```bash
git clone https://github.com/Danielkis97/Age-Gender-and-Emotion-Recognition-Local-Edge-Application.git
cd Age-Gender-and-Emotion-Recognition-Local-Edge-Application
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

## Smoke Test (after install)

Run this once to confirm core imports:

```bash
python -c "import cv2, tensorflow, deepface; print('imports ok')"
```

If this fails, make sure you are on Python 3.12/3.11 and run `.\setup.ps1` again.

## Reproducibility (fresh clone)

From the project root, create a venv with Python 3.12 (or 3.11) and install:

**Windows (PowerShell)**

```powershell
cd path\to\age-gender-emotion-edge
Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

(Use `py -3.11` if you do not have 3.12.)

Or run the helper script (same folder):

```powershell
.\setup.ps1
```

If `Activate.ps1` is blocked, run once (current user only):

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**Linux / macOS**

```bash
cd age-gender-emotion-edge
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Privacy

Use only data you are allowed to process. Prefer **anonymized** or stock images; avoid identifying real people without consent.

## Install (short)

Same as **Reproducibility** above. After `pip install -r requirements.txt`, optional evaluation example:

```bash
python evaluate.py --images_dir dataset/images --labels_csv dataset/labels.csv
```

The first run of DeepFace may **download model weights** (one time). `tf-keras` is needed on many Windows setups for DeepFace’s RetinaFace stack.

## Expected outputs (evaluation)

After a successful evaluation run, these files should exist under `results/`:

- `evaluation_results.csv`
- `predictions.csv`
- `metrics.csv`
- `confusion_gender.csv`
- `confusion_age.csv`
- `confusion_emotion.csv`

## Run — interactive menu (default)

```bash
python main.py
```

**Rich menu**

1. **Webcam** — realtime (CLAHE on face crops before DeepFace; predictions every 10 frames, cached).  
2. **Single image** — file picker or typed path, OpenCV overlay.  
3. **Evaluation** — folder + labels CSV (up to 20 images); validates count/labels (warns if not 20 or missing fields); progress bar; summary + per image tables; **sklearn** confusion matrices + `classification_report` for **Male/Female**, **Adult/Elderly** (age ≥ 50), **Happy/Sad** (non-happy → Sad); writes `evaluation_results.csv`, `predictions.csv`, `confusion_gender.csv`, `confusion_age.csv`, `confusion_emotion.csv`.  
4. **Run Performance Benchmark** — compares CPU vs GPU vs Edge and also includes a TFLite deployment demo row. Saves a Rich table, `results/performance.csv`, and (if matplotlib is installed) `results/performance_plot.png`.  
5. **Exit**

**OpenCV keys:** `q` quit; `p` toggles predictions (webcam only).

### Command line (skip menu)

```bash
python main.py --webcam
python main.py --image path/to/photo.jpg
```

## How it works

1. **Face detection:** OpenCV Haar cascade.  
2. **Attributes:** DeepFace `analyze` on a **cropped face** for `age`, `gender`, `emotion`.  
3. **Webcam:** model **not** every frame — every N frames + **cache**; **CLAHE** on crops for low light.

**Canonical labels (UI + eval):** see `utils/label_mapping.py` — age group **Adult** if predicted age &lt; 50 else **Elderly**; gender **Male** / **Female**; emotion **Happy** if dominant class is happy else **Sad**.

## Batch evaluation (CLI)

CSV columns: `filename`, `true_gender`, `true_emotion`, `true_age_group`. Template: [data/labels_example.csv](data/labels_example.csv).

```bash
python evaluate.py --images_dir path/to/images --labels_csv path/to/labels.csv --out_csv results/evaluation_results.csv --predictions_csv results/predictions.csv
```

`--plain` disables Rich formatting. Outputs:

- **evaluation_results.csv** — full detail + timing per image.  
- **predictions.csv** — compact view (true vs predicted triplets, correctness flags).  
- **confusion_gender.csv**, **confusion_age.csv**, **confusion_emotion.csv** — 2×2 (or 2-class) confusion counts for sklearn-aligned metrics.

## Performance benchmark (CLI)

Needs face images in the folder (for benchmarking, any folder with a few `.jpg/.png` face photos is enough).

```bash
python performance.py --images_dir dataset/images --n_runs 3 --out_csv results/performance.csv --plot results/performance_plot.png
```

`--no-plot` skips matplotlib. GPU time is measured when TensorFlow sees a GPU; otherwise the code falls back to a simple simulation.

## Project layout

```
main.py
evaluate.py
performance.py
requirements.txt
utils/
  face_detect.py
  deepface_predict.py
  drawing.py
  label_mapping.py
data/
  labels_example.csv
results/          (created when you run eval / benchmark)
```

## Edge AI Explanation

In this project, the Edge device is your **local laptop/PC**. All inference for the app and the evaluation runs **on-device** (no cloud APIs).
This reflects the idea of *edge computing*: the model runs close to the data source so results are produced without uploading images to a remote server.

## Mobile Deployment

To demonstrate mobile compatibility (without building a real mobile app), the project includes a **TensorFlow Lite (TFLite) deployment demo**:

- `tflite_export.py` creates a small demo TensorFlow model and converts it to a TFLite model at `models/model.tflite`.
- `tflite_inference.py` loads `models/model.tflite`, runs inference on local face crops, and saves timing to `results/tflite_performance.csv`.

This shows how a lightweight model can be executed on a smartphone-class runtime. No full mobile app build is required.

You can run the TFLite timing directly:

```bash
python tflite_export.py
python tflite_inference.py --images_dir dataset/images --iterations 3 --out_csv results/tflite_performance.csv
```

## Important Separation: Evaluation vs Deployment Performance

- **Evaluation metrics (Accuracy/Precision/Recall/F1)** are computed using the **DeepFace model only** in `evaluate.py`.
- **TFLite is not used** in `evaluate.py` and does not influence accuracy metrics.
- TFLite is used only to provide a deployment/performance perspective.

## FAQ

- **Do I need Colab steps in this README?**  
  No. This README is intentionally local-first.
- **Should results be kept in the repository?**  
  Yes, keeping selected result files/plots is useful as reproducibility evidence.

## Limitations

- Lighting and image quality can impact face detection and recognition quality.
- Bias: pretrained models may not be equally accurate across all demographics.
- Small dataset: evaluation can be limited if you only run small image batches (default is 20 images).
