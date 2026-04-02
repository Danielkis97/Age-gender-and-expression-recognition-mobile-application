# CPU vs. Colab-GPU Comparison Notes

## Setup

- CPU source: `C:\Users\danie\PycharmProjects\age-gender-emotion-edge\results\Results CPU PYCHARM`
- GPU source: `C:\Users\danie\PycharmProjects\age-gender-emotion-edge\results\RESULTS GPU TF Google Collab`
- Same `20`-image test set and same label format were used for both runs.

## Quality Metrics

| Scope | Accuracy CPU | Accuracy GPU | Delta | Precision CPU | Precision GPU | Recall CPU | Recall GPU | F1 CPU | F1 GPU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 0.7833 | 0.7833 | 0.0000 | 0.8441 | 0.8441 | 0.7833 | 0.7833 | 0.7658 | 0.7658 |
| gender | 0.8500 | 0.8500 | 0.0000 | 0.8846 | 0.8846 | 0.8500 | 0.8500 | 0.8465 | 0.8465 |
| emotion | 0.8500 | 0.8500 | 0.0000 | 0.8535 | 0.8535 | 0.8500 | 0.8500 | 0.8496 | 0.8496 |
| age | 0.6500 | 0.6500 | 0.0000 | 0.7941 | 0.7941 | 0.6500 | 0.6500 | 0.6011 | 0.6011 |

## Runtime Metrics

| Metric | CPU | Colab-GPU | Delta | Delta % |
|---|---:|---:|---:|---:|
| Mean inference time per image (s) | 3.4774 | 3.4467 | -0.0307 | -0.8828% |
| Total runtime (20 images, s) | 69.5483 | 68.9342 | -0.6141 | -0.8830% |

### Warm-up-Aware Metrics (from `evaluation_results.csv`)

| Metric | CPU | Colab-GPU | Delta | Delta % |
|---|---:|---:|---:|---:|
| Median inference time (s) | 0.7312 | 0.5517 | -0.1795 | -24.5466% |
| Mean excluding first image (s) | 1.1582 | 1.3810 | 0.2228 | 19.2390% |
| Trimmed mean without min/max (s) | 1.1907 | 1.4459 | 0.2552 | 21.4348% |
| P90 inference time (s) | 1.6876 | 2.4798 | 0.7922 | 46.9402% |
| Min/Max (s) | 0.5734 / 47.5424 | 0.2131 / 42.6945 | - | - |

## Prediction Consistency (Image Level)

- Prediction tuples `(pred_gender, pred_emotion, pred_age_group)` were compared for all `20` images.
- Number of different prediction tuples between CPU and GPU: **0**.

## Figures

### 1) Quality Overview (single-matrix view)

![Quality parity panel](figures_cpu_vs_colab/01_quality_parity_panel.png)

*Figure 1 shows one clean quality matrix and a short delta summary. Since all deltas are zero, separate CPU/GPU grids are not needed.*

### 2) Runtime Profile (Dumbbell Plot)

![Timing dumbbell clean](figures_cpu_vs_colab/02_timing_dumbbell_clean.png)

*Figure 2 compares the main runtime KPIs and makes the per-row CPU-GPU delta directly visible.*

### 3) Latency Distribution (Box + Points, Log Scale)

![Latency boxstrip](figures_cpu_vs_colab/03_latency_distribution_boxstrip.png)

*Figure 3 shows the latency spread per image. The marked warm-up point is clearly above the steady-state range.*

### 4) Latency by Image Order

![Latency by order](figures_cpu_vs_colab/04_latency_by_image_order.png)

*Figure 4 shows inference time by image index and highlights the first-image warm-up effect in both runs.*

### 5) Per-Image Runtime Delta (Lollipop)

![Per-image delta lollipop](figures_cpu_vs_colab/05_image_delta_lollipop.png)

*Figure 5 shows `CPU - GPU` per image; positive values indicate a GPU speed advantage.*

## Takeaway

- Classification quality is identical across all reported scopes in this experiment.
- Observed differences are primarily runtime-related and strongly influenced by warm-up behavior.
- For a fair comparison, show both end-to-end metrics and warm-up-aware metrics together.
