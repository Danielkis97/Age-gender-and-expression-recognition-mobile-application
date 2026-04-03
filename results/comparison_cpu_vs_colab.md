# CPU vs Colab-GPU vs Mobile Edge Comparison Notes

## Setup

- CPU source: `results/Results CPU PYCHARM`
- GPU source: `results/RESULTS GPU TF Google Collab`
- Mobile Edge source: `results/Results mobile metrics`
- Same `20`-image test set is used for all three timing tracks.

## Quality Metrics

| Scope | Accuracy CPU | Accuracy GPU | Accuracy Mobile | Precision CPU | Precision GPU | Precision Mobile | Recall CPU | Recall GPU | Recall Mobile | F1 CPU | F1 GPU | F1 Mobile |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 0.7833 | 0.7833 | N/A | 0.8441 | 0.8441 | N/A | 0.7833 | 0.7833 | N/A | 0.7658 | 0.7658 | N/A |
| gender | 0.8500 | 0.8500 | N/A | 0.8846 | 0.8846 | N/A | 0.8500 | 0.8500 | N/A | 0.8465 | 0.8465 | N/A |
| emotion | 0.8500 | 0.8500 | N/A | 0.8535 | 0.8535 | N/A | 0.8500 | 0.8500 | N/A | 0.8496 | 0.8496 | N/A |
| age | 0.6500 | 0.6500 | N/A | 0.7941 | 0.7941 | N/A | 0.6500 | 0.6500 | N/A | 0.6011 | 0.6011 | N/A |

Mobile quality fields are `N/A` in this demo path because the browser TFLite run is timing-focused.

## Runtime Metrics

| Metric | CPU (ms) | Colab-GPU (ms) | Mobile Edge (ms) |
|---|---:|---:|---:|
| Mean inference time per image | 3477.40 | 3446.70 | 3.60 |
| Total runtime (20 images) | 69548.30 | 68934.20 | 71.00 |
| Median inference time | 731.17 | 551.69 | 3.50 |
| Mean excluding first image | 1158.21 | 1381.03 | 3.47 |
| P90 inference time | 2310.74 | 3552.57 | 4.10 |

## Figures (updated with Mobile Edge)

### 1) Quality overview

![Quality parity panel](figures_cpu_vs_colab/01_quality_parity_panel.png)

### 2) Timing KPI comparison

![Timing dumbbell clean](figures_cpu_vs_colab/02_timing_dumbbell_clean.png)

### 3) Latency distribution

![Latency boxstrip](figures_cpu_vs_colab/03_latency_distribution_boxstrip.png)

### 4) Latency by image order

![Latency by order](figures_cpu_vs_colab/04_latency_by_image_order.png)

### 5) Image-wise latency delta vs CPU

![Per-image delta lollipop](figures_cpu_vs_colab/05_image_delta_lollipop.png)

### 6) Additional three-way speedup view

![Three-way speedup panel](figures_cpu_vs_colab/06_three_way_speedup_panel.png)

## Takeaway

- CPU and Colab-GPU remain close on end-to-end mean latency in this dataset.
- Mobile Edge (iPhone on-device Safari run) is substantially faster in raw latency for this TFLite demo path.
- Keep quality comparison and timing comparison separated when interpreting conclusions.
