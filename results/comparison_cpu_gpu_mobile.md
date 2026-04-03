# CPU vs Colab-GPU vs iPhone (On-Device) Comparison

## Runtime Metrics

| Metric | CPU (ms) | Colab-GPU (ms) | iPhone Safari (ms) |
|---|---:|---:|---:|
| Mean (all images) | 3477.40 | 3446.70 | 3.60 |
| Median | 731.17 | 551.69 | 3.50 |
| Mean excl. first image | 1158.21 | 1381.03 | 3.47 |
| P90 | 2310.74 | 3552.57 | 4.10 |
| Total runtime (20 images) | 69548.30 | 68934.20 | 71.00 |

## Quality Metrics (same schema)

| Scope | CPU Acc | GPU Acc | Mobile Acc | CPU Prec | GPU Prec | Mobile Prec | CPU Recall | GPU Recall | Mobile Recall | CPU F1 | GPU F1 | Mobile F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 0.7833 | 0.7833 | N/A | 0.8441 | 0.8441 | N/A | 0.7833 | 0.7833 | N/A | 0.7658 | 0.7658 | N/A |
| gender | 0.8500 | 0.8500 | N/A | 0.8846 | 0.8846 | N/A | 0.8500 | 0.8500 | N/A | 0.8465 | 0.8465 | N/A |
| emotion | 0.8500 | 0.8500 | N/A | 0.8535 | 0.8535 | N/A | 0.8500 | 0.8500 | N/A | 0.8496 | 0.8496 | N/A |
| age | 0.6500 | 0.6500 | N/A | 0.7941 | 0.7941 | N/A | 0.6500 | 0.6500 | N/A | 0.6011 | 0.6011 | N/A |

## Figure

![Three-way timing KPI chart](figures_three_way/01_timing_kpis_three_way.png)

## Notes

- CPU and Colab-GPU include full quality metrics from the DeepFace evaluation pipeline.
- Mobile timing is measured on iPhone Safari with on-device TFLite browser inference.
- Mobile quality fields stay `N/A` for this demo TFLite deployment path.
