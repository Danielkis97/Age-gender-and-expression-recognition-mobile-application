# CPU vs Colab-GPU vs iPhone (On-Device) Comparison

## Quick Takeaways

- Mean latency: **CPU 3477.40 ms**, **Colab-GPU 3446.70 ms**, **iPhone 3.60 ms**.
- This three-way view is intended for deployment timing comparison across environments.
- CPU and Colab-GPU keep full quality metrics from the DeepFace evaluation pipeline.
- Mobile Edge run is timing-focused in this demo path, so quality fields are reported as `N/A`.

## Runtime Metrics

| Metric | CPU (ms) | Colab-GPU (ms) | iPhone Safari (ms) |
|---|---:|---:|---:|
| Mean (all images) | 3477.40 | 3446.70 | 3.60 |
| Median | 731.17 | 551.69 | 3.50 |
| Mean excl. first image | 1158.21 | 1381.03 | 3.47 |
| P90 | 2310.74 | 3552.57 | 4.10 |
| Total runtime (20 images) | 69548.30 | 68934.20 | 71.00 |

## Quality Metrics (CPU/GPU comparable)

| Scope | CPU Accuracy | GPU Accuracy | Delta | CPU F1 | GPU F1 | Delta |
|---|---:|---:|---:|---:|---:|---:|
| overall | 0.7833 | 0.7833 | 0.0000 | 0.7658 | 0.7658 | 0.0000 |
| gender | 0.8500 | 0.8500 | 0.0000 | 0.8465 | 0.8465 | 0.0000 |
| emotion | 0.8500 | 0.8500 | 0.0000 | 0.8496 | 0.8496 | 0.0000 |
| age | 0.6500 | 0.6500 | 0.0000 | 0.6011 | 0.6011 | 0.0000 |

## Figure

![Three-way timing KPI chart](figures_three_way/01_timing_kpis_three_way.png)

## Notes

- Mobile timing is measured on iPhone Safari with on-device TFLite browser inference.
- Quality metrics here are intentionally CPU/GPU-only, because the mobile browser run
  in this demo path does not emit equivalent label predictions.
