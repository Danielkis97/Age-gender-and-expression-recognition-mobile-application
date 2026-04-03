# Appendix: Project Progress Gantt Chart

The following timeline summarizes the implementation and validation progress of the project.
Use this as an appendix figure or progress reference.

```mermaid
gantt
    title Age-Gender-Emotion Edge Application - Project Progress
    dateFormat  YYYY-MM-DD
    axisFormat  %d %b

    section Setup and Core Pipeline
    Repository setup and environment configuration         :done, a1, 2026-03-28, 2d
    Local app pipeline (`main.py`)                        :done, a2, 2026-03-29, 3d
    Dataset and label preparation (`dataset/labels.csv`)  :done, a3, 2026-03-31, 2d

    section Evaluation and Benchmarking
    Evaluation outputs (`evaluate.py`, confusion matrices):done, b1, 2026-04-01, 2d
    CPU vs Colab-GPU experiments and charting             :done, b2, 2026-04-02, 2d
    README professionalization and reproducibility cleanup :done, b3, 2026-04-02, 2d

    section Mobile Edge Extension
    TFLite export/inference integration                   :done, c1, 2026-04-02, 2d
    iPhone Safari on-device timing pipeline              :done, c2, 2026-04-03, 1d
    Three-way comparison (CPU/GPU/Mobile)                :done, c3, 2026-04-03, 1d

    section Finalization
    Final consistency and figure readability pass         :active, d1, 2026-04-03, 2d
    Appendix integration (charts + narrative)             :d2, 2026-04-04, 2d
    Final packaging and submission check                  :d3, 2026-04-05, 1d
```

## Notes

- This chart reflects the current project state and can be updated with exact institutional milestone dates.
- If your report requires strict calendar precision, replace the start dates/durations with your official schedule.
