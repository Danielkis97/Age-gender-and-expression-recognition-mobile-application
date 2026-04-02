# CPU vs. Colab-GPU Vergleichsbericht

## Datengrundlage

- CPU-Quelle: `C:\Users\danie\PycharmProjects\age-gender-emotion-edge\results\Results CPU PYCHARM`
- GPU-Quelle: `C:\Users\danie\PycharmProjects\age-gender-emotion-edge\results\RESULTS GPU TF Google Collab`
- Grundlage: identisches Testset mit `20` Bildern und identischer Label-Struktur.

## Qualitätsmetriken

| Scope | Accuracy CPU | Accuracy GPU | Delta | Precision CPU | Precision GPU | Recall CPU | Recall GPU | F1 CPU | F1 GPU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 0.7833 | 0.7833 | 0.0000 | 0.8441 | 0.8441 | 0.7833 | 0.7833 | 0.7658 | 0.7658 |
| gender | 0.8500 | 0.8500 | 0.0000 | 0.8846 | 0.8846 | 0.8500 | 0.8500 | 0.8465 | 0.8465 |
| emotion | 0.8500 | 0.8500 | 0.0000 | 0.8535 | 0.8535 | 0.8500 | 0.8500 | 0.8496 | 0.8496 |
| age | 0.6500 | 0.6500 | 0.0000 | 0.7941 | 0.7941 | 0.6500 | 0.6500 | 0.6011 | 0.6011 |

## Laufzeitmetriken

| Kennzahl | CPU | Colab-GPU | Delta | Delta % |
|---|---:|---:|---:|---:|
| Mittlere Inferenzzeit pro Bild (s) | 3.4774 | 3.4467 | -0.0307 | -0.8828% |
| Gesamtlaufzeit (20 Bilder, s) | 69.5483 | 68.9342 | -0.6141 | -0.8830% |

### Warm-up-bewusste Kennzahlen (aus `evaluation_results.csv`)

| Kennzahl | CPU | Colab-GPU | Delta | Delta % |
|---|---:|---:|---:|---:|
| Median Inferenzzeit (s) | 0.7312 | 0.5517 | -0.1795 | -24.5466% |
| Mittelwert ohne erstes Bild (s) | 1.1582 | 1.3810 | 0.2228 | 19.2390% |
| Getrimmter Mittelwert ohne Min/Max (s) | 1.1907 | 1.4459 | 0.2552 | 21.4348% |
| P90 Inferenzzeit (s) | 1.6876 | 2.4798 | 0.7922 | 46.9402% |
| Min/Max (s) | 0.5734 / 47.5424 | 0.2131 / 42.6945 | - | - |

## Konsistenz auf Bildebene

- Vergleich der Vorhersagen über `(pred_gender, pred_emotion, pred_age_group)` für alle `20` Bilder.
- Anzahl abweichender Vorhersage-Tupel zwischen CPU und GPU: **0**.

## Abbildungen für Bericht/Anhang

### 1) Qualitäts-Parität (CPU vs. GPU vs. Delta)

![Quality parity panel](figures_cpu_vs_colab/01_quality_parity_panel.png)

*Abbildung 1 zeigt die identischen Qualitätsmetriken beider Läufe sowie die Delta-Matrix mit durchgehend `0.0000`.*

### 2) Laufzeitprofil (Dumbbell-Plot)

![Timing dumbbell clean](figures_cpu_vs_colab/02_timing_dumbbell_clean.png)

*Abbildung 2 stellt die zentralen Laufzeit-KPIs gegenüber; je Zeile ist der Abstand zwischen CPU und GPU direkt als Delta erkennbar.*

### 3) Latenzverteilung (Box + Punkte, log-Skala)

![Latency boxstrip](figures_cpu_vs_colab/03_latency_distribution_boxstrip.png)

*Abbildung 3 visualisiert Verteilung und Streuung der Einzelinferenzzeiten; der markierte Warm-up-Punkt liegt deutlich oberhalb des stationären Bereichs.*

### 4) Latenz nach Bildreihenfolge

![Latency by order](figures_cpu_vs_colab/04_latency_by_image_order.png)

*Abbildung 4 zeigt den zeitlichen Verlauf pro Bildindex und macht den initialen Warm-up-Effekt in beiden Läufen sichtbar.*

### 5) Bildweise Laufzeitdifferenz (Lollipop)

![Per-image delta lollipop](figures_cpu_vs_colab/05_image_delta_lollipop.png)

*Abbildung 5 zeigt die Differenz `CPU - GPU` je Bild; positive Werte entsprechen einem Geschwindigkeitsvorteil der GPU.*

## Fazit

- Die Klassifikationsqualität ist im vorliegenden Versuch über alle berichteten Scopes identisch.
- Unterschiede zeigen sich primär im Laufzeitverhalten und sind stark vom Warm-up-Effekt beeinflusst.
- Für belastbare Aussagen sollten End-to-End-Kennzahlen und warm-up-bereinigte Kennzahlen gemeinsam berichtet werden.
