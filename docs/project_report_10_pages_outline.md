# Projektbericht (Bachelor) - 10 Seiten Strukturvorlage

## Arbeitstitel (Vorschlag)
Entwicklung und Evaluation einer lokalen Edge-Anwendung zur Alters-, Geschlechts- und Emotionsklassifikation mit mobilem Browser-Benchmark

## Zielumfang
- Textteil: 10 Seiten (ohne Verzeichnisse/Anhang)
- Stil: sachlich, praezise, dritte Person
- Kernnachweis: Produktentwicklung + Evaluation + reflektierte Einordnung

## Empfohlene Seitenverteilung (10 Seiten)
- 1 Einleitung und Projektvorbereitung: 1.2 Seiten
- 2 Theoretische Grundlagen und methodischer Rahmen: 1.5 Seiten
- 3 Projektdurchfuehrung (Konzeption bis Umsetzung): 2.5 Seiten
- 4 Evaluation und Ergebnisse: 2.2 Seiten
- 5 Reflexion (Schwierigkeiten, Risiken, Verbesserungen): 1.4 Seiten
- 6 Fazit und Ausblick: 1.2 Seiten

---

## Dokumentstruktur fuer Word (mit Abschnittszielen)

### Titelblatt
- Modultitel, Studiengang, Matrikelnummer, Name, Abgabedatum
- Projekttitel

### Inhaltsverzeichnis

### Abbildungs- und Tabellenverzeichnis

### Abkuerzungsverzeichnis (Beispiel)
- CPU, GPU, Edge, TFLite, F1-Score, KPI

## 1 Einleitung und Projektvorbereitung (ca. 10-15 %)

### 1.1 Ausgangslage und Problemstellung
Schreibe klar, welches Praxisproblem adressiert wurde:
- automatische Erkennung von Alter/Geschlecht/Emotion als prototypische Decision-Support-Funktion
- Anforderung: lokale Verarbeitung (Edge-Naehe), reproduzierbare Bewertung

### 1.2 Projektziele und Abgrenzung
Pflichtziele explizit nennen:
- Entwicklung einer funktionsfaehigen Anwendung fuer Alter/Geschlecht/Emotion
- Edge-orientierte Umsetzung mit mobilem Bezug
- Leistungsbericht CPU vs GPU vs mobile Edge
- Evaluation auf 20 Bildern mit vorgegebenem Balancing

Abgrenzung:
- mobile Browser-Pfad primär fuer Laufzeitmessung
- vollstaendige Qualitaetsmetriken ueber die DeepFace-Desktoppipeline

### 1.3 Vorgehensweise und Berichtsaufbau
- kurzer Methodenüberblick (Design, Implementierung, Evaluation, Reflexion)
- Kapitelueberblick in 4-6 Saetzen

## 2 Theoretische Grundlagen und methodischer Rahmen (ca. 15 %)

### 2.1 Fachlicher Hintergrund
- Face Analysis in der Praxis (Retail, Sicherheit, klinische Orientierung)
- Relevante Konzepte: Klassifikation, Ground Truth, Konfusionsmatrix, Precision/Recall/F1

### 2.2 Edge-Computing und mobile Inferenz
- Begriffsklaerung Edge (lokal, geringe Latenz, Datenschutzvorteile)
- Unterschied zwischen Browser-Laufzeitmessung und vollstaendiger Modell-Evaluation

### 2.3 Methodenbegruendung
- Warum DeepFace-Pipeline fuer Qualitaetsmetriken
- Warum TFLite/mobile Browser fuer on-device Timing
- Warum dieser Methodenmix fuer das gegebene Setup (iPhone + Windows) geeignet ist

## 3 Projektdurchfuehrung (ca. 25 %)

### 3.1 Projektkontext, Rollen, Ressourcen
- Projektgruppe/Einzelarbeit, Rollen und Verantwortlichkeiten
- Ressourcen: Hardware (Windows-PC, iPhone), Software (Python, TensorFlow, OpenCV, DeepFace, Colab)

### 3.2 Implementierungsphasen
- Phase 1: lokale Kernanwendung (`main.py`)
- Phase 2: Batch-Evaluation (`evaluate.py`) inkl. Metrikexport und Konfusionsmatrizen
- Phase 3: Performancebenchmark (`performance.py`) CPU/GPU/Edge/TFLite
- Phase 4: mobile Browser-Metrikerfassung (`mobile_eval_server.py`)

### 3.3 Datenbasis und Labeling
- Datensatz: 20 Bilder
- Balancing nach Aufgabenstellung:
  - 10 Adult / 10 Elderly
  - 10 Male / 10 Female
  - 10 Happy / 10 Sad
- Ground-Truth-CSV und Qualitaetssicherung der Label

### 3.4 Meilensteine und Verlauf
- nutze Gantt-Appendix als Zeitreferenz
- nenne 3-5 entscheidende Meilensteine

## 4 Evaluation und Ergebnisse (ca. 22 %)

### 4.1 Evaluationsdesign
- Qualitaetsmetriken: Accuracy, Precision, Recall, F1
- Laufzeitmetriken: Mean, Median, P90, Total Runtime
- Vergleichsachsen:
  - Local CPU
  - Colab GPU
  - iPhone Safari (mobile Edge, timing-focused)

### 4.2 Qualitaetsergebnisse (CPU/GPU)
Tabellarisch berichten (Beispielwerte aus Projekt):
- overall accuracy 0.7833, F1 0.7658
- gender accuracy 0.8500
- emotion accuracy 0.8500
- age accuracy 0.6500
- CPU/GPU in diesem Lauf identisch

### 4.3 Laufzeitergebnisse (CPU/GPU/Mobile)
Tabellarisch berichten:
- Mean: CPU 3477.40 ms, GPU 3446.70 ms, Mobile 3.60 ms
- Median: CPU 731.17 ms, GPU 551.69 ms, Mobile 3.50 ms
- Total runtime (20 Bilder): CPU 69548.30 ms, GPU 68934.20 ms, Mobile 71.00 ms

### 4.4 Ergebnisinterpretation
- Trenne klar zwischen Qualitaetsvergleich und Timingvergleich
- Begruende, warum mobile Messung hier als Deployment-/Latenzbenchmark gewertet wird
- Aussage: fuer vollstaendig vergleichbare mobile Qualitaetsmetriken waeren browserkompatible Modelle und Revalidierung notwendig

## 5 Reflexion: Schwierigkeiten, Risiken, Verbesserungen (ca. 14 %)

### 5.1 Was gut gelaufen ist
- reproduzierbare CPU/GPU-Qualitaetsauswertung
- konsistente Ergebnisartefakte (CSV, Plots, Vergleichsberichte)
- erfolgreicher echter iPhone-on-device Timinglauf

### 5.2 Schwierigkeiten und Umgang damit
- Plattformgrenze: iPhone + Windows erschwert native mobile App-Pipeline
- Technische Divergenz: DeepFace-Desktoppipeline vs Browser-TFLite-Pfad
- Umgebungsrisiken: Python-/venv-/Tooling-Probleme nach Systemaenderungen
- Umgang: klare Scope-Trennung, dokumentierte Reproduktionsschritte, fallback-faehige Skripte

### 5.3 Was eher schlecht lief / Grenzen
- mobile Browser-Pfad liefert im aktuellen Stand keine direkt vergleichbaren Qualitaetslabels
- kleine Stichprobe (20 Bilder) begrenzt Generalisierbarkeit
- Altersklassifikation schwaecher als Gender/Emotion

### 5.4 Kontinuierliche Verbesserungen
- browserfaehige Multi-Head-Modelle fuer Face/Age/Gender/Emotion evaluieren
- Preprocessing zwischen Desktop und Mobile harmonisieren
- Datensatz erweitern (Licht, Pose, Hauttoene, Alterssegmente)
- systematische Revalidierung und Kalibrierung gegen Ground Truth

## 6 Fazit und Ausblick (ca. 10-15 %)

### 6.1 Fazit zur Zielerreichung
- Aufgabenkern wurde erreicht: funktionsfaehige Anwendung + strukturierte Evaluation + mobile Edge-Timingnachweise
- Zielerreichung differenziert darstellen: Qualitaet (CPU/GPU) erfolgreich, mobile Qualitaet bewusst als Folgearbeit definiert

### 6.2 Transfer und berufliche Relevanz
- uebertragbare Kompetenzen: MLOps-Grundprinzipien, Experimentdesign, wissenschaftliche Ergebnisdarstellung
- Relevanz fuer spaetere Praxis: datengetriebene Entscheidungsfindung unter realen Ressourcenrestriktionen

### 6.3 Ausblick
- naechster Umsetzungsschritt: vollstaendige browserbasierte Qualitaets-Evaluation
- optional: native mobile Runtime (CoreML/Android NN) als Vergleich

---

## Tabellen- und Abbildungsvorschlaege (fuer hohe Bewertungsqualitaet)
- Tabelle 1: Zielsystem und Randbedingungen
- Tabelle 2: Datensatzverteilung (20 Bilder, Balancing)
- Tabelle 3: Qualitaetsmetriken CPU vs GPU
- Tabelle 4: Laufzeitmetriken CPU vs GPU vs Mobile
- Abbildung 1: Architekturuebersicht
- Abbildung 2: Vergleichsplot Qualitaet
- Abbildung 3: Vergleichsplot Laufzeit
- Abbildung 4: Gantt-Auszug (Appendix)

## Anhang (empfohlen)
- A1 Quellcodeverweise / Repository-Struktur
- A2 Vollstaendige CSV-Outputs
- A3 Zusatzplots
- A4 Gantt-Chart
- A5 Erklaerung zum Einsatz von KI-Tools (transparent, knapp, sachlich)

## Kurzer Textbaustein: KI-Tool-Transparenz (anpassbar)
"Im Projekt wurden KI-gestuetzte Werkzeuge fuer sprachliche Ueberarbeitung, Strukturierung und Plausibilitaetspruefung von Dokumentation eingesetzt. Die technische Umsetzung, Experimentdurchfuehrung, Ergebnispruefung und finale fachliche Bewertung erfolgten durch die Projektgruppe. Alle berichteten Messwerte wurden anhand der erzeugten Ergebnisdateien verifiziert."

---

## Bewertungs-Checkliste (IU-Kriterien auf 100 % ausrichten)
- Transfer (15 %): Theoriebezug und Ableitung konkreter Schlussfolgerungen klar sichtbar
- Dokumentation (10 %): konsistente Sprache, saubere Tabellen/Abbildungen, korrektes Zitieren
- Ressourcen (10 %): Werkzeugwahl begruendet (Windows + iPhone + Colab), Grenzen transparent
- Prozess (25 %): Vorgehen schrittweise und nachvollziehbar mit Meilensteinen dokumentiert
- Kreativitaet (15 %): pragmatische Hybridstrategie (CPU/GPU-Qualitaet + mobile On-Device-Timing)
- Qualitaet (25 %): differenzierte Reflexion, Risiken, Limitationen, Verbesserungsplan

