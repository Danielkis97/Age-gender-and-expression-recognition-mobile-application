# Project Report (Bachelor) - 10-Page Structure Template

## Working Title (Suggested)
Development and Evaluation of a Local Edge Application for Age, Gender, and Emotion Recognition with Mobile Browser Benchmarking

## Target Scope
- Main text: 10 pages (excluding lists and appendices)
- Writing style: objective, precise, third person
- Core evidence: product development, evaluation, and reflective analysis

## Recommended Page Allocation (10 Pages)
- 1 Introduction and project preparation: 1.2 pages
- 2 Theoretical foundations and methodological framework: 1.5 pages
- 3 Project execution (from design to implementation): 2.5 pages
- 4 Evaluation and results: 2.2 pages
- 5 Reflection (challenges, risks, improvements): 1.4 pages
- 6 Conclusion and outlook: 1.2 pages

---

## Word Document Structure (with section purpose)

### Title Page
- Module title, study program, student ID, name, submission date
- Project title

### Table of Contents

### List of Figures and Tables

### List of Abbreviations (Example)
- CPU, GPU, Edge, TFLite, F1 score, KPI

## 1 Introduction and Project Preparation (about 10-15%)

### 1.1 Initial situation and problem statement
Clearly define the practical problem addressed:
- automatic age/gender/emotion recognition as a prototype decision-support function
- requirement: local processing (edge context) and reproducible evaluation

### 1.2 Project objectives and scope boundaries
State mandatory objectives explicitly:
- development of a working application for age/gender/emotion recognition
- edge-oriented implementation with mobile relevance
- performance reporting across CPU, GPU, and mobile edge
- evaluation on 20 images with the required balancing

Scope boundary:
- mobile browser path is primarily used for runtime measurement
- full quality metrics are generated through the DeepFace desktop pipeline

### 1.3 Procedure and report structure
- brief methodological overview (design, implementation, evaluation, reflection)
- chapter overview in 4-6 sentences

## 2 Theoretical Foundations and Methodological Framework (about 15%)

### 2.1 Domain background
- face analysis use cases in practice (retail, security, clinical routing)
- relevant concepts: classification, ground truth, confusion matrix, precision/recall/F1

### 2.2 Edge computing and mobile inference
- define edge execution (local processing, low latency, data protection advantages)
- explain the difference between browser runtime measurement and full model evaluation

### 2.3 Method justification
- why the DeepFace pipeline is used for quality metrics
- why TFLite/mobile browser is used for on-device timing
- why this mixed approach fits the available setup (iPhone + Windows)

## 3 Project Execution (about 25%)

### 3.1 Project context, roles, and resources
- project group or individual setup, roles, and responsibilities
- resources: hardware (Windows PC, iPhone), software (Python, TensorFlow, OpenCV, DeepFace, Colab)

### 3.2 Implementation phases
- Phase 1: local core application (`main.py`)
- Phase 2: batch evaluation (`evaluate.py`) including metric export and confusion matrices
- Phase 3: performance benchmarking (`performance.py`) across CPU/GPU/Edge/TFLite
- Phase 4: mobile browser metric collection (`mobile_eval_server.py`)

### 3.3 Dataset and labeling
- dataset: 20 images
- balancing according to assignment requirements:
  - 10 adult / 10 elderly
  - 10 male / 10 female
  - 10 happy / 10 sad
- ground-truth CSV and label quality assurance

### 3.4 Milestones and timeline
- use the Gantt appendix as timeline reference
- describe 3-5 critical milestones

## 4 Evaluation and Results (about 22%)

### 4.1 Evaluation design
- quality metrics: accuracy, precision, recall, F1
- runtime metrics: mean, median, P90, total runtime
- comparison axes:
  - local CPU
  - Colab GPU
  - iPhone Safari (mobile edge, timing-focused)

### 4.2 Quality results (CPU/GPU)
Report in tables (example values from this project):
- overall accuracy 0.7833, F1 0.7658
- gender accuracy 0.8500
- emotion accuracy 0.8500
- age accuracy 0.6500
- CPU/GPU values are identical in this run

### 4.3 Runtime results (CPU/GPU/Mobile)
Report in tables:
- mean: CPU 3477.40 ms, GPU 3446.70 ms, mobile 3.60 ms
- median: CPU 731.17 ms, GPU 551.69 ms, mobile 3.50 ms
- total runtime (20 images): CPU 69548.30 ms, GPU 68934.20 ms, mobile 71.00 ms

### 4.4 Result interpretation
- clearly separate quality comparison from timing comparison
- justify why mobile measurement is treated as a deployment/latency benchmark
- state that fully comparable mobile quality metrics would require browser-compatible models and full revalidation

## 5 Reflection: Challenges, Risks, and Improvements (about 14%)

### 5.1 What worked well
- reproducible CPU/GPU quality evaluation
- consistent output artifacts (CSV files, plots, comparison reports)
- successful real iPhone on-device timing run

### 5.2 Challenges and mitigation
- platform constraint: iPhone + Windows makes native mobile app workflow harder
- technical divergence: DeepFace desktop pipeline vs browser TFLite path
- environment risks: Python/venv/tooling issues after system reset
- mitigation: clear scope separation, documented reproduction steps, fallback-capable scripts

### 5.3 What worked less well / limitations
- current mobile browser path does not output directly comparable quality labels
- small sample size (20 images) limits generalizability
- age classification is weaker than gender and emotion classification

### 5.4 Continuous improvement plan
- evaluate browser-compatible multi-head models for face/age/gender/emotion
- align preprocessing between desktop and mobile paths
- expand dataset diversity (lighting, pose, skin tones, age segments)
- perform systematic recalibration and revalidation against ground truth

## 6 Conclusion and Outlook (about 10-15%)

### 6.1 Conclusion on objective achievement
- core assignment goals were achieved: functional application, structured evaluation, and mobile edge timing evidence
- present differentiated achievement: quality (CPU/GPU) completed, mobile quality intentionally defined as follow-up work

### 6.2 Transfer and professional relevance
- transferable competencies: MLOps fundamentals, experiment design, scientific reporting
- future relevance: data-driven decision making under real resource constraints

### 6.3 Outlook
- next implementation step: full browser-based quality evaluation
- optional extension: native mobile runtime comparison (CoreML/Android NN)

---

## Suggested Tables and Figures (for high assessment quality)
- Table 1: target system and constraints
- Table 2: dataset distribution (20 images, balanced design)
- Table 3: quality metrics CPU vs GPU
- Table 4: runtime metrics CPU vs GPU vs mobile
- Figure 1: architecture overview
- Figure 2: quality comparison plot
- Figure 3: runtime comparison plot
- Figure 4: Gantt extract (appendix)

## Appendix (recommended)
- A1 source code references / repository structure
- A2 complete CSV outputs
- A3 additional plots
- A4 Gantt chart
- A5 AI tool usage statement (transparent, concise, objective)

## Short text block: AI tool transparency (adaptable)
"AI-assisted tools were used in this project for language polishing, structure support, and documentation plausibility checks. Technical implementation, experiment execution, result verification, and final domain assessment were carried out by the project team. All reported measurement values were validated against generated result files."

---

## Assessment Checklist (aligned to IU criteria)
- Transfer (15%): explicit theory-to-practice linkage and concrete conclusions
- Documentation (10%): consistent language, clean tables/figures, correct referencing
- Resources (10%): justified tool selection (Windows + iPhone + Colab), transparent constraints
- Process (25%): coherent step-by-step procedure with traceable milestones
- Creativity (15%): pragmatic hybrid strategy (CPU/GPU quality + mobile on-device timing)
- Quality (25%): deep reflection, risk handling, limitations, and improvement plan

