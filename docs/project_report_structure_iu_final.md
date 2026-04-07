# IU Project Report Structure (Bachelor, 7-10 Pages)

## Purpose
This template gives a ready-to-use English structure for your final project report and aligns with:
- IU project report guidance (goal, structure, evaluation)
- IU formal formatting rules
- IU APA citation rules (including IU-specific stricter requirements)
- Your Task 2 implementation (age, gender, expression on edge/mobile)

---

## A) Formal Setup (Word)

- Paper size: `A4`
- Margins: `2.00 cm` on all sides
- Font: `Arial` (or similar sans-serif, e.g., Calibri), black
- Body text: `11 pt`
- Line spacing: `1.5`
- Body alignment: `Justified`
- Headings and lists: `Left`
- Footnotes: `10 pt`, justified
- Paragraphs:
  - no first-line indent
  - `6 pt` spacing after each paragraph
- Heading formatting:
  - Level 1: `16 pt`, bold, `12 pt` before / `12 pt` after
  - Level 2: `14 pt`, bold, `12 pt` before / `6 pt` after
  - Level 3: `11 pt`, bold, `12 pt` before / `6 pt` after
- Maximum heading depth: 3 levels (`1`, `1.1`, `1.1.1`)

### Pagination rules
- Front matter: Roman numerals (`I, II, III ...`), title page counted but number not printed
- Main text: Arabic numbering starts at `1` on `1 Introduction`
- Continue Arabic numbering through appendices
- Position: centered footer

---

## B) Mandatory Section Order

1. Title page  
2. Optional acknowledgment  
3. Optional gender disclaimer  
4. Abstract (if required for your module)  
5. Table of contents  
6. List of figures (if 3+ figures)  
7. List of tables (if 3+ tables)  
8. List of abbreviations  
9. Main text (Introduction, Main Part, Conclusion)  
10. References  
11. Appendix list (if appendices are used)  
12. Appendices

Notes:
- The title page is not listed as a TOC entry.
- The 7-10 page requirement applies to main text only, starting at `1 Introduction`.
- Figures/tables inside main text count toward main-text page length.

---

## C) Writing and Citation Rules

- Write in formal academic English.
- Use third person.
- Avoid first person (`I`, `we`) and avoid informal filler style.
- Keep wording precise and concise.

### Citation (IU + APA)
- Use APA 7.
- In-text citation should include author, year, and page number (IU requirement, including paraphrases where possible).
- Keep direct quotes limited.
- Every in-text source must appear in the reference list.
- Reference list: hanging indent `1.27 cm` from line 2 onward.
- Avoid using course slides/webinars as core academic sources.

### Figures and tables
- Every figure/table must have number + title.
- Add source line below each figure/table.
- If adapted, mark as adapted.
- If AI-generated visuals are used, report tool/prompt/source/license in IU-compliant form.

---

## D) 10-Page Main Text Plan

- `1 Introduction`: ~1.2 pages (10-15%)
- `2 Theoretical and technical foundation`: ~1.8 pages
- `3 Project execution`: ~2.4 pages
- `4 Evaluation and results`: ~2.2 pages
- `5 Reflection and improvements`: ~1.2 pages
- `6 Conclusion`: ~1.2 pages (10-15%)

Total target: ~10 pages

---

## E) Chapter Structure (ready to paste into report)

## 1 Introduction

### 1.1 Initial situation and problem context
- Explain practical relevance of age/gender/expression recognition.
- Introduce real use contexts (retail, security, healthcare routing).

### 1.2 Project objectives
- Define the required outputs and success criteria.
- Link explicitly to Task 2 requirements.

### 1.3 Scope and boundaries
- Define what is included and excluded.
- Clarify mobile-browser scope (timing evidence) vs desktop scope (quality metrics).

### 1.4 Planned approach and report roadmap
- Briefly explain methods and chapter logic.

## 2 Theoretical and Technical Foundation

### 2.1 Core recognition and evaluation concepts
- Classification tasks (age, gender, expression)
- Ground truth, confusion matrix, accuracy, precision, recall, F1

### 2.2 Edge deployment principles
- Why edge deployment matters (latency, privacy, local execution)
- Mobile/browser constraints

### 2.3 Method rationale
- Why desktop DeepFace pipeline for quality metrics
- Why mobile browser/TFLite path for on-device timing

## 3 Project Execution

### 3.1 Resources, environment, and tools
- Hardware/software setup (Windows, iPhone, Python, Colab, etc.)
- Roles/stakeholders (if relevant)

### 3.2 Development phases
- Phase 1: interactive app
- Phase 2: batch evaluation pipeline
- Phase 3: CPU/GPU benchmarking
- Phase 4: mobile browser timing pipeline

### 3.3 Data collection and labeling
- 20-image dataset collected with mobile device
- Balancing design and label quality checks

### 3.4 Milestones and progress
- Key milestones with short status notes
- Reference to Gantt chart in appendix

## 4 Evaluation and Results

### 4.1 Evaluation design
- Dataset, metrics, and test setup

### 4.2 Quality results (CPU vs GPU)
- Present one compact metrics table

### 4.3 Runtime results (CPU vs GPU vs Mobile Edge)
- Present one compact timing table (mean, median, P90, total)

### 4.4 Interpretation
- Explain findings, trade-offs, and what was validated
- Keep quality interpretation separate from timing interpretation

## 5 Reflection (Challenges, Risks, Improvements)

### 5.1 What worked well
- Reproducible outputs, stable pipeline, successful on-device timing run

### 5.2 Challenges and limitations
- Platform constraints and model-path differences
- Environment reset/tooling risks and their impact

### 5.3 Improvement plan
- Full browser-compatible quality path
- Larger and more diverse dataset
- Calibration/revalidation extensions

## 6 Conclusion

### 6.1 Objective achievement summary
- State exactly which goals were met

### 6.2 Transfer and practical relevance
- Show theory-to-practice transfer and module relevance

### 6.3 Future work
- Priority next steps with clear technical direction

---

## F) Task 2 Compliance Table (include in report)

| Task requirement | Status | Evidence |
|---|---|---|
| Deploy on edge/mobile | Completed | iPhone Safari on-device benchmark workflow |
| Report expected performance on CPU/GPU/edge | Completed | Comparison tables/figures in results folder |
| Evaluate 20 balanced images | Completed | `dataset/labels.csv` + evaluation outputs |
| Include source code in appendix context | Completed | Full repository file structure and references |
| Add Gantt chart (recommended) | Completed | Gantt file in docs/appendix |
| State AI tool usage | Completed | AI disclosure statement in report |

---

## G) Minimum tables and figures

- Table 1: dataset composition and balancing
- Table 2: quality metrics (CPU vs GPU)
- Table 3: runtime metrics (CPU vs GPU vs mobile edge)
- Figure 1: system architecture
- Figure 2: quality comparison chart
- Figure 3: runtime comparison chart
- Figure 4: Gantt chart extract (appendix)

---

## H) Appendix package

- Appendix A: source code structure overview
- Appendix B: key CSV result extracts
- Appendix C: supplementary figures
- Appendix D: Gantt chart
- Appendix E: AI usage declaration

---

## I) AI Usage Statement (formal, ready to use)

"AI-assisted tools were used for language polishing, structural drafting, and documentation plausibility checks.
Technical implementation, experiment execution, result validation, and final interpretation were performed by the project team.
All reported quantitative values were verified against generated project output files."

---

## J) Final Quality Gate (before submission)

- Main text length is between 7 and 10 pages
- Main text starts at `1 Introduction`
- Third-person academic style is consistent
- No German text remains
- In-text citations and reference list entries are fully consistent
- Every figure/table has a source line
- Appendices are referenced in main text (e.g., "see Appendix D")
