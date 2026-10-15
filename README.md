# AI & Applied Machine Learning Projects

A collection of desktop AI applications developed for University Computer Science & Engineering Internal Assessments, featuring offline-first architectures, Tkinter graphical user interfaces, and automated diagnostic pipelines.

---

## 📚 Projects in this Repository

| # | Project | Domain | Technologies | Directory |
|---|---|---|---|---|
| 1 | **[Intelligent Spell & Grammar Checker](#1-intelligent-spell--grammar-checker)** | Natural Language Processing (NLP) | Python, Tkinter, `pyspellchecker`, `TextBlob`, `LanguageTool` | [`spell_grammar_checker/`](spell_grammar_checker/) |
| 2 | **[PhytoScan AI — Plant Leaf Disease Diagnostic System](#2-phytoscan-ai--plant-leaf-disease-prediction--diagnostic-system)** | Computer Vision & Machine Learning | Python, Tkinter, PIL, `scikit-learn` (Random Forest), NumPy | [`plant_disease_predictor/`](plant_disease_predictor/) |

---

## 1. Intelligent Spell & Grammar Checker

An offline-capable desktop application built with Python and Tkinter for intelligent spell checking, grammar correction, and style diagnostics.

![Intelligent Spell & Grammar Checker](spell_grammar_checker/screenshot.png)

### 📌 Highlights
- **Multi-Tier NLP Engine:** Combines frequency-based spell checking (`pyspellchecker`), n-gram corrections (`TextBlob`), syntax heuristics, and optional LanguageTool support.
- **Visual Error Highlighting:** Highlights detected errors directly in the text editor (Red for spelling, Amber for grammar/style).
- **Interactive Diagnostic Inspector:** Displays issue numbers, error types, flagged words, suggestions, and explanations in a structured table.
- **One-Click Auto-Correction & Selective Fixes:** Instantly cleans up and rewrites the entire text, or allows applying individual fixes.

### 🚀 Quick Start
```bash
cd spell_grammar_checker
pip install -r requirements.txt
python app.py
```
👉 *For full details and testing instructions, see [spell_grammar_checker/README.md](spell_grammar_checker/README.md).*

---

## 2. PhytoScan AI — Plant Leaf Disease Prediction & Diagnostic System

An offline-capable Computer Vision and Machine Learning desktop application for instant plant leaf pathology diagnosis, lesion heatmap segmentation, and agronomic treatment prescription.

![PhytoScan AI](plant_disease_predictor/screenshot.png)

### 📌 Highlights
- **Dual-Tier Hybrid Architecture:** 100% offline Tier 2 engine (Multi-spectral HSV/RGB Computer Vision + 120-tree Random Forest Classifier) with optional Tier 1 MobileNetV2 support.
- **Dual-Spectrum Visual Studio:** Side-by-side comparison of original leaf specimen and AI Lesion Segmentation Map (color-coded for Healthy, Necrosis, Chlorosis, and Fungal Mycelium).
- **Differential Diagnosis & Severity Quantification:** Displays top-3 candidate disease probabilities with biophysical spectral breakdown and precise leaf infection percentage (None, Mild, Moderate, Severe).
- **Agronomic Prescriptions:** Provides comprehensive treatment plans spanning Symptoms & Biomarkers, Organic Treatment, Chemical Control, and Prevention Protocols.
- **Automated Verification:** Includes 7 procedural sample leaves and a built-in test suite (`test_suite.py`).

### 🚀 Quick Start
```bash
cd plant_disease_predictor
pip install -r requirements.txt
python app.py
```
👉 *For full details, architecture pipeline, testing guide, and viva questions, see [plant_disease_predictor/README.md](plant_disease_predictor/README.md).*

---

## 🛠️ General Requirements

- Python 3.9 through 3.12+
- Standard Tkinter support (included with standard Python installers on Windows/macOS)
