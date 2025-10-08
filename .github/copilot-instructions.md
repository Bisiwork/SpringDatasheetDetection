# Copilot Instructions for Spring Datasheet Detection

## Project Overview
This repository benchmarks models for spring datasheet detection and prepares labeled reference data. It is organized for rapid experimentation and evaluation of model performance on real-world datasheets.

## Key Components
- `benchmark_models.py`: Main entry for running benchmarks, generating reports, and preparing labeling files.
- `src/`: Contains core logic for evaluation (`benchmark_eval.py`), CLI (`benchmark_cli.py`), data extraction (`extract_predictions.py`), and utilities.
- `data/`: Contains input files (PDFs, images) and reference JSON objects for spring parameters.
- `reports/` and `results/`: Output directories for generated metrics, labeled data, and model outputs (ignored by Git).

## Developer Workflows
- **Install dependencies:**
  - `pip install -r requirements.txt`
  - On macOS: `brew install poppler` (required for PDF image conversion)
- **Run benchmarks:**
  - `python benchmark_models.py` (extract predictions for all models, generate aggregate metrics and plots, create ground truth labeling Excel file with ground truth column empty for manual labeling)
- **Debugging:**
  - Most logic is in `src/`; use CLI flags for verbose output and targeted runs.

## Project-Specific Patterns
- **Data references:**
  - JSON objects in `data/json_objects/` define spring parameters and are used for evaluation and labeling.
- **PDF/Image handling:**
  - Uses `pdf2image` and Poppler binaries for PDF conversion; ensure Poppler is installed on macOS.
- **Output conventions:**
  - All generated results and reports are written to `results/` and `reports/`.
  - Output files are ignored by Git to keep the repo clean.
- **Modular utilities:**
  - Utility functions are in `src/utilities/` and `src/utils/`.

## Integration Points
- **External dependencies:**
  - `pdf2image`, Poppler (for PDF conversion)
  - Standard Python data science stack (see `requirements.txt`)
- **Cross-component communication:**
  - Main scripts in root and `src/` import utilities and data definitions from submodules.

## Example Usage
```bash
# Run all 
default: python benchmark_models.py

```

## References
- See `README.md` for more details on setup and usage.
- Key logic: `src/benchmark_eval.py`, `src/benchmark_cli.py`, `src/extract_predictions.py`
- Data definitions: `data/json_objects/`
- Output: `reports/`, `results/`

---
_Keep instructions concise and focused on actual project conventions. Update this file if workflows or architecture change._
