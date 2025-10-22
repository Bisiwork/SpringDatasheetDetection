"""
benchmark_runner.py - Script per benchmark e utility di supporto.

Questo modulo fornisce funzionalità di supporto per il benchmarking:
- Esegue estrazione predizioni per modelli su dataset ridotto (data/sketch)
- Genera report benchmark con metriche, grafici e selezione best model
- Crea workbook Excel per labeling manuale basato su predizioni di un modello

Input: Nessuno (usa costanti predefinite)
Output: File in results/, reports/
Dipendenze: model_evaluator.py, excel_labeling_generator.py, llm_parameter_extractor.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.model_evaluator import run_benchmark_report
from src.excel_labeling_generator import build_labeling_workbook
from src.llm_parameter_extractor import run_for_model

# Constants for configuration
MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-5",
]

# Default behavior constants
DEFAULT_MODELS_LIST = MODELS
DEFAULT_PRICING_CSV = None
DEFAULT_FORCE = False
DEFAULT_REPORT = False
DEFAULT_LABELING_MODEL = None
DEFAULT_WORKBOOK_PATH = "reports/ground_truth_labeling.xlsx"
DEFAULT_INPUT_DIR = "data/sketch"
DEFAULT_OUTPUT_ROOT = "results"

def _run_models(models: list[str], pricing_csv: Path | str | None, *, force: bool = False) -> None:
    for m in models:
        print(f"\n===== RUN {m} =====")
        run_for_model(
            m,
            category="Text tokens - Standard",
            input_dir="data/sketch",
            output_root="results",
            pricing_csv=pricing_csv,
            force=force,
        )


def main() -> None:
    # Default behavior: run models
    if DEFAULT_REPORT:
        run_benchmark_report()
        return

    if DEFAULT_LABELING_MODEL:
        model = DEFAULT_LABELING_MODEL
        build_labeling_workbook(
            input_dir=DEFAULT_INPUT_DIR,
            outputs_dir=f"{DEFAULT_OUTPUT_ROOT}/{model}/outputs",
            workbook_path=DEFAULT_WORKBOOK_PATH,
        )
        print(f"Labeling workbook salvato in {DEFAULT_WORKBOOK_PATH}")
        return

    _run_models(
        list(dict.fromkeys(DEFAULT_MODELS_LIST)),
        DEFAULT_PRICING_CSV,
        force=DEFAULT_FORCE,
    )


if __name__ == "__main__":
    main()
