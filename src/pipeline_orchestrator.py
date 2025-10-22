"""
pipeline_orchestrator.py - Orchestratore principale della pipeline Spring Datasheet Detection.

Questo modulo coordina l'intera pipeline end-to-end per l'estrazione di parametri
da datasheet di molle utilizzando modelli LLM (OpenAI). Esegue sequenzialmente:
1. Estrazione predizioni per modelli specificati (chiama llm_parameter_extractor.py)
2. Mostra tariffe modelli (chiama openai_pricing_manager.py)
3. Genera workbook Excel per labeling (chiama excel_labeling_generator.py)
4. Crea report benchmark con grafici (chiama model_evaluator.py)
5. Analizza tradeoff costo/accuratezza (chiama cost_accuracy_analyzer.py)

Input: Nessuno (usa costanti predefinite)
Output: File in results/, reports/, data/json_objects/
Dipendenze: Tutti gli altri moduli src/
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Sequence

# Ensure project root is importable when running as a script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.model_evaluator import run_benchmark_report
from src.excel_labeling_generator import build_labeling_workbook
from src.cost_accuracy_analyzer import main as run_tradeoff_cli
from src.llm_parameter_extractor import resolve_pricing_csv, run_for_model
from src.openai_pricing_manager import Pricing

# Constants for configuration
DEFAULT_MODELS: Sequence[str] = (
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-5",
)
DEFAULT_CATEGORY = "Text tokens - Standard"
DEFAULT_INPUT_DIR = Path("data/all")
DEFAULT_OUTPUT_ROOT = Path("results")
DEFAULT_WORKBOOK_PATH = "reports/spring_datasheet_detection.xlsx"

# Pipeline configuration constants
PIPELINE_MODELS = DEFAULT_MODELS
PIPELINE_CATEGORY = DEFAULT_CATEGORY
PIPELINE_INPUT_DIR = DEFAULT_INPUT_DIR
PIPELINE_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
PIPELINE_OUTPUTS_DIR = None
PIPELINE_WORKBOOK_PATH = DEFAULT_WORKBOOK_PATH
PIPELINE_LABELING_MODEL = None
PIPELINE_STRATEGY = "utopia"
PIPELINE_FORCE = True
PIPELINE_SKIP_PRICING = False
PIPELINE_SKIP_LABELING = False
PIPELINE_SKIP_REPORT = False
PIPELINE_SKIP_TRADEOFF = False


def _dedupe(models: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for name in models:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def run_extraction(
    models: Sequence[str],
    *,
    category: str = DEFAULT_CATEGORY,
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    pricing_csv: str | Path | None = None,
    force: bool = False,
) -> None:
    for model in _dedupe(models):
        print(f"\n===== EXTRACT {model} =====")
        run_for_model(
            model,
            category=category,
            input_dir=input_dir,
            output_root=output_root,
            pricing_csv=pricing_csv,
            force=force,
        )


def show_pricing_table(
    models: Sequence[str],
    *,
    category: str = DEFAULT_CATEGORY,
    pricing_csv: str | Path | None = None,
) -> None:
    pricing_path = resolve_pricing_csv(pricing_csv)
    pricer = Pricing(pricing_path, category=category)
    for model in _dedupe(models):
        rates = pricer.get_rates(model)
        print(f"{model}: IN={rates['in']} USD/1M tok  |  OUT={rates['out']} USD/1M tok")


def generate_labeling_workbook(
    model: str,
    *,
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    outputs_dir: str | Path | None = None,
    workbook_path: str | None = None,
    fields_priority: Sequence[str] | None = None,
) -> Path:
    outputs_dir = outputs_dir or Path("results") / model / "outputs"
    if workbook_path is None:
        # Determine the next version id by checking existing report directories
        base_report_dir = Path("reports")
        existing_versions = [d for d in base_report_dir.iterdir() if d.is_dir() and d.name.startswith("v")]
        max_id = 0
        for d in existing_versions:
            try:
                vid = int(d.name.lstrip("v"))
                if vid > max_id:
                    max_id = vid
            except ValueError:
                continue
        next_id = max_id + 1
        workbook_path = Path(DEFAULT_WORKBOOK_PATH.format(id=next_id, model=model))
    else:
        workbook_path = Path(workbook_path)

    result = build_labeling_workbook(
        input_dir=input_dir,
        outputs_dir=outputs_dir,
        workbook_path=workbook_path,
        fields_priority=fields_priority,
    )
    print(f"Workbook creato: {result}")
    return result


def generate_benchmark_report(
    *,
    models: Sequence[str] | None = None,
    strategy: str = "utopia",
) -> None:
    best = run_benchmark_report(models=list(models) if models else None, strategy=strategy)
    if not best:
        print("Nessun dato disponibile per il report.")
    else:
        print(f"Best model selezionato: {best.get('model')}")


def run_tradeoff_analysis() -> None:
    run_tradeoff_cli()


def run_pipeline() -> None:
    models = _dedupe(PIPELINE_MODELS)
    if not models:
        raise SystemExit("Nessun modello specificato per la pipeline.")

    run_extraction(
        models,
        category=PIPELINE_CATEGORY,
        input_dir=PIPELINE_INPUT_DIR,
        output_root=PIPELINE_OUTPUT_ROOT,
        pricing_csv=None,  # Use default
        force=PIPELINE_FORCE,
    )

    if not PIPELINE_SKIP_PRICING:
        show_pricing_table(models, category=PIPELINE_CATEGORY, pricing_csv=None)

    if not PIPELINE_SKIP_LABELING:
        labeling_model = PIPELINE_LABELING_MODEL or "gpt-5-mini"
        generate_labeling_workbook(
            labeling_model,
            input_dir=PIPELINE_INPUT_DIR,
            outputs_dir=PIPELINE_OUTPUTS_DIR,
            workbook_path=PIPELINE_WORKBOOK_PATH,
        )

    if not PIPELINE_SKIP_REPORT:
        generate_benchmark_report(models=models, strategy=PIPELINE_STRATEGY)

    if not PIPELINE_SKIP_TRADEOFF:
        run_tradeoff_analysis()


def main() -> None:
    # Run the default pipeline
    run_pipeline()


if __name__ == "__main__":
    main()
