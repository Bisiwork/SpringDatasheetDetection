import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.benchmark_eval import run_benchmark_report
from src.build_labeling_excel import build_labeling_workbook
from src.extract_predictions import run_for_model

MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-5",

]

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
    parser = argparse.ArgumentParser(description="Benchmark & utilities for Spring Datasheet Detection")
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        help="Elenco modelli da eseguire (default: tutti)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Genera report aggregato con metriche, grafici e best model",
    )
    parser.add_argument(
        "--labeling",
        metavar="MODEL",
        help="Genera l'Excel di labeling usando le predizioni del modello indicato",
    )
    parser.add_argument(
        "--pricing-csv",
        help="Path al file openai_pricing.csv (opzionale, default: autodetect)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forza le chiamate OpenAI anche se esistono già i risultati",
    )
    args = parser.parse_args()

    if args.report:
        run_benchmark_report()
        return

    if args.labeling:
        model = args.labeling
        build_labeling_workbook(
            input_dir="data/sketch",
            outputs_dir=f"results/{model}/outputs",
            workbook_path="reports/ground_truth_labeling.xlsx",
        )
        print("Labeling workbook salvato in reports/ground_truth_labeling.xlsx")
        return

    _run_models(
        list(dict.fromkeys(args.models)),
        args.pricing_csv,
        force=args.force,
    )


if __name__ == "__main__":
    main()
