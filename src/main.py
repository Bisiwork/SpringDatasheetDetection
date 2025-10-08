from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence

# Ensure project root is importable when running as a script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.benchmark_eval import run_benchmark_report
from src.build_labeling_excel import build_labeling_workbook
from src.evaluate_tradeoff import main as run_tradeoff_cli
from src.extract_predictions import resolve_pricing_csv, run_for_model
from src.pricing import Pricing

DEFAULT_MODELS: Sequence[str] = (
    # "gpt-4o-mini",
    # "gpt-4o",
    # # "gpt-5-nano",
    # "gpt-5-mini",
    "gpt-5",
)
DEFAULT_CATEGORY = "Text tokens - Standard"
DEFAULT_INPUT_DIR = Path("data/all")
DEFAULT_OUTPUT_ROOT = Path("results")
DEFAULT_WORKBOOK_PATH = "reports/v{id}/{model}/spring_datasheet_detection.xlsx"


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
        workbook_path = DEFAULT_WORKBOOK_PATH.format(id=next_id, model=model)
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


def run_pipeline(args: argparse.Namespace) -> None:
    models = _dedupe(args.models or DEFAULT_MODELS)
    if not models:
        raise SystemExit("Nessun modello specificato per la pipeline.")

    run_extraction(
        models,
        category=args.category,
        input_dir=args.input_dir,
        output_root=args.output_root,
        pricing_csv=args.pricing_csv,
        force=getattr(args, "force", False),
    )

    if not args.skip_pricing:
        show_pricing_table(models, category=args.category, pricing_csv=args.pricing_csv)

    if not args.skip_labeling:
        labeling_model = args.labeling_model or "gpt-5-mini"
        generate_labeling_workbook(
            labeling_model,
            input_dir=args.input_dir,
            outputs_dir=args.outputs_dir,
            workbook_path=args.workbook_path,
        )

    if not args.skip_report:
        generate_benchmark_report(models=models, strategy=args.strategy)

    if not args.skip_tradeoff:
        run_tradeoff_analysis()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Utility CLI per Spring Datasheet Detection")
    parser.add_argument(
        "--pricing-csv",
        help="Percorso al file openai_pricing.csv (default: autodetect)",
    )

    subparsers = parser.add_subparsers(dest="command", required=False)

    extract = subparsers.add_parser("extract", help="Esegue l'estrazione per i modelli indicati")
    extract.add_argument("--models", nargs="+", help="Lista modelli da eseguire")
    extract.add_argument("--category", default=DEFAULT_CATEGORY)
    extract.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    extract.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    extract.add_argument(
        "--force",
        action="store_true",
        help="Forza le chiamate OpenAI anche se esistono già i risultati",
    )

    pricing = subparsers.add_parser("pricing", help="Mostra le tariffe dei modelli")
    pricing.add_argument("--models", nargs="+", help="Lista modelli (default: configurazione standard)")
    pricing.add_argument("--category", default=DEFAULT_CATEGORY)

    labeling = subparsers.add_parser("labeling", help="Genera l'Excel per il labeling")
    labeling.add_argument("model", help="Modello di cui usare le predizioni")
    labeling.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    labeling.add_argument("--outputs-dir", help="Directory delle predizioni (default: results/<model>/outputs)")
    labeling.add_argument("--workbook-path", default=str(DEFAULT_WORKBOOK_PATH))
    labeling.add_argument("--fields-priority", nargs="*", help="Ordine preferito per i campi")

    report = subparsers.add_parser("report", help="Genera report e grafici di benchmark")
    report.add_argument("--models", nargs="+", help="Filtra i modelli considerati")
    report.add_argument(
        "--strategy",
        choices=["utopia", "acc_per_dollar", "pareto_first"],
        default="utopia",
        help="Criterio per il best model",
    )

    tradeoff = subparsers.add_parser("tradeoff", help="Analisi trade-off costo/accuratezza")

    pipeline = subparsers.add_parser("pipeline", help="Esegue l'intera pipeline end-to-end")
    pipeline.add_argument("--models", nargs="+", help="Modelli da processare")
    pipeline.add_argument("--category", default=DEFAULT_CATEGORY)
    pipeline.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    pipeline.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    pipeline.add_argument("--outputs-dir", help="Directory delle predizioni da usare per il labeling")
    pipeline.add_argument("--workbook-path", default=str(DEFAULT_WORKBOOK_PATH))
    pipeline.add_argument("--labeling-model", help="Modello da usare per il labeling (default: primo della lista)")
    pipeline.add_argument("--strategy", choices=["utopia", "acc_per_dollar", "pareto_first"], default="utopia")
    pipeline.add_argument("--skip-pricing", action="store_true")
    pipeline.add_argument("--skip-labeling", action="store_true")
    pipeline.add_argument("--skip-report", action="store_true")
    pipeline.add_argument("--skip-tradeoff", action="store_true")
    pipeline.add_argument(
        "--force",
        action="store_true",
        help="Forza le chiamate OpenAI anche se esistono già i risultati",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        args.command = "pipeline"
        args.models = None
        args.category = DEFAULT_CATEGORY
        args.input_dir = DEFAULT_INPUT_DIR
        args.output_root = DEFAULT_OUTPUT_ROOT
        args.outputs_dir = None
        args.workbook_path = DEFAULT_WORKBOOK_PATH
        args.labeling_model = None
        args.strategy = "utopia"
        args.force = False
        args.skip_pricing = False
        args.skip_labeling = False
        args.skip_report = False
        args.skip_tradeoff = False

    if args.command == "extract":
        run_extraction(
            args.models or DEFAULT_MODELS,
            category=args.category,
            input_dir=args.input_dir,
            output_root=args.output_root,
            pricing_csv=args.pricing_csv,
            force=args.force,
        )
    elif args.command == "pricing":
        show_pricing_table(
            args.models or DEFAULT_MODELS,
            category=args.category,
            pricing_csv=args.pricing_csv,
        )
    elif args.command == "labeling":
        generate_labeling_workbook(
            args.model,
            input_dir=args.input_dir,
            outputs_dir=args.outputs_dir,
            workbook_path=args.workbook_path,
            fields_priority=args.fields_priority,
        )
    elif args.command == "report":
        generate_benchmark_report(models=args.models, strategy=args.strategy)
    elif args.command == "tradeoff":
        run_tradeoff_analysis()
    elif args.command == "pipeline":
        run_pipeline(args)
    else:
        parser.error(f"Comando non riconosciuto: {args.command}")


if __name__ == "__main__":
    main()
