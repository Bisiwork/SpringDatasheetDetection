#!/usr/bin/env python3
"""
Script per costruire il report Excel di labeling direttamente dai risultati esistenti di gpt-5,
senza ricalcolare i modelli.
"""

from pathlib import Path
import sys

# Ensure project root is importable
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.main import generate_labeling_workbook

def main():
    model = "gpt-5"
    input_dir = Path("data/all")
    outputs_dir = Path("results") / model / "outputs"

    if not outputs_dir.exists():
        print(f"Directory outputs non trovata: {outputs_dir}")
        return

    # Il workbook_path sarà generato automaticamente con l'id versione successivo
    workbook_path = generate_labeling_workbook(
        model=model,
        input_dir=input_dir,
        outputs_dir=outputs_dir,
    )

    print(f"Report Excel creato: {workbook_path}")

if __name__ == "__main__":
    main()
