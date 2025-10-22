

"""
image_to_pdf_converter.py - Utility per conversione immagini JPG/PNG a PDF.

Questo script di utilità converte tutti i file JPG e PNG nella directory
data/sketch in PDF, preservando l'aspetto e la qualità. Utile per preparare
dataset di test con formati PDF.

Input: File JPG/PNG in data/sketch/
Output: File PDF corrispondenti in data/sketch/
Dipendenze: PIL (Pillow)
"""
from pathlib import Path
from PIL import Image

input_dir = Path("data/sketch")

# Cerca ricorsivamente tutti i file .jpg e .png
for ext in ("*.jpg", "*.png"):
    for img_file in input_dir.rglob(ext):
        try:
            img = Image.open(img_file).convert("RGB")
            pdf_path = img_file.with_suffix(".pdf")
            img.save(pdf_path, "PDF", resolution=100.0)
            print(f"Creato: {pdf_path}")
        except Exception as e:
            print(f"Errore su {img_file}: {e}")