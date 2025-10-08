

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