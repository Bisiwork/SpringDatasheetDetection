import os
import sys
import base64
import mimetypes
from pathlib import Path
from difflib import SequenceMatcher
from openai import OpenAI
from dotenv import load_dotenv

# =========================
# CONFIG
# =========================
IMAGE_PATH = Path("/Users/stefanoroybisignano/Desktop/W_SimplexRapid/SpringDatasheetDetection/data/all backup/134.604-A.page1.png")
PDF_PATH   = Path("/Users/stefanoroybisignano/Desktop/W_SimplexRapid/SpringDatasheetDetection/data/all backup/134.604-A.page1.pdf")

MODEL = "gpt-5-mini"
PROMPT = (
    "Estrai tutte le informazioni utili presenti nel file: testo, etichette, dimensioni, "
    "annotazioni, simboli, unità di misura, riferimenti tecnici e significato. "
    "Riporta un elenco strutturato e completo. "
    "Se mancano dati, indica esplicitamente cosa non è leggibile."
)
OUT_DIR = (IMAGE_PATH.parent if IMAGE_PATH.exists() else PDF_PATH.parent) / "dual_outputs"
OUT_DIR.mkdir(exist_ok=True)

# =========================
# OpenAI helpers
# =========================
def require_api_key():
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Errore: variabile d'ambiente OPENAI_API_KEY non impostata.", file=sys.stderr)
        sys.exit(1)
    return api_key

def upload_pdf_return_id(client: OpenAI, file_path: Path) -> str:
    # PDF → upload come file di contesto per Responses
    with open(file_path, "rb") as f:
        up = client.files.create(file=f, purpose="user_data")
    return up.id

def image_path_to_data_url(image_path: Path) -> str:
    # Converte l'immagine locale in un data URL base64 (schema supportato dalla Responses API)
    mime, _ = mimetypes.guess_type(image_path)
    if not mime or not mime.startswith("image/"):
        print(f"Formato immagine non valido per data URL: {image_path}", file=sys.stderr)
        sys.exit(1)
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

# Per IMMAGINE (PNG/JPEG): usare input_image con image_url = stringa (URL o data URL)
def ask_with_image_dataurl(client: OpenAI, model: str, data_url: str, prompt: str) -> str:
    resp = client.responses.create(
        model=model,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": data_url}
            ]
        }]
    )
    return resp.output_text or ""

# Per PDF: usare input_file con file_id
def ask_with_pdf_fileid(client: OpenAI, model: str, file_id: str, prompt: str) -> str:
    resp = client.responses.create(
        model=model,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_file", "file_id": file_id}
            ]
        }]
    )
    return resp.output_text or ""

# =========================
# Main
# =========================
def main():
    # 1) Controlli base
    if not IMAGE_PATH.exists():
        print(f"File immagine non trovato: {IMAGE_PATH}", file=sys.stderr)
        sys.exit(1)
    if not PDF_PATH.exists():
        print(f"File PDF non trovato: {PDF_PATH}", file=sys.stderr)
        sys.exit(1)

    # 2) Client
    client = OpenAI(api_key=require_api_key())

    # 3) Prepara i due input adatti alla Responses API
    print("↑ Preparazione input...")
    image_data_url = image_path_to_data_url(IMAGE_PATH)   # immagine → data URL base64 (NO file_id)
    pdf_file_id = upload_pdf_return_id(client, PDF_PATH)  # PDF → upload, ottieni file_id

    print(f"   image: data URL (lunghezza {len(image_data_url)} caratteri)")
    print(f"   pdf  : file_id = {pdf_file_id}")

    # 4) Richieste al modello
    print("\n▶ Richiesta sul file IMMAGINE:")
    out_image = ask_with_image_dataurl(client, MODEL, image_data_url, PROMPT)
    print("\n===== OUTPUT IMMAGINE =====\n")
    print(out_image)

    print("\n▶ Richiesta sul file PDF:")
    out_pdf = ask_with_pdf_fileid(client, MODEL, pdf_file_id, PROMPT)
    print("\n===== OUTPUT PDF =====\n")
    print(out_pdf)

    # 5) Similarità
    ratio = SequenceMatcher(None, out_image, out_pdf).ratio()
    print(f"\n=== Similarità (SequenceMatcher ratio) ===\n{ratio:.4f}")

    # 6) Salvataggi
    img_out = OUT_DIR / "image_output.txt"
    pdf_out = OUT_DIR / "pdf_output.txt"
    img_out.write_text(out_image, encoding="utf-8")
    pdf_out.write_text(out_pdf, encoding="utf-8")
    print(f"\n✔ Salvati:\n - {img_out}\n - {pdf_out}")

if __name__ == "__main__":
    main()
