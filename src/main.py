#!/usr/bin/env python3
"""
main.py – Step-by-step extraction using OpenAI *Responses* API + JSON Schema.
Automatically generates JSON schemas if missing.
"""
from __future__ import annotations
import base64
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Type

from dotenv import load_dotenv
import openai

from springs import (
    SpringBase, SpringType, CylindricalSpring, ConicalSpring,
    BiconicalSpring, CustomSpring
)


# torsione vs compression vs garter 


# ────────────────────────────── CONFIGURATION ────────────────────────────── #
DEBUG = True
MODEL = "gpt-4o"
# Aggiungi il supporto ai PDF -----------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".gif"}
PDF_EXTS = {".pdf"}
INPUT_EXTS = IMG_EXTS | PDF_EXTS

INPUT_DIR = Path("data/sketch")
SCHEMA_DIR = Path("data/json_objects")
OUTPUT_DIR = Path("results")
USAGE_DIR = Path("data/usage")

# ────────────────────────────── ENVIRONMENT ──────────────────────────────── #
def setup_environment():
    """Load environment variables and set OpenAI API key."""
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY") or sys.exit("❌  OPENAI_API_KEY missing")

# ────────────────────────────── DIRECTORIES ──────────────────────────────── #
def prepare_directories():
    """Ensure all required directories exist."""
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    USAGE_DIR.mkdir(parents=True, exist_ok=True)

# ────────────────────────────── SCHEMA GENERATION ────────────────────────── #
def ensure_json_schemas_exist():
    """Ensure all required JSON schema files exist, generate if missing."""
    required = {
        "spring_type", "wire_material", "wire_diameter", "free_length", "total_coils",
        "initial_closed_coils", "final_closed_coils", "pitch_insertion_coils", "pitch_retraction_coils",
        "external_diameter", "body_diameter_correction", "minimum_diameter", "maximum_diameter",
        "concavity_convexity", "initial_diameter", "central_diameter", "final_diameter",
        "initial_conical_coils", "final_conical_coils", "initial_coils_curvature", "final_coils_curvature",
        "note"
    }
    missing = [f for f in required if not (SCHEMA_DIR / f"{f}.json").exists()]
    if missing:
        print(f"🔧 Missing JSON schema(s): {', '.join(missing)}. Generating...")
        import subprocess
        subprocess.run([sys.executable, "src/make_json_templates.py"], check=True)
        print("✅  JSON schemas generated.")

ensure_json_schemas_exist()

# ────────────────────────────── HELPERS ──────────────────────────────────── #
# Unifica la codifica di immagini e PDF in un’unica funzione
def encode_input(path: Path) -> Dict[str, Any]:
    """
    Restituisce il dizionario da inserire in `messages[*].content`
    secondo la sintassi OpenAI *Responses*:

      • Immagini  →  {"type": "input_image", …}
      • PDF       →  {"type": "input_file",  …}
    """
    ext = path.suffix.lower()

    # --- Immagini ----------------------------------------------------------
    if ext in IMG_EXTS:
        mime = f"image/{ext.lstrip('.')}"
        b64 = base64.b64encode(path.read_bytes()).decode()
        return {
            "type": "input_image",
            "image_url": f"data:{mime};base64,{b64}",
            "detail": "auto",
        }

    # --- PDF ---------------------------------------------------------------
    if ext in PDF_EXTS:
        b64 = base64.b64encode(path.read_bytes()).decode()
        return {
            "type": "input_file",
            "filename": path.name,
            "file_data": f"data:application/pdf;base64,{b64}",
        }

    # ----------------------------------------------------------------------
    raise ValueError(f"Unsupported file type: {path.suffix}")

def _sanitize(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a copy of the schema without external references:
    - Removes $id and $ref if they point outside
    - Deduplicates fields in 'required'
    """
    out = dict(schema)
    out.pop("$id", None)
    out.pop("$ref", None)
    if "required" in out:
        out["required"] = list(dict.fromkeys(out["required"]))
    return out

def load_schema(*fields: str) -> Dict[str, Any]:
    """Merge the templates of individual parameters into a valid schema."""
    merged: Dict[str, Any] | None = None
    for f in fields:
        raw = json.loads((SCHEMA_DIR / f"{f}.json").read_text())
        sch = _sanitize(raw)
        if merged is None:
            merged = sch
        else:
            merged["properties"].update(sch["properties"])
            merged["required"].extend(sch["required"])
    return merged or {}

# ─── NEW: costanti di listino (Apr-2025) ────────────────────────────────────
PRICE = {
    "gpt-4o": {"in": 0.005, "out": 0.015},   # USD / 1K tok
}
USD_TO_EUR = 0.92

GRAND_TOTAL_USD: float = 0.0
GRAND_TOTAL_REQ: int   = 0

def cost_usd(model: str, usage) -> float:
    """Ritorna il costo in USD di una response."""
    price = PRICE.get(model, {"in":0,"out":0})
    return (usage.input_tokens / 1000) * price["in"] + \
           (usage.output_tokens / 1000) * price["out"]

def ask(messages: List[Dict[str, Any]], schema: Dict[str, Any], name: str, acc: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Send a prompt to OpenAI and return the parsed JSON response. Log usage."""
    if DEBUG:
        print("──────────────────── OPENAI REQUEST")
        for m in messages:
            role = m["role"].upper()
            txt = m["content"]
            if isinstance(txt, list):
                txt = next((c["text"] for c in txt if c["type"] == "input_text"), "")
            print(f"[{role}] {txt}\n")
        print("──────────────────── SCHEMA")
        print(json.dumps(schema, indent=2, ensure_ascii=False), "\n")

    rsp = openai.responses.create(
        model=MODEL,
        input=messages,
        temperature=0.0,
        text={
            "format": {
                "type": "json_schema",
                "name": name,
                "schema": schema,
                "strict": True
            }
        }
    )
    usage = rsp.usage
    usd   = cost_usd(MODEL, usage)
    eur   = usd * USD_TO_EUR

    acc.append({
        "name": name,
        "in_tokens":  usage.input_tokens,
        "out_tokens": usage.output_tokens,
        "usd":        round(usd,4),
        "eur":        round(eur,4),
    })

    global GRAND_TOTAL_USD, GRAND_TOTAL_REQ
    GRAND_TOTAL_USD += usd
    GRAND_TOTAL_REQ += 1

    raw_json = rsp.output_text
    if DEBUG:
        print("──────────────────── OPENAI RESPONSE")
        print(raw_json, "\n")
    return json.loads(raw_json)

# ────────────────────────────── PROMPT BUILDERS ──────────────────────────── #
def prompt_type_and_material(img: Dict[str, Any], acc: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Prompt for spring type and wire material.
    """
    schema = load_schema("spring_type", "wire_material")
    sys = (
        "You are a Simplex Rapid engineer. Analyze the spring image, the tables, the notes."
    )
    return ask(
        [
            {"role": "system", "content": sys},
            {"role": "user", "content": [
                {"type": "input_text", "text": 
                    "Detect the following:\n"
                    "• spring_type (enum: cylindrical | conical | biconical | custom): "
                    "the geometric family of the spring, based on its shape.\n"
                    "• wire_material (enum: stainless_steel | chrome_silicon_steel | music_wire_steel): "
                    "the material used for the spring wire.\n"
                    "\nReturn the JSON object."
                },
                img
            ]}
        ],
        schema,
        name=short_schema_name("spring_and_material", ["spring_type", "wire_material"]),
        acc=acc
    )

def prompt_params(fields: Sequence[str], img: Dict[str, Any], model_cls: Type[SpringBase], acc: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Prompt for one or more spring parameters, using improved descriptions.
    """
    schema = load_schema(*fields)
    desc = []
    for f in fields:
        fld = model_cls.model_fields[f]
        # Improved, more formal descriptions
        if f == "wire_diameter":
            d = "Wire diameter (mm): The thickness of the spring wire."
        elif f == "free_length":
            d = "Free length (mm): The overall length of the spring when unloaded."
        elif f == "total_coils":
            d = "Total number of coils: The sum of all coils, including closed and open."
        elif f == "initial_closed_coils":
            d = "Initial closed coils: Number of coils at the beginning that are fully closed."
        elif f == "final_closed_coils":
            d = "Final closed coils: Number of coils at the end that are fully closed."
        elif f == "pitch_insertion_coils":
            d = "Insertion pitch coils: Number of coils with increasing pitch at the start."
        elif f == "pitch_retraction_coils":
            d = "Retraction pitch coils: Number of coils with decreasing pitch at the end."
        elif f == "external_diameter":
            d = "External diameter (mm): The constant outer diameter of the spring body."
        elif f == "body_diameter_correction":
            d = "Body diameter correction (mm): Adjustment to the nominal body diameter."
        elif f == "minimum_diameter":
            d = "Minimum diameter (mm): The smallest diameter at the narrow end of the spring."
        elif f == "maximum_diameter":
            d = "Maximum diameter (mm): The largest diameter at the wide end of the spring."
        elif f == "concavity_convexity":
            d = "Concavity/convexity (mm): The overall curvature of the spring profile."
        elif f == "initial_diameter":
            d = "Initial diameter (mm): Diameter at the first end of the spring."
        elif f == "central_diameter":
            d = "Central diameter (mm): Diameter at the central section of the spring."
        elif f == "final_diameter":
            d = "Final diameter (mm): Diameter at the second end of the spring."
        elif f == "initial_conical_coils":
            d = "Initial conical coils: Number of conical coils at the first end."
        elif f == "final_conical_coils":
            d = "Final conical coils: Number of conical coils at the second end."
        elif f == "initial_coils_curvature":
            d = "Initial coils curvature (mm): Curvature of the initial coils (concave/convex)."
        elif f == "final_coils_curvature":
            d = "Final coils curvature (mm): Curvature of the final coils (concave/convex)."
        elif f == "note":
            d = "Notes: Free-form notes for special geometries or parameters."
        else:
            d = f"{f}: {fld.description or ''}"
        desc.append(d)
    sys = "You are a Simplex Rapid engineer. Analyze the spring image, the tables, the notes."

    name = fields[0] if len(fields) == 1 else short_schema_name("p", list(fields))
    return ask(
        [
            {"role": "system", "content": sys},
            {"role": "user", "content": [
                {"type": "input_text", "text": 
                    "Provide the following values.\n" + " • ".join(desc) + "\n\n"
                    "Fill in the JSON."
                },
                img
            ]}
        ],
        schema,
        name=name,
        acc=acc
    )

# ────────────────────────────── FIELDS TO EXTRACT ────────────────────────── #
COMMON_SINGLE = ["wire_diameter", "free_length", "total_coils"]
COMMON_COUPLES = [
    ["initial_closed_coils", "final_closed_coils"],
    ["pitch_insertion_coils", "pitch_retraction_coils"]
]
SPEC_FIELDS = {
    SpringType.CYLINDRICAL: ["external_diameter", "body_diameter_correction"],
    SpringType.CONICAL: ["minimum_diameter", "maximum_diameter", "concavity_convexity"],
    SpringType.BICONICAL: [
        "initial_diameter", "central_diameter", "final_diameter",
        "initial_conical_coils", "final_conical_coils",
        "initial_coils_curvature", "final_coils_curvature"
    ],
}
MODEL_MAP = {
    SpringType.CYLINDRICAL: CylindricalSpring,
    SpringType.CONICAL: ConicalSpring,
    SpringType.BICONICAL: BiconicalSpring,
    SpringType.CUSTOM: CustomSpring,
}
ALIASES = {"compression": SpringType.CYLINDRICAL}

# ────────────────────────────── PROCESSING LOGIC ─────────────────────────── #
def calculate_costs(model: str, usage) -> tuple[float, float]:
    """Return the cost in USD and EUR of a response."""
    price = PRICE.get(model, {"in":0,"out":0})
    usd = (usage.input_tokens / 1000) * price["in"] + \
          (usage.output_tokens / 1000) * price["out"]
    eur = usd * USD_TO_EUR
    return usd, eur

def log_usage(acc: list[dict[str,Any]], path: Path):
    """Log usage statistics to a CSV file."""
    usage_file = USAGE_DIR / f"{path.stem}.csv"
    with usage_file.open("w") as fp:
        fp.write("stage,in_tok,out_tok,USD,EUR\n")
        for row in acc:
            fp.write(f"{row['name']},{row['in_tokens']},{row['out_tokens']},{row['usd']},{row['eur']}\n")
        tot_eur = sum(r["eur"] for r in acc)
        fp.write(f"TOTAL,,,,{round(tot_eur,4)}\n")
    print(f"💰  Costo {path.stem}: {round(tot_eur,4)} €  "
          f"({round(tot_eur/USD_TO_EUR,4)} USD)")

def extract_spring_data(img: Dict[str, Any], path: Path, acc: list[dict[str,Any]]) -> dict:
    """Extract all spring data from the image using OpenAI prompts."""
    data: Dict[str, Any] = {}

    # 1) spring_type & wire_material
    info = prompt_type_and_material(img, acc)
    raw_type = info["spring_type"].lower()
    st_enum = ALIASES.get(raw_type, raw_type)
    try:
        st_enum = SpringType(st_enum)
    except ValueError:
        raise ValueError(f"spring_type «{raw_type}» is not valid")
    data["spring_type"] = st_enum.value
    data["wire_material"] = info["wire_material"]

    model_cls = MODEL_MAP[st_enum]

    # 2) Common parameters
    for f in COMMON_SINGLE:
        data.update(prompt_params([f], img, model_cls, acc))
    for duo in COMMON_COUPLES:
        data.update(prompt_params(duo, img, model_cls, acc))

    # 3) Geometry-specific parameters
    if st_enum != SpringType.CUSTOM:
        data.update(prompt_params(SPEC_FIELDS[st_enum], img, model_cls, acc))

    return data

def save_output(data: dict, path: Path):
    """Save the extracted data to a JSON file."""
    out = OUTPUT_DIR / f"{path.stem}.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    try:
        rel = out.relative_to(Path.cwd())
    except ValueError:
        rel = out
    print(f"✅  Saved → {rel}")
    if DEBUG:
        print("📝  RESULT:", json.dumps(data, indent=2, ensure_ascii=False))

def process_input(path: Path):
    """Process a single file (image o PDF): extract data, save output, log usage."""
    out = OUTPUT_DIR / f"{path.stem}.json"
    if out.exists():
        print(f"⏭️  Skipping {path.name}: output already exists.")
        return
    acc: list[dict[str,Any]] = []
    print(f"\n🔄  {path.name}")
    file_arg = encode_input(path)
    try:
        data = extract_spring_data(file_arg, path, acc)
        save_output(data, path)
        log_usage(acc, path)
    except Exception as e:
        print(f"❌  {path.name}: {e}", file=sys.stderr)

def run_batch_processing():
    """Process all file di input (immagini + PDF) presenti in INPUT_DIR."""
    files = [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in INPUT_EXTS]
    if not files:
        sys.exit(f"No valid input files found in {INPUT_DIR}")
    for p in files:
        process_input(p)

    print("\n═══════════════════════════════════════════════════════")
    print(f"📊  Totale run: {GRAND_TOTAL_REQ} request – "
          f"{round(GRAND_TOTAL_USD*USD_TO_EUR,4)} € "
          f"({round(GRAND_TOTAL_USD,4)} USD)")
    try:
        usage_dir_disp = USAGE_DIR.relative_to(Path.cwd())
    except ValueError:
        usage_dir_disp = USAGE_DIR
    print(f"📁  Dettaglio per immagine in {usage_dir_disp}")
    print("═══════════════════════════════════════════════════════")

def short_schema_name(prefix: str, fields: list[str]) -> str:
    digest = hashlib.sha1("_".join(fields).encode("utf-8")).hexdigest()[:12]
    base = f"{prefix}_{digest}"
    return base[:64]  # Ensure the name is within 64 characters

def main() -> None:
    setup_environment()
    prepare_directories()
    ensure_json_schemas_exist()
    run_batch_processing()

if __name__ == "__main__":
    main()
