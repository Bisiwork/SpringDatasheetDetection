#!/usr/bin/env python3
"""
main.py – Estrazione step-by-step con OpenAI *Responses* API + JSON Schema.
- Modello: gpt-5-mini (reasoning minimal, verbosity low)
- Descrizioni prese da springs.py (Pydantic)
- Flusso condizionale: spring_function -> (compression only) spring_type + params
- CSV 'usage': token, costo, tempo stage, totale API e wall-clock
"""

from __future__ import annotations
import base64
import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Sequence, Type

from dotenv import load_dotenv
import openai

from springs import (
    SpringBase, SpringType,
    CylindricalSpring, ConicalSpring, BiconicalSpring, CustomSpring,
)

# ────────────────────────────── CONFIG ────────────────────────────── #
DEBUG = True
MODEL = "gpt-5-mini"  # modello economico/veloce
IMG_EXTS = {".jpg", ".jpeg", ".png", ".gif"}
PDF_EXTS = {".pdf"}
INPUT_EXTS = IMG_EXTS | PDF_EXTS

INPUT_DIR = Path("data/sketch")
SCHEMA_DIR = Path("data/json_objects")
OUTPUT_DIR = Path("results")
USAGE_DIR = Path("results/usage")

# Prezzi indicativi; aggiorna col tuo listino
PRICE = {"gpt-5-mini": {"in": 0.005, "out": 0.015}}  # USD / 1K token (placeholder)
USD_TO_EUR = 0.92

GRAND_TOTAL_USD: float = 0.0
GRAND_TOTAL_REQ: int = 0

# ────────────────────────────── ENV / FS ─────────────────────────── #
def setup_environment():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY") or sys.exit("❌  OPENAI_API_KEY missing")

def prepare_directories():
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    USAGE_DIR.mkdir(parents=True, exist_ok=True)

def ensure_json_schemas_exist():
    """Garantisce la presenza dei template JSON Schema usati dalle chiamate."""
    required = {
        # Metadati
        "spring_function", "wire_material", "wire_diameter",
        # Compressione – comuni
        "spring_type", "free_length", "total_coils",
        "initial_closed_coils", "final_closed_coils",
        "pitch_insertion_coils", "pitch_retraction_coils",
        # Compressione – specifici
        "external_diameter", "body_diameter_correction",
        "minimum_diameter", "maximum_diameter", "concavity_convexity",
        "initial_diameter", "central_diameter", "final_diameter",
        "initial_conical_coils", "final_conical_coils",
        "initial_coils_curvature", "final_coils_curvature",
    }
    missing = [f for f in required if not (SCHEMA_DIR / f"{f}.json").exists()]
    if missing:
        print(f"🔧 Missing JSON schema(s): {', '.join(missing)}. Generating...")
        import subprocess
        subprocess.run([sys.executable, "src/make_json_templates.py"], check=True)
        print("✅  JSON schemas generated.")

# ────────────────────────────── SCHEMAS / ENCODING ───────────────── #
def _sanitize(schema: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(schema)
    out.pop("$id", None)
    out.pop("$ref", None)
    if "required" in out:
        out["required"] = list(dict.fromkeys(out["required"]))
    return out

def load_schema(*fields: str) -> Dict[str, Any]:
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

def encode_input(path: Path) -> Dict[str, Any]:
    ext = path.suffix.lower()
    if ext in IMG_EXTS:
        mime = f"image/{ext.lstrip('.')}"
        b64 = base64.b64encode(path.read_bytes()).decode()
        return {"type": "input_image", "image_url": f"data:{mime};base64,{b64}", "detail": "auto"}
    if ext in PDF_EXTS:
        b64 = base64.b64encode(path.read_bytes()).decode()
        return {"type": "input_file", "filename": path.name, "file_data": f"data:application/pdf;base64,{b64}"}
    raise ValueError(f"Unsupported file type: {path.suffix}")

def short_schema_name(prefix: str, fields: Sequence[str]) -> str:
    digest = hashlib.sha1("_".join(fields).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"[:64]  # vincolo API: max 64 char

# ────────────────────────────── COSTI ────────────────────────────── #
def cost_usd(model: str, usage) -> float:
    price = PRICE.get(model, {"in": 0, "out": 0})
    return (usage.input_tokens / 1000) * price["in"] + (usage.output_tokens / 1000) * price["out"]

# ────────────────────────────── OPENAI CALL ──────────────────────── #
def ask(
    messages: List[Dict[str, Any]],
    schema: Dict[str, Any],
    schema_name: str,
    acc: List[Dict[str, Any]],
    *,
    stage_label: str,
) -> Dict[str, Any]:
    if DEBUG:
        print("──────────────────── OPENAI REQUEST")
        for m in messages:
            role = m["role"].upper()
            txt = m["content"]
            if isinstance(txt, list):
                txt = next((c.get("text") for c in txt if c.get("type") == "input_text"), "")
            print(f"[{role}] {txt}\n")
        print("──────────────────── SCHEMA")
        print(json.dumps(schema, indent=2, ensure_ascii=False), "\n")

    t0 = perf_counter()
    rsp = openai.responses.create(
        model=MODEL,
        input=messages,
        reasoning={"effort": "minimal"},
        text={
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": schema,
                "strict": True,
            },
        },
    )
    dt = perf_counter() - t0  # seconds for this stage

    usd = cost_usd(MODEL, rsp.usage)
    eur = usd * USD_TO_EUR

    acc.append({
        "stage": stage_label,                       # label leggibile
        "in_tokens": rsp.usage.input_tokens,
        "out_tokens": rsp.usage.output_tokens,
        "usd": round(usd, 4),
        "eur": round(eur, 4),
        "secs": round(dt, 3),                       # tempo per stage
    })

    global GRAND_TOTAL_USD, GRAND_TOTAL_REQ
    GRAND_TOTAL_USD += usd
    GRAND_TOTAL_REQ += 1

    raw = rsp.output_text
    if DEBUG:
        print("──────────────────── OPENAI RESPONSE")
        print(raw, "\n")
    return json.loads(raw)

# ────────────────────────────── PROMPTS ──────────────────────────── #
def prompt_function_and_material(img: Dict[str, Any], acc: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Estrae solo metadati comuni a TUTTE le molle:
    - spring_function (compression|torsion)
    - wire_material
    """
    schema = load_schema("spring_function", "wire_material")
    sys_msg = "Sei un ingegnere Simplex Rapid. Analizza immagine, eventuali tabelle e note."
    return ask(
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": [
                {"type": "input_text", "text":
                    "Rileva i seguenti campi e restituisci SOLO il JSON:\n"
                    "• spring_function (enum: compression | torsion)\n"
                    "• wire_material (enum: stainless_steel | chrome_silicon_steel | music_wire_steel)"
                },
                img
            ]}
        ],
        schema=schema,
        schema_name=short_schema_name("function_and_material", ["spring_function", "wire_material"]),
        acc=acc,
        stage_label="spring_function+wire_material",
    )

def prompt_spring_type(img: Dict[str, Any], acc: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Chiamata separata: solo per molle a compressione (spring_type)."""
    schema = load_schema("spring_type")
    sys_msg = "Sei un ingegnere Simplex Rapid. Analizza immagine, eventuali tabelle e note."
    return ask(
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": [
                {"type": "input_text", "text":
                    "Per una molla a COMPRESSIONE, rileva e restituisci SOLO il JSON:\n"
                    "• spring_type (enum: cylindrical | conical | biconical | custom)"
                },
                img
            ]}
        ],
        schema=schema,
        schema_name="spring_type",
        acc=acc,
        stage_label="spring_type",
    )

def prompt_params(
    fields: Sequence[str],
    img: Dict[str, Any],
    model_cls: Type[SpringBase],
    acc: List[Dict[str, Any]],
) -> Dict[str, Any]:
    schema = load_schema(*fields)
    # Descrizioni direttamente da springs.py
    lines: List[str] = []
    for f in fields:
        fld = model_cls.model_fields[f]
        unit = " (mm)" if f.endswith(("diameter", "length")) else ""
        lines.append(f"{f}{unit}: {fld.description}")
    sys_msg = "Sei un ingegnere Simplex Rapid. Analizza immagine, eventuali tabelle e note."
    stage_label = "+".join(fields)  # leggibile nel CSV
    schema_name = fields[0] if len(fields) == 1 else short_schema_name("blk", list(fields))
    return ask(
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": [
                {"type": "input_text", "text": "Fornisci i seguenti valori e restituisci SOLO il JSON.\n" + " • ".join(lines)},
                img
            ]}
        ],
        schema=schema,
        schema_name=schema_name,
        acc=acc,
        stage_label=stage_label,
    )

# ────────────────────────────── CAMPI ────────────────────────────── #
# Metadati per qualunque molla
META_SINGLE = ["wire_diameter"]  # possiamo chiederlo sempre

# Parametri comuni alle MOLLE A COMPRESSIONE
COMMON_SINGLE = ["free_length", "total_coils"]
COMMON_COUPLES = [
    ["initial_closed_coils", "final_closed_coils"],
    ["pitch_insertion_coils", "pitch_retraction_coils"],
]
# Parametri specifici per geometria (compressione)
SPEC_FIELDS = {
    SpringType.CYLINDRICAL: ["external_diameter", "body_diameter_correction"],
    SpringType.CONICAL: ["minimum_diameter", "maximum_diameter", "concavity_convexity"],
    SpringType.BICONICAL: [
        "initial_diameter", "central_diameter", "final_diameter",
        "initial_conical_coils", "final_conical_coils",
        "initial_coils_curvature", "final_coils_curvature",
    ],
    SpringType.CUSTOM: [],
}
MODEL_MAP = {
    SpringType.CYLINDRICAL: CylindricalSpring,
    SpringType.CONICAL: ConicalSpring,
    SpringType.BICONICAL: BiconicalSpring,
    SpringType.CUSTOM: CustomSpring,
}

# ────────────────────────────── PIPELINE ─────────────────────────── #
def extract_spring_data(img: Dict[str, Any], path: Path, acc: List[Dict[str, Any]]) -> Dict[str, Any]:
    data: Dict[str, Any] = {}

    # 0) Funzione + materiale (valido per tutte le molle)
    meta = prompt_function_and_material(img, acc)
    data.update(meta)  # spring_function, wire_material

    # 0.1) Metadato: wire_diameter (lo chiediamo sempre)
    data.update(prompt_params(META_SINGLE, img, SpringBase, acc))

    # Se NON è compressione → STOP qui (evitiamo di interrogare campi “a compressione”)
    if (data.get("spring_function") or "").lower() != "compression":
        return data

    # 1) Tipo geometrico (solo compressione)
    st_info = prompt_spring_type(img, acc)
    raw_type = st_info["spring_type"].lower()
    try:
        st_enum = SpringType(raw_type)
    except ValueError:
        raise ValueError(f"spring_type «{raw_type}» non valido")
    data["spring_type"] = st_enum.value

    model_cls = MODEL_MAP[st_enum]

    # 2) Parametri comuni (compressione)
    for f in COMMON_SINGLE:
        data.update(prompt_params([f], img, model_cls, acc))
    for duo in COMMON_COUPLES:
        data.update(prompt_params(duo, img, model_cls, acc))

    # 3) Specifici per geometria (compressione)
    if SPEC_FIELDS.get(st_enum):
        data.update(prompt_params(SPEC_FIELDS[st_enum], img, model_cls, acc))

    return data

def save_output(data: Dict[str, Any], path: Path):
    out = OUTPUT_DIR / f"{path.stem}.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    try:
        rel = out.relative_to(Path.cwd())
    except ValueError:
        rel = out
    print(f"✅  Saved → {rel}")
    if DEBUG:
        print("📝  RESULT:", json.dumps(data, indent=2, ensure_ascii=False))

def log_usage(acc: List[Dict[str, Any]], path: Path, wall_secs: float):
    usage_file = USAGE_DIR / f"{path.stem}.csv"
    total_eur = sum(r["eur"] for r in acc)
    total_secs = sum(r["secs"] for r in acc)  # solo tempo API cumulato
    with usage_file.open("w", encoding="utf-8") as fp:
        fp.write("stage,in_tok,out_tok,USD,EUR,secs\n")
        for r in acc:
            fp.write(f"{r['stage']},{r['in_tokens']},{r['out_tokens']},{r['usd']},{r['eur']},{r['secs']}\n")
        fp.write(f"TOTAL,,,,{round(total_eur,4)},{round(total_secs,3)}\n")
        fp.write(f"WALL,,,,,{round(wall_secs,3)}\n")
    print(f"💰  Costo {path.stem}: {round(total_eur,4)} €  ({round(total_eur/USD_TO_EUR,4)} USD)")
    print(f"🕒  Tempi {path.stem}: API {round(total_secs,2)} s  |  Wall {round(wall_secs,2)} s")

def process_input(path: Path):
    out = OUTPUT_DIR / f"{path.stem}.json"
    if out.exists():
        print(f"⏭️  Skipping {path.name}: output already exists.")
        return
    acc: List[Dict[str, Any]] = []
    print(f"\n🔄  {path.name}")
    wall_t0 = perf_counter()
    file_arg = encode_input(path)
    try:
        data = extract_spring_data(file_arg, path, acc)
        save_output(data, path)
        wall_dt = perf_counter() - wall_t0
        log_usage(acc, path, wall_dt)
    except Exception as e:
        print(f"❌  {path.name}: {e}", file=sys.stderr)

def run_batch_processing():
    files = [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in INPUT_EXTS]
    if not files:
        sys.exit(f"No valid input files found in {INPUT_DIR}")
    for p in files:
        process_input(p)

    print("\n═══════════════════════════════════════════════════════")
    print(f"📊  Totale run: {GRAND_TOTAL_REQ} request – "
          f"{round(GRAND_TOTAL_USD*USD_TO_EUR,4)} €  ({round(GRAND_TOTAL_USD,4)} USD)")
    try:
        usage_dir_disp = USAGE_DIR.relative_to(Path.cwd())
    except ValueError:
        usage_dir_disp = USAGE_DIR
    print(f"📁  Dettaglio per file in {usage_dir_disp}")
    print("═══════════════════════════════════════════════════════")

def main() -> None:
    setup_environment()
    prepare_directories()
    ensure_json_schemas_exist()
    run_batch_processing()

if __name__ == "__main__":
    main()
