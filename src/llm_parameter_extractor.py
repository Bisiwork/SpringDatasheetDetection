"""
llm_parameter_extractor.py - Modulo per estrazione parametri da datasheet utilizzando LLM OpenAI.

Questo modulo gestisce l'estrazione automatica di parametri tecnici da immagini/PDF
di datasheet di molle utilizzando l'API OpenAI Responses con JSON Schema validation.
Implementa un flusso condizionale basato su spring_function e spring_type, con
prompt engineering specifico per ciascun campo.

Caratteristiche principali:
- Supporto multi-modello (GPT-4o, GPT-5 family)
- Schema JSON strutturato basato su Pydantic models (spring_models.py)
- Tracing dettagliato e logging costi/tempi
- Gestione batch e caching risultati

Input: Immagini/PDF in data/sketch o data/all
Output: JSON predizioni in results/{model}/outputs/, CSV usage in results/{model}/usage/
Dipendenze: openai_pricing_manager.py, utilities/spring_models.py, utilities/json_schema_generator.py
"""

from __future__ import annotations
import base64
import hashlib
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Sequence, Type

from dotenv import load_dotenv
import openai

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.openai_pricing_manager import Pricing
from src.utilities.spring_models import (
    SpringBase,
    SpringType,
    CylindricalSpring,
    ConicalSpring,
    BiconicalSpring,
    CustomSpring,
)

# ────────────────────────────── CONFIG ────────────────────────────── #
DEBUG = True
MODEL = "gpt-4o"  # default, sarà sovrascritto da run_for_model()
IMG_EXTS = {".jpg", ".jpeg", ".png", ".gif"}
PDF_EXTS = {".pdf"}
INPUT_EXTS = IMG_EXTS | PDF_EXTS

# Queste directory verranno reimpostate da run_for_model() per modello
INPUT_DIR = Path("data/sketch")
SCHEMA_DIR = Path("data/json_objects")
OUTPUT_DIR = Path("results")      # verrà mutato in results/{model}/outputs
USAGE_DIR = Path("results/usage") # verrà mutato in results/{model}/usage

# Carica prezzi (per default usiamo "Text tokens - Standard" dal tuo CSV)
PRICING_CANDIDATES = [
    Path("openai_pricing.csv"),
    ROOT_DIR / "openai_pricing.csv",
    Path(__file__).resolve().parent / "utilities" / "openai_pricing.csv",
]


def resolve_pricing_csv(explicit: Path | str | None = None) -> Path:
    if explicit is not None:
        path = Path(explicit)
        if path.exists():
            return path
        raise FileNotFoundError(f"File pricing non trovato: {path}")

    for candidate in PRICING_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Impossibile trovare 'openai_pricing.csv'. Posizionalo nella root del progetto "
        "o in src/utilities/openai_pricing.csv."
    )


PRICER = Pricing(resolve_pricing_csv(), category="Text tokens - Standard")

USD_TO_EUR = 0.92

GRAND_TOTAL_USD: float = 0.0
GRAND_TOTAL_REQ: int = 0
FORCE_REGEN = False

GPT5_FAMILY = ("gpt-5", "gpt-5-mini", "gpt-5-nano")

# ────────────────────────────── ENV / FS ─────────────────────────── #
def setup_environment():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY") or sys.exit("❌  OPENAI_API_KEY missing")

def prepare_directories():
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
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

        script = Path(__file__).resolve().parent / "utilities" / "make_json_templates.py"
        subprocess.run([sys.executable, str(script)], check=True)
        print("✅  JSON schemas generated.")


def _collect_input_files(directory: Path) -> List[Path]:
    """Ritorna i file di input validi (immagini/PDF) ordinati alfabeticamente, uno per stem (preferisci jpg su pdf)."""
    try:
        files = [
            p
            for p in directory.iterdir()
            if p.suffix.lower() in INPUT_EXTS and "thumb" not in p.stem
        ]
    except FileNotFoundError:
        sys.exit(f"Input directory non trovata: {directory}")
    # Raggruppa per stem
    from collections import defaultdict
    grouped = defaultdict(list)
    for p in files:
        grouped[p.stem].append(p)
    # Per ogni gruppo, preferisci jpg, altrimenti pdf
    selected = []
    for stem, paths in grouped.items():
        jpgs = [p for p in paths if p.suffix.lower() in IMG_EXTS]
        pdfs = [p for p in paths if p.suffix.lower() in PDF_EXTS]
        if jpgs:
            selected.append(jpgs[0])
        elif pdfs:
            selected.append(pdfs[0])
    return sorted(selected, key=lambda p: p.name)



# ────────────────────────────── SCHEMAS / ENCODING ───────────────── #
def _sanitize(schema: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(schema)
    out.pop("$id", None)
    out.pop("$ref", None)
    if "required" in out:
        out["required"] = list(dict.fromkeys(out["required"]))
    # Add nullable for optional fields like JS
    nullable_fields = {"wire_material", "wire_diameter", "spring_type"}
    if "properties" in out:
        for field in nullable_fields:
            if field in out["properties"]:
                out["properties"][field]["nullable"] = True
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
    # Responses API addebita ai prezzi del modello scelto; gestiamo cached tokens.
    return PRICER.cost_usd(model, usage)

def is_gpt5(model: str) -> bool:
    return any(model.startswith(prefix) for prefix in GPT5_FAMILY)

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

    print(f"📝 Logging request: method={MODEL}, path={stage_label}")

    t0 = perf_counter()

    text_cfg = {
        "format": {
            "type": "json_schema",
            "name": schema_name,
            "strict": True,
            "schema": schema
        }
    }
    req = {
        "model": MODEL,
        "input": messages,
        "text": text_cfg,
    }
    if is_gpt5(MODEL):
        req["reasoning"] = {"effort": "medium"}
        text_cfg["verbosity"] = "medium"

    rsp = openai.responses.create(**req)
    dt = perf_counter() - t0

    usd = cost_usd(MODEL, rsp.usage)
    eur = usd * USD_TO_EUR
    acc.append({
        "stage": stage_label,
        "in_tokens": rsp.usage.input_tokens,
        "out_tokens": rsp.usage.output_tokens,
        "usd": round(usd, 4),
        "eur": round(eur, 4),
        "secs": round(dt, 3),
    })

    global GRAND_TOTAL_USD, GRAND_TOTAL_REQ
    GRAND_TOTAL_USD += usd
    GRAND_TOTAL_REQ += 1

    parsed = getattr(rsp, 'output_parsed', None)
    if parsed is not None:
        if isinstance(parsed, list) and len(parsed) == 1:
            data = parsed[0]
        else:
            data = parsed
    else:
        text_out = getattr(rsp, 'output_text', '')
        if not text_out:
            chunks = []
            for item in getattr(rsp, 'output', []) or []:
                for part in getattr(item, 'content', []) or []:
                    chunks.append(getattr(part, 'text', ''))
            text_out = ''.join(chunks)
        data = json.loads(text_out) if text_out else {}
    if DEBUG:
        print("──────────────────── OPENAI RESPONSE")
        print(json.dumps(data, indent=2, ensure_ascii=False), "\n")
    return data

# ────────────────────────────── UTILS ──────────────────────────── #
def normalize_spring_function(func: str) -> str:
    """Normalize spring_function like JS."""
    if not func:
        return "not_a_spring"
    lower = func.lower().strip()
    if lower in ("compression", "torsion"):
        return lower
    if lower == "not_supported":
        return "other"
    if lower == "not_a_spring":
        return lower
    return "not_a_spring"

# ────────────────────────────── PROMPTS ──────────────────────────── #
def prompt_meta(img: Dict[str, Any], acc: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Combined meta extraction: spring_function, wire_material, wire_diameter, spring_type
    """
    schema = load_schema("spring_function", "wire_material", "wire_diameter", "spring_type")
    sys_msg = "Sei un ingegnere Simplex Rapid. Analizza immagine, eventuali tabelle e note."
    text = "Rileva i seguenti campi e restituisci SOLO il JSON:\n• spring_function: Spring operating mode\n• wire_material: Wire material\n• wire_diameter: Wire diameter (mm)\n• spring_type: Geometric family"
    return ask(
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": [
                {"type": "input_text", "text": text},
                img
            ]}
        ],
        schema=schema,
        schema_name="meta",
        acc=acc,
        stage_label="meta",
    )



def prompt_common_params(
    img: Dict[str, Any],
    model_cls: Type[SpringBase],
    acc: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Combined call for all common compression parameters."""
    common_fields = [
        "free_length", "total_coils",
        "initial_closed_coils", "final_closed_coils",
        "pitch_insertion_coils", "pitch_retraction_coils"
    ]
    schema = load_schema(*common_fields)
    lines: List[str] = []
    for f in common_fields:
        fld = model_cls.model_fields[f]
        unit = " (mm)" if f.endswith(("diameter", "length")) else ""
        lines.append(f"• {f}{unit}: {fld.description}")
    sys_msg = "Sei un ingegnere Simplex Rapid. Analizza immagine, eventuali tabelle e note."
    text = "Fornisci i seguenti valori per una molla a COMPRESSIONE e restituisci SOLO il JSON.\n" + "\n".join(lines)
    return ask(
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": [
                {"type": "input_text", "text": text},
                img
            ]}
        ],
        schema=schema,
        schema_name="common_params",
        acc=acc,
        stage_label="common_params",
    )

def prompt_specific_params(
    spring_type: SpringType,
    img: Dict[str, Any],
    model_cls: Type[SpringBase],
    acc: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Combined call for type-specific parameters."""
    spec_fields = SPEC_FIELDS.get(spring_type, [])
    if not spec_fields:
        return {}

    schema = load_schema(*spec_fields)
    lines: List[str] = []
    for f in spec_fields:
        fld = model_cls.model_fields[f]
        unit = " (mm)" if f.endswith(("diameter", "length")) else ""
        lines.append(f"• {f}{unit}: {fld.description}")

    type_name = spring_type.value.upper()
    sys_msg = "Sei un ingegnere Simplex Rapid. Analizza immagine, eventuali tabelle e note."
    text = f"Fornisci i seguenti valori per una molla a COMPRESSIONE di tipo {type_name} e restituisci SOLO il JSON.\n" + "\n".join(lines)
    return ask(
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": [
                {"type": "input_text", "text": text},
                img
            ]}
        ],
        schema=schema,
        schema_name=f"{spring_type.value}_params",
        acc=acc,
        stage_label=f"{spring_type.value}_params",
    )

def prompt_params(
    fields: Sequence[str],
    img: Dict[str, Any],
    model_cls: Type[SpringBase],
    acc: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Legacy function for backward compatibility."""
    schema = load_schema(*fields)
    lines: List[str] = []
    for f in fields:
        fld = model_cls.model_fields[f]
        unit = " (mm)" if f.endswith(("diameter", "length")) else ""
        lines.append(f"• {f}{unit}: {fld.description}")
    sys_msg = "Sei un ingegnere Simplex Rapid. Analizza immagine, eventuali tabelle e note."
    stage_label = "+".join(fields)
    schema_name = fields[0] if len(fields) == 1 else short_schema_name("blk", list(fields))
    text = "Fornisci i seguenti valori e restituisci SOLO il JSON.\n" + "\n".join(lines)
    return ask(
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": [
                {"type": "input_text", "text": text},
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

# Parametri comuni alle MOLLE A COMPRESSIONE (ora combinati in una chiamata)
COMMON_FIELDS = [
    "free_length", "total_coils",
    "initial_closed_coils", "final_closed_coils",
    "pitch_insertion_coils", "pitch_retraction_coils"
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

    try:
        # 0) Combined meta: spring_function, wire_material, wire_diameter, spring_type
        meta = prompt_meta(img, acc)
        data.update(meta)

        # Normalize spring_function
        data["spring_function"] = normalize_spring_function(data.get("spring_function", ""))

        # If not compression, stop
        if data["spring_function"] != "compression":
            return data

        # If spring_type is null, skip specific params
        if not data.get("spring_type"):
            return data

        raw_type = data["spring_type"].lower()
        try:
            st_enum = SpringType(raw_type)
        except ValueError:
            raise ValueError(f"spring_type «{raw_type}» non valido")
        data["spring_type"] = st_enum.value

        model_cls = MODEL_MAP[st_enum]

        # Sequential calls for common and specific params to avoid threading issues
        common_data = prompt_common_params(img, model_cls, acc)
        specific_data = prompt_specific_params(st_enum, img, model_cls, acc)

        data.update(common_data)
        data.update(specific_data)

        return data
    except Exception as e:
        raise

def save_output(data: Dict[str, Any], path: Path, acc: List[Dict[str, Any]]):
    # Add merged usage
    total_in = sum(r["in_tokens"] for r in acc)
    total_out = sum(r["out_tokens"] for r in acc)
    total_usd = sum(r["usd"] for r in acc)
    total_eur = sum(r["eur"] for r in acc)
    # For sequential calls, but treating as if parallel: secs is meta + max(common, specific)
    if len(acc) >= 3:  # meta + common + specific
        meta_secs = acc[0]["secs"]
        common_secs = acc[1]["secs"]
        specific_secs = acc[2]["secs"]
        total_secs = meta_secs + max(common_secs, specific_secs)
    else:
        total_secs = sum(r["secs"] for r in acc)
    data["usage"] = {
        "in_tokens": total_in,
        "out_tokens": total_out,
        "usd": round(total_usd, 4),
        "eur": round(total_eur, 4),
        "secs": round(total_secs, 3),
    }
    out = OUTPUT_DIR / f"{path.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
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
    if OUTPUT_DIR.exists() and not FORCE_REGEN:
        print(f"⏭️  Skipping {path.name}: output already exists.")
        return
    acc: List[Dict[str, Any]] = []
    print(f"\n🔄  [{MODEL}] {path.name}")
    wall_t0 = perf_counter()
    file_arg = encode_input(path)
    try:
        data = extract_spring_data(file_arg, path, acc)
        save_output(data, path, acc)
        wall_dt = perf_counter() - wall_t0
        log_usage(acc, path, wall_dt)
    except Exception as e:
        print(f"❌  [{MODEL}] {path.name}: {e}", file=sys.stderr)

def run_batch_processing(files: Sequence[Path] | None = None):
    if files is None:
        candidates = _collect_input_files(INPUT_DIR)
        if not candidates:
            sys.exit(f"No valid input files found in {INPUT_DIR}")
    else:
        candidates = list(files)
        if not candidates:
            return
    for p in candidates:
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

def run_for_model(
    model: str,
    *,
    category: str = "Text tokens - Standard",
    input_dir: Path | str = "data/sketch",
    output_root: Path | str = "results",
    pricing_csv: Path | str | None = None,
    force: bool = False,
):
    """Esegue l'intera pipeline sul dataset per un dato modello e salva in results/{model}/..."""
    global MODEL, INPUT_DIR, OUTPUT_DIR, USAGE_DIR, PRICER, GRAND_TOTAL_USD, GRAND_TOTAL_REQ, FORCE_REGEN
    MODEL = model
    INPUT_DIR = Path(input_dir)
    output_root = Path(output_root)
    OUTPUT_DIR = output_root / model / "outputs"
    USAGE_DIR = output_root / model / "usage"
    FORCE_REGEN = force

    inputs = _collect_input_files(INPUT_DIR)
    if not inputs:
        sys.exit(f"No valid input files found in {INPUT_DIR}")
    if not force:
        pending = [p for p in inputs if not (OUTPUT_DIR / f"{p.stem}.json").exists()]
    else:
        pending = inputs

    if not pending:
        print(f"⏭️  {model}: risultati già presenti, salto le chiamate OpenAI.")
        return

    if force:
        print(f"♻️  {model}: rielaboro {len(pending)} file.")
    elif len(pending) < len(inputs):
        print(f"ℹ️  {model}: {len(pending)} file senza output (su {len(inputs)}).")

    setup_environment()
    pricing_path = resolve_pricing_csv(pricing_csv)
    PRICER = Pricing(pricing_path, category=category)
    GRAND_TOTAL_USD = 0.0
    GRAND_TOTAL_REQ = 0
    prepare_directories()
    ensure_json_schemas_exist()
    run_batch_processing(pending)


def main() -> None:
    models = ["gpt-4o", "gpt-5-nano", "gpt-5-mini", "gpt-5", "gpt-4o-mini"]
    for model in models:
        run_for_model(model)

if __name__ == "__main__":
    main()
