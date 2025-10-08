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

from src.pricing import Pricing
from src.utilities.springs import (
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

# ────────────────────────────── TRACING ──────────────────────────── #
class Tracer:
    def __init__(self):
        self.trail = []
        self.now = lambda: __import__('datetime').datetime.now().isoformat()

    def start(self, label, meta=None):
        t0 = __import__('time').perf_counter()
        self.trail.append({"label": label, "at": self.now(), "status": "start", "meta": meta or {}})
        return TracerStep(self, label, t0)

class TracerStep:
    def __init__(self, tracer, label, t0):
        self.tracer = tracer
        self.label = label
        self.t0 = t0

    def log(self, data):
        self.tracer.trail.append({"label": self.label, "at": self.tracer.now(), "status": "log", "data": data})

    def ok(self, data=None):
        secs = round((__import__('time').perf_counter() - self.t0), 3)
        self.tracer.trail.append({"label": self.label, "at": self.tracer.now(), "status": "ok", "secs": secs, "data": data or {}})

    def fail(self, err):
        secs = round((__import__('time').perf_counter() - self.t0), 3)
        error = {
            "name": getattr(err, 'name', type(err).__name__),
            "message": str(getattr(err, 'message', err)),
            "stack": str(getattr(err, 'stack', '')).split('\n')[:4]
        }
        self.tracer.trail.append({"label": self.label, "at": self.tracer.now(), "status": "fail", "secs": secs, "error": error})

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
    return ask(
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": [
                {"type": "input_text", "text":
                    "Rileva i seguenti campi e restituisci SOLO il JSON:\n"
                    "• spring_function (enum: compression | torsion | other | not_a_spring)\n"
                    "• wire_material (enum: stainless_steel | chrome_silicon_steel | music_wire_steel)\n"
                    "• wire_diameter (numero, mm)\n"
                    "• spring_type (enum: cylindrical | conical | biconical | custom)"
                },
                img
            ]}
        ],
        schema=schema,
        schema_name=short_schema_name("meta", ["spring_function", "wire_material", "wire_diameter", "spring_type"]),
        acc=acc,
        stage_label="meta",
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
def extract_spring_data(img: Dict[str, Any], path: Path, acc: List[Dict[str, Any]], tracer: Tracer) -> Dict[str, Any]:
    step = tracer.start("extract_spring_data", {"path": str(path)})
    data: Dict[str, Any] = {}

    try:
        # 0) Combined meta: spring_function, wire_material, wire_diameter, spring_type
        meta = prompt_meta(img, acc)
        data.update(meta)
        step.log({"meta": meta})

        # Normalize spring_function
        data["spring_function"] = normalize_spring_function(data.get("spring_function", ""))

        # If not compression, stop
        if data["spring_function"] != "compression":
            step.ok({"reason": "not_compression", "function": data["spring_function"]})
            return data

        # If spring_type not provided or null, call separate prompt
        if not data.get("spring_type"):
            st_info = prompt_spring_type(img, acc)
            data["spring_type"] = st_info["spring_type"]
            step.log({"spring_type_from_separate": st_info})

        raw_type = data["spring_type"].lower()
        try:
            st_enum = SpringType(raw_type)
        except ValueError:
            raise ValueError(f"spring_type «{raw_type}» non valido")
        data["spring_type"] = st_enum.value

        model_cls = MODEL_MAP[st_enum]

        # Common params
        for f in COMMON_SINGLE:
            data.update(prompt_params([f], img, model_cls, acc))
        for duo in COMMON_COUPLES:
            data.update(prompt_params(duo, img, model_cls, acc))

        # Specific params
        if SPEC_FIELDS.get(st_enum):
            data.update(prompt_params(SPEC_FIELDS[st_enum], img, model_cls, acc))

        step.ok({"data": data})
        return data
    except Exception as e:
        step.fail(e)
        raise

def save_output(data: Dict[str, Any], path: Path, tracer: Tracer, acc: List[Dict[str, Any]]):
    data["debugTrail"] = tracer.trail
    # Add merged usage
    total_in = sum(r["in_tokens"] for r in acc)
    total_out = sum(r["out_tokens"] for r in acc)
    total_usd = sum(r["usd"] for r in acc)
    total_eur = sum(r["eur"] for r in acc)
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
    tracer = Tracer()
    print(f"\n🔄  [{MODEL}] {path.name}")
    wall_t0 = perf_counter()
    file_arg = encode_input(path)
    try:
        data = extract_spring_data(file_arg, path, acc, tracer)
        save_output(data, path, tracer, acc)
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
