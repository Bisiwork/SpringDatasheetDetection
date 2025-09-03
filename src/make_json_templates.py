#!/usr/bin/env python3
"""
make_json_templates.py
----------------------
Genera per ciascun parametro di springs.py un JSON object
(stile “translationsDocument.json” nel tuo esempio) con:
- $id: "<param>.json#"
- type: "object"
- $ref: "springsBaseFields.json#"
- properties: { <param>: { … } }
- required: [<param>]
- additionalProperties: false

Applica inoltre:
- per “spring_type” e “wire_material” gli enum corretti;
- “integer” vs “number” a seconda del tipo;
- “minimum”: 0 per tutti i campi numerici ≥0.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Set, Type, get_origin, get_args

from springs import SpringBase, SpringType, WireMaterial

OUT_DIR = Path("data/json_objects")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------- utils
def collect_models() -> Set[Type[SpringBase]]:
    seen: Set[Type[SpringBase]] = set()
    stack = [SpringBase]
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
    return seen

def base_py_type(ann):
    """Estrai int/float/str anche da Optional[...]"""
    origin = get_origin(ann)
    if origin is None:
        return ann
    args = [a for a in get_args(ann) if a is not type(None)]
    return args[0] if args else str

def json_type(py: type) -> str:
    if py is int:
        return "integer"
    if py is float:
        return "number"
    return "string"

# -------------------------------------------------- build per-param schema
def build_json_object(param: str, field) -> Dict[str, Any]:
    py = base_py_type(field.annotation)
    jtype = json_type(py)
    prop: Dict[str, Any] = {"type": jtype}

    # descrizione
    if field.description:
        prop["description"] = field.description

    # enum per spring_type e wire_material
    if param == "spring_type":
        prop["enum"] = [e.value for e in SpringType]
    elif param == "wire_material":
        prop["enum"] = [e.value for e in WireMaterial]

    # minimum per numeri ≥ 0
    if jtype in ("integer", "number"):
        prop["minimum"] = 0

    return {
        "$id": f"{param}.json#",
        "type": "object",
        "$ref": "springsBaseFields.json#",
        "properties": {
            param: prop
        },
        "required": [param],
        "additionalProperties": False
    }

# -------------------------------------------------- main
def main() -> None:
    # raccogli un solo Field per nome parametro
    param_seen: Dict[str, Any] = {}
    for model in collect_models():
        for name, fld in model.model_fields.items():  # type: ignore[attr-defined]
            param_seen.setdefault(name, fld)

    # genera e salva
    for name, fld in sorted(param_seen.items()):
        schema = build_json_object(name, fld)
        out_path = OUT_DIR / f"{name}.json"
        out_path.write_text(json.dumps(schema, indent=2, ensure_ascii=False))
        # print(f"🔧 JSON object per `{name}`:\n")
        # print(json.dumps(schema, indent=2, ensure_ascii=False))
        # print()

    print(f"✅  Creati {len(param_seen)} file JSON in {OUT_DIR}")

if __name__ == "__main__":
    main()
