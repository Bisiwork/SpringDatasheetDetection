"""
json_schema_generator.py - Generatore schemi JSON per validazione API OpenAI.

Questo script genera schemi JSON strutturati per ciascun parametro definito
nei modelli Pydantic di spring_models.py. Gli schemi sono utilizzati dall'API OpenAI
Responses per validare e strutturare le risposte dei modelli LLM.

Caratteristiche:
- Genera schema per ogni campo (spring_function, wire_diameter, etc.)
- Include enum per campi categorici, vincoli minimi per numerici
- Formato compatibile con JSON Schema e OpenAI API
- Riferimento a base fields comune

Input: Modelli Pydantic in spring_models.py
Output: File JSON in data/json_objects/ per ogni parametro
Dipendenze: spring_models.py
"""

from __future__ import annotations
import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Set, Type, get_args, get_origin, Literal

from springs import ENUM_FIELD_MAP, SpringBase

OUT_DIR = Path("data/json_objects")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------- utils
def collect_models() -> list[Type[SpringBase]]:
    seen: Set[Type[SpringBase]] = set()
    ordered: list[Type[SpringBase]] = []
    stack: list[Type[SpringBase]] = [SpringBase]
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        ordered.append(cls)
        stack.extend(cls.__subclasses__())
    return ordered


def base_py_type(ann):
    origin = get_origin(ann)
    if origin is None:
        if isinstance(ann, Enum):
            return type(ann)
        if isinstance(ann, type):
            return ann
        return type(ann)

    if origin is Literal:
        for arg in get_args(ann):
            if arg is type(None):
                continue
            if isinstance(arg, Enum):
                return type(arg)
            return type(arg)
        return str

    args = [a for a in get_args(ann) if a is not type(None)]
    if not args:
        return str

    return base_py_type(args[0])


def json_type(py: type) -> str:
    if py is int:
        return "integer"
    if py is float:
        return "number"
    return "string"


def enum_values_for(param: str, annotation) -> list[Any] | None:
    if param in ENUM_FIELD_MAP:
        enum_cls = ENUM_FIELD_MAP[param]
        return [member.value for member in enum_cls]

    origin = get_origin(annotation)
    if origin is None:
        if isinstance(annotation, type) and issubclass(annotation, Enum):
            return [member.value for member in annotation]
        if isinstance(annotation, Enum):
            return [annotation.value]
        return None

    values: list[Any] = []
    for arg in get_args(annotation):
        if arg is type(None):
            continue
        if isinstance(arg, Enum):
            values.append(arg.value)
        elif isinstance(arg, type) and issubclass(arg, Enum):
            values.extend(member.value for member in arg)
        else:
            nested = enum_values_for(param, arg)
            if nested:
                values.extend(nested)

    if not values:
        return None
    return list(dict.fromkeys(values))


# -------------------------------------------------- build per-param schema
def build_json_object(param: str, field) -> Dict[str, Any]:
    py = base_py_type(field.annotation)
    jtype = json_type(py)
    prop: Dict[str, Any] = {"type": jtype}

    if field.description:
        prop["description"] = field.description

    enums = enum_values_for(param, field.annotation)
    if enums:
        prop["enum"] = enums

    if jtype in ("integer", "number"):
        prop["minimum"] = 0

    return {
        "$id": f"{param}.json#",
        "type": "object",
        "$ref": "springsBaseFields.json#",
        "properties": {param: prop},
        "required": [param],
        "additionalProperties": False,
    }


# -------------------------------------------------- main
def main() -> None:
    param_seen: Dict[str, Any] = {}
    for model in collect_models():
        for name, fld in model.model_fields.items():  # type: ignore[attr-defined]
            param_seen.setdefault(name, fld)

    for name, fld in sorted(param_seen.items()):
        schema = build_json_object(name, fld)
        out_path = OUT_DIR / f"{name}.json"
        out_path.write_text(json.dumps(schema, indent=2, ensure_ascii=False))

    print(f"✅  Creati {len(param_seen)} file JSON in {OUT_DIR}")


if __name__ == "__main__":
    main()
