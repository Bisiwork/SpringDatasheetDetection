#!/usr/bin/env python3

import os
import sys
import json
import base64
from pathlib import Path

from dotenv import load_dotenv
import openai


# ───────── CONFIG ─────────
DEBUG      = True
MODEL      = "gpt-4o"
INPUT_DIR  = Path("data/sketch")
OUTPUT_DIR = Path("data/output")
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".gif"}

def setup_environment():
    """Load environment variables and set OpenAI API key."""
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY") or sys.exit("❌  OPENAI_API_KEY missing")

def encode_image(path: Path) -> dict:
    """
    Encode an image file as a base64 data URL for the OpenAI API.
    """
    mime = f"image/{path.suffix.lstrip('.').lower()}"
    b64  = base64.b64encode(path.read_bytes()).decode()
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "auto"}
    }

def detect_type_and_material(img: dict) -> dict:
    """
    Detect only spring_type and wire_material (JSON mode).
    Respond ONLY with a JSON like:
      { "spring_type": "...", "wire_material": "..." }
    Allowed enums are:
      spring_type: cylindrical | conical | biconical | custom
      wire_material: stainless_steel | chrome_silicon_steel | music_wire_steel
    """
    sys_prompt = (
        "You are a mechanical engineer at Simplex Rapid. "
        "Detect spring_type and wire_material from the drawing. "
        "Allowed values:\n"
        "- spring_type: cylindrical | conical | biconical | custom\n"
        "- wire_material: stainless_steel | chrome_silicon_steel | music_wire_steel\n"
        "Answer ONLY with JSON."
    )
    user_content = [
        {"type": "text", "text": "Return {\"spring_type\":\"…\",\"wire_material\":\"…\"}"},
        img
    ]
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role":   "user", "content": user_content}
    ]

    if DEBUG:
        print("──────────────────── REQUEST")
        for m in messages:
            role = m["role"].upper()
            content = m["content"]
            if isinstance(content, list):
                txt = next((c.get("text","") for c in content if c.get("type")=="text"), "")
            else:
                txt = content
            print(f"[{role}] {txt}")
        print()

    rsp = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"}
    )

    payload = rsp.choices[0].message.content
    if DEBUG:
        print("──────────────────── RESPONSE")
        print(payload, "\n")

    return json.loads(payload)


def main():
    setup_environment()

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    imgs = [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in IMG_EXTS]
    if not imgs:
        sys.exit("No images found in data/sketch")

    for img_path in imgs:
        print(f"\n🔄  {img_path.name}")
        img_ref = encode_image(img_path)

        # 1) spring_type + wire_material
        info = detect_type_and_material(img_ref)

        # Save output to data/output/
        out_path = OUTPUT_DIR / f"{img_path.stem}_output.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)



if __name__ == "__main__":
    main()
