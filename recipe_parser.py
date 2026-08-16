"""Tarif .txt dosyası okuma."""
import os
import re

from text_utils import normalize

MAX_UPLOAD_BYTES = 200_000
MAX_TITLE_LEN = 200
MAX_TEXT_LEN = 50_000
MAX_INGREDIENTS = 40
MAX_INGREDIENT_LEN = 80
MAX_FILENAME_LEN = 100


def safe_filename(name):
    base = os.path.basename(str(name or "tarif.txt")).strip() or "tarif.txt"
    base = re.sub(r"[^\w.\- ]+", "_", base, flags=re.UNICODE)
    base = base.strip("._ ") or "tarif.txt"
    if not base.lower().endswith(".txt"):
        base = base[: MAX_FILENAME_LEN - 4] + ".txt"
    return base[:MAX_FILENAME_LEN]


def clean_text(value, max_len):
    text = str(value or "").replace("\x00", "").strip()
    return text[:max_len]


def clean_ingredients(items):
    cleaned = []
    for item in items or []:
        if not isinstance(item, str):
            item = str(item)
        item = clean_text(item, MAX_INGREDIENT_LEN)
        if item:
            cleaned.append(item)
        if len(cleaned) >= MAX_INGREDIENTS:
            break
    return cleaned


def parse_recipe_bytes(filename, raw_bytes):
    if raw_bytes is None:
        raise ValueError("Dosya boş.")
    data = bytes(raw_bytes)
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Dosya çok büyük (en fazla 200 KB).")
    if not data.strip():
        raise ValueError("Dosya boş.")

    text = data.decode("utf-8", errors="ignore").replace("\x00", "").strip()
    text = text[:MAX_TEXT_LEN]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    title = clean_text(lines[0] if lines else filename, MAX_TITLE_LEN)

    ingredients = []
    for line in lines:
        if normalize(line).startswith("malzemeler"):
            part = line.split(":", 1)[-1] if ":" in line else line
            ingredients = [item.strip(" .") for item in part.split(",") if item.strip(" .")]
            break
    if not ingredients:
        ingredients = [item.strip() for item in re.findall(r"[^,.\n]+", text)[:8] if item.strip()]

    return title, text, clean_ingredients(ingredients)


def extract_yapilis(full_text):
    text = clean_text(full_text, MAX_TEXT_LEN)
    for marker in ("Yapılışı:", "Yapilisi:", "Yapılış:", "Yapilis:"):
        idx = text.find(marker)
        if idx != -1:
            return text[idx + len(marker) :].strip()[:MAX_TEXT_LEN]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text
