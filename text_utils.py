"""Türkçe metin işleme."""
import re

VEGETABLES = {
    "sebze", "patlican", "kabak", "domates", "biber", "sogan", "havuc", "patates",
    "salatalik", "maydanoz", "fasulye", "bezelye", "ispanak", "lahana", "brokoli",
    "kereviz", "pirasa", "turp", "sarimsak", "roka", "marul",
}
MEAT = {
    "kiyma", "tavuk", "iskembe", "dana", "kuzu", "hindi", "balik", "sosis", "sucuk",
    "pastirma", "jambon", "biftek", "et", "kofte", "bonfile", "salam",
}
CATEGORIES = {
    "sebze": VEGETABLES,
    "et": MEAT,
    "etli": MEAT,
    "tatli": {"tatli", "baklava", "sutlac", "brownie", "cikolata", "kurabiye", "helva"},
}
DIET_WORDS = {
    "vegan": "vegan",
    "veganlar": "vegan",
    "vejetaryen": "vegetarian",
    "vejeteryan": "vegetarian",
    "vegetarian": "vegetarian",
    "vejetaryenler": "vegetarian",
}
STOP = {
    "var", "yok", "kac", "tane", "adet", "tarif", "tarifler", "yapabilirim", "oner",
    "onerir", "misin", "ne", "bir", "ile", "icin", "bu", "ve", "mi", "mu", "benim",
    "bende", "elimde", "nasil", "yapilir", "yapilisi", "onerisi", "lutfen",
}


def normalize(text):
    if text is None:
        return ""
    return (
        str(text)
        .replace("İ", "i")
        .replace("I", "i")
        .lower()
        .replace("ı", "i")
        .replace("ğ", "g")
        .replace("ü", "u")
        .replace("ş", "s")
        .replace("ö", "o")
        .replace("ç", "c")
        .replace("\u0307", "")
    )


def stem(word):
    for suffix in ("larim", "lerim", "um", "im", "m", "lar", "ler"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if word.endswith("ug") and len(word) >= 4:
        word = word[:-1] + "k"
    return word


def expand_terms(terms):
    out = set(terms)
    for term in list(terms):
        out |= CATEGORIES.get(term, set())
    return out


def has_term(text, term):
    return bool(re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", normalize(text)))


def parse_query(question):
    q = normalize(question)
    include, exclude = set(), set()
    diet = None

    for word in re.findall(r"[a-z]+", q):
        if word in DIET_WORDS:
            mapped = DIET_WORDS[word]
            if diet != "vegan":
                diet = mapped
            continue
        if word in STOP or len(word) < 3:
            continue
        if word.endswith(("siz", "suz")) and len(word) - 3 >= 2:
            exclude.add(stem(word[:-3]))
        else:
            include.add(stem(word))

    for match in re.finditer(r"([a-z]{3,})\s+olmadan", q):
        exclude.add(stem(match.group(1)))

    return expand_terms(include), expand_terms(exclude), diet
