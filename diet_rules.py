"""Diyet etiketleri."""
from text_utils import MEAT, has_term, normalize

ANIMAL_NON_VEGAN = {
    "sut", "yogurt", "peynir", "kasar", "tereyag", "yumurta", "bal", "krema",
    "ayran", "kaymak", "labne",
}


def ingredient_blob(recipe):
    return normalize(" ".join(recipe.get("ingredients", [])))


def is_vegetarian(recipe):
    text = ingredient_blob(recipe)
    return not any(has_term(text, term) for term in MEAT)


def is_vegan(recipe):
    if not is_vegetarian(recipe):
        return False
    text = ingredient_blob(recipe)
    return not any(has_term(text, term) for term in ANIMAL_NON_VEGAN)


def diet_label(recipe):
    if is_vegan(recipe):
        return "vegan"
    if is_vegetarian(recipe):
        return "vejetaryen"
    return "et içeren"


def matches_diet(recipe, diet):
    if diet == "vegan":
        return is_vegan(recipe)
    if diet == "vegetarian":
        return is_vegetarian(recipe)
    return True
