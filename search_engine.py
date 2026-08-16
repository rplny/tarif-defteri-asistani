"""Tarif arama ve cevap metni."""
import database
import diet_rules
import recipe_parser
import text_utils
from embeddings import local_embed
from retrieval import cosine_similarity


def recipe_text(recipe):
    return text_utils.normalize(
        recipe["title"] + " " + " ".join(recipe["ingredients"]) + " " + recipe["full_text"]
    )


def search_recipes(conn, question, limit=8):
    question = str(question or "")[:300]
    limit = max(1, min(int(limit), 20))
    include, exclude, diet = text_utils.parse_query(question)
    recipes = []
    for recipe in database.get_all_recipes(conn):
        if not diet_rules.matches_diet(recipe, diet):
            continue
        text = recipe_text(recipe)
        if any(text_utils.has_term(text, term) for term in exclude):
            continue
        recipes.append(recipe)

    if (diet or exclude) and not include:
        return recipes[:limit]

    scored = []
    query_vec = local_embed(question)
    for recipe in recipes:
        text = recipe_text(recipe)
        hits = sum(1 for term in include if text_utils.has_term(text, term))
        vec = recipe.get("embedding") or local_embed(text)
        sim = cosine_similarity(query_vec, vec) if hits else 0.0
        scored.append((hits, sim, recipe))

    if include:
        keyword_hits = [item for item in scored if item[0] > 0]
        keyword_hits.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [recipe for _, _, recipe in keyword_hits[:limit]]
    if diet or exclude:
        return recipes[:limit]
    return []


def format_recipe_detail(recipe):
    ingredients = ", ".join(str(item) for item in recipe.get("ingredients", []))
    yapilis = recipe_parser.extract_yapilis(recipe.get("full_text", ""))
    title = str(recipe.get("title", ""))
    label = diet_rules.diet_label(recipe)
    return (
        title
        + " ["
        + label
        + "]\n"
        + "Malzemeler: "
        + ingredients
        + "\n"
        + "Yapılışı: "
        + yapilis
    )


def build_answer(question, recipes):
    if not str(question or "").strip():
        return "Boş soru gönderildi."
    if not recipes:
        return "Bu konuda tarif bulunamadı."

    details = "\n\n".join(format_recipe_detail(recipe) for recipe in recipes)
    short_lines = [
        "- " + str(recipe["title"]) + " [" + diet_rules.diet_label(recipe) + "]"
        for recipe in recipes
    ]
    q = text_utils.normalize(question)

    if any(token in q for token in ("kac", "tane", "adet")):
        return (
            "Uygun "
            + str(len(recipes))
            + " tarif var:\n"
            + "\n".join(short_lines)
            + "\n\n"
            + details
        )
    if any(token in q for token in ("nasil", "yapilis")) and len(recipes) == 1:
        return format_recipe_detail(recipes[0])
    return "Bulunan tarifler:\n\n" + details
