"""Malzeme listesine göre tarif önerisi."""
import database
import text_utils


def match_by_ingredients(conn, ingredients_text, top_n=5):
    top_n = max(1, min(int(top_n), 20))
    user = {
        text_utils.normalize(item.strip())[:80]
        for item in str(ingredients_text or "").split(",")[:40]
        if item.strip()
    }
    if not user:
        return []

    results = []
    for recipe in database.get_all_recipes(conn):
        recipe_set = {text_utils.normalize(item) for item in recipe["ingredients"]}
        if not recipe_set:
            continue

        matched = user & recipe_set
        # örn. "un" ile "galeta unu", "tavuk" ile "tavuk gogsu"
        partial = set()
        for item in recipe_set:
            if item in matched:
                continue
            for piece in user:
                if piece == item:
                    continue
                if len(piece) >= 2 and (
                    item.startswith(piece + " ")
                    or item.endswith(" " + piece)
                    or f" {piece} " in f" {item} "
                ):
                    partial.add(item)
                    break
                if len(item) >= 3 and len(piece) > len(item) and (
                    piece.startswith(item)
                    or piece.endswith(item)
                    or f" {item} " in f" {piece} "
                ):
                    partial.add(item)
                    break
        matched = matched | partial
        if not matched:
            continue

        missing = recipe_set - matched
        percent = round(100 * len(matched) / len(recipe_set))
        item = dict(recipe)
        item["overlap_percentage"] = percent
        item["missing_ingredients"] = [
            name for name in recipe["ingredients"]
            if text_utils.normalize(name) in missing
        ]
        results.append(item)

    results.sort(key=lambda recipe: recipe["overlap_percentage"], reverse=True)
    return results[:top_n]
