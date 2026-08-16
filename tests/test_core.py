"""Temel arama, diyet ve güvenlik testleri."""
import recipe_parser
import search_engine
import text_utils
from diet_rules import diet_label, is_vegan, is_vegetarian
from recommender import match_by_ingredients


def test_stem_tavugum():
    assert text_utils.stem("tavugum") == "tavuk"


def test_has_term_avoids_false_positives():
    assert text_utils.has_term("serbet", "et") is False
    assert text_utils.has_term("koftesi", "kofte") is False
    assert text_utils.has_term("tavuk sote", "tavuk") is True


def test_empty_question_does_not_crash(conn):
    hits = search_engine.search_recipes(conn, "")
    assert hits == []
    assert "Boş soru" in search_engine.build_answer("", [])


def test_unknown_topic_says_not_found(conn):
    hits = search_engine.search_recipes(conn, "uzay mekiği nasıl tamir edilir xyzzy")
    answer = search_engine.build_answer("uzay mekiği nasıl tamir edilir xyzzy", hits)
    assert "yeterli bilgi" in answer.lower() or "bulunamadı" in answer.lower()


def test_search_menemen(conn):
    hits = search_engine.search_recipes(conn, "menemen nasıl yapılır")
    assert hits
    assert hits[0]["title"] == "Menemen"


def test_search_vegan(conn):
    hits = search_engine.search_recipes(conn, "vegan tarif öner")
    assert hits
    assert all(is_vegan(recipe) for recipe in hits)


def test_search_sebzesiz(conn):
    hits = search_engine.search_recipes(conn, "sebzesiz tarif öner")
    assert hits
    from text_utils import VEGETABLES, has_term

    for recipe in hits:
        blob = " ".join(recipe["ingredients"])
        assert not any(has_term(blob, term) for term in VEGETABLES)


def test_database_schema_idempotent(tmp_path):
    import database

    db_file = str(tmp_path / "twice.db")
    first = database.get_connection(db_file)
    second = database.get_connection(db_file)
    names = {row["name"] for row in second.execute("PRAGMA table_info(recipes)")}
    assert "embedding" in names
    first.close()
    second.close()


def test_search_etli(conn):
    hits = search_engine.search_recipes(conn, "etli tarif")
    assert hits
    assert all(not is_vegetarian(recipe) for recipe in hits)


def test_diet_labels(conn):
    recipes = {recipe["title"]: recipe for recipe in __import__("database").get_all_recipes(conn)}
    assert diet_label(recipes["İmam Bayıldı"]) == "vegan"
    assert diet_label(recipes["Peynirli Omlet"]) == "vejetaryen"
    assert diet_label(recipes["Izgara Köfte"]) == "et içeren"


def test_safe_filename_blocks_traversal():
    name = recipe_parser.safe_filename("../../etc/passwd.txt")
    assert ".." not in name
    assert name.endswith(".txt")


def test_upload_size_limit():
    try:
        recipe_parser.parse_recipe_bytes("buyuk.txt", b"x" * 300_000)
        assert False, "beklenen ValueError gelmedi"
    except ValueError as exc:
        assert "büyük" in str(exc).lower() or "200" in str(exc)


def test_recommender_sut_not_su(conn):
    matches = match_by_ingredients(conn, "sut", top_n=10)
    titles = [item["title"] for item in matches]
    assert "Fırın Sütlaç" in titles
    for item in matches:
        norms = {text_utils.normalize(name) for name in item["ingredients"]}
        # eşleşme yalnızca süt/süt içeren malzemeden gelmeli; yalnız "su" yetmez
        assert "sut" in norms or any("sut" in name for name in norms)
