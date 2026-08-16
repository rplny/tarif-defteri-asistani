"""SQLite veritabanı."""
import json
import os
import sqlite3

import recipe_parser

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "recipes.db")

SAMPLES = [
    ("mercimek_corbasi.txt", "Mercimek Çorbası", ["kırmızı mercimek", "soğan", "havuç", "patates", "tereyağ", "un", "tuz"],
     "Mercimek Çorbası\nMalzemeler: kırmızı mercimek, soğan, havuç, patates, tereyağ, un, tuz.\nYapılışı: Sebzeleri kavur, mercimek ve su ekle, pişir, blenderdan geçir."),
    ("menemen.txt", "Menemen", ["yumurta", "domates", "biber", "soğan", "zeytinyağı", "tuz"],
     "Menemen\nMalzemeler: yumurta, domates, biber, soğan, zeytinyağı, tuz.\nYapılışı: Soğan ve biberi sotele, domates ekle, yumurtaları kırıp pişir."),
    ("karniyarik.txt", "Karnıyarık", ["patlıcan", "kıyma", "soğan", "domates", "biber", "sarımsak", "zeytinyağı"],
     "Karnıyarık\nMalzemeler: patlıcan, kıyma, soğan, domates, biber, sarımsak, zeytinyağı.\nYapılışı: Patlıcanları kızart, kıymalı harcı doldur, fırınla."),
    ("imam_bayildi.txt", "İmam Bayıldı", ["patlıcan", "soğan", "domates", "sarımsak", "zeytinyağı", "maydanoz"],
     "İmam Bayıldı\nMalzemeler: patlıcan, soğan, domates, sarımsak, zeytinyağı, maydanoz.\nYapılışı: Patlıcanları kızart, soğanlı harcı doldur, demlendir."),
    ("kofte.txt", "Izgara Köfte", ["kıyma", "soğan", "sarımsak", "galeta unu", "yumurta", "kimyon", "tuz"],
     "Izgara Köfte\nMalzemeler: kıyma, soğan, sarımsak, galeta unu, yumurta, kimyon, tuz.\nYapılışı: Yoğur, şekil ver, ızgarada pişir."),
    ("pilav.txt", "Tereyağlı Pirinç Pilavı", ["pirinç", "tereyağ", "şehriye", "su", "tuz"],
     "Tereyağlı Pirinç Pilavı\nMalzemeler: pirinç, tereyağ, şehriye, su, tuz.\nYapılışı: Şehriyeyi kavur, pirinç ve su ekle, demlendir."),
    ("makarna_salcasi.txt", "Salçalı Makarna", ["makarna", "domates salçası", "zeytinyağı", "sarımsak", "tuz", "kaşar"],
     "Salçalı Makarna\nMalzemeler: makarna, domates salçası, zeytinyağı, sarımsak, tuz, kaşar.\nYapılışı: Makarnayı haşla, salçalı sosla karıştır."),
    ("lahmacun.txt", "Lahmacun", ["un", "kıyma", "soğan", "domates", "biber", "maydanoz", "sarımsak"],
     "Lahmacun\nMalzemeler: un, kıyma, soğan, domates, biber, maydanoz, sarımsak.\nYapılışı: Hamur aç, harcı sür, yüksek ateşte pişir."),
    ("gozleme.txt", "Patatesli Gözleme", ["un", "patates", "soğan", "zeytinyağı", "tuz"],
     "Patatesli Gözleme\nMalzemeler: un, patates, soğan, zeytinyağı, tuz.\nYapılışı: Hamur aç, patatesli iç koy, sacda pişir."),
    ("cacik.txt", "Cacık", ["yoğurt", "salatalık", "sarımsak", "zeytinyağı", "nane", "tuz"],
     "Cacık\nMalzemeler: yoğurt, salatalık, sarımsak, zeytinyağı, nane, tuz.\nYapılışı: Yoğurdu çırp, salatalık ve sarımsak ekle."),
    ("kisir.txt", "Kısır", ["bulgur", "domates salçası", "soğan", "maydanoz", "nar ekşisi", "zeytinyağı"],
     "Kısır\nMalzemeler: bulgur, domates salçası, soğan, maydanoz, nar ekşisi, zeytinyağı.\nYapılışı: Bulguru ıslat, malzemeleri yoğur."),
    ("mercimek_koftesi.txt", "Mercimek Köftesi", ["kırmızı mercimek", "bulgur", "soğan", "domates salçası", "maydanoz"],
     "Mercimek Köftesi\nMalzemeler: kırmızı mercimek, bulgur, soğan, domates salçası, maydanoz.\nYapılışı: Mercimeği haşla, bulgurla yoğur, şekil ver."),
    ("baklava.txt", "Cevizli Baklava", ["yufka", "ceviz", "tereyağ", "şeker", "su", "limon"],
     "Cevizli Baklava\nMalzemeler: yufka, ceviz, tereyağ, şeker, su, limon.\nYapılışı: Yufkaları diz, pişir, şerbet dök."),
    ("sutlac.txt", "Fırın Sütlaç", ["süt", "pirinç", "şeker", "nişasta", "vanilya"],
     "Fırın Sütlaç\nMalzemeler: süt, pirinç, şeker, nişasta, vanilya.\nYapılışı: Pirinci sütte pişir, fırında üstünü kızart."),
    ("omlet.txt", "Peynirli Omlet", ["yumurta", "beyaz peynir", "tereyağ", "tuz"],
     "Peynirli Omlet\nMalzemeler: yumurta, beyaz peynir, tereyağ, tuz.\nYapılışı: Yumurtayı çırp, peynir ekle, tavada pişir."),
    ("sebze_guvec.txt", "Sebze Güveç", ["patlıcan", "kabak", "patates", "domates", "biber", "soğan", "zeytinyağı"],
     "Sebze Güveç\nMalzemeler: patlıcan, kabak, patates, domates, biber, soğan, zeytinyağı.\nYapılışı: Sebzeleri doğra, güveçte pişir."),
    ("tavuk_sote.txt", "Tavuk Sote", ["tavuk göğsü", "biber", "soğan", "domates", "sarımsak", "zeytinyağı", "kekik"],
     "Tavuk Sote\nMalzemeler: tavuk göğsü, biber, soğan, domates, sarımsak, zeytinyağı, kekik.\nYapılışı: Tavuğu sotele, sebzeleri ekle, pişir."),
    ("manti.txt", "Mantı", ["un", "yumurta", "kıyma", "soğan", "yoğurt", "sarımsak", "tereyağ"],
     "Mantı\nMalzemeler: un, yumurta, kıyma, soğan, yoğurt, sarımsak, tereyağ.\nYapılışı: Hamura iç koy, haşla, yoğurtla servis et."),
    ("iskembe_corbasi.txt", "İşkembe Çorbası", ["işkembe", "un", "yumurta", "limon", "sarımsak", "tereyağ"],
     "İşkembe Çorbası\nMalzemeler: işkembe, un, yumurta, limon, sarımsak, tereyağ.\nYapılışı: İşkembeyi pişir, terbiye ekle, sarımsaklı yağ gezdir."),
    ("brownie.txt", "Çikolatalı Brownie", ["bitter çikolata", "tereyağ", "şeker", "yumurta", "un", "kakao", "ceviz"],
     "Çikolatalı Brownie\nMalzemeler: bitter çikolata, tereyağ, şeker, yumurta, un, kakao, ceviz.\nYapılışı: Erit, karıştır, fırında pişir."),
]


def get_connection(db_path=DB_PATH):
    folder = os.path.dirname(db_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA secure_delete=ON")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS recipes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            full_text TEXT NOT NULL,
            ingredients TEXT NOT NULL,
            embedding TEXT
        )
        """
    )
    names = set()
    for row in conn.execute("PRAGMA table_info(recipes)").fetchall():
        try:
            names.add(row["name"])
        except (IndexError, KeyError, TypeError):
            names.add(row[1])
    if "embedding" not in names:
        try:
            conn.execute("ALTER TABLE recipes ADD COLUMN embedding TEXT")
        except sqlite3.OperationalError:
            pass
    conn.commit()
    return conn


def row_to_recipe(row):
    try:
        ingredients = json.loads(row["ingredients"])
        if not isinstance(ingredients, list):
            ingredients = []
    except (TypeError, json.JSONDecodeError):
        ingredients = []
    ingredients = recipe_parser.clean_ingredients(ingredients)
    embedding = []
    try:
        raw = row["embedding"]
        if raw:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                embedding = [float(x) for x in parsed]
    except (TypeError, ValueError, json.JSONDecodeError, IndexError, KeyError):
        embedding = []
    return {
        "id": row["id"],
        "source_file": recipe_parser.safe_filename(row["source_file"]),
        "title": recipe_parser.clean_text(row["title"], recipe_parser.MAX_TITLE_LEN),
        "full_text": recipe_parser.clean_text(row["full_text"], recipe_parser.MAX_TEXT_LEN),
        "ingredients": ingredients,
        "embedding": embedding,
    }


def get_all_recipes(conn):
    rows = conn.execute("SELECT * FROM recipes ORDER BY id").fetchall()
    return [row_to_recipe(row) for row in rows]


def upsert_recipe(conn, source_file, title, full_text, ingredients, embedding=None):
    source_file = recipe_parser.safe_filename(source_file)
    title = recipe_parser.clean_text(title, recipe_parser.MAX_TITLE_LEN)
    full_text = recipe_parser.clean_text(full_text, recipe_parser.MAX_TEXT_LEN)
    ingredients = recipe_parser.clean_ingredients(ingredients)
    if embedding is None:
        from embeddings import local_embed
        from search_engine import recipe_text

        embedding = local_embed(
            recipe_text(
                {"title": title, "ingredients": ingredients, "full_text": full_text}
            )
        )
    conn.execute(
        """
        INSERT INTO recipes (source_file, title, full_text, ingredients, embedding)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(source_file) DO UPDATE SET
            title=excluded.title,
            full_text=excluded.full_text,
            ingredients=excluded.ingredients,
            embedding=excluded.embedding
        """,
        (
            source_file,
            title,
            full_text,
            json.dumps(ingredients, ensure_ascii=False),
            json.dumps(embedding),
        ),
    )
    conn.commit()


def seed_if_empty(conn):
    count = conn.execute("SELECT COUNT(*) AS n FROM recipes").fetchone()["n"]
    if count:
        return
    for source, title, ingredients, text in SAMPLES:
        upsert_recipe(conn, source, title, text, ingredients)
