"""Tarif Defteri RAG: bul, ekle, üret (Foundry Local)."""
import re
import sys
import time
from pathlib import Path

from foundry_local_sdk import Configuration, FoundryLocalManager

from diet_rules import is_vegan, is_vegetarian
from ingestion import load_chunk_records
from knowledge_store import KB_FOLDER, get_connection, index_documents, load_records
from retrieval import format_context, keyword_rank_chunks, part_in_query, rank_chunks, tighten_hits
from text_utils import STOP, VEGETABLES, has_term, normalize

SYSTEM_PROMPT = (
    "Sadece aşağıdaki context'teki bilgiden cevap ver. Kısa yaz. "
    "Context'te geçen tarif adlarını olduğu gibi kullan; yeni tarif veya hikaye ekleme. "
    "Cevabında kaynak dosya adını söyle (örnek: diyet.txt dosyasına göre ...). "
    "Context boşsa açıkça 'Bu bilgi context'te yok.' de."
)


def load_knowledge_items():
    """knowledge/*.txt parçalarını kaynak adıyla yükler."""
    return load_chunk_records(KB_FOLDER)


CATALOGS = (
    ("sebzesiz", "sebzesiz tarifler:"),
    ("vegan", "vegan tarifler:"),
    ("etsiz", "vejetaryen tarifler:"),
    ("vejetaryen", "vejetaryen tarifler:"),
    ("tatli", "tatlilar:"),
    ("kahvalti", "kahvalti tarifler:"),
    ("meze", "meze tarifler:"),
    ("corba", "corba tarifler:"),
    ("etli", "etli tarifler:"),
)
LIST_HINTS = ("oner", "listele", "neler var", "hangi tarif")
HOWTO_HINTS = ("nasil", "yapilir", "yapilisi", "hangi malzeme")
INVENTORY_HINTS = ("kac tane", "kac tarif", "kac adet", "elimde", "bende", "yapabilirim")


def recipe_file_stems():
    return [path.stem for path in sorted(KB_FOLDER.glob("*.txt")) if path.stem != "diyet"]


def _stem_parts(stem):
    return [part for part in stem.split("_") if len(part) >= 4] or [
        part for part in stem.split("_") if len(part) >= 3
    ]


def named_recipe_sources(query):
    """Soruda geçen tarif dosya adlarını döndürür; en uzun ad önce gelir."""
    found = []
    for stem in recipe_file_stems():
        parts = _stem_parts(stem)
        if parts and all(part_in_query(query, part) for part in parts):
            found.append(f"{stem}.txt")
    found.sort(key=len, reverse=True)
    return found


def partial_recipe_sources(query):
    """Tek parça eşleşmesi: mercimek → çorba ve köfte; tavuk → tavuk sote."""
    found = []
    for stem in recipe_file_stems():
        parts = _stem_parts(stem)
        if any(part_in_query(query, part) for part in parts):
            found.append(f"{stem}.txt")
    found.sort(key=len, reverse=True)
    return found


def is_howto_query(query):
    q = normalize(query)
    return any(hint in q for hint in HOWTO_HINTS)


def is_yesno_query(query):
    q = normalize(query).rstrip("?.! ")
    return q.endswith((" midir", " mudur", " mi", " mu"))


def is_inventory_query(query):
    """Elimdeki malzeme / kaç tarif: 'tavuğum var kaç tane tarif yapabilirim'."""
    if is_yesno_query(query):
        return False
    if is_howto_query(query) and named_recipe_sources(query):
        return False
    q = normalize(query)
    if any(hint in q for hint in INVENTORY_HINTS):
        return True
    padded = f" {q} "
    if " var " in padded or q.endswith(" var"):
        return not named_recipe_sources(query) and catalog_label(query) is None
    return False


def has_specific_food(query):
    if named_recipe_sources(query) or partial_recipe_sources(query):
        return True
    from text_utils import CATEGORIES, MEAT, VEGETABLES

    foods = set(MEAT) | set(VEGETABLES) | set(CATEGORIES.get("tatli", set()))
    return any(part_in_query(query, term) or has_term(query, term) for term in foods if len(term) >= 4)


def catalog_label(query):
    q = normalize(query)
    for key, marker in CATALOGS:
        if key in q:
            return marker
    return None


def extra_query_tokens(query):
    """Kategori/öneri kalıpları dışında kalan kelimeler: 'gözleme öner' → gözleme."""
    skip = set(STOP)
    skip.update(("tarif", "tarifler", "oner", "listele", "neler", "hangi"))
    skip.update(key for key, _marker in CATALOGS)
    return [
        word
        for word in normalize(query).split()
        if len(word) >= 4 and word not in skip
    ]


def is_list_query(query):
    """Kategori önerisi: 'vegan tarif öner'. 'köfte öner' veya 'Menemen vegan mı?' değil."""
    if is_yesno_query(query) or is_inventory_query(query):
        return False
    named = named_recipe_sources(query)
    if is_howto_query(query) and named:
        return False
    q = normalize(query)
    has_hint = any(hint in q for hint in LIST_HINTS)
    has_cat = catalog_label(query) is not None
    if named and not has_cat:
        return False
    if has_cat and not named:
        return True
    if has_hint and has_cat:
        return True
    if has_hint and not named and not has_specific_food(query) and not extra_query_tokens(query):
        return True
    return False


def catalog_marker(query):
    if not is_list_query(query):
        return None
    return catalog_label(query)


def _hit(index, doc, source, score=1.0):
    return {"index": index, "score": score, "content": doc, "source": source or ""}


def hits_for_sources(wanted, docs, sources):
    wanted = set(wanted or [])
    if not wanted:
        return []
    sources = sources or [""] * len(docs)
    return [
        _hit(index, docs[index], sources[index])
        for index, src in enumerate(sources)
        if src in wanted and index < len(docs)
    ]


def generic_recipe_list(docs, sources):
    names = []
    seen = set()
    for src in sources or []:
        if not src or src == "diyet.txt" or src in seen:
            continue
        seen.add(src)
        names.append(Path(src).stem.replace("_", " "))
    if not names:
        return []
    content = "Bu defterde tarifler: " + ", ".join(names) + "."
    return [_hit(0, content, "diyet.txt")]


def catalog_hits(query, docs, sources=None):
    if not is_list_query(query) or not docs:
        return []
    marker = catalog_label(query)
    sources = sources or [""] * len(docs)
    if not marker:
        return generic_recipe_list(docs, sources)
    found = []
    for index, doc in enumerate(docs):
        blob = normalize(doc)
        if marker not in blob:
            continue
        if marker == "vegan tarifler:" and "vegan degildir" in blob:
            continue
        found.append(_hit(index, doc, sources[index] if index < len(sources) else ""))
    found.sort(key=lambda hit: 0 if hit.get("source") == "diyet.txt" else 1)
    return found


def get_top_chunks(query, embedding_client, docs, doc_embeddings, sources=None, top_k=3, min_score=0.45):
    """Önce kategori/tarif adı, yetmezse kosinüs, en sonda kelime örtüşmesi."""
    listed = catalog_hits(query, docs, sources=sources)
    if listed:
        return listed
    named = named_recipe_sources(query)
    if named:
        found = hits_for_sources([named[0]], docs, sources)
        if found:
            return found
    partial = partial_recipe_sources(query)
    if partial:
        found = hits_for_sources(partial, docs, sources)
        if found:
            return found
    query_response = embedding_client.generate_embedding(query)
    query_embedding = query_response.data[0].embedding
    hits = rank_chunks(
        query_embedding,
        docs,
        doc_embeddings,
        sources=sources,
        top_k=top_k,
        min_score=min_score,
    )
    if hits:
        return tighten_hits(query, hits)
    return tighten_hits(query, keyword_rank_chunks(query, docs, sources=sources, top_k=top_k))


def retrieve_context(query, embedding_client, docs, doc_embeddings, sources=None, top_k=3, min_score=0.45):
    hits = get_top_chunks(
        query,
        embedding_client,
        docs,
        doc_embeddings,
        sources=sources,
        top_k=top_k,
        min_score=min_score,
    )
    return format_context(select_hits(query, hits))


GENERIC = STOP | {
    "defterde", "yoktur", "icermez", "olabilir", "gibi", "tariflerde",
    "malzemeler", "kaynak", "dosya", "dosyasina", "gore", "sebze",
    "sebzeler", "iceren", "bunlar", "bunlardan", "ilgili", "adi",
}


def select_hits(query, hits):
    if not hits:
        return hits
    marker = catalog_marker(query)
    if not marker:
        return hits

    def blob(hit):
        return normalize(hit.get("content") or "")

    listed = [hit for hit in hits if marker in blob(hit)]
    if marker == "vegan tarifler:":
        listed = [hit for hit in listed if "vegan degildir" not in blob(hit)]
        if not listed:
            listed = [
                hit
                for hit in hits
                if "vegan" in blob(hit) and "vegan degildir" not in blob(hit)
            ]
    if not listed:
        return hits
    listed.sort(key=lambda hit: 0 if hit.get("source") == "diyet.txt" else 1)
    return listed


def _content_tokens(text):
    return [
        word
        for word in normalize(text).split()
        if len(word) >= 4 and word not in GENERIC
    ]


def _token_in_ctx(word, ctx):
    if word in ctx:
        return True
    return any(
        item.startswith(word) or word.startswith(item)
        for item in ctx
        if min(len(item), len(word)) >= 4
    )


def is_grounded(answer, hits):
    if not (answer or "").strip() or not hits:
        return False
    ctx = set()
    for hit in hits:
        ctx.update(_content_tokens(hit.get("content") or ""))
    answer_words = _content_tokens(answer)
    if not answer_words:
        return False
    overlap = [word for word in answer_words if _token_in_ctx(word, ctx)]
    novel = [word for word in answer_words if not _token_in_ctx(word, ctx)]
    if len(set(overlap)) < 3:
        return False
    if len(set(overlap)) / len(set(answer_words)) < 0.5:
        return False
    if novel and len(set(novel)) / len(set(answer_words)) > 0.25:
        return False
    return True


def list_sentence(text):
    text = (text or "").strip()
    parts = []
    buf = []
    for ch in text:
        buf.append(ch)
        if ch in ".!?":
            parts.append("".join(buf).strip())
            buf = []
    if buf:
        parts.append("".join(buf).strip())
    for part in parts:
        if "tarifler:" in part.lower() or "tatlilar:" in normalize(part):
            sentence = part.strip().rstrip(".!?")
            if sentence.lower().startswith("bu defterde "):
                sentence = sentence[12:]
            return sentence
    if parts:
        return parts[0].strip().rstrip(".!?")
    return text


def quote_hits(hits):
    hit = hits[0]
    source = hit.get("source") or "kaynak"
    text = list_sentence(hit.get("content") or "")
    return f"{source} dosyasına göre {text}"


def ingredients_from_hits(hits):
    text = " ".join(hit.get("content") or "" for hit in hits)
    match = re.search(r"Malzemeler:\s*(.+?)(?:\.|$)", text, flags=re.IGNORECASE)
    if not match:
        return []
    return [part.strip() for part in re.split(r",| ve ", match.group(1)) if part.strip()]


def format_recipe_answer(hits, query=""):
    if not hits:
        return "Bu bilgi context'te yok."
    wanted = named_recipe_sources(query) or partial_recipe_sources(query)
    if wanted:
        use = [hit for hit in hits if hit.get("source") in wanted] or hits
    else:
        use = hits
    grouped = []
    seen_sources = []
    for hit in use:
        source = hit.get("source") or "kaynak"
        if source not in seen_sources:
            seen_sources.append(source)
            grouped.append((source, []))
        for name, texts in grouped:
            if name == source:
                content = (hit.get("content") or "").strip()
                if content and content not in texts:
                    texts.append(content)
                break
    parts = [f"{source} dosyasına göre {' '.join(texts)}" for source, texts in grouped]
    if len(parts) > 1:
        return "Bu adda birden fazla tarif var.\n\n" + "\n\n".join(parts)
    return parts[0]


def yesno_answer(query, hits):
    source = (hits[0].get("source") if hits else "") or "kaynak"
    title = Path(source).stem.replace("_", " ")
    recipe = {"ingredients": ingredients_from_hits(hits)}
    blob = normalize(" ".join(hit.get("content") or "" for hit in hits))
    q = normalize(query)
    if "vegan" in q:
        ok = is_vegan(recipe) if recipe["ingredients"] else (
            "vegan tariftir" in blob and "vegan degildir" not in blob
        )
        return f"{source} dosyasına göre {title} {'vegan' if ok else 'vegan değil'}."
    if "vejetaryen" in q or "etsiz" in q:
        ok = is_vegetarian(recipe) if recipe["ingredients"] else (
            "et yoktur" in blob or "et icermez" in blob
        )
        return f"{source} dosyasına göre {title} {'vejetaryen' if ok else 'vejetaryen değil'}."
    if "etli" in q:
        if not recipe["ingredients"]:
            return format_recipe_answer(hits, query)
        ok = not is_vegetarian(recipe)
        return f"{source} dosyasına göre {title} {'etli' if ok else 'etli değil'}."
    if "sebzesiz" in q:
        if not recipe["ingredients"]:
            return format_recipe_answer(hits, query)
        ok = not any(has_term(" ".join(recipe["ingredients"]), veg) for veg in VEGETABLES)
        return f"{source} dosyasına göre {title} {'sebzesiz' if ok else 'sebzesiz değil'}."
    if "tatli" in q:
        ok = "tatli" in blob or source in {"baklava.txt", "sutlac.txt", "brownie.txt"}
        return f"{source} dosyasına göre {title} {'tatlı' if ok else 'tatlı değil'}."
    return format_recipe_answer(hits, query)


def is_extractive_query(query):
    return (
        is_list_query(query)
        or is_howto_query(query)
        or is_yesno_query(query)
        or is_inventory_query(query)
        or bool(named_recipe_sources(query))
        or bool(partial_recipe_sources(query))
    )


def extract_answer(query, hits):
    if not hits:
        return "Bu bilgi context'te yok."
    if is_yesno_query(query):
        return yesno_answer(query, hits)
    if is_list_query(query):
        return quote_hits(hits)
    return format_recipe_answer(hits, query)


def finalize_answer(answer, hits, query=""):
    if hits and is_list_query(query):
        return quote_hits(hits)
    if hits and is_extractive_query(query):
        return extract_answer(query, hits)
    text = (answer or "").strip()
    if hits and not is_grounded(text, hits):
        return extract_answer(query, hits)
    return text


def build_messages(query, context):
    return [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\nContext:\n" + context},
        {"role": "user", "content": query},
    ]


def stream_answer(chat_client, messages, writer=print):
    writer("Cevap: ", end="", flush=True)
    parts = []
    for chunk in chat_client.complete_streaming_chat(messages):
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        content = getattr(delta, "content", None) if delta else None
        if content:
            parts.append(content)
            writer(content, end="", flush=True)
    writer("\n")
    return "".join(parts).strip()


def unique_sources(sources):
    seen = set()
    out = []
    for source in sources or []:
        if not source or source in seen:
            continue
        seen.add(source)
        out.append(source)
    return out


def inventory_answer(query, conn=None):
    import database
    import search_engine

    if conn is None:
        conn = database.get_connection()
        database.seed_if_empty(conn)
    found = search_engine.search_recipes(conn, query, limit=8)
    if found:
        return search_engine.build_answer(query, found), unique_sources(
            recipe["source_file"] for recipe in found
        )
    if not has_specific_food(query):
        return "Elindeki malzemeleri yaz: örneğin yumurta, un, süt.", []
    return "Bu bilgi context'te yok.", []


def resolve_answer(
    query,
    embedding_client,
    doc_embeddings,
    docs=None,
    sources=None,
    top_k=3,
    chat_client=None,
    min_score=0.45,
    conn=None,
):
    query = (query or "").strip()
    if not query:
        return "Boş soru gönderildi.", []
    if is_inventory_query(query):
        return inventory_answer(query, conn=conn)
    docs = docs if docs is not None else []
    hits = select_hits(
        query,
        get_top_chunks(
            query,
            embedding_client,
            docs,
            doc_embeddings,
            sources=sources,
            top_k=top_k,
            min_score=min_score,
        ),
    )
    context = format_context(hits)
    hit_sources = unique_sources(hit["source"] for hit in hits if hit.get("source"))
    if context.strip():
        if chat_client is None or is_extractive_query(query):
            return extract_answer(query, hits), hit_sources
        answer = stream_answer(
            chat_client, build_messages(query, context), writer=lambda *a, **k: None
        )
        return finalize_answer(answer, hits, query), hit_sources
    if conn is not None:
        import search_engine

        found = search_engine.search_recipes(conn, query, limit=8)
        if found:
            return search_engine.build_answer(query, found), unique_sources(
                recipe["source_file"] for recipe in found
            )
    return "Bu bilgi context'te yok.", []


def answer_query(
    query,
    embedding_client,
    doc_embeddings,
    docs=None,
    sources=None,
    top_k=3,
    chat_client=None,
    min_score=0.45,
    conn=None,
):
    answer, _sources = resolve_answer(
        query,
        embedding_client,
        doc_embeddings,
        docs=docs,
        sources=sources,
        top_k=top_k,
        chat_client=chat_client,
        min_score=min_score,
        conn=conn,
    )
    return answer


def load_models():
    config = Configuration(app_name="foundry_local_rag")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embedding_model.download(
        lambda p: print(f"\rEmbedding modeli indiriliyor: {p:.1f}%", end="", flush=True)
    )
    print()
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    chat_model = manager.catalog.get_model("qwen2.5-0.5b")
    chat_model.download(
        lambda p: print(f"\rSohbet modeli indiriliyor: {p:.1f}%", end="", flush=True)
    )
    print()
    chat_model.load()
    chat_client = chat_model.get_chat_client()
    return embedding_model, embedding_client, chat_model, chat_client


def _index_key(rows):
    """Kaynak + metin imzası; dosya eklenince veya içeriği değişince yeniden indeksle."""
    return [(row.get("source") or "", row.get("content") or "") for row in rows]


def prepare_index(embedding_client, conn=None):
    items = load_knowledge_items()
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    records = load_records(conn)
    probe = embedding_client.generate_embedding("tarif")
    dim = len(probe.data[0].embedding) if probe.data else 0
    stale_dim = bool(records) and len(records[0].get("embedding") or []) != dim
    if _index_key(records) != _index_key(items) or stale_dim:
        index_documents(conn, items, embedding_client)
        records = load_records(conn)
    docs = [item["content"] for item in records]
    embeddings = [item["embedding"] for item in records]
    sources = [item["source"] for item in records]
    print(f"{len(docs)} parça indekslendi ({len({s for s in sources if s})} kaynak dosya).")
    if own_conn:
        conn.close()
    return docs, embeddings, sources


def _print_hits(hits):
    if not hits:
        print("Arama: uygun parça yok (eşik altı).")
        return
    print("Arama:")
    for hit in hits:
        preview = hit["content"].replace("\n", " ")[:90]
        print(f"  - {hit['source'] or 'kaynak yok'} ({hit['score']:.2f}): {preview}")


def run_question(query, embedding_client, chat_client, docs, embeddings, sources):
    query = (query or "").strip()
    if not query:
        print("Boş soru gönderildi.")
        return "Boş soru gönderildi.", 0.0
    started = time.perf_counter()
    if is_inventory_query(query):
        answer, _sources = inventory_answer(query)
        print("Cevap:", answer)
        elapsed = time.perf_counter() - started
        print(f"(yanıt süresi: {elapsed:.1f} sn)\n")
        return answer, elapsed
    hits = select_hits(
        query,
        get_top_chunks(query, embedding_client, docs, embeddings, sources=sources, top_k=3),
    )
    _print_hits(hits)
    context = format_context(hits)
    if not context.strip():
        import database
        import search_engine

        recipe_conn = database.get_connection()
        database.seed_if_empty(recipe_conn)
        found = search_engine.search_recipes(recipe_conn, query, limit=8)
        if found:
            answer = search_engine.build_answer(query, found)
            print("Cevap:", answer)
            elapsed = time.perf_counter() - started
            print(f"(yanıt süresi: {elapsed:.1f} sn)\n")
            return answer, elapsed
        print("Cevap: Bu bilgi context'te yok.")
        elapsed = time.perf_counter() - started
        print(f"(yanıt süresi: {elapsed:.1f} sn)\n")
        return "Bu bilgi context'te yok.", elapsed
    if chat_client is None or is_extractive_query(query):
        answer = extract_answer(query, hits)
    else:
        raw = stream_answer(chat_client, build_messages(query, context), writer=lambda *a, **k: None)
        answer = finalize_answer(raw, hits, query)
    print("Cevap:", answer)
    elapsed = time.perf_counter() - started
    print(f"(yanıt süresi: {elapsed:.1f} sn)\n")
    return answer, elapsed


def run_demo(embedding_client, chat_client, docs, embeddings, sources):
    print("\n--- Demo senaryosu ---")
    cases = [
        ("cevaplanabilir", "Menemen nasıl yapılır?"),
        ("vegan", "vegan tarif öner"),
        ("sebzesiz", "sebzesiz tarif öner"),
        ("cevaplanamaz", "Bu uygulamanın aylık barındırma maliyeti nedir?"),
        ("boş soru", ""),
    ]
    for label, question in cases:
        print(f"[{label}] Soru: {question or '(boş)'}")
        run_question(question, embedding_client, chat_client, docs, embeddings, sources)


def main():
    embedding_model, embedding_client, chat_model, chat_client = load_models()
    docs, embeddings, sources = prepare_index(embedding_client)

    print("\nModeller yüklendi. Sorularınızı bekliyorum.")
    print('Çıkmak için "quit" yazın.\n')

    if "--demo" in sys.argv:
        run_demo(embedding_client, chat_client, docs, embeddings, sources)
    else:
        while True:
            query = input("Soru: ").strip()
            if query.lower() == "quit":
                break
            run_question(query, embedding_client, chat_client, docs, embeddings, sources)

    embedding_model.unload()
    chat_model.unload()
    print("Modeller kapatıldı. Bitti!")


if __name__ == "__main__":
    main()
