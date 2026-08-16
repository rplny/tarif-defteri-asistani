"""Tarif Defteri RAG: bul, ekle, üret (Foundry Local)."""
import sys
import time

from foundry_local_sdk import Configuration, FoundryLocalManager

from ingestion import load_chunk_records
from knowledge_store import KB_FOLDER, get_connection, index_documents, load_records
from retrieval import format_context, keyword_rank_chunks, rank_chunks
from text_utils import STOP, normalize

SYSTEM_PROMPT = (
    "Sadece aşağıdaki context'teki bilgiden cevap ver. Kısa yaz. "
    "Context'te geçen tarif adlarını olduğu gibi kullan; yeni tarif veya hikaye ekleme. "
    "Cevabında kaynak dosya adını söyle (örnek: diyet.txt dosyasına göre ...). "
    "Context boşsa açıkça 'Bu bilgi context'te yok.' de."
)


def load_knowledge_items():
    """knowledge/*.txt parçalarını kaynak adıyla yükler."""
    return load_chunk_records(KB_FOLDER)


def get_top_chunks(query, embedding_client, docs, doc_embeddings, sources=None, top_k=3, min_score=0.45):
    """Soruyu göm, SQLite vektörleriyle kosinüs hesapla, en alakalı 2–3 parçayı döndür."""
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
        return hits
    return keyword_rank_chunks(query, docs, sources=sources, top_k=top_k)


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


def is_list_query(query):
    q = normalize(query)
    return any(key in q for key in ("oner", "vegan", "sebzesiz", "vejetaryen"))


def select_hits(query, hits):
    q = normalize(query)
    if not hits:
        return hits

    def blob(hit):
        return normalize(hit.get("content") or "")

    if "sebzesiz" in q:
        matched = [hit for hit in hits if "sebzesiz" in blob(hit)]
    elif "vegan" in q:
        matched = [
            hit
            for hit in hits
            if "vegan" in blob(hit) and "vegan degildir" not in blob(hit)
        ]
    elif "vejetaryen" in q:
        matched = [hit for hit in hits if "vejetaryen" in blob(hit)]
    else:
        return hits
    if not matched:
        return hits
    matched.sort(key=lambda hit: 0 if hit.get("source") == "diyet.txt" else 1)
    return matched


def is_grounded(answer, hits):
    if not (answer or "").strip() or not hits:
        return False
    ctx = set()
    for hit in hits:
        for word in normalize(hit.get("content") or "").split():
            if len(word) >= 4 and word not in GENERIC:
                ctx.add(word)
    overlap = [
        word
        for word in normalize(answer).split()
        if len(word) >= 4 and word not in GENERIC and word in ctx
    ]
    return any(len(word) >= 5 for word in overlap)


def quote_hits(hits):
    hit = hits[0]
    source = hit.get("source") or "kaynak"
    text = (hit.get("content") or "").strip()
    return f"{source} dosyasına göre {text}"


def finalize_answer(answer, hits, query=""):
    if hits and is_list_query(query):
        return quote_hits(hits)
    text = (answer or "").strip()
    if hits and not is_grounded(text, hits):
        return quote_hits(hits)
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
    hit_sources = [hit["source"] for hit in hits if hit.get("source")]
    if context.strip():
        if chat_client is None or is_list_query(query):
            return (quote_hits(hits) if is_list_query(query) else context), hit_sources
        answer = stream_answer(
            chat_client, build_messages(query, context), writer=lambda *a, **k: None
        )
        return finalize_answer(answer, hits, query), hit_sources
    if conn is not None:
        import search_engine

        found = search_engine.search_recipes(conn, query, limit=8)
        if found:
            return search_engine.build_answer(query, found), [
                recipe["source_file"] for recipe in found
            ]
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
    if _index_key(records) != _index_key(items):
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
    if is_list_query(query):
        answer = quote_hits(hits)
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
