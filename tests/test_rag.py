"""RAG arama ve get_top_chunks testleri."""
from embeddings import LocalEmbeddingClient
from main import answer_query, get_top_chunks
from retrieval import cosine_similarity, find_relevant, format_context, rank_chunks

SAMPLE_DOCS = [
    "Menemen kahvaltılık bir yumurta yemeğidir. Malzemeler: yumurta, domates, biber.",
    "Ankara Türkiye'nin başkentidir.",
]


def test_cosine_identical_vectors():
    vec = [1.0, 0.0, 0.0]
    assert abs(cosine_similarity(vec, vec) - 1.0) < 1e-9


def test_find_relevant_picks_closest():
    docs = [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]
    ranked = find_relevant([1.0, 0.0], docs, top_k=2)
    assert ranked[0][0] == 0
    assert ranked[0][1] > ranked[1][1]


def test_rank_chunks_drops_weak_scores():
    hits = rank_chunks(
        [1.0, 0.0],
        ["a", "b"],
        [[1.0, 0.0], [0.0, 1.0]],
        sources=["a.txt", "b.txt"],
        top_k=2,
        min_score=0.8,
    )
    assert len(hits) == 1
    assert hits[0]["source"] == "a.txt"


def test_format_context_includes_source():
    text = format_context([{"source": "menemen.txt", "content": "Menemen yumurta ile yapılır."}])
    assert "[Kaynak: menemen.txt]" in text
    assert "Menemen yumurta ile yapılır." in text


def test_unknown_topic_without_context():
    client = LocalEmbeddingClient()
    docs = ["Menemen kahvaltılık bir yumurta yemeğidir."]
    embs = [item.embedding for item in client.generate_embeddings(docs).data]
    answer = answer_query(
        "Bu uygulamanın aylık barındırma maliyeti nedir?",
        client,
        embs,
        docs=docs,
        min_score=0.45,
    )
    assert "context'te yok" in answer.lower()


def test_vegan_and_sebzesiz_hit_diyet():
    from main import load_knowledge_items

    client = LocalEmbeddingClient()
    items = load_knowledge_items()
    docs = [item["content"] for item in items]
    sources = [item["source"] for item in items]
    embs = [item.embedding for item in client.generate_embeddings(docs).data]
    vegan = get_top_chunks("vegan tarif öner", client, docs, embs, sources=sources)
    sebzesiz = get_top_chunks("sebzesiz tarif öner", client, docs, embs, sources=sources)
    assert vegan
    assert any("vegan" in hit["content"].lower() or hit["source"] == "diyet.txt" for hit in vegan)
    assert sebzesiz
    assert any("sebzesiz" in hit["content"].lower() or hit["source"] == "diyet.txt" for hit in sebzesiz)


def test_keyword_fallback_finds_menemen():
    from main import load_knowledge_items

    client = LocalEmbeddingClient()
    items = load_knowledge_items()
    docs = [item["content"] for item in items]
    sources = [item["source"] for item in items]
    embs = [item.embedding for item in client.generate_embeddings(docs).data]
    hits = get_top_chunks(
        "Menemen nasıl yapılır?",
        client,
        docs,
        embs,
        sources=sources,
        min_score=0.45,
    )
    assert hits
    assert hits[0]["source"] == "menemen.txt"


def test_recipe_db_fallback_when_knowledge_empty(conn):
    client = LocalEmbeddingClient()
    answer = answer_query(
        "Menemen nasıl yapılır?",
        client,
        [],
        docs=[],
        min_score=0.45,
        conn=conn,
    )
    assert "Menemen" in answer
    assert "context'te yok" not in answer.lower()


def test_hosting_cost_stays_unknown_with_fallback(conn):
    client = LocalEmbeddingClient()
    answer = answer_query(
        "Bu uygulamanın aylık barındırma maliyeti nedir?",
        client,
        [],
        docs=[],
        min_score=0.45,
        conn=conn,
    )
    assert "context'te yok" in answer.lower()


def test_empty_query_returns_message():
    client = LocalEmbeddingClient()
    embs = [item.embedding for item in client.generate_embeddings(SAMPLE_DOCS).data]
    assert answer_query("", client, embs, docs=SAMPLE_DOCS) == "Boş soru gönderildi."


def test_cli_rag_known_document():
    client = LocalEmbeddingClient()
    embs = [item.embedding for item in client.generate_embeddings(SAMPLE_DOCS).data]
    answer = answer_query(
        "Menemen nasıl yapılır?",
        client,
        embs,
        docs=SAMPLE_DOCS,
        min_score=0.0,
    )
    assert answer
    assert "Menemen" in answer or "yumurta" in answer


def test_get_top_chunks_returns_source():
    client = LocalEmbeddingClient()
    sources = ["menemen.txt", "unrelated.txt"]
    embs = [item.embedding for item in client.generate_embeddings(SAMPLE_DOCS).data]
    hits = get_top_chunks(
        "Menemen nasıl yapılır?",
        client,
        SAMPLE_DOCS,
        embs,
        sources=sources,
        top_k=2,
        min_score=0.0,
    )
    assert hits
    assert hits[0]["source"] == "menemen.txt"


def test_select_hits_keeps_sebzesiz_paragraph():
    from main import select_hits

    hits = [
        {"source": "diyet.txt", "content": "Vegan tarifler: kısır ve imam bayıldı."},
        {
            "source": "diyet.txt",
            "content": "Sebzesiz tarifler: peynirli omlet, tereyağlı pirinç pilavı ve cevizli baklava.",
        },
    ]
    picked = select_hits("sebzesiz tarif öner", hits)
    assert len(picked) == 1
    assert "omlet" in picked[0]["content"]


def test_select_hits_drops_vegan_degildir():
    from main import select_hits

    hits = [
        {"source": "baklava.txt", "content": "Cevizli baklava. Vejetaryendir, vegan değildir."},
        {"source": "diyet.txt", "content": "Vegan tarifler: kısır, imam bayıldı ve sebze güveç."},
    ]
    picked = select_hits("vegan tarif öner", hits)
    assert picked
    assert all(hit["source"] != "baklava.txt" for hit in picked)
    assert picked[0]["source"] == "diyet.txt"


def test_vegan_list_answer_not_baklava():
    from main import answer_query, load_knowledge_items
    from text_utils import normalize

    client = LocalEmbeddingClient()
    items = load_knowledge_items()
    docs = [item["content"] for item in items]
    sources = [item["source"] for item in items]
    embs = [item.embedding for item in client.generate_embeddings(docs).data]
    answer = normalize(answer_query("vegan tarif öner", client, embs, docs=docs, sources=sources))
    assert "kisir" in answer
    assert "baklava" not in answer


def test_cacik_keyword_prefers_cacik_file():
    from main import load_knowledge_items

    client = LocalEmbeddingClient()
    items = load_knowledge_items()
    docs = [item["content"] for item in items]
    sources = [item["source"] for item in items]
    embs = [item.embedding for item in client.generate_embeddings(docs).data]
    hits = get_top_chunks("Cacıkta hangi malzemeler var?", client, docs, embs, sources=sources)
    assert hits
    assert hits[0]["source"] == "cacik.txt"
    assert all(hit["source"] == "cacik.txt" for hit in hits)


def test_ungrounded_chat_falls_back_to_chunk():
    from main import finalize_answer

    hits = [
        {
            "source": "diyet.txt",
            "content": "Sebzesiz tariflerde sebze yoktur. Bu defterde sebzesiz tarifler: peynirli omlet, tereyağlı pirinç pilavı ve cevizli baklava.",
        }
    ]
    garbage = (
        "Bilginizeceğim kaynak dosyanın adı 'diyet.txt' ile ilgili bir saade bulunuyoruz. "
        "Seçtiğiniz sahnedeki sebzeleri doğru şekilde yazım."
    )
    answer = finalize_answer(garbage, hits)
    assert "omlet" in answer.lower()
    assert "baklava" in answer.lower()
    assert answer.startswith("diyet.txt dosyasına göre")


def test_grounded_chat_is_kept():
    from main import finalize_answer

    hits = [
        {
            "source": "diyet.txt",
            "content": "Sebzesiz tarifler: peynirli omlet, tereyağlı pirinç pilavı ve cevizli baklava.",
        }
    ]
    good = "diyet.txt dosyasına göre peynirli omlet, pilav ve baklava."
    assert finalize_answer(good, hits) == good


def test_list_query_uses_chunk_not_short_model_line():
    from main import finalize_answer

    hits = [
        {
            "source": "diyet.txt",
            "content": "Sebzesiz tariflerde sebze yoktur. Bu defterde sebzesiz tarifler: peynirli omlet, tereyağlı pirinç pilavı ve cevizli baklava.",
        }
    ]
    answer = finalize_answer("Bu defterde sebize yoktur.", hits, "sebzesiz tarif öner")
    assert "omlet" in answer.lower()
    assert "baklava" in answer.lower()
    assert "sebize" not in answer.lower()
