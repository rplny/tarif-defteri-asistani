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
