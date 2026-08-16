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
    assert "salatalık" not in answer.lower()
    assert answer.startswith("diyet.txt dosyasına göre")


def test_tatli_oner_uses_first_sentence():
    from main import finalize_answer

    hits = [
        {
            "source": "baklava.txt",
            "content": "Cevizli baklava şerbetli tatlıdır. Malzemeler: yufka, ceviz, tereyağ, şeker, su ve limon. Sebze yoktur. Vejetaryendir, vegan değildir.",
        }
    ]
    answer = finalize_answer("", hits, "tatlı öner")
    assert "baklava" in answer.lower()
    assert "malzemeler" not in answer.lower()
    assert "vegan" not in answer.lower()
    assert answer.startswith("baklava.txt dosyasına göre")


def test_tatli_oner_lists_notebook_desserts():
    from main import answer_query, load_knowledge_items
    from text_utils import normalize

    client = LocalEmbeddingClient()
    items = load_knowledge_items()
    docs = [item["content"] for item in items]
    sources = [item["source"] for item in items]
    embs = [item.embedding for item in client.generate_embeddings(docs).data]
    answer = normalize(answer_query("tatlı öner", client, embs, docs=docs, sources=sources))
    assert "baklava" in answer
    assert "sutlac" in answer
    assert "brownie" in answer
    assert "malzemeler" not in answer


def test_category_lists_use_diyet_catalog():
    from main import answer_query, load_knowledge_items
    from text_utils import normalize

    client = LocalEmbeddingClient()
    items = load_knowledge_items()
    docs = [item["content"] for item in items]
    sources = [item["source"] for item in items]
    embs = [item.embedding for item in client.generate_embeddings(docs).data]

    vegetarian = normalize(answer_query("vejetaryen tarif öner", client, embs, docs=docs, sources=sources))
    assert "menemen" in vegetarian
    assert "cacik" in vegetarian
    assert "malzemeler" not in vegetarian
    assert vegetarian.startswith("diyet.txt")

    breakfast = normalize(answer_query("kahvaltı öner", client, embs, docs=docs, sources=sources))
    assert "menemen" in breakfast
    assert "omlet" in breakfast
    assert breakfast.startswith("diyet.txt")

    mezze = normalize(answer_query("meze öner", client, embs, docs=docs, sources=sources))
    assert "cacik" in mezze
    assert "kisir" in mezze
    assert "koftesi" in mezze
    assert mezze.startswith("diyet.txt")

    soup = normalize(answer_query("çorba öner", client, embs, docs=docs, sources=sources))
    assert "mercimek" in soup
    assert "iskembe" in soup
    assert "omlet" not in soup
    assert soup.startswith("diyet.txt")

    meat = normalize(answer_query("etli tarif öner", client, embs, docs=docs, sources=sources))
    assert "kofte" in meat
    assert "tavuk" in meat
    assert "baklava" not in meat
    assert meat.startswith("diyet.txt")

    meatless = normalize(answer_query("etsiz tarif öner", client, embs, docs=docs, sources=sources))
    assert "menemen" in meatless
    assert "cacik" in meatless
    assert meatless.startswith("diyet.txt")

    no_veg = normalize(answer_query("sebzesiz tarif öner", client, embs, docs=docs, sources=sources))
    assert "omlet" in no_veg
    assert "sutlac" in no_veg
    assert "brownie" in no_veg
    assert no_veg.startswith("diyet.txt")


def test_howto_uses_recipe_file_not_lookalike():
    from main import load_knowledge_items

    client = LocalEmbeddingClient()
    items = load_knowledge_items()
    docs = [item["content"] for item in items]
    sources = [item["source"] for item in items]
    embs = [item.embedding for item in client.generate_embeddings(docs).data]
    tavuk = get_top_chunks("Tavuk sote nasıl yapılır?", client, docs, embs, sources=sources)
    assert tavuk
    assert tavuk[0]["source"] == "tavuk_sote.txt"
    kofte = get_top_chunks("Köfte nasıl yapılır?", client, docs, embs, sources=sources)
    assert kofte
    assert kofte[0]["source"] == "kofte.txt"
    iskembe = get_top_chunks("İşkembe çorbası nasıl yapılır?", client, docs, embs, sources=sources)
    assert iskembe
    assert iskembe[0]["source"] == "iskembe_corbasi.txt"
    assert all(hit["source"] == "iskembe_corbasi.txt" for hit in iskembe)
    mercimek_kofte = get_top_chunks("Mercimek köftesi nasıl yapılır?", client, docs, embs, sources=sources)
    assert mercimek_kofte
    assert mercimek_kofte[0]["source"] == "mercimek_koftesi.txt"
    assert all(hit["source"] == "mercimek_koftesi.txt" for hit in mercimek_kofte)


def test_keyword_does_not_match_sotele_to_sote():
    from retrieval import keyword_rank_chunks

    docs = [
        "Soğan ve biberi zeytinyağında sotele. Yumurtaları kır.",
        "Tavuk sote tavuk göğsüyle yapılan etli bir yemektir.",
    ]
    sources = ["menemen.txt", "tavuk_sote.txt"]
    hits = keyword_rank_chunks("Tavuk sote nasıl yapılır?", docs, sources)
    assert hits
    assert hits[0]["source"] == "tavuk_sote.txt"
    assert all(hit["source"] != "menemen.txt" for hit in hits)


def test_select_hits_vejetaryen_prefers_catalog():
    from main import select_hits

    hits = [
        {"source": "baklava.txt", "content": "Cevizli baklava. Vejetaryendir, vegan değildir."},
        {
            "source": "diyet.txt",
            "content": "Bu defterde vejetaryen tarifler: menemen, cacık ve peynirli omlet.",
        },
    ]
    picked = select_hits("vejetaryen tarif öner", hits)
    assert picked
    assert picked[0]["source"] == "diyet.txt"
    assert "menemen" in picked[0]["content"]
