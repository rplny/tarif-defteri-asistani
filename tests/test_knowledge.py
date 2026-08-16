"""SQLite bilgi tabanı testleri."""
from embeddings import LocalEmbeddingClient
from ingestion import load_chunk_records
from knowledge_store import get_connection, index_documents, load_chunks, load_records, save_chunk
from main import load_knowledge_items

SAMPLE_ITEMS = [
    {"source": "a.txt", "content": "Menemen yumurta ve domates ile yapılır."},
    {"source": "b.txt", "content": "Cacık yoğurt ve salatalık içerir."},
    {"source": "c.txt", "content": "Kısır vegan bir mezedir."},
]


def test_save_and_load_chunks(tmp_path):
    conn = get_connection(tmp_path / "kb.db")
    save_chunk(conn, "hello", [0.1, 0.2], source="demo.txt")
    docs, embs = load_chunks(conn)
    records = load_records(conn)
    assert docs == ["hello"]
    assert embs == [[0.1, 0.2]]
    assert records[0]["source"] == "demo.txt"
    conn.close()


def test_index_from_txt_folder():
    items = load_knowledge_items()
    assert len(items) >= 8
    texts = [item["content"] for item in items]
    sources = {item["source"] for item in items}
    assert any("Menemen" in text or "yumurta" in text for text in texts)
    assert "menemen.txt" in sources
    assert "cacik.txt" in sources
    assert "diyet.txt" in sources
    assert any("vegan" in text.lower() for text in texts)
    assert any("sebzesiz" in text.lower() for text in texts)


def test_index_documents_with_local_embed(tmp_path):
    conn = get_connection(tmp_path / "kb.db")
    client = LocalEmbeddingClient()
    index_documents(conn, SAMPLE_ITEMS, client)
    docs, embs = load_chunks(conn)
    assert len(docs) == 3
    assert len(embs) == 3
    assert len(embs[0]) > 0
    conn.close()


def test_folder_loader(tmp_path):
    (tmp_path / "a.txt").write_text("Paragraf bir.\n\nParagraf iki.", encoding="utf-8")
    records = load_chunk_records(tmp_path)
    assert records[0]["source"] == "a.txt"
    assert [item["content"] for item in records] == ["Paragraf bir.", "Paragraf iki."]
