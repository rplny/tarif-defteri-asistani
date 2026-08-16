"""SQLite bilgi tabanı: kaynak adı + metin + embedding."""
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "knowledge_base.db"
KB_FOLDER = ROOT / "knowledge"


def get_connection(db_path=None):
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            source TEXT,
            content TEXT,
            embedding TEXT
        )
        """
    )
    columns = {row[1] for row in conn.execute("PRAGMA table_info(documents)")}
    if "source" not in columns:
        conn.execute("ALTER TABLE documents ADD COLUMN source TEXT DEFAULT ''")
    conn.commit()
    return conn


def save_chunk(conn, content, embedding_vector, source=""):
    conn.execute(
        "INSERT INTO documents (source, content, embedding) VALUES (?, ?, ?)",
        (source or "", content, json.dumps(list(embedding_vector))),
    )
    conn.commit()


def load_records(conn):
    rows = conn.execute(
        "SELECT COALESCE(source, ''), content, embedding FROM documents"
    ).fetchall()
    return [
        {"source": row[0], "content": row[1], "embedding": json.loads(row[2])}
        for row in rows
    ]


def load_chunks(conn):
    records = load_records(conn)
    docs = [item["content"] for item in records]
    embeddings = [item["embedding"] for item in records]
    return docs, embeddings


def clear_chunks(conn):
    conn.execute("DELETE FROM documents")
    conn.commit()


def _normalize_items(items):
    normalized = []
    for item in items or []:
        if isinstance(item, dict):
            content = (item.get("content") or "").strip()
            source = (item.get("source") or "").strip()
        else:
            content = str(item).strip()
            source = ""
        if content:
            normalized.append((source, content))
    return normalized


def index_documents(conn, items, embedding_client):
    clear_chunks(conn)
    pairs = _normalize_items(items)
    if not pairs:
        return
    texts = [content for _, content in pairs]
    response = embedding_client.generate_embeddings(texts)
    for (source, text), item in zip(pairs, response.data):
        save_chunk(conn, text, item.embedding, source)
