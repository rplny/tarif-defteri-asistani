""".txt klasöründen paragraf parçaları; her parçaya kaynak dosya adı eklenir."""
from pathlib import Path

MAX_CHUNK_CHARS = 50_000


def _clean(text):
    return str(text or "").replace("\x00", "").strip()[:MAX_CHUNK_CHARS]


def _split_paragraphs(text):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs and text.strip():
        return [text.strip()]
    return paragraphs


def load_chunk_records(folder_path, chunk_size=500):
    """Her kayıt: kaynak dosya adı ve metin."""
    folder = Path(folder_path)
    if not folder.exists():
        return []
    records = []
    for path in sorted(folder.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for para in _split_paragraphs(text):
            para = _clean(para)
            if not para:
                continue
            if len(para) <= chunk_size:
                records.append({"source": path.name, "content": para})
                continue
            for i in range(0, len(para), chunk_size):
                piece = para[i : i + chunk_size].strip()
                if piece:
                    records.append({"source": path.name, "content": piece})
    return records


if __name__ == "__main__":
    from embeddings import get_embedding_client
    from knowledge_store import KB_FOLDER, get_connection, index_documents, load_records

    items = load_chunk_records(KB_FOLDER)
    conn = get_connection()
    index_documents(conn, items, get_embedding_client())
    records = load_records(conn)
    conn.close()
    print(f"Parçalama bitti: {len(records)} parça, {len({r['source'] for r in records})} dosya.")
