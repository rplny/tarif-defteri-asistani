"""Kosinüs benzerliği: soru vektörü ile SQLite'taki parça vektörlerini karşılaştırır."""
import math

from text_utils import STOP, normalize, stem


def cosine_similarity(a, b):
    """İki embedding arasındaki açı benzerliği; 1.0 = aynı yön, 0.0 = ilgisiz."""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for i in range(n):
        x = float(a[i])
        y = float(b[i])
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if not norm_a or not norm_b:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def find_relevant(query_embedding, doc_embeddings, top_k=2):
    """Tüm parçaları skorlar, en yüksek top_k sonucu döner."""
    scores = []
    for i, doc_emb in enumerate(doc_embeddings):
        score = cosine_similarity(query_embedding, doc_emb)
        scores.append((i, score))
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[: max(1, int(top_k))]


def rank_chunks(query_embedding, docs, doc_embeddings, sources=None, top_k=3, min_score=0.45):
    """SQLite'tan yüklenen vektörleri kosinüs ile sıralar; zayıf eşleşmeleri eler."""
    sources = sources or [""] * len(docs)
    ranked = find_relevant(query_embedding, doc_embeddings, top_k=top_k)
    hits = []
    for index, score in ranked:
        if score < min_score:
            continue
        hits.append(
            {
                "index": index,
                "score": score,
                "content": docs[index],
                "source": sources[index] if index < len(sources) else "",
            }
        )
    return hits


def keyword_rank_chunks(query, docs, sources=None, top_k=3):
    """Kosinüs yetmezse parça metninde kelime örtüşmesiyle arar."""
    sources = sources or [""] * len(docs)
    tokens = []
    for word in normalize(query).split():
        if len(word) < 4 or word in STOP:
            continue
        tokens.append(word)
        stemmed = stem(word)
        if stemmed != word and len(stemmed) >= 3:
            tokens.append(stemmed)
    if not tokens:
        return []
    scored = []
    for index, doc in enumerate(docs):
        blob = normalize(doc)
        source_name = normalize(sources[index] if index < len(sources) else "")
        hits = sum(1 for token in tokens if token in blob)
        if any(token in source_name for token in tokens):
            hits += 2
        if hits:
            scored.append((hits, index))
    scored.sort(key=lambda item: item[0], reverse=True)
    out = []
    for hits, index in scored[: max(1, int(top_k))]:
        out.append(
            {
                "index": index,
                "score": min(1.0, 0.5 + 0.1 * hits),
                "content": docs[index],
                "source": sources[index] if index < len(sources) else "",
            }
        )
    return out


def format_context(hits):
    lines = []
    for hit in hits:
        source = hit.get("source") or "bilinmeyen"
        lines.append(f"[Kaynak: {source}]\n{hit['content']}")
    return "\n\n".join(lines)
