"""Kosinüs benzerliği: soru vektörü ile SQLite'taki parça vektörlerini karşılaştırır."""
import math
from pathlib import Path

from text_utils import STOP, has_term, normalize, stem


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


QUERY_STOP = STOP | {
    "hangi", "malzeme", "malzemeler", "nedir", "icerir", "icinde", "olan",
}


def token_in_text(text, token):
    blob = normalize(text)
    if has_term(blob, token):
        return True
    return any(stem(word) == token for word in blob.split() if word)


def keyword_rank_chunks(query, docs, sources=None, top_k=3):
    """Kosinüs yetmezse parça metninde kelime örtüşmesiyle arar."""
    sources = sources or [""] * len(docs)
    qnorm = normalize(query)
    want_meat = "etli" in qnorm and "etsiz" not in qnorm
    tokens = []
    for word in qnorm.split():
        if len(word) < 4 or word in QUERY_STOP:
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
        source_stem = normalize(Path(sources[index] or "").stem)
        hits = 0
        meatless = "et yoktur" in blob or "et icermez" in blob
        for token in tokens:
            if token == "vegan" and "vegan degildir" in blob:
                continue
            if want_meat and meatless:
                continue
            if token_in_text(blob, token):
                hits += 1
        if source_stem and len(source_stem) >= 4:
            if any(
                token == source_stem
                or token.startswith(source_stem)
                or source_stem.startswith(token)
                or token in source_name
                for token in tokens
            ):
                hits += 2
        elif any(token_in_text(source_name, token) or token in source_name for token in tokens):
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


def part_in_query(query, part):
    q = normalize(query)
    if has_term(q, part):
        return True
    return any(tok == part or stem(tok) == part for tok in q.split())


def tighten_hits(query, hits):
    """Dosya adı sorudaki yemeğe uyuyorsa diğer parçaları bırakır."""
    if not hits:
        return hits
    named = []
    for hit in hits:
        name = normalize(Path(hit.get("source") or "").stem)
        parts = [part for part in name.split("_") if len(part) >= 4]
        if parts and all(part_in_query(query, part) for part in parts):
            named.append(hit)
    return named or hits


def format_context(hits):
    lines = []
    for hit in hits:
        source = hit.get("source") or "bilinmeyen"
        lines.append(f"[Kaynak: {source}]\n{hit['content']}")
    return "\n\n".join(lines)
