"""Yerel embedding: Foundry Local açıksa onu, yoksa kelime vektörünü kullanır."""
import os
from collections import Counter

import text_utils

DIM = 256


def _hash_token(token):
    h = 2166136261
    for ch in token:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h % DIM


def local_embed(text):
    tokens = text_utils.normalize(text).split()
    vec = [0.0] * DIM
    if not tokens:
        return vec
    counts = Counter(tokens)
    for token, count in counts.items():
        if len(token) < 2:
            continue
        vec[_hash_token(token)] += float(count)
        stem = text_utils.stem(token)
        if stem != token:
            vec[_hash_token(stem)] += float(count) * 0.7
    return vec


class LocalEmbeddingClient:
    def generate_embedding(self, text):
        return _Wrap([_Item(local_embed(text))])

    def generate_embeddings(self, texts):
        return _Wrap([_Item(local_embed(t)) for t in texts])


class _Item:
    def __init__(self, embedding):
        self.embedding = embedding


class _Wrap:
    def __init__(self, data):
        self.data = data


def foundry_enabled():
    value = os.environ.get("FOUNDRY_RAG", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def get_embedding_client():
    if foundry_enabled():
        from foundry_local_sdk import Configuration, FoundryLocalManager

        config = Configuration(app_name="foundry_local_rag")
        FoundryLocalManager.initialize(config)
        manager = FoundryLocalManager.instance
        model = manager.catalog.get_model("qwen3-embedding-0.6b")
        if hasattr(model, "download"):
            model.download(lambda p: print(f"\rEmbedding modeli: {p:.1f}%", end="", flush=True))
            print()
        model.load()
        return model.get_embedding_client()
    return LocalEmbeddingClient()
