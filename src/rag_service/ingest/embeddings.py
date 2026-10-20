"""Embedding backends.

Three backends, selected by the ``embedding_model`` setting:

* an OpenAI model name (``text-embedding-3-small``, ...) -- production default
* ``hf:<model-id>`` -- local sentence-transformers, no API key
* ``hashing`` -- deterministic hashed bag-of-words, no key and no download

``hashing`` exists so the sweep, the tests and CI can run end to end without a
secret. It is a lexical baseline, not a semantic model: use it to compare
*chunking* mechanics reproducibly, not to judge absolute answer quality.
"""

import hashlib
import math
import re
from collections import Counter

from langchain_core.embeddings import Embeddings

from rag_service.config import Settings, get_settings

HASHING_BACKEND = "hashing"
HF_PREFIX = "hf:"

# Unicode-aware word tokenizer: keeps Arabic and Latin runs, drops punctuation.
_TOKEN = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


class HashingEmbeddings(Embeddings):
    """Feature-hashed bag of words, L2-normalised.

    Deterministic across processes and machines (BLAKE2b, not Python's salted
    ``hash``), so a sweep run today is comparable with one run tomorrow.
    """

    def __init__(self, dim: int = 512, ngram: int = 4):
        self.dim = dim
        self.ngram = ngram

    def _bucket(self, token: str) -> tuple[int, int]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % self.dim, 1 if (value >> 63) & 1 else -1

    def _features(self, text: str) -> Counter:
        tokens = [t.lower() for t in _TOKEN.findall(text)]
        features = Counter(tokens)
        # Character n-grams add robustness to Arabic affixes and inflection.
        squashed = "".join(tokens)
        for i in range(len(squashed) - self.ngram + 1):
            features[squashed[i : i + self.ngram]] += 1
        return features

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token, count in self._features(text).items():
            index, sign = self._bucket(token)
            # sqrt damps very frequent terms, as tf-idf's log would.
            vector[index] += sign * math.sqrt(count)

        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    """Build the embedding backend named by ``settings.embedding_model``."""
    settings = settings or get_settings()
    name = settings.embedding_model

    if name == HASHING_BACKEND:
        return HashingEmbeddings(dim=settings.hashing_dim)

    if name.startswith(HF_PREFIX):
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=name[len(HF_PREFIX) :],
            encode_kwargs={"normalize_embeddings": True},
        )

    from langchain_openai import OpenAIEmbeddings

    if not settings.openai_api_key:
        raise RuntimeError(
            f"embedding_model={name!r} needs OPENAI_API_KEY. Set it, or use "
            f"RAG_EMBEDDING_MODEL={HASHING_BACKEND} for an offline run."
        )
    return OpenAIEmbeddings(model=name, openai_api_key=settings.openai_api_key)
