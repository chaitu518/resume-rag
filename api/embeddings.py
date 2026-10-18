"""Pluggable embedding providers: OpenAI, Cohere, HuggingFace, or a local
offline fallback (TF-IDF + SVD, i.e. latent semantic analysis).

Choose with EMBEDDING_PROVIDER=openai|cohere|huggingface|local|auto.
"auto" picks OpenAI -> Cohere -> HuggingFace -> local, based on which API
key / package is available, so the project always runs.

Each provider exposes calibrated cosine bounds (sim_floor, sim_ceil) that
the matcher uses to map raw similarity onto a 0-1 score: different models
produce very different similarity ranges, so a raw 0.45 means "strong"
for one model and "weak" for another.
"""
from __future__ import annotations

import os
import sys
import pickle
from pathlib import Path

import numpy as np


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


class BaseEmbedder:
    name = "base"
    sim_floor, sim_ceil = 0.0, 1.0
    needs_fit = False

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    def fit(self, texts: list[str]):  # only the local model needs this
        pass

    def save(self, directory: Path):
        pass

    def load(self, directory: Path) -> bool:
        return True


class OpenAIEmbedder(BaseEmbedder):
    sim_floor, sim_ceil = 0.15, 0.62

    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self.name = f"openai:{model}"

    def embed_documents(self, texts):
        out = []
        for i in range(0, len(texts), 100):
            resp = self.client.embeddings.create(model=self.model, input=texts[i:i + 100])
            out.extend(d.embedding for d in resp.data)
        return _normalize(np.array(out))


class CohereEmbedder(BaseEmbedder):
    sim_floor, sim_ceil = 0.20, 0.65

    def __init__(self, model: str = "embed-english-v3.0"):
        import cohere
        self.client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
        self.model = model
        self.name = f"cohere:{model}"

    def _embed(self, texts, input_type):
        out = []
        for i in range(0, len(texts), 90):
            r = self.client.embed(texts=texts[i:i + 90], model=self.model, input_type=input_type,
                                  embedding_types=["float"])
            out.extend(r.embeddings.float_)
        return _normalize(np.array(out))

    def embed_documents(self, texts):
        return self._embed(texts, "search_document")

    def embed_query(self, text):
        return self._embed([text], "search_query")[0]


class HuggingFaceEmbedder(BaseEmbedder):
    """sentence-transformers model, runs locally on CPU. Default: BGE-small (384-d)."""
    sim_floor, sim_ceil = 0.35, 0.80

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        self.model_name = model
        self.model = SentenceTransformer(model)
        self.name = f"huggingface:{model}"
        # BGE models expect an instruction prefix on queries
        self.query_prefix = "Represent this sentence for searching relevant passages: " if "bge" in model.lower() else ""
        if "minilm" in model.lower():
            self.sim_floor, self.sim_ceil = 0.10, 0.65

    def embed_documents(self, texts):
        return _normalize(self.model.encode(texts, batch_size=32, show_progress_bar=False))

    def embed_query(self, text):
        return self.embed_documents([self.query_prefix + text])[0]


class LocalLSAEmbedder(BaseEmbedder):
    """Offline fallback: TF-IDF (word 1-2 grams) -> TruncatedSVD -> L2 norm.

    Captures co-occurrence semantics ("PyTorch" near "deep learning") without
    any download. Must be fit on the corpus at ingestion time.
    """
    sim_floor, sim_ceil = 0.05, 0.60
    needs_fit = True

    def __init__(self, dims: int = 128):
        self.dims = dims
        self.name = f"local:tfidf-lsa-{dims}"
        self.vectorizer = None
        self.svd = None

    def fit(self, texts):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1,
                                          stop_words="english", token_pattern=r"(?u)\b[\w+#./-]{1,}\b")
        x = self.vectorizer.fit_transform(texts)
        k = max(2, min(self.dims, x.shape[0] - 1, x.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=42).fit(x)

    def embed_documents(self, texts):
        if self.vectorizer is None:
            raise RuntimeError("Local embedder not fitted. Run ingestion first.")
        return _normalize(self.svd.transform(self.vectorizer.transform(texts)))

    def save(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        with open(Path(directory) / "local_lsa.pkl", "wb") as f:
            pickle.dump((self.vectorizer, self.svd), f)

    def load(self, directory):
        p = Path(directory) / "local_lsa.pkl"
        if not p.exists():
            return False
        with open(p, "rb") as f:
            self.vectorizer, self.svd = pickle.load(f)
        return True


def get_embedder(provider: str | None = None) -> BaseEmbedder:
    provider = (provider or os.getenv("EMBEDDING_PROVIDER", "auto")).lower()
    if provider == "openai":
        return OpenAIEmbedder(os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"))
    if provider == "cohere":
        return CohereEmbedder(os.getenv("COHERE_EMBED_MODEL", "embed-english-v3.0"))
    if provider in ("huggingface", "hf"):
        return HuggingFaceEmbedder(os.getenv("HF_EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    if provider == "local":
        return LocalLSAEmbedder()
    # auto
    if os.getenv("OPENAI_API_KEY"):
        try:
            return OpenAIEmbedder()
        except Exception as e:
            print(f"[embeddings] OpenAI unavailable ({e}); trying next", file=sys.stderr)
    if os.getenv("COHERE_API_KEY"):
        try:
            return CohereEmbedder()
        except Exception as e:
            print(f"[embeddings] Cohere unavailable ({e}); trying next", file=sys.stderr)
    try:
        return HuggingFaceEmbedder(os.getenv("HF_EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    except Exception as e:
        print(f"[embeddings] HuggingFace unavailable ({type(e).__name__}); using local TF-IDF/LSA", file=sys.stderr)
    return LocalLSAEmbedder()
