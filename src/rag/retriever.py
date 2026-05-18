from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import torch

log = logging.getLogger(__name__)

ScoreFunction = Literal["cos_sim", "dot"]


def _mean_pool(model_output, attention_mask: torch.Tensor) -> np.ndarray:
    token_embs = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(token_embs.size()).float()
    summed = torch.sum(token_embs * mask, 1)
    counts = torch.clamp(mask.sum(1), min=1e-9)
    return (summed / counts).detach().cpu().numpy()


class Retriever:
    """
    Dense retrieval with FAISS IndexFlatIP.

    Supports GTR-base (via SentenceTransformer) and Contriever
    (via HuggingFace with mean pooling).

    For the attack loop, score_doc() computes a single doc–query similarity
    without touching the FAISS index — needed to check if a candidate
    adversarial document would be retrieved.

    Heavy dependencies (faiss, sentence_transformers, transformers) are
    imported lazily inside __init__ so tests can mock the class without
    those packages being installed.
    """

    def __init__(
        self,
        model_name: str,
        score_function: ScoreFunction = "cos_sim",
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.score_function = score_function
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._is_contriever = "contriever" in model_name.lower()

        log.info("Loading retrieval model: %s", model_name)
        if self._is_contriever:
            from transformers import AutoModel, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(model_name).to(self.device)
            self._model.eval()
            self._st = None
        else:
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer(model_name, device=self.device)
            self._tok = None
            self._model = None

        self._index = None
        self._doc_ids: list[str] = []
        self._corpus: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Returns L2-normalized embeddings, shape (N, D)."""
        if self._is_contriever:
            all_embs: list[np.ndarray] = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                enc = self._tok(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                ).to(self.device)
                with torch.no_grad():
                    out = self._model(**enc)
                emb = _mean_pool(out, enc["attention_mask"])
                all_embs.append(emb)
            embs = np.vstack(all_embs).astype(np.float32)
        else:
            embs = self._st.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=False,
                convert_to_numpy=True,
            ).astype(np.float32)

        # L2-normalize so IndexFlatIP gives cosine similarity
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        return embs / norms

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def build_index(self, corpus: dict[str, str]) -> None:
        """
        Build a FAISS index from a corpus dict {doc_id: text}.
        Corpus texts are embedded in batches.
        """
        import faiss

        self._corpus = corpus
        self._doc_ids = list(corpus.keys())
        texts = [corpus[d] for d in self._doc_ids]

        log.info("Building FAISS index for %d documents...", len(texts))
        embs = self._embed(texts)
        dim = embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(embs)
        self._index = index
        log.info("Index built: %d vectors, dim=%d", self._index.ntotal, dim)

    def save_index(self, path: Path) -> None:
        import faiss

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        cpu_index = (
            faiss.index_gpu_to_cpu(self._index)
            if hasattr(self._index, "getDevice")
            else self._index
        )
        faiss.write_index(cpu_index, str(path / "index.faiss"))
        with open(path / "meta.pkl", "wb") as f:
            pickle.dump({"doc_ids": self._doc_ids, "corpus": self._corpus}, f)
        log.info("Index saved to %s", path)

    def load_index(self, path: Path) -> None:
        import faiss

        path = Path(path)
        cpu_index = faiss.read_index(str(path / "index.faiss"))
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self._index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            self._index = cpu_index
        with open(path / "meta.pkl", "rb") as f:
            meta = pickle.load(f)
        self._doc_ids = meta["doc_ids"]
        self._corpus = meta["corpus"]
        log.info("Index loaded: %d vectors", self._index.ntotal)

    def build_or_load_index(self, corpus: dict[str, str], index_dir: Path) -> None:
        index_dir = Path(index_dir)
        faiss_path = index_dir / "index.faiss"
        if faiss_path.exists():
            log.info("Found existing index at %s — loading.", index_dir)
            self.load_index(index_dir)
        else:
            self.build_index(corpus)
            self.save_index(index_dir)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        """Returns top-k doc_ids by cosine similarity."""
        assert self._index is not None, "Call build_index() or load_index() first."
        q_emb = self._embed([query])
        _, ids = self._index.search(q_emb, k)
        return [self._doc_ids[i] for i in ids[0] if i >= 0]

    def retrieve_with_scores(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        assert self._index is not None
        q_emb = self._embed([query])
        scores, ids = self._index.search(q_emb, k)
        return [
            (self._doc_ids[i], float(scores[0][j]))
            for j, i in enumerate(ids[0])
            if i >= 0
        ]

    def score_doc(self, doc_text: str, query_text: str) -> float:
        """
        Cosine similarity between a single document and a query.
        Used by the attack loop to check if a candidate would be retrieved
        without touching the FAISS index.
        """
        embs = self._embed([doc_text, query_text])
        return float(np.dot(embs[0], embs[1]))

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Public access to the L2-normalized embedding function."""
        return self._embed(texts)

    def get_doc_text(self, doc_id: str) -> str:
        return self._corpus.get(doc_id, "")
