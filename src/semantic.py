"""Hybrid and semantical search module."""


from pathlib import Path

import numpy as np
from bm25s import Any
from sentence_transformers import SentenceTransformer

from .model import MinimalSource
from .utils import get_minimal_sources, get_path, secure_embed

MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"
EMBEDDIN_DIR = Path(get_path("data", "processed", "database.npy"))


class SemanticEngine:
    """Class to perform an Hybrid search.

    An hybrid search will only occure if the flag
    `hybrid` search is activated.
    """

    def __init__(
            self, chunk_file: Path, K: int = 60
    ) -> None:
        """Initiate an hybrid instance class."""
        self.model = SentenceTransformer(model_name_or_path=MODEL_NAME)
        self.chunk_file = chunk_file
        self.K = K

    def embed(self, corpus: list[str]) -> None:
        """Embed our corpuse."""
        if EMBEDDIN_DIR.exists():
            self.vector_matrix = np.load(EMBEDDIN_DIR)
        else:
            self.vector_matrix = secure_embed(
                    self.model, corpus
                    )
            np.save(EMBEDDIN_DIR, self.vector_matrix)

    def search(
            self, query: str, k: int, sources: list[MinimalSource]
    ) -> Any:
        """Execute  dens search for the query.

        Args:
            query_vector: vector represnting the query.
            embedding_vecs: vector of the corpuse.
            k: top k results to retrieve.
        Returns:
            res: the results.
        """
        query_vector = self.model.encode_query(query)
        self.embed(
                [source.chunk for source in sources]
                )
        q_norm = query_vector / np.linalg.norm(query_vector)
        doc_normed = self.vector_matrix / np.linalg.norm(
                self.vector_matrix, axis=1, keepdims=True
                )
        scores = doc_normed @ q_norm
        top_k = np.argsort(-scores)[:k]
        extracte_sources: list[MinimalSource] = [
                sources[idx]
                for idx in top_k
                ]
        for index, el in enumerate(extracte_sources, start=1):
            el.vector_rank = index
        return extracte_sources

    def rrf(self, minimal_sources: list[MinimalSource]) -> list[MinimalSource]:
        """Execute an rrf scoring.

        Using the rank of the document retrieved
        from semantic and bm25, we take the
        minimum based on those rank.

        Args:
            minimal_sources: a list of minimal_sources
        Returns:
            combined_source: a list of minimal source after rrf.
        """
        sources: list[MinimalSource] = []
        for source in minimal_sources:
            bm_score = 1 / (self.K + source.bm_rank)
            vector_score = 1 / (self.K + source.vector_rank)
            source.score = bm_score + vector_score
        return sorted(sources, key=lambda source: -source.score)
