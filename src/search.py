import json
from pathlib import Path
from typing import Any

from bm25s import BM25, tokenize

from .model import (
    MinimalSearchResults,
    MinimalSource,
    UnansweredQuestion,
)
from .utils import get_minimal_sources


class SearchEngine:
    def __init__(self, index_dir: Path, chunk_file: Path) -> None:
        self.index_dir: Path = index_dir
        self.chunk_file: Path = chunk_file
        self.retriever: BM25 | None = None
        self.search_cache: dict[str, Any] = {}

    def search(self, query: str, k: int) -> dict[str, Any]:
        cache_key = f"{query}: {k}"
        cache_value = self.search_cache.get(cache_key)
        if cache_value:
            return json.loads(cache_value)
        unanswered_question = UnansweredQuestion(question=query)
        if self.retriever is None:
            self.retriever = BM25.load(
                    self.index_dir, load_corpus=True, mmap=True
                    )
        minimal_sources = get_minimal_sources(self.chunk_file)
        results, scores = self.retriever.retrieve(
                tokenize(query.replace("_", " ")), k=k,
                corpus=minimal_sources, sorted=True
                )
        retrieved_sources: list[MinimalSource] = []
        for i in range(scores.shape[1]):
            match_source = results[0, i]
            retrieved_sources.append(match_source)
        minimal_search_results = MinimalSearchResults(
                question_id=unanswered_question.question_id,
                question=unanswered_question.question,
                retrieved_sources=retrieved_sources)
        self.search_cache[cache_key] = cache_value
        return minimal_search_results.model_dump()

    def search_dataset(
            self, dataset_path: str,
            save_directory: str, k: int
    ) -> list[dict[str, Any]]:
        ...
