from pathlib import Path
from typing import Any

from bm25s import BM25, tokenize

from .model import (
    MinimalSearchResults,
    MinimalSource,
    UnansweredQuestion,
)
from .utils import get_minimal_sources, get_path


class SearchEngine:
    def __init__(self) -> None:
        self.index_dir: Path = Path(get_path("data", "processed", "bm_index"))
        self.retriever: BM25 | None = None

    def search(self, query: str, k: int, chunk_file: Path) -> dict[str, Any]:
        unanswered_question = UnansweredQuestion(question=query)
        if self.retriever is None:
            self.retriever = BM25.load(
                    self.index_dir, load_corpus=True, mmap=True
                    )
        minimal_sources = get_minimal_sources(chunk_file)
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
        return minimal_search_results.model_dump()

    def search_dataset(
            self, dataset_path: str,
            save_directory: str, k: int
    ) -> list[dict[str, Any]]:
        ...
