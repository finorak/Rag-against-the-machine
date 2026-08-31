import json
from typing import Any

from bm25s import Path

from .answer import AnswerEngine
from .index import IndexEngine
from .search import SearchEngine
from .utils import get_path


class App:
    def __init__(self) -> None:
        data_path: str = get_path("data", "raw")
        self.chunk_file: Path = Path("data", "processed", "chunk.json")
        self.index_engine = IndexEngine(data_path, self.chunk_file)
        self.search_engine = SearchEngine()
        self.answer_engine = AnswerEngine()
        self.search_cache: dict[str, Any] = {}
        self.answer_cache: dict[str, Any] = {}

    def index(self, max_chunk_size: int = 2000) -> None:
        self.index_engine.index(max_chunk_size)

    def search(self, query: str, k: int = 10) -> dict[str, Any]:
        cache_key = f"{query}: {k}"
        cache_value = self.search_cache.get(cache_key)
        if cache_value:
            return json.loads(cache_value)
        search_results = self.search_engine.search(query, k, self.chunk_file)
        self.search_cache[cache_key] = cache_value
        return search_results

    def search_dataset(
            self, dataset_path: str,
            save_directory: str,
            k: int = 10
    ) -> list[dict[str, Any]]:
        search_dataset_results = self.search_engine.search_dataset(
                dataset_path, save_directory, k)
        return search_dataset_results

    def answer(self, query: str, k: int = 10) -> dict[str, Any]:
        cache_key = "{query}: {k}"
        cache_value = self.answer_cache.get(cache_key)
        if cache_value:
            return json.loads(cache_value)
        response_results = self.answer_engine.answer(query, k)
        self.answer_cache[cache_key] = cache_value
        return response_results

    def answer_dataset(
            self, student_search_results_path: str,
            dataset_path: str
    ) -> list[dict[str, Any]]:
        ...
