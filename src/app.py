"""Rag app combiner.

This module help us combine all what we need to
achieve our retrieval and answer in a way that it is
portable and not very hard to improve later as
it's class has it's own module and this just combine
those classes.
"""


import sys
from typing import Any

from bm25s import Path

from .answer import AnswerEngine
from .index import IndexEngine
from .search import SearchEngine
from .utils import get_path


class App:
    """An app used to combine all needed to achieve our Rag."""

    def __init__(self) -> None:
        """Initiate an App instance."""
        data_path: str = get_path("data", "raw")
        processed_dir: Path = Path(get_path("data", "raw", "processed"))
        chunk_file: Path = Path("data", "processed", "chunk.json")
        index_dir: Path = Path(get_path("data", "processed", "bm_index"))
        self.index_engine = IndexEngine(
                data_path, processed_dir,
                index_dir, chunk_file
                )
        self.search_engine = SearchEngine(index_dir, chunk_file)
        self.answer_engine = AnswerEngine(self.search_engine)

    def index(self, max_chunk_size: int = 2000) -> None:
        """Execute an indexation on the provided data.

        Args:
            max_chunk_size: the chunk size for our corpus \
the default is `2000`
        """
        if 200 < max_chunk_size or max_chunk_size > 2000:
            print("Index size must be between 200 and 2000", file=sys.stderr)
            sys.exit(1)
        self.index_engine.index(max_chunk_size)

    def search(self, query: str, k: int = 10) -> dict[str, Any] | Any:
        """Exectute a search based on user's query.

        Args:
            query: what the user requested.
            k: top-k needed to retreieve the default is `10`.
        Returns:
            search_result: the results of our retrieval.
        """
        search_results = self.search_engine.search(query, k)
        return search_results

    def search_dataset(
            self, dataset_path: str,
            save_directory: str,
            k: int = 10
    ) -> dict[str, Any] | Any:
        """Exectute a batch search based on the dataset.

        Args:
            dataset_path: path to our dataset.
            save_directory: where to save our search_results.
            k: top-k needed to retreieve the default is `10`.
        Returns:
            search_result: the results of our retrieval.
        """
        search_dataset_results = self.search_engine.search_dataset(
                dataset_path, save_directory, k)
        return search_dataset_results

    def answer(self, query: str, k: int = 10) -> dict[str, Any] | Any:
        """Exectute a retrieval and answer based on user's query.

        Args:
            query: what the user requested.
            k: top-k needed to retreieve the default is `10`.
        Returns:
            answer_result: the results of our retrieval and answer.
        """
        response_results = self.answer_engine.answer(query, k)
        return response_results

    def answer_dataset(
            self, student_search_results_path: str,
            dataset_path: str
    ) -> dict[str, Any] | Any:
        """Exectute a batch retrieval and answer based on user's query.

        Args:
            student_search_results_path: path to our dataset.
            dataset_path: where to save our answer results.
        Returns:
            answer_result: the results of our retrieval and answer.
        """
        response_results = self.answer_engine.answer_dataset(
                student_search_results_path, dataset_path
                )
        return response_results
