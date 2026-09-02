"""Search engine module."""


import sys
from pathlib import Path
from typing import Any

from bm25s import BM25, tokenize
from tqdm.std import tqdm

from .model import (
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
    UnansweredQuestion,
)
from .semantic import SemanticEngine
from .utils import get_minimal_sources, get_path, secure_open


class SearchEngine:
    """Search engine class.

    This help us retrieve minimal sources to help
    us answer the user's query.
    """

    def __init__(
            self, index_dir: Path,
            chunk_file: Path,
            semantic_engine: SemanticEngine
    ) -> None:
        """Initiate Search Engine instance.

        Args:
            index_dir: path to where the indexe folder is located.
            chunk_file: path to where the chunk file is located.
        """
        self.index_dir: Path = index_dir
        self.chunk_file: Path = chunk_file
        self.retriever: BM25 | None = None
        self.cache: dict[str, dict[str, Any]] = {}
        self.semantic_engine = semantic_engine

    def search(
            self, query: str, k: int,
            question_id: str = "",
            hybrid_search: bool = False
    ) -> dict[str, Any] | Any:
        """Search user's query.

        Searching best sources that match the query.
        Insead of recompute everytime, we cache the query, k
        values into a string and see if it's already in
        the cache, if yes, we return the retrieved_sources imediatly,
        this avoid redoing the same question everytime for example,
        as we only embed the query a single time and just retrieve the
        result of the searched before.

        Args:
            query: What the user asked for.
            k: top-k result to retrieve.
            question_id: the id of the question if it is \
provided.
        Returns:
            search_results: the result of our query.
        """
        query = query.replace("_", " ").strip()
        if not query or k < 0:
            print("Query can't be empty or k < 0", file=sys.stderr)
            return {}
        cache_key = f"{query}: {k}"
        cache_value = self.cache.get(cache_key)
        if cache_value:
            return cache_value
        unanswered_question = UnansweredQuestion(question=query)
        if question_id:
            unanswered_question.question_id = question_id
        if self.retriever is None:
            self.retriever = BM25.load(
                    self.index_dir, load_corpus=True, mmap=True
                    )
        minimal_sources = get_minimal_sources(self.chunk_file)
        if hybrid_search:
            self.semantic_engine.search(
                    query, k, minimal_sources
                    )
        results, scores = self.retriever.retrieve(
                tokenize(query), k=k,
                corpus=minimal_sources, sorted=True
                )
        retrieved_sources: list[MinimalSource] = []
        for i in range(scores.shape[1]):
            match_source = results[0, i]
            match_source.bm_rank = i + 1
            retrieved_sources.append(match_source)
        minimal_search_results = MinimalSearchResults(
                question_id=unanswered_question.question_id,
                question=unanswered_question.question,
                retrieved_sources=retrieved_sources)
        self.cache[cache_key] = minimal_search_results.model_dump()
        return minimal_search_results.model_dump()

    def search_dataset(
            self, dataset_path: str,
            save_directory: str, k: int,
            hybrid_search: bool = False
    ) -> dict[str, Any] | Any:
        """Execute a search in a batch.

        Using the provided dataset we query it and
        execute a search for each element of the dataset.

        Args:
            dataset_path: where the dataset is located.
            save_directory: where to save the results.
        search_results: the result of our batch query.
        """
        raw_data = secure_open(dataset_path)
        rag_questions = raw_data['rag_questions']
        search_lst: list[MinimalSearchResults] = []
        for prompt in tqdm(rag_questions, desc="Retrieve dataset"):
            search_results = self.search(
                    prompt['question'], k,
                    prompt['question_id'],
                    hybrid_search
                    )
            if not search_results:
                continue
            sources = search_results['retrieved_sources']
            minimal_sources = [
                    MinimalSource(**data)
                    for data in sources
                    ]
            minimal_search = MinimalSearchResults(
                    question_id=prompt["question_id"],
                    question=prompt['question'],
                    retrieved_sources=minimal_sources
                    )
            search_lst.append(minimal_search)
        student_search_results = StudentSearchResults(
                search_results=search_lst, k=k
                )
        data_path: Path = Path(dataset_path)
        save_path: Path = Path(save_directory)
        save_file: Path = Path(
                get_path(
                    str(save_path), data_path.name
                    )
                )
        save_file.parent.mkdir(parents=True, exist_ok=True)
        secure_open(
                save_file, mode="w",
                data=student_search_results.model_dump()
                )
        return student_search_results.model_dump()
