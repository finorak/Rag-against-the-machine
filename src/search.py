import json
from pathlib import Path
from typing import Any

from bm25s import BM25, tokenize

from .model import (
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
    UnansweredQuestion,
)
from .utils import get_minimal_sources, get_path


class SearchEngine:
    def __init__(self, index_dir: Path, chunk_file: Path) -> None:
        self.index_dir: Path = index_dir
        self.chunk_file: Path = chunk_file
        self.retriever: BM25 | None = None
        self.cache: dict[str, dict[str, Any]] = {}

    def search(self, query: str, k: int, question_id: str = "") -> dict[str, Any]:
        query = query.replace("_", " ").strip()
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
        results, scores = self.retriever.retrieve(
                tokenize(query), k=k,
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
        self.cache[cache_key] = minimal_search_results.model_dump()
        return minimal_search_results.model_dump()

    def search_dataset(
            self, dataset_path: str,
            save_directory: str, k: int
    ) -> dict[str, Any]:
        with open(dataset_path, mode="r", encoding="utf-8") as f:
            raw_data = json.load(f)
        rag_questions = raw_data['rag_questions']
        search_lst: list[MinimalSearchResults] = []
        for prompt in rag_questions:
            search_results = self.search(prompt['question'], k, prompt['question_id'])
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
                search_results=search_lst,
                k=k
                )
        data_path: Path = Path(dataset_path)
        save_path: Path = Path(save_directory)
        save_file: Path = Path(get_path(save_path._raw_paths[0], data_path.name))
        save_file.parent.mkdir(parents=True, exist_ok=True)
        with open(save_file, mode="w", encoding="utf-8") as f:
            json.dump(student_search_results.model_dump(), fp=f, indent=4)
        return student_search_results.model_dump()
