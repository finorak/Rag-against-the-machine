"""Answer engine module."""


import json
import sys
from pathlib import Path
from typing import Any

from langchain_ollama import ChatOllama
from tqdm.std import tqdm

from .model import (
    AnsweredQuestion,
    MinimalAnswer,
    MinimalSource,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)
from .search import SearchEngine
from .utils import get_path, secure_answer, secure_open


class AnswerEngine:
    """Answer engine class.

    This let us answer user's query and
    it is way more portable than the first
    one.
    """

    def __init__(
            self, search_engine: SearchEngine,
            model_name: str = "qwen3:0.6b"
    ) -> None:
        """Answer engine initializer.

        Args:
            search_engine: Search engine instance \
to help us retrieve the best source to respond the query.
            model_name: the model used to answer the query.
        """
        self.search_engine = search_engine
        self.cache: dict[str, Any] = {}
        try:
            self.model = ChatOllama(model=model_name)
        except Exception as e:
            print(e, file=sys.stderr)
            sys.exit(1)

    def answer(
            self, query: str, k: int,
            hybrid_search: bool
    ) -> dict[str, Any] | Any:
        """Answer user's query.

        We answer the user's query using the `qwen3:0.6` model
        as the default.
        Args:
            query: (str) the user's request.
            k: (int) top-k to use to answer the query.
            hybrid_search: weather to perform an hybrid_search
        Returns:
            response: (dict[str, Any]) retrieved sources and answer.
        """
        query = query.strip().replace("_", " ").strip()
        if not query or k <= 0:
            print("Query can't be empty or k < 0", file=sys.stderr)
            return {}
        cache_key = f"{query}: {k} {hybrid_search}"
        cache_value = self.cache.get(cache_key)
        if cache_value:
            return cache_value
        unanswered_question = UnansweredQuestion(question=query)
        search_results = self.search_engine.search(
                query=query, k=k,
                hybrid_search=hybrid_search
                )
        sources: list[str] = [
                data['chunk']
                for data in search_results['retrieved_sources']
                ]
        prompt = self._prompt_augmenter(query, sources)
        response = secure_answer(self.model, prompt)
        answered_question = AnsweredQuestion(
                question_id=unanswered_question.question_id,
                question=query,
                sources=[
                    MinimalSource(**data)
                    for data in search_results['retrieved_sources']
                    ],
                answer=response
                )
        self.cache[cache_key] = answered_question.model_dump()
        return self.cache[cache_key]

    def answer_dataset(
            self, student_search_results_path: str,
            dataset_path: str
    ) -> dict[str, Any] | Any:
        """Answer query in a batch.

        Given a dataset, we call the `answer` function on
        each query and write those response into a file.
        Args:
            student_search_restults_path: the path of the dataset.
            dataset_path: where to save the results.
        Returns:
            restults: a dict containing the search resutls.
        """
        raw_data = secure_open(student_search_results_path)
        results = raw_data.get('search_results')
        k = raw_data.get("k")
        if not results or not k:
            print("k or restults missing", file=sys.stderr)
            sys.exit(1)
        search_results: list[MinimalAnswer] = []
        for search in tqdm(results, desc="Answering dataset"):
            minimal_sources = [
                    MinimalSource(**data)
                    for data in search['retrieved_sources']
                    ]
            prompt = self._prompt_augmenter(
                    search['question'],
                    [source.chunk for source in minimal_sources]
                    )
            answer = secure_answer(self.model, prompt)
            minimal_answer = MinimalAnswer(
                    question_id=search['question_id'],
                    question=search['question'],
                    retrieved_sources=minimal_sources,
                    answer=answer
                    )
            search_results.append(minimal_answer)
        student_search_results_and_answer = StudentSearchResultsAndAnswer(
                search_results=search_results,
                k=k
                )
        file_name = Path(student_search_results_path).name
        save_file: Path = Path(get_path(dataset_path, file_name))
        save_file.parent.mkdir(parents=True, exist_ok=True)
        secure_open(
                save_file, mode="w",
                data=student_search_results_and_answer.model_dump()
                )
        return student_search_results_and_answer.model_dump()

    def _prompt_augmenter(self, query: str, sources: list[str]) -> str:
        context: str = ""
        for index, el in enumerate(sources):
            context += f"- {el}\n" if index < len(sources) else f"- {el}"
        return f"""
Answer the user's query using only the provided context bellow.

<context>
{json.dumps(context)}
</context>
User's query: {json.dumps(query)}
"""
