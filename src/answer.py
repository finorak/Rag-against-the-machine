"""Answer engine module."""


import json
import sys
from typing import Any

from langchain_ollama import ChatOllama

from .model import AnsweredQuestion, MinimalSource, UnansweredQuestion
from .search import SearchEngine


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
        self.model = ChatOllama(model=model_name)

    def answer(self, query: str, k: int) -> dict[str, Any] | Any:
        """Answer user's query.

        We answer the user's query using the `qwen3:0.6` model
        as the default.
        Args:
            query: (str) the user's request.
            k: (int) top-k to use to answer the query.
        Returns:
            response: (dict[str, Any]) retrieved sources and answer.
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
        search_results = self.search_engine.search(query, k)
        sources: list[str] = [
                data['chunk']
                for data in search_results['retrieved_sources']
                ]
        prompt = self._prompt_augmenter(query, sources)
        response = self.model.invoke(prompt)
        answered_question = AnsweredQuestion(
                question_id=unanswered_question.question_id,
                question=query,
                sources=[
                    MinimalSource(**data)
                    for data in search_results['retrieved_sources']
                    ],
                answer=str(response.content)
                )
        self.cache[cache_key] = answered_question.model_dump()
        return answered_question.model_dump()

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
        with open(
                student_search_results_path,
                mode="r", encoding="utf-8"
        ) as f:
            raw_data = json.load(f)
        return {}

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
