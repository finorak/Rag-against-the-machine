import json
from typing import Any

from langchain_ollama import ChatOllama

from .model import AnsweredQuestion, MinimalSource, UnansweredQuestion
from .search import SearchEngine


class AnswerEngine:
    def __init__(self, search_engine: SearchEngine, model_name: str = "qwen3:0.6b") -> None:
        self.search_engine = search_engine
        self.cache: dict[str, Any] = {}
        self.model = ChatOllama(model=model_name)

    def answer(self, query: str, k: int) -> dict[str, Any]:
        query = query.replace("_", " ").strip()
        cache_key = "{query}: {k}"
        cache_value = self.cache.get(cache_key)
        if cache_value:
            return json.loads(cache_value)
        unanswered_question = UnansweredQuestion(question=query)
        search_results = self.search_engine.search(query, k)
        sources: list[str] = [
                data['chunk']
                for data in search_results['retrieved_sources']
                ]
        prompt = self._prompt_augmenter(query, sources)
        response = self.model.invoke(prompt)
        self.cache[cache_key] = response.content
        answered_question = AnsweredQuestion(
                question_id=unanswered_question.question_id,
                question=query,
                sources=[
                    MinimalSource(**data)
                    for data in search_results['retrieved_sources']
                    ],
                answer=str(response.content)
                )
        return answered_question.model_dump()

    def answer_dataset(
            self, student_search_results_path: str,
            dataset_path: str
    ) -> list[dict[str, Any]]:
        ...

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
