from typing import Any


class AnswerEngine:
    def __init__(self) -> None:
        pass

    def answer(self, query: str, k: int) -> dict[str, Any]:
        ...

    def answer_dataset(
            self, student_search_results_path: str,
            dataset_path: str
    ) -> list[dict[str, Any]]:
        ...
