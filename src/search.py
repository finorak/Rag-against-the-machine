from typing import Any


class SearchEngine:
    def __init__(self) -> None:
        pass

    def search(self, query: str, k: int) -> dict[str, Any]:
        print(query, k)
        return {}

    def search_dataset(
            self, dataset_path: str,
            save_directory: str, k: int
    ) -> list[dict[str, Any]]:
        ...
