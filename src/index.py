from bm25s import BM25


class IndexEngine:
    def __init__(self, data_path: str) -> None:
        self.data_path = data_path
        self.retriever: BM25 | None = None

    def index(self, max_chunk_size: int) -> None:
        print(max_chunk_size)
