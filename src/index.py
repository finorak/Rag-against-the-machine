import json
from pathlib import Path
from typing import Any

from bm25s import BM25, tokenize
from langchain_core.documents import Document
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

from .model import MinimalSource
from .utils import get_path


class IndexEngine:
    def __init__(self, data_path: str, chunk_file: Path) -> None:
        self.data_path = Path(data_path)
        self.chunk_file: Path = chunk_file
        self.processed_dir = Path(get_path("data", "processed"))
        self.index_dir: Path = Path(get_path("data", "processed", "bm_index"))
        self.retriever: BM25 = BM25()
        self.sources: list[dict[str, Any]] = []

    def index(self, max_chunk_size: int) -> None:
        self._explore(max_chunk_size)
        self.chunk_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.chunk_file, mode="w", encoding="utf-8") as f:
            json.dump(self.sources, fp=f, indent=4)
        self.index_dir.parent.mkdir(parents=True, exist_ok=True)
        self._retriver(self.sources)

    def _explore(self, max_chunk_size: int) -> None:
        python_files = self.data_path.rglob("*.py")
        markdown_files = self.data_path.rglob("*.md")
        text_files = self.data_path.rglob("*.txt")
        for file in [*python_files, *markdown_files, *text_files]:
            language = Language.MARKDOWN
            if file in python_files:
                language = Language.PYTHON
            chunks = self._get_chunks(
                    file._raw_paths[0], max_chunk_size, language
                    )
            self._extract_chunk(file._raw_paths[0], chunks)

    def _get_chunks(
            self, file_path: str,
            max_chunk_size: int,
            language: Language
    ) -> list[Document]:
        with open(file_path, mode="r", encoding="utf-8") as f:
            data = f.read()
        text_splitter = RecursiveCharacterTextSplitter.from_language(
                language=language,
                chunk_size=max_chunk_size,
                chunk_overlap=int(0.2 * max_chunk_size),
                add_start_index=True
                )
        chunks = text_splitter.create_documents([data])
        for chunk in chunks:
            end_index = chunk.metadata['start_index'] + len(chunk.page_content)
            chunk.metadata['end_index'] = end_index
        return chunks

    def _extract_chunk(self, file_path: str, chunks: list[Document]) -> None:
        for chunk in chunks:
            source = MinimalSource(
                    file_path=file_path,
                    chunk=chunk.page_content,
                    first_character_index=chunk.metadata['start_index'],
                    last_character_index=chunk.metadata['end_index']
                    )
            self.sources.append(source.model_dump())

    def _retriver(self, sources: list[dict[str, Any]]) -> None:
        corpuse: list[str] = [
                source['chunk'].replace("_", " ").strip()
                for source in sources
                ]
        self.retriever.index(tokenize(corpuse), show_progress=True)
        self.retriever.save(self.index_dir, corpus=corpuse)
