"""Index engine module."""


import sys
from pathlib import Path
from typing import Any

from bm25s import BM25, tokenize
from langchain_core.documents import Document
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

from .model import MinimalSource
from .semantic import SemanticEngine
from .utils import secure_open


class IndexEngine:
    """Index engine class.

    THis let us index the corpuse in a way that
    we achieve the requirement asked by the subject.
    """

    def __init__(
            self, data_path: str,
            processed_dir: Path,
            index_dir: Path,
            chunk_file: Path,
            semantic_engine: SemanticEngine
    ) -> None:
        """Initialize an IndexEngine instance.

        Args:
            data_path: path to where our data `vllm` is located.
            processed_dir: where to store our processed dir.
            index_dir: where the index directory is located.
            chunk_file: path to where the chunk file is located.
        """
        self.data_path: Path = Path(data_path)
        self.processed_dir: Path = processed_dir
        self.chunk_file: Path = chunk_file
        self.index_dir: Path = index_dir
        self.retriever: BM25 = BM25()
        self.semantic_engine = semantic_engine
        self.sources: list[dict[str, Any]] = []

    def index(self, max_chunk_size: int, hybrid_search: bool) -> None:
        """Index the corpuse.

        Args:
            max_chunk_size: the chunk size to use for chunking.
        """
        self._explore(max_chunk_size)
        self.chunk_file.parent.mkdir(parents=True, exist_ok=True)
        secure_open(self.chunk_file, mode="w", data=self.sources)
        self.index_dir.parent.mkdir(parents=True, exist_ok=True)
        if hybrid_search:
            self.semantic_engine.embed(
                    [source['chunk'] for source in self.sources]
                    )
        self._retriver(self.sources)

    def _explore(self, max_chunk_size: int) -> None:
        """Explore raw data directory.

        Given a directory, we retrieve all the files
        we need to achieve our goal, `python`, `markdown`
        and `text` using rglob and then.
        """
        python_files = self.data_path.rglob("*.py")
        markdown_files = self.data_path.rglob("*.md")
        text_files = self.data_path.rglob("*.txt")
        for file in [*python_files, *markdown_files, *text_files]:
            language = Language.MARKDOWN
            if file in python_files:
                language = Language.PYTHON
            chunks = self._get_chunks(
                    file, max_chunk_size, language
                    )
            self._extract_chunk(str(file), chunks)

    def _get_chunks(
            self, file_path: Path,
            max_chunk_size: int,
            language: Language
    ) -> list[Document]:
        """Chunk file based on provided argument.

        Given a file, we chunk it based on the provided
        argument so that it isn't repetitive with a lot
        of sequence of if statement.

        Args:
            file_path: path to where the file is located.
            max_chunk_size: the chunk size to use.
            language: the language used to chunk the file.
        Returns:
            chunks: a list of chunk `Document`
        """
        try:
            with open(file_path, mode="r", encoding="utf-8") as f:
                data = f.read()
        except Exception as e:
            print(e, file=sys.stderr)
            sys.exit(1)
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
        """Extract chunk from provided chunks.

        Given a list of chunk, we extract the metadata
        and the content of each chunk so that we can
        create a MinimalSource source from it.

        Args:
            file_path: path to where the file is located.
            chunks: a list of chunk `Document`
        """
        for chunk in chunks:
            source = MinimalSource(
                    file_path=file_path,
                    chunk=chunk.page_content,
                    first_character_index=chunk.metadata['start_index'],
                    last_character_index=chunk.metadata['end_index']
                    )
            self.sources.append(source.model_dump())

    def _retriver(self, sources: list[dict[str, Any]]) -> None:
        """Retrive the corpus.

        using this function, we save the index in the
        index_dir after indexing the corpus.

        Args:
            sources: a list of dictionary containing our desired \
chunk.
        """
        corpuse: list[str] = [
                source['chunk'].replace("_", " ").strip()
                for source in sources
                ]
        self.retriever.index(tokenize(corpuse), show_progress=True)
        try:
            self.retriever.save(self.index_dir, corpus=corpuse)
        except Exception as e:
            print(e, file=sys.stderr)
            sys.exit(1)
