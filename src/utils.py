"""Utilities module."""


import json
from os.path import join

from bm25s import Path

from .model import MinimalSource


def get_path(*arg: str) -> str:
    """Get path based on system.

    Args:
        arg: a sequence of string.
    Returns:
        path: the joined path
    """
    return join(*arg)


def get_minimal_sources(chunk_file: Path) -> list[MinimalSource]:
    """Extract minimal source from chunk file.

    Args:
        chunk_file: path to chunk file.
    Returns:
        retrieved_sources: a list of minimal source.
    """
    with open(chunk_file, mode="r", encoding="utf-8") as f:
        raw_data = json.load(f)
    return [MinimalSource(**data) for data in raw_data]
