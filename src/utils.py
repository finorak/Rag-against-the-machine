import json
from os.path import join

from bm25s import Path

from .model import MinimalSource


def get_path(*arg: str) -> str:
    return join(*arg)


def get_minimal_sources(chunk_file: Path) -> list[MinimalSource]:
    with open(chunk_file, mode="r", encoding="utf-8") as f:
        raw_data = json.load(f)
    return [MinimalSource(**data) for data in raw_data]
