"""Utilities module."""


import json
import sys
from os.path import join
from pathlib import Path
from typing import Any

import numpy as np
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer

from .model import MinimalSource


def get_path(*arg: str) -> str:
    """Get path based on system.

    Args:
        arg: a sequence of string.
    Returns:
        path: the joined path
    """
    return join(*arg)


def secure_open(
        file_path: Path | str, mode: str = "r",
        encoding: str = "utf-8", data: Any = None
) -> Any:
    """Securely open file to avoid errors.

    Args:
        file_path: the path of the file to open
        mode: mode of the file descriptor.
        encoding: str the encoding for the file descriptor.
        data: the data to be writen inside the file.
    Returns:
        extracted: the data we extracted.
    """
    try:
        with open(file_path, mode=mode, encoding=encoding) as f:
            if mode == "w":
                json.dump(data, fp=f, indent=4)
            elif mode == "r":
                return json.load(f)
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)


def get_minimal_sources(chunk_file: Path) -> list[MinimalSource]:
    """Extract minimal source from chunk file.

    Args:
        chunk_file: path to chunk file.
    Returns:
        retrieved_sources: a list of minimal source.
    """
    raw_data = secure_open(chunk_file)
    return [MinimalSource(**data) for data in raw_data]


def secure_answer(model: ChatOllama, prompt: str) -> str:
    """Securely answer question.

    During my first eval, I encountered an issue where
    the chat ollama won't respond sometimes. So before that
    we check if we can generate a response before continuing.

    Args:
        model: the model used to respond our prompt.
        prompt: the prompt to be answered.
    Returns:
        response: the generated text.
    """
    try:
        response = model.invoke(prompt)
        return str(response.content)
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)


def secure_embed(
        model: SentenceTransformer, corpuse: list[str]
) -> list[np.ndarray] | Any:
    """Securely embed our corpuses.

    To avoid error at maximum we put them in separate function

    Args:
        model: the model used to embed our corpuse
        corpuse: corpuse
    Returns:
        embed: a list of vector
    """
    try:
        embed = model.encode(
                corpuse, show_progress_bar=True,
                batch_size=16
                )
        return embed
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)
