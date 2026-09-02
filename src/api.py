"""Api exposer module."""


from typing import Any

from fastapi import FastAPI

from .app import App

app = FastAPI()
rag_application = App()


@app.post("/answer")
def query(query: str, k: int, hybrid: bool = False) -> Any:
    """Answer api to expose the rag answer API.

    Args:
        query: the user's request.
        k: top-k used for retrieval.
        hybrid: weather to perform an hybrid search or not.
    """
    try:
        response = rag_application.answer(query, k, hybrid)
        return response
    except Exception as e:
        print(e)
        return e.__str__()
