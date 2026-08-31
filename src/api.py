from typing import Any

from fastapi import FastAPI

from .app import App

app = FastAPI()
rag_application = App()

@app.post("/answer")
def query(query: str, k: int) -> Any:
    try:
        response = rag_application.answer(query, k)
        return response
    except Exception as e:
        return e.__str__()
