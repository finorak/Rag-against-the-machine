"""Data validator.

This modul provies all the model we need to validate
our data, and to facilitate data validation.
"""


import uuid

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Class to validate the source retrieved.

    Parameters:
    ----------
    file_path: str
        the path of the source.
    chunk: str
        chunk extracted, and used for source.
    first_character_index: int
        where the chunk start
    last_character_index: int
        where the chunk end
    bm_rank: index of the source in the bm rank results.
    """

    file_path: str = Field(...)
    chunk: str = Field(...)
    first_character_index: int = Field(...)
    last_character_index: int = Field(...)
    bm_rank: float = Field(default=0.0, exclude=True)
    vector_rank: float = Field(default=0.0, exclude=True)
    score: float = Field(default=0.0, exclude=True)


class UnansweredQuestion(BaseModel):
    """Class to store unanswered question.

    Parameters:
    ----------
    question_id: str
        the id of the question
    question: str
        the question that is unanswered
    """

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Class to store answered question.

    Parameters:
    ----------
    sources: list[MinimalSource]
        sources retrieved needed to answer the question.
    answer: str
        the answer provided by our model.
    """

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Class containing (un)answered question.

    This class isn't used in our project.
    """

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Class containing minimal search result for a single query.

    Parameters:
    ----------
    question_id: str
        id of the question
    question: str
        the question to be retrieved it's answer.
    retrieved_sources: list[MinimalSource]
        sources retrieved needed that match the question.
    """

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Class containing minimal search results for a sinlg query.

    Parameters:
    ----------
    answer: str
        the answer to the question, provided by the model.
    """

    answer: str


class StudentSearchResults(BaseModel):
    """Class containing minimal search results for a batch search.

    Parameters:
    ----------
    search_results: list[MinimalSearchResults]
        list of search results results.
    k: int
        top-k used to for each question, how many resources needed.
    """

    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Class containing minimal search results and answer for a batch search.

    Parameters:
    ----------
    search_results: list[MinimalSearchResults]
        list of search results results.
    k: int
        top-k used to for each question, how many resources needed.
    """

    search_results: list[MinimalAnswer]
    k: int
