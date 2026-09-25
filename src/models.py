"""Data models for the RAG pipeline stages."""

from pydantic import BaseModel, Field
from typing import List
import uuid


class MinimalSource(BaseModel):
    """Represent a specific text snippet retrieved from the codebase.

    Attributes:
        file_path: Relative path to the original source file.
        first_character_index: Starting character position in the file.
        last_character_index: Ending character position in the file.
    """

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Represent a question that has not been answered yet.

    Attributes:
        question_id: Unique identifier for the question.
        question: The text of the question.
    """

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represent a question along with its sources and generated answer.

    Attributes:
        sources: List of minimal sources supporting the answer.
        answer: The text of the generated answer.
    """

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Represent a collection of RAG questions, answered or unanswered.

    Attributes:
        rag_questions: List containing answered or unanswered questions.
    """

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Represent the retrieved sources for a single question.

    Attributes:
        question_id: Identifier of the queried question.
        question: Text of the question.
        retrieved_sources: List of minimal sources retrieved for this question.
    """

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Represent a question with retrieved sources and the generated answer.

    Attributes:
        answer: Text of the answer produced by the language model.
    """

    answer: str


class StudentSearchResults(BaseModel):
    """Represent the complete search output for a dataset.

    Attributes:
        search_results: List of retrieved search results per question.
        k: Number of retrieved candidates requested.
    """

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Represent the complete output containing answers for a dataset.

    Attributes:
        search_results: List of search results with answers per question.
        k: Number of retrieved candidates requested.
    """

    search_results: List[MinimalAnswer]
    k: int


class Chunk(BaseModel):
    """Represent an indexed text chunk with its source location.

    Attributes:
        content: The text content of the chunk.
        source: MinimalSource containing file path and character span.
    """
    content: str
    source: MinimalSource
