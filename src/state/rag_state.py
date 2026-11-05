
from typing import List
from pydantic import BaseModel, Field
from langchain_core.documents import Document


class RAGState(BaseModel):

    # Original user question
    question: str

    # All passages retrieved from FAISS
    retrieved_docs: List[Document] = Field(default_factory=list)

    # Passages that pass relevance grading
    relevant_docs: List[Document] = Field(default_factory=list)

    # Relevance decision
    is_relevant: bool = False

    # Final generated answer
    answer: str = ""
