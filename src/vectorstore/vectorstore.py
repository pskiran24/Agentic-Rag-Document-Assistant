
"""Vector store module for document embedding and retrieval."""

from typing import List

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document


class VectorStore:
    """Manages vector store operations."""

    def __init__(self):
        """Initialize vector store with local Hugging Face embeddings."""

        self.embedding = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.vectorstore = None
        self.retriever = None

    def create_vectorstore(self, documents: List[Document]):
        """Create FAISS vector store from documents."""

        self.vectorstore = FAISS.from_documents(
            documents,
            self.embedding
        )

        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

    def get_retriever(self):
        """Return the retriever instance."""

        if self.retriever is None:
            raise ValueError(
                "Vector store not initialized. "
                "Call create_vectorstore first."
            )

        return self.retriever

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve top-k relevant documents."""

        if self.vectorstore is None:
            raise ValueError(
                "Vector store not initialized. "
                "Call create_vectorstore first."
            )

        return self.vectorstore.similarity_search(
            query,
            k=k
        )
