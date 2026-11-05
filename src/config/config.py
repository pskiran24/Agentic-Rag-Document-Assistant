
"""Configuration module for Agentic RAG system."""

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for RAG system."""

    # Model Configuration
    LLM_MODEL = "llama3.2:latest"

    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # Default URLs
    # Document sources
    DEFAULT_SOURCES = [
    "data/attention.pdf"]

    @classmethod
    def get_llm(cls):
        """Initialize and return the local Ollama LLM."""

        return ChatOllama(
            model=cls.LLM_MODEL,
            temperature=0
        )
