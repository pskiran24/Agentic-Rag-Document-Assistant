
"""Document processing module for loading and splitting documents."""

from pathlib import Path
from typing import List, Union

from langchain.schema import Document
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    PyPDFDirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """Handles document loading and processing."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def load_from_url(self, url: str) -> List[Document]:
        """Load documents from a webpage."""
        loader = WebBaseLoader(url)
        return loader.load()

    def load_from_pdf_dir(
        self, directory: Union[str, Path]
    ) -> List[Document]:
        """Load all PDFs from a directory."""
        loader = PyPDFDirectoryLoader(str(directory))
        return loader.load()

    def load_from_pdf(
        self, file_path: Union[str, Path]
    ) -> List[Document]:
        """Load a single PDF file."""
        loader = PyPDFLoader(str(file_path))
        return loader.load()

    def load_from_txt(
        self, file_path: Union[str, Path]
    ) -> List[Document]:
        """Load a TXT file."""
        loader = TextLoader(str(file_path), encoding="utf-8")
        return loader.load()

    def load_documents(self, sources: List[str]) -> List[Document]:
        """Load documents from URLs, PDF files, PDF directories or TXT files."""

        docs: List[Document] = []

        for src in sources:

            # Case 1: Website URL
            if src.startswith(("http://", "https://")):
                docs.extend(self.load_from_url(src))
                continue

            # Case 2: Local file or directory
            path = Path(src)

            if not path.exists():
                raise FileNotFoundError(f"Source not found: {src}")

            if path.is_dir():
                docs.extend(self.load_from_pdf_dir(path))

            elif path.suffix.lower() == ".pdf":
                docs.extend(self.load_from_pdf(path))

            elif path.suffix.lower() == ".txt":
                docs.extend(self.load_from_txt(path))

            else:
                raise ValueError(f"Unsupported source type: {src}")

        return docs

    def split_documents(
        self, documents: List[Document]
    ) -> List[Document]:
        """Split loaded documents into smaller chunks."""
        return self.splitter.split_documents(documents)

    def process_sources(self, sources: List[str]) -> List[Document]:
        """Load documents and split them into chunks."""
        docs = self.load_documents(sources)
        return self.split_documents(docs)
