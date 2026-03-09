from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader

from src.chroma_db import ChromaDatabase
from src.chunking_manager import ChunkingManager
from src.embedding_manager import EmbeddingManager


class DocumentIngestion:

    def __init__(
        self,
        path_to_document: str,
        vector_store: ChromaDatabase,
        embedding_manager: EmbeddingManager,
    ) -> None:
        self.path_to_document: str = path_to_document
        self.documents = None
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def load_document(self) -> None:
        """Load documents from the configured directory."""
        if not self.path_to_document:
            raise ValueError("path_to_document must be provided")

        docs_dir = Path(self.path_to_document)

        print(f"Loading document from path {docs_dir}")

        try:
            self.documents = DirectoryLoader(
                path=str(docs_dir),
                glob="**/*.pdf",
                loader_cls=PyMuPDFLoader,
                show_progress=False,
            ).load()

            print(f"Document loaded from path {docs_dir}")

        except FileNotFoundError:
            print(f"Directory not found: {docs_dir}")
            raise
        except Exception as e:
            print(f"Loading document failed with exception \n{e}")
            raise

    def save_document(self) -> None:
        """Chunk, embed, and persist loaded documents into the vector store."""
        if not self.documents:
            raise ValueError("No documents loaded")

        print(f"Saving document from path {self.path_to_document}")

        try:
            chunking_manager = ChunkingManager(documents=self.documents)
            chunks = chunking_manager.chunk_documents()

            text_chunks = [chunk.page_content for chunk in chunks]
            embedded_chunks = self.embedding_manager.generate_embeddings(text_chunks)

            self.vector_store.add_documents(
                documents=chunks,
                embeddings=embedded_chunks,
            )

            print("Document saved")

        except Exception as e:
            print(f"Saving document failed with exception \n{e}")
            raise
