from pathlib import Path

from src.vector_database.chroma_db import ChromaDatabase
from src.ingestion.document_ingestion import DocumentIngestion
from src.embedding_manager.embedding_manager import EmbeddingManager


def run() -> None:
    """Run the end-to-end data ingestion pipeline."""

    print("Running data ingestion pipeline...")

    base_dir = Path(__file__).resolve().parents[1]
    docs_dir = base_dir / "data" / "docs"

    vector_store = ChromaDatabase(collection_name="research-papers")
    embedding_manager = EmbeddingManager()

    data_ingestion = DocumentIngestion(
        vector_store=vector_store,
        embedding_manager=embedding_manager,
        path_to_document=str(docs_dir),
    )

    data_ingestion.load_document()
    data_ingestion.save_document()
    print("Completed data ingestion pipeline...")


if __name__ == "__main__":
    run()