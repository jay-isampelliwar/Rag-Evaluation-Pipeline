import os
from pathlib import Path

from src.chroma_db import ChromaDatabase
from src.document_ingestion import DocumentIngestion
from src.embedding_manager import EmbeddingManager


DEFAULT_EVAL_DOCS = "data/docs"
CHROMA_COLLECTION=os.getenv("CHROMA_COLLECTION")

def run() -> None:
    """Run the end-to-end data ingestion pipeline."""

    print("Running data ingestion pipeline...")

    embedding_manager = EmbeddingManager()
    vector_store = ChromaDatabase(collection_name=CHROMA_COLLECTION)

    data_ingestion = DocumentIngestion(
        vector_store=vector_store,
        embedding_manager=embedding_manager,
        path_to_document=str(DEFAULT_EVAL_DOCS),
    )

    data_ingestion.load_document()
    data_ingestion.save_document()
    print("Completed data ingestion pipeline...")


if __name__ == "__main__":
    run()