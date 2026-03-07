
from src.vector_database.chroma_db import ChromaDatabase
from src.embedding_manager.embedding_manager import EmbeddingManager
from model.retriever_response_model import RetrieverResponseModel, SourceData , DocumentData
from typing import List

class Retriever:

    def __init__(self , vector_store : ChromaDatabase, embedding_manager: EmbeddingManager ):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve_docs(self, query_text: str, top_k: int = 5 , ) -> RetrieverResponseModel:

        try:
            print("Embedding user input..")
            query_embeddings = self.embedding_manager.generate_embeddings([query_text])

            results = self.vector_store.get_documents(
                query_embeddings=query_embeddings,
                top_k=top_k,
                query_text=query_text
            )

            retrieved_docs = self._format_chroma_results(results=results)
            sources = self._format_metadata(result_docs=results)
            scores = self._get_scores_from_results(results=results)

            return RetrieverResponseModel(
                document_data=retrieved_docs,
                sources_data= sources,
                scores= scores
            )

        except Exception as e:
            print("Failed to retrieve documents!")
            raise e

    def _format_chroma_results(self, results: dict) -> list[DocumentData]:
        """Format ChromaDB search results into a clean list of documents."""
        formatted = []

        documents = results.get("documents", [[]])[0] or []
        ids = results.get("ids", [[]])[0] or []


        for i, (doc, doc_id) in enumerate(zip(documents, ids)):

            formatted.append(
                DocumentData(
                    content=doc.strip(),
                    rank=i + 1,
                    id=doc_id,
                )
            )

        return formatted

    def _format_metadata(self, result_docs: dict) -> list[SourceData]:

        sources = []

        metadatas = result_docs.get("metadatas", [None])[0]


        for index, metadata in enumerate(metadatas):
            source = metadata.get("source", "Unknown")
            page = metadata.get("page", "Unknown")
            title = metadata.get("title", "Unknown")
            authors = ", ".join(metadata.get("author", "Unknown").split(";"))

            sources.append(
                SourceData(
                    source=source,
                    page=page,
                    title=title,
                    authors=authors
                )
            )

        return sources

    def _get_scores_from_results(self, results: dict) -> list[float]:
        scores = results.get("scores", [[]])[0] or []
        return scores