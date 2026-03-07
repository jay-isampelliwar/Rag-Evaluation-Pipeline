
from src.vector_database.chroma_db import ChromaDatabase
from src.embedding_manager.embedding_manager import EmbeddingManager
from model.retriever_response_model import RetrieverResponseModel, SourceData , DocumentData
from typing import List

class Retriever:

    def __init__(self , vector_store : ChromaDatabase, embedding_manager: EmbeddingManager ):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve_docs(self, query_text: str, top_k: int = 5 , ) -> List[RetrieverResponseModel]:

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

            final_results = [RetrieverResponseModel(
                document_data=document_data,
                sources_data= sources,
                scores=scores
            ) for i, (document_data, source_data, scores) in enumerate(zip(retrieved_docs, sources, scores))]

            return final_results

        except Exception as e:
            print("Failed to retrieve documents!")
            raise e

    def _format_chroma_results(self, results: dict) -> list[DocumentData]:
        """Format ChromaDB search results into a clean list of documents."""
        formatted = []

        documents = results.get("documents", [[]])[0] or []
        scores = results.get("scores", [[]])[0] or []
        ids = results.get("ids", [[]])[0] or []
        metadatas_raw = results.get("metadatas", [None])[0]  # ← fixed key
        metadatas = metadatas_raw if isinstance(metadatas_raw, list) else [None] * len(documents)

        for i, (doc, score, doc_id, metadata) in enumerate(zip(documents, scores, ids, metadatas)):

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

        for index, doc in enumerate(result_docs):
            source = doc["metadata"].get("source", "Unknown").split("/")[-1]
            page = doc["metadata"].get("page", "Unknown")
            title = doc["metadata"].get("title", "Unknown")
            authors = ", ".join(doc["metadata"].get("author", "Unknown").split(";"))

            sources.append(
                SourceData(
                    source=source,
                    page=page,
                    title=title,
                    authors=authors
                )
            )

        return sources

    def _get_scores_from_results(self, results: dict) -> list[dict]:
        scores = results.get("scores", [[]])[0] or []
        return scores