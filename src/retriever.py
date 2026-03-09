from src.chroma_db import ChromaDatabase
from src.embedding_manager import EmbeddingManager
from model.retriever_response_model import RetrieverResponseModel, SourceData, DocumentData
from cohere import ClientV2


COHERE_MODEL = "rerank-v3.5"

class Retriever:

    def __init__(self , vector_store : ChromaDatabase, embedding_manager: EmbeddingManager ):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve_docs(self, query_text: str, top_k: int = 5, enable_rerank: bool = False) -> RetrieverResponseModel:

        try:
            print("Embedding user input..")
            query_embeddings = self.embedding_manager.generate_embeddings([query_text])

            results = self.vector_store.get_documents(
                query_embeddings=query_embeddings,
                top_k= top_k if not enable_rerank else top_k * 5,
                query_text=query_text
            )

            sources = self._format_metadata(result_docs=results)
            scores = self._get_scores_from_results(results=results)

            if not enable_rerank:

                retrieved_docs = self._format_chroma_results(results=results)

                return RetrieverResponseModel(
                    document_data=retrieved_docs,
                    sources_data=sources,
                    scores=scores
                )
            else:
                reranked_docs = self._rerank_docs(
                    user_query=query_text,
                    retrieved_docs= results,
                    top_k=top_k
                )

                return RetrieverResponseModel(
                    document_data=reranked_docs,
                    sources_data=sources,
                    scores=scores
                )

        except Exception as e:
            print("Failed to retrieve documents!")
            raise e


    def _rerank_docs(self, user_query : str, retrieved_docs : dict, top_k: int = 5) -> list[DocumentData]:

        try:

            documents = retrieved_docs.get("documents", [[]])[0] or []
            ids = retrieved_docs.get("ids", [[]])[0] or []

            co = ClientV2()

            response = co.rerank(
                model= COHERE_MODEL,
                query=user_query,
                documents=documents,
                top_n=top_k,
            )

            result_indices = [result.index for result in response.results]

            return [ DocumentData(
                content=documents[index].strip(),
                rank=i + 1,
                id=ids,
            ) for i, (index, ids) in enumerate(zip(result_indices, ids))]

        except Exception as e:
            raise


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