import os
from typing import List, Any

from chromadb import Schema, SparseVectorIndexConfig, K, Search, Knn, Rrf
import chromadb
import numpy as np
import uuid
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import Bm25EmbeddingFunction

load_dotenv()


CHROMA_API_KEY=os.getenv("CHROMA_API_KEY")
CHROMA_TENANT=os.getenv("CHROMA_TENANT")
CHROMA_DATABASE=os.getenv("CHROMA_DATABASE")
SPARSE_EMBEDDING_KEY = "sparse_embedding"

class ChromaDatabase:

    def __init__(self, collection_name: str):

        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialize()


    def _initialize(self):

        try:

            if not CHROMA_API_KEY or len(CHROMA_API_KEY) == 0:
                raise Exception("CHROMA_API_KEY not initialized")

            if not CHROMA_TENANT or len(CHROMA_TENANT) == 0:
                raise Exception("CHROMA_DB_TENANT_KEY not initialized")

            print("Initializing Chroma Database Client...")
            self.client = chromadb.CloudClient(
                api_key=CHROMA_API_KEY,
                tenant=CHROMA_TENANT,
                database=CHROMA_DATABASE,

            )
            print("Successfully initialized Chroma Database Client")

            print("Initializing Chroma Database Collection...")

            sparse_ef = Bm25EmbeddingFunction()
            schema = Schema()
            schema.create_index(
                config=SparseVectorIndexConfig(
                    bm25=True,
                    source_key=K.DOCUMENT,
                    embedding_function=sparse_ef
                ),
                key=SPARSE_EMBEDDING_KEY
            )

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
                schema=schema,
            )

        except Exception as e:
            print(f"Failed to initialize Chroma Database Client \n{e}")
            raise e

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):

        try:
            if not self.collection:
                raise Exception("Collection not initialized")

            if len(documents) != len(embeddings):
                raise ValueError("Length of embeddings should be equal to length of documents")

            ids = []
            embedding_list = []
            document_texts = []
            metadatas = []

            for i, (current_doc , embedding) in enumerate(zip(documents, embeddings)):

                doc_id = f"doc_{uuid.uuid4()}"
                ids.append(doc_id)

                metadata = current_doc.metadata
                metadata["index"] = i
                metadata["content_length"] = len(current_doc.page_content)

                metadatas.append(metadata)

                document_texts.append(current_doc.page_content)
                embedding_list.append(embedding.tolist())


            print(f"Adding Documents to Collection {self.collection_name}")

            batch_size = 300
            collection = self.client._collection

            for i in range(0, len(ids), batch_size):
                collection.add(
                    ids=ids[i:i + batch_size],
                    documents=document_texts[i:i + batch_size],
                    embeddings=embedding_list[i:i + batch_size],
                    metadatas=metadatas[i:i + batch_size]
                )

            print(f"Successfully added Documents to Collection {self.collection_name}")

        except Exception as e:
            print(f"Failed to add Documents to Collection {self.collection_name} \n{e}")
            raise e

    def get_documents(self,query_text: str, query_embeddings: np.ndarray, top_k : int = 5, ):
        try:

            if not self.collection:
                raise Exception("Collection not initialized")

            if query_embeddings is None or len(query_embeddings) == 0:
                raise Exception("Length of query_embeddings should be equal to length of documents")


            print("Querying Documents...")

            hybrid_rank = Rrf(
                ranks=[
                    Knn(query=query_embeddings, return_rank=True),
                    Knn(query=query_text, key=SPARSE_EMBEDDING_KEY, return_rank=True)
                ],
                weights=[0.5, 0.5],
                k=60
            )

            results = self.collection.search(
                Search()
                .rank(hybrid_rank)
                .limit(top_k)
                .select(K.DOCUMENT, K.SCORE)
            )

            print(f"Successfully queried {len(results)} Documents")

            return results

        except Exception as e:
            print(f"Failed to queried Documents \n{e}")
            raise e