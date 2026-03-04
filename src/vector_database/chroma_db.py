import os
from typing import List, Any

import chromadb
import numpy as np
import uuid
from dotenv import load_dotenv

load_dotenv()


CHROMA_DB_API_KEY=os.getenv("CHROMA_DB_API_KEY")
CHROMA_DB_TENANT_KEY=os.getenv("CHROMA_DB_TENANT_KEY")

class ChromaDatabase:

    def __init__(self, collection_name: str, database_name: str = "Docs-Test'"):

        self.collection_name = collection_name
        self.database_name = database_name
        self.client = None
        self.collection = None
        self._initialize()


    def _initialize(self):

        try:

            if not CHROMA_DB_API_KEY or len(CHROMA_DB_API_KEY) == 0:
                raise Exception("CHROMA_DB_API_KEY not initialized")

            if not CHROMA_DB_TENANT_KEY or len(CHROMA_DB_TENANT_KEY) == 0:
                raise Exception("CHROMA_DB_TENANT_KEY not initialized")

            client = chromadb.CloudClient(
                api_key=CHROMA_DB_API_KEY,
                tenant=CHROMA_DB_TENANT_KEY,
                database=self.database_name,
            )
            self.collection = client.create_collection(name=self.collection_name)
            self.client = client

        except Exception as e:
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

                metadata = current_doc["metadata"]
                metadata["index"] = i
                metadata["content_length"] = len(current_doc.page_content)

                metadatas.append(metadata)

                document_texts.append(current_doc.page_content)
                embedding_list.append(embedding.tolist())


            self.collection.add(
                ids=ids,
                embeddings=embedding_list,
                documents=document_texts,
                metadatas=metadatas,
            )

        except Exception as e:
            raise e


    def get_documents(self, query_embeddings: np.ndarray, top_k : int = 5):
        try:

            if not self.collection:
                raise Exception("Collection not initialized")

            if query_embeddings is None or len(query_embeddings) == 0:
                raise Exception("Length of query_embeddings should be equal to length of documents")


            results = self.collection.querr(
                query_embeddings=query_embeddings,
                top_k=top_k,

            )

            return results

        except Exception as e:
            raise e