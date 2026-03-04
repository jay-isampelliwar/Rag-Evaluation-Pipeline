import os
from typing import List, Any

import chromadb
import numpy as np
import uuid
from dotenv import load_dotenv

load_dotenv()


CHROMA_API_KEY=os.getenv("CHROMA_API_KEY")
CHROMA_TENANT=os.getenv("CHROMA_TENANT")

class ChromaDatabase:

    def __init__(self, collection_name: str, database_name: str = "Docs-Test"):

        self.collection_name = collection_name
        self.database_name = database_name
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
                database=self.database_name,
            )
            print("Successfully initialized Chroma Database Client")


            print("Creating Collection...")
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            print("Successfully created Collection")

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

                metadata = current_doc["metadata"]
                metadata["index"] = i
                metadata["content_length"] = len(current_doc.page_content)

                metadatas.append(metadata)

                document_texts.append(current_doc.page_content)
                embedding_list.append(embedding.tolist())


            print(f"Adding Documents to Collection {self.collection_name}")

            self.collection.add(
                ids=ids,
                embeddings=embedding_list,
                documents=document_texts,
                metadatas=metadatas,
            )

            print(f"Successfully added Documents to Collection {self.collection_name}")

        except Exception as e:
            print(f"Failed to add Documents to Collection {self.collection_name} \n{e}")
            raise e


    def get_documents(self, query_embeddings: np.ndarray, top_k : int = 5):
        try:

            if not self.collection:
                raise Exception("Collection not initialized")

            if query_embeddings is None or len(query_embeddings) == 0:
                raise Exception("Length of query_embeddings should be equal to length of documents")


            print("Querying Documents...")

            results = self.collection.querr(
                query_embeddings=query_embeddings,
                top_k=top_k,
            )

            print(f"Successfully queried {len(results)} Documents")

            return results

        except Exception as e:
            print(f"Failed to queried Documents \n{e}")
            raise e