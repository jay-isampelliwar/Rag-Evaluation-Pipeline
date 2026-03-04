from os import path
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from ..vector_database.chroma_db import ChromaDatabase
from ..processing.chunking_manager import ChunkingManager
from ..embedding_manager.embedding_manager import EmbeddingManager

class DocumentIngestion:

    def __init__(self, path_to_document, vector_store : ChromaDatabase , embedding_manager : EmbeddingManager):
        self.path_to_document = path_to_document
        self.documents = None
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager


    def load_document(self):
        try:
            if self.path_to_document is None or len(self.path_to_document) == 0:
                raise Exception("No documents loaded")


            print(f"Loading document from path {self.path_to_document}" )

            self.documents  = DirectoryLoader(
                path=f"{self.path_to_document}",
                glob="**/*.pdf",
                loader_cls=PyMuPDFLoader,
                show_progress=False,
            ).load()

            print(f"Document loaded from path {self.path_to_document}" )

        except Exception as e:
            print(f"Loading document failed with exception \n{e}")
            raise e

    def save_document(self):

        try:
            if self.documents is None or len(self.documents) == 0:
                raise Exception("No documents loaded")

            print(f"Saving document from path {self.path_to_document}" )

            chunking_manager = ChunkingManager(documents=self.documents)
            chunks = chunking_manager.chunk_documents()

            text_chunks = [document.page_content for document in self.documents]
            embedded_chunks = self.embedding_manager.generate_embeddings(text_chunks)

            self.vector_store.add_documents(
                documents=chunks,
                embeddings=embedded_chunks,
            )

            print("Document saved")

        except Exception as e:
            print(f"Saving document failed with exception \n{e}")
            raise e

