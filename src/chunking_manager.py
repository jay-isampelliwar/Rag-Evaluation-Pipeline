from langchain_text_splitters import RecursiveCharacterTextSplitter


class ChunkingManager:
    def __init__(self, documents) -> None:
        self.documents = documents

    def chunk_documents(self, chunk_size: int = 800, chunk_overlap: int = 100):
        try:
            if not self.documents:
                raise ValueError("Document is empty")

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            chunks = splitter.split_documents(documents=self.documents)

            return chunks

        except Exception:
            raise

