from langchain_text_splitters import RecursiveCharacterTextSplitter

class ChunkingManager:

    def __init__(self, documents):
        self.documents = documents


    def chunk_documents(self , chunk_size=300, chunk_overlap=50):

        try:

            if not self.documents or self.documents.empty:
                raise Exception("Document is empty")

            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = splitter.split_documents(documents=self.documents)

            return chunks

        except Exception as e:
            raise e


