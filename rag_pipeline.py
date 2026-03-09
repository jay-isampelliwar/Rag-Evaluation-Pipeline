from src.retriever import Retriever
from src.generation import Generation

class RagPipeline:

    def __init__(self , retriever: Retriever):
        self.retriever = retriever
        self.generation = Generation()

    def execute(self, user_query: str) -> str:
        try:
            result_response = self.retriever.retrieve_docs(user_query)
            llm_answer = self.generation.invoke(user_query=user_query, result_response=result_response)
            return llm_answer
        except Exception as e:
            print(e)
            raise