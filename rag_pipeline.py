from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

from model.result_model import ResultModel
from src.evaluator import Evaluator
from src.retriever import Retriever
from src.generation import Generation

class RagPipeline:

    def __init__(self , retriever: Retriever):
        self.retriever = retriever
        self.generation = Generation()
        self.evaluator = Evaluator()

    def execute(self, evaluation_dataset: List[Dict[str, str]], rerank: bool = False):
        try:

            questions = [item["question"] for item in evaluation_dataset]
            expected_answers = [item["answer"] for item in evaluation_dataset]
            # sources = [item["source"] for item in evaluation_dataset]
            # page_numbers = [item["page"] for item in evaluation_dataset]

            with ThreadPoolExecutor(max_workers=10) as executor:
                results: List[ResultModel] = list(
                    executor.map(
                        self.evaluate,
                        questions,
                        expected_answers,
                        # sources,
                        # page_numbers,
                        [rerank] * len(questions),
                    )
                )

            for i, result in enumerate(results):
                result_emoji = "✅" if result.status else "❌"
                print(f"{result_emoji} Q {i + 1}: {result.question}: \n")
                print(f"Response: {result.response}\n")
                print(f"Expected Answer: {result.expected_answer}\n")
                print("--------------------------------")

            number_correct = sum(result.status for result in results)
            print(f"✨ Total Score: {number_correct}/{len(results)}")

        except Exception as e:
            print(e)
            raise


    def evaluate(
        self,
        question: str,
        expected_answer: str,
        # source: str,
        # page_number: str,
        rerank: bool = False,
    ):

        result_response = self.retriever.retrieve_docs(
            query_text=question,
            enable_rerank=rerank,
        )
        llm_answer = self.generation.invoke(
            user_query=question,
            result_response=result_response,
        )
        return self.evaluator.evaluate(
            user_query=question,
            expected_answer=expected_answer,
            response=llm_answer,
            # source=source,
            # page_number=str(page_number),
        )
