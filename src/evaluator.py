from model.evaluation_response_model import EvaluationResponseModel
from model.result_model import ResultModel
from src.generation import Generation

class Evaluator:

    def __init__(self):
        self.generation = Generation()

    def evaluate(self, user_query: str, response: str, expected_answer: str):

        response_result : EvaluationResponseModel   = self.generation.invoke_for_evaluation(
            response=response,
            expected_answer=expected_answer,
        )

        if response_result is not None:
            status = response_result.result
        else:
            status = False

        return ResultModel(
            question=user_query,
            response=response,
            expected_answer=expected_answer,
            status=status
        )