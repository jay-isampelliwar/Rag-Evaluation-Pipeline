class ResultModel:
    question: str
    response: str
    expected_answer: str
    status: bool

    def __init__(self, question: str, response: str, expected_answer: str, status: bool):
        self.question = question
        self.response = response
        self.expected_answer = expected_answer
        self.status = status