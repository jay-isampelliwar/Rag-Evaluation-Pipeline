from pydantic import BaseModel

class EvaluationResponseModel(BaseModel):
    result: bool