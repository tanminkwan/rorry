from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field

class QuizResponse(BaseModel):
    question: str = Field(description="문제")
    answer: str = Field(description="정답, 모범 답안")