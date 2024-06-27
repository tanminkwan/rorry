from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field

class EvaluationByStudent(BaseModel):
    student_id: str = Field(description="학생 이름 또는 학생 ID")
    evaluation: str = Field(description="학생의 답변에 대한 평가 내용")
    score: int = Field(description="학생의 답변에 대한 평가 점수")

class EvaluationResponse(BaseModel):
    evaluation_by_student: List[EvaluationByStudent] = Field(description="학생별 세부 평가 내용")
    overall: str = Field(description="종합평가")