import sys
import json
from reader import ExcelReader
from writer import ExcelWriter
from agent import Agent
from output_schema import EvaluationResponse

# 명령줄 인자(argument) 개수 확인
num_args = len(sys.argv)

if num_args != 3:
    print("사용법: python evaluater.py [소스 excel 파일명] [타겟 excel 파일명]")
    exit(0)

src_file_name = sys.argv[1]
target_file_name = sys.argv[2]

agent = Agent()

with ExcelWriter(target_file_name) as writer:
    with ExcelReader(src_file_name) as reader:
        for row_data in reader.get_iterator("구분", "질문", "답","answer","sds_answer","bge_answer"):

            responses = dict(
                tom = row_data["answer"],
                sds = row_data["sds_answer"],
                bge = row_data["bge_answer"],
            )

            arguments = dict(
                question = row_data["질문"],
                answer = row_data["답"],
                responses = responses,
                pydantic_object = EvaluationResponse,
            )

            evaluation = agent.get_evaluation(**arguments)

            eval = evaluation.evaluation_by_student

            writer.set_row(
                category = row_data["구분"],
                question = row_data["질문"],
                answer = row_data["답"], 
                tom = row_data["answer"],
                sds = row_data["sds_answer"],
                bge = row_data["bge_answer"],
                model_1 = eval[0].student_id,
                evaluation_1 = eval[0].evaluation,
                score_1 = eval[0].score,
                model_2 = eval[1].student_id,
                evaluation_2 = eval[1].evaluation,
                score_2 = eval[1].score,
                model_3 = eval[2].student_id,
                evaluation_3 = eval[2].evaluation,
                score_3 = eval[2].score,
            )

