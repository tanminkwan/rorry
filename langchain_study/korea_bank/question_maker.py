import sys
import json
from reader import Reader
from writer import Writer
from agent import Agent
from output_schema import QuizResponse

# 명령줄 인자(argument) 개수 확인
num_args = len(sys.argv)

if num_args != 3:
    print("사용법: python question_maker.py [소스 pdf 파일명] [타겟 excel 파일명]")
    exit(0)

src_file_name = sys.argv[1]
target_file_name = sys.argv[2]

contents = Reader(src_file_name).get_iterator(chunk_size=3000, chunk_overlap=300)

agent = Agent()

with Writer(target_file_name) as writer:
    for i, content in enumerate(contents):
        
        arguments = dict(
            content = content,
            pydantic_object = QuizResponse,
        )

        result = agent.get_quiz(**arguments)

        print(result)
        writer.set_row(
            chunk_num = i,
            question = result.question,
            answer = result.answer, 
            content = content,
        )

