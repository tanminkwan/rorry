from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import PromptTemplate
from langchain import OpenAI, LLMChain
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from typing import List, Union
import re

# OpenAI API 키 설정 (실제 사용 시 환경 변수로 관리하는 것이 좋습니다)
#import os
#os.environ["OPENAI_API_KEY"] = "your-api-key-here"
from dotenv import load_dotenv
load_dotenv(dotenv_path='./../.env')

# 에이전트 1: 질문 생성기
class QuestionGeneratorAgent(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        return AgentFinish(
            return_values={"output": llm_output.strip()},
            log=llm_output,
        )

# 에이전트 2: 답변 생성기
class AnswerGeneratorAgent(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        return AgentFinish(
            return_values={"output": llm_output.strip()},
            log=llm_output,
        )

# 질문 생성기 프롬프트 템플릿
question_template = """당신은 질문 생성기입니다. 주어진 주제에 대해 흥미로운 질문을 생성해주세요.

주제: {topic}

질문:"""

question_prompt = PromptTemplate(
    template=question_template,
    input_variables=["topic"]
)

# 답변 생성기 프롬프트 템플릿
answer_template = """당신은 답변 생성기입니다. 주어진 질문에 대해 정보를 제공하는 답변을 생성해주세요.

질문: {question}

답변:"""

answer_prompt = PromptTemplate(
    template=answer_template,
    input_variables=["question"]
)

# LLM 및 에이전트 설정
llm = OpenAI(temperature=0.9)
question_agent = LLMSingleActionAgent(
    llm_chain=LLMChain(llm=llm, prompt=question_prompt),
    output_parser=QuestionGeneratorAgent(),
    stop=["\nHuman:"],
)
answer_agent = LLMSingleActionAgent(
    llm_chain=LLMChain(llm=llm, prompt=answer_prompt),
    output_parser=AnswerGeneratorAgent(),
    stop=["\nHuman:"],
)

# 에이전트 실행기 설정
question_agent_executor = AgentExecutor.from_agent_and_tools(agent=question_agent, tools=[], verbose=True)
answer_agent_executor = AgentExecutor.from_agent_and_tools(agent=answer_agent, tools=[], verbose=True)

# 멀티 에이전트 대화 실행
def run_multi_agent_conversation(topic, num_turns=3):
    for i in range(num_turns):
        print(f"\n--- 대화 턴 {i+1} ---")
        
        # 질문 생성
        question_result = question_agent_executor.run(topic=topic)
        print(f"질문: {question_result}")
        
        # 답변 생성
        answer_result = answer_agent_executor.run(question=question_result)
        print(f"답변: {answer_result}")
        
        # 다음 주제 설정 (여기서는 간단히 답변의 일부를 사용)
        topic = answer_result.split()[:3]
        topic = " ".join(topic)

# 대화 실행
run_multi_agent_conversation("인공지능의 미래")