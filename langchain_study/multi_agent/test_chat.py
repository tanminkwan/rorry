from langchain import OpenAI, LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# OpenAI API 키 설정 (실제 사용 시 환경 변수로 관리하는 것이 좋습니다)
import os
#os.environ["OPENAI_API_KEY"] = "your-api-key-here"
from dotenv import load_dotenv
load_dotenv(dotenv_path='./../.env')
os.environ["LANGCHAIN_PROJECT"] = "langchain_study" # Langsmith project 명

# LLM 설정
llm = OpenAI(temperature=0.7)

# Agent 1: 질문자
questioner_template = """당신은 호기심 많은 질문자입니다. 주어진 주제에 대해 대화를 이어나가며 흥미로운 질문을 해주세요.

현재 주제: {topic}
이전 대화:
{history}
인간: {input}
AI: 

질문자의 다음 질문:"""

questioner_prompt = PromptTemplate(
    input_variables=["topic", "history", "input"],
    template=questioner_template
)

questioner_memory = ConversationBufferMemory(input_key="input", memory_key="history")

questioner_chain = LLMChain(
    llm=llm, 
    prompt=questioner_prompt,
    memory=questioner_memory
)

# Agent 2: 답변자
answerer_template = """당신은 지식이 풍부한 답변자입니다. 질문자의 질문에 대해 정보를 제공하고 대화를 이어나가는 답변을 해주세요.

현재 주제: {topic}
이전 대화:
{history}
인간: {input}
AI: 

답변자의 답변:"""

answerer_prompt = PromptTemplate(
    input_variables=["topic", "history", "input"],
    template=answerer_template
)

answerer_memory = ConversationBufferMemory(input_key="input", memory_key="history")

answerer_chain = LLMChain(
    llm=llm, 
    prompt=answerer_prompt,
    memory=answerer_memory
)

def run_conversation(topic, num_turns=5):
    print(f"대화 주제: {topic}\n")
    current_input = "대화를 시작해주세요."
    
    for i in range(num_turns):
        print(f"--- 대화 턴 {i+1} ---")
        
        # 질문자의 턴
        question_result = questioner_chain.run(topic=topic, input=current_input)
        print(f"질문자: {question_result}")
        
        # 답변자의 턴
        answer_result = answerer_chain.run(topic=topic, input=question_result)
        print(f"답변자: {answer_result}")
        
        # 다음 턴을 위해 현재 입력 업데이트
        current_input = answer_result
        
        print()  # 가독성을 위한 빈 줄

# 대화 실행
run_conversation("인공지능의 윤리적 문제")