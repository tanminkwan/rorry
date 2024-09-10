from typing import Annotated, Sequence, TypedDict, Union, List, Tuple
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolInvocation, ToolExecutor

# OpenAI API 키 설정 (실제 사용 시 환경 변수로 관리하는 것이 좋습니다)
import os
#os.environ["OPENAI_API_KEY"] = "your-api-key-here"
from dotenv import load_dotenv
load_dotenv(dotenv_path='./../.env')
os.environ["LANGCHAIN_PROJECT"] = "langchain_study" # Langsmith project 명

# 상태 정의
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    next: Annotated[str, "The next agent to run"]
    topic: Annotated[str, "The topic of discussion"]

# Agent 1: 질문자
questioner_system_text = """
당신은 호기심 많은 질문자입니다.
주어진 주제에 대해 이미 한 질문을 제외하고 한가지 질문을 해주세요.
"""
questioner_prompt = ChatPromptTemplate.from_messages([
    ("system", questioner_system_text),
    ("human", "주제: {topic}\n\n이전 대화:\n{chat_history}\n\n다음 질문:"),
])

# Agent 2: 답변자
answerer_system_text = """
당신은 지식이 풍부한 답변자입니다. 
질문자의 질문에 대해 정보를 제공하고 대화를 이어나가는 답변을 해주세요. \
단, 답변은 100자 이내로 짧게 이전에 답하지 않은 한가지 내용만 답합니다.
"""
answerer_prompt = ChatPromptTemplate.from_messages([
    ("system", answerer_system_text),
    ("human", "주제: {topic}\n\n이전 대화:\n{chat_history}\n\n질문: {question}\n\n답변:"),
])

# Agent 3: 요약자
summarizer_system_text = """
당신은 대화 요약자입니다. 
먼저 당신은 대화가 충분한지를 판단합니다.

충분한 대화란 다음을 만족해야 합니다.
 - questioner와 answerer간의 대화가 5회 이상 진행되어야 함

대화가 충분한 경우 대화를 분석하여 요약을 제공해주세요.
"""
summarizer_prompt = ChatPromptTemplate.from_messages([
    ("system", summarizer_system_text),
    ("human", "주제: {topic}\n\n대화:\n{chat_history}\n\n이 충분히 대화했나요? 그렇다면 '예'라고 답하고 대화를 요약해주세요. 그렇지 않다면 '아니오'라고만 답해주세요."),
])

# LLM 설정
llm = ChatOpenAI(temperature=0.7)

# Agent 함수 정의
def questioner(state):
    messages = state['messages']
    topic = state['topic']
    chat_history = "\n".join([f"{m.name}: {m.content}" for m in messages])
    response = llm.invoke(questioner_prompt.format_messages(topic=topic, chat_history=chat_history))
    return {
        "messages": messages + [AIMessage(content=response.content, name="questioner")],
        "next": "answerer"
    }

def answerer(state):
    messages = state['messages']
    topic = state['topic']
    chat_history = "\n".join([f"{m.name}: {m.content}" for m in messages[:-1]])
    question = messages[-1].content
    response = llm.invoke(answerer_prompt.format_messages(topic=topic, chat_history=chat_history, question=question))
    return {
        "messages": messages + [AIMessage(content=response.content, name="answerer")],
        "next": "summarizer"
    }

def summarizer(state):
    messages = state['messages']
    topic = state['topic']
    chat_history = "\n".join([f"{m.type}: {m.content}" for m in messages])
    response = llm.invoke(summarizer_prompt.format_messages(topic=topic, chat_history=chat_history))
    
    if response.content.lower().startswith("예"):
        return {
            "messages": messages + [AIMessage(content=response.content)],
            "next": END
        }
    else:
        return {
            "messages": messages,
            "next": "questioner"
        }

# 그래프 정의
workflow = StateGraph(AgentState)

# 노드 추가
workflow.add_node("questioner", questioner)
workflow.add_node("answerer", answerer)
workflow.add_node("summarizer", summarizer)

# 엣지 추가
workflow.add_edge("questioner", "answerer")
workflow.add_edge("answerer", "summarizer")
workflow.add_edge("summarizer", "questioner")

# 시작 노드 설정
workflow.set_entry_point("questioner")

# 그래프 컴파일
app = workflow.compile()

# 실행
initial_state = {
    "messages": [],
    "next": "questioner",
    "topic": "인공지능의 윤리적 문제"
}

for output in app.stream(initial_state):
    # 출력의 첫 번째 (유일한) 키를 가져옵니다
    agent = list(output.keys())[0]
    agent_output = output[agent]
    
    if agent_output['next'] == END:
        print("충분한 정보가 수집되어 대화를 종료합니다.")
        print("최종 요약:", agent_output['messages'][-1].content)
        break
    else:
        print(f"{agent.capitalize()}: {agent_output['messages'][-1].content}")