from langchain.tools import tool
from langchain_community.utilities import SerpAPIWrapper
from langchain.chains import LLMMathChain
from langchain_openai.chat_models import ChatOpenAI

import os
from config import OPENAI_API_KEY, SERPAPI_API_KEY

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['SERPAPI_API_KEY'] = SERPAPI_API_KEY

llm = ChatOpenAI(model="gpt-3.5-turbo-1106", streaming=True)
# serp api
# google search를 위해서 google-search-results 패키지를 설치해야 한다.
# SERPAPI_API_KEY에 serpapi key 값을 환경 변수로 등록해야 한다.
@tool("search")
def search_api(query : str) -> str:
    """Searchs the api for the query""" # tool decorator를 사용하면 docstring으로 이에 대한 설명을 적는다.
    #search = SerpAPIWrapper()
    #result = search.run(query)
    result = "이름은 거거거이고 나이는 18세"
    return result

@tool("math")
def math(query : str) -> str:
    """useful for when you need to answer questions about math"""
    llm_math_chain = LLMMathChain(llm=llm, verbose=True)
    return llm_math_chain.run(query)

tools = [search_api, math]

# prompt 정의
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are very powerful assistant, but bad at calculating lengths of words."),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"), # 중간 과정 전달
    ]
)

# tool을 사용할 수 있게 openai function으로 변경
from langchain_core.utils.function_calling import convert_to_openai_function

openai_functions = [convert_to_openai_function(t) for t in tools]
print(openai_functions)
chat_model_with_tools = llm.bind(functions = [convert_to_openai_function(t) for t in tools])

# agent 생성
from langchain.schema.runnable.passthrough import RunnablePassthrough
from langchain.schema.runnable import RunnableLambda
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents.format_scratchpad import format_to_openai_functions
from operator import itemgetter

agent = (
    RunnablePassthrough.assign(
        agent_scratchpad=RunnableLambda(itemgetter('intermediate_steps'))
        | format_to_openai_functions
    )
    | prompt
    | chat_model_with_tools
    | OpenAIFunctionsAgentOutputParser()
)

# query 
res = agent.invoke({"input": "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?", "intermediate_steps": []})
print(res)

# agent executor 사용
from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
res = agent_executor.invoke({"input": "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"})
print(res)