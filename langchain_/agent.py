from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

import os
import re

class Agent:
    def __init__(self) -> None:

        #llm = Ollama(model="llama3:instruct", temperature=0.2)
        #llm = Ollama(model="bnksys/llama3-ko-8b", temperature=0.2)
        llm = Ollama(model="aya", temperature=0.2)
        #llm = Ollama(model="gemma", temperature=0.2)
        
        output_parser = StrOutputParser()

        # 프롬프트 템플릿 설정
        prompt = ChatPromptTemplate.from_messages(
            [
                #("system", "You are a human being. Say like a man."),
                ("system", "You are an AI that creates a report by organizing the content that the user tells you"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{user_input}"),
            ]
        )

        self.memory = ConversationBufferMemory(return_messages=True)
        
        self.__chain = (
            RunnablePassthrough.assign(
                history=RunnableLambda(self.memory.load_memory_variables) | itemgetter("history")
            )
            | prompt
            | llm
            | output_parser
        )

    def invoke(self, input):
        return self.__chain.invoke(input)