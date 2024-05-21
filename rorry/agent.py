from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from operator import itemgetter
import os
import re
from system_prompt import worldview_prompt, system_prompts
from legacy_engine import LegacyEngine

class Agent:
    def __init__(self, system_name) -> None:

        os.environ['OPENAI_API_KEY']=""
        llm = ChatOpenAI(model="gpt-4o", temperature=0)

        prompt_str = worldview_prompt.format(system_name=system_name) + system_prompts[system_name]
        print(prompt_str)
        # 프롬프트 템플릿 설정
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_str),
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
        )

        self.system_name = system_name
        self.legacy = LegacyEngine(self.system_name)

    def __parse_content(self, content):

        # 정규 표현식을 사용하여 모든 <system_command> 태그 사이의 텍스트 추출
        matches = re.findall(r'<system_command>(.*?)</system_command>', content, re.DOTALL)

        # 각 추출된 텍스트를 리스트로 만듦
        extracted_commands = [match.strip() for match in matches]

        return extracted_commands

    def __call_legacy(self, command):
        rtn, message = self.legacy.command(command)
        return rtn, message

    def invoke(self, input):
        response = self.__chain.invoke(input)
        commands = self.__parse_content(response.content)

        return_messages = [response.content]

        if commands:
            return_messages.append('\n\n Messages from Legacies :')
            for command in commands:
                rtn, message = self.__call_legacy(command)
                return_messages.append('```'+message+'```')
                if rtn < 1:
                    break

        return '\n\n'.join(return_messages)