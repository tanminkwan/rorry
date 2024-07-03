from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from prompt import system_prompt
from langchain_core.pydantic_v1 import BaseModel
import os

class Agent:
    def __init__(self) -> None:
        self.__load_env__()
        self.__create_llm_client()

    def __create_llm_client(self) -> None:
        self.__llm = ChatOpenAI(model="gpt-4o", temperature=0)

    def __load_env__(self) -> None:
        load_dotenv(dotenv_path='./../.env')
        #os.environ["LANGCHAIN_PROJECT"] = "langchain_study" # Langsmith project 명

    def __get_output_parser(self, pydantic_object) -> PydanticOutputParser:
        return PydanticOutputParser(pydantic_object=pydantic_object)

    def get_quiz(self, content: str, pydantic_object: BaseModel ) -> dict:
        
        output_parger = self.__get_output_parser(pydantic_object)

        eval_prompt = \
            PromptTemplate(
                template="""

                    System : {system_prompt}
    
                    content : {content}

                    Format :
                    {format}
                """,
                input_variables=["content"],
                partial_variables={
                    "system_prompt" : system_prompt,
                    "format" : output_parger.get_format_instructions(),
                },
            )
        
        self.__chain = (
            eval_prompt
            | self.__llm
            | output_parger
        )

        response = self.__chain.invoke({"content":content})
        return response
    
if __name__ == "__main__":
    agent = Agent()