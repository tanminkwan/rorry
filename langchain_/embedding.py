import os
from config import OPENAI_API_KEY

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

from langchain_community.vectorstores import LanceDB

db = LanceDB(uri='./lancedb', embedding=embeddings)

question = "러시아의 푸틴과 그 당시 우크라이나 대통령에 대한 이야기를 해줘"

"""
query = "What did the president say about Ketanji Brown Jackson"
"""

docs = db.similarity_search(question, k=3)

for doc in docs:
    print(doc.page_content, doc.metadata['_distance'])

#print(docs[0].page_content)

retriever = db.as_retriever(search_kwargs={'k': 3})

def format_docs(_docs):
    return "\n\n".join(doc.page_content for doc in _docs)

prompt_template = """

You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
가능하다면 한국어로 답변해줘.

Question : {question}

Context : {context}
"""
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama

llm = Ollama(model="aya:8b-23-q8_0", temperature=0.2)
#llm = Ollama(model="llama3:8b-instruct-q8_0", temperature=0.2)

output_parser = StrOutputParser()

prompt = PromptTemplate.from_template(prompt_template)

chain = (
    {"context": retriever | format_docs, "question" : RunnablePassthrough()}
    | prompt
    | llm
    | output_parser
)

responce = chain.invoke(question)
print(responce)