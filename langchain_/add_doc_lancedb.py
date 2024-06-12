import os
from config import OPENAI_API_KEY

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

"""
import lancedb

db = lancedb.connect("lancedb")
table = db.create_table(
    "my_table",
    data=[
        {
            "vector": embeddings.embed_query("Hello World"),
            "text": "Hello World",
            "id": "1",
        }
    ],
    mode="overwrite",
)
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import LanceDB

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
raw_documents = TextLoader('./state_of_the_union.txt', encoding='UTF8').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(raw_documents)

vectorstore = LanceDB(uri='./lancedb', embedding=embeddings)
vectorstore.add_documents(documents)

query = "What did the president say about Ketanji Brown Jackson"

db = LanceDB(uri='./lancedb', embedding=embeddings)
docs = db.similarity_search(query, k=3)

for doc in docs:
    print(doc.page_content, doc.metadata['_distance'])
