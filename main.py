from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv, dotenv_values
import os

#Setting up paths and OpenAI Chatbot Model
DATA_PATH = r"/Users/MarshallRhodes/Documents/Python/RAG Chatbot/PDFs"
CHROMA_PATH = "chroma"
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(api_key = OPENAI_API_KEY)

#Loading PDFs folder in Langchain PDF directory loader
loader = PyPDFDirectoryLoader(DATA_PATH)
documents = loader.load()

#Splitting documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 500, add_start_index = True)
chunks = text_splitter.split_documents(documents)

#Building a vectorstore
vectorestore = Chroma.from_documents(documents = chunks, embedding=OpenAIEmbeddings(api_key=OPENAI_API_KEY))
retriever = vectorestore.as_retriever()

#Setting up prompt and ragchain
prompt = hub.pull("rlm/rag-prompt", api_key=OPENAI_API_KEY)

def format_docs(docs):
    return "/n/n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

while True:
    query = input("What would you like to know from your PDFs?")
    if query == "q":
        break
    else:
        answer = rag_chain.invoke(query)
        print(answer)