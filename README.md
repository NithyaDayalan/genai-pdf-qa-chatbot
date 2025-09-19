## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM :
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT :
The objective is to build a chatbot that answers questions from the content of a PDF document. Using LangChain and OpenAI, the system processes the PDF, retrieves relevant information, and provides accurate responses to user queries.

### DESIGN STEPS :
#### STEP 1 :
Load and Process PDF – Import the PDF document, extract its content, and split it into smaller text chunks for efficient handling.
#### STEP 2 :
Embed and Store Content – Convert text chunks into embeddings and store them in an in-memory vector database for fast retrieval.
#### STEP 3 :
Build Question-Answering System – Use LangChain’s RetrievalQA with an OpenAI model to retrieve relevant chunks and generate accurate answers to user queries.

### PROGRAM :
```
import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

def build_pdf_qa(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    db = DocArrayInMemorySearch.from_documents(chunks, OpenAIEmbeddings())
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        retriever=db.as_retriever()
    )

qa = build_pdf_qa("docs/cs229_lectures/gen_ai_paper.pdf")
query = "What are the main topics covered in this document?"
answer = qa.run(query)
print("Q:", query)
print("A:", answer)

loader = PyPDFLoader("docs/cs229_lectures/gen_ai_paper.pdf")
pages = loader.load()
print(f"Loaded {len(pages)} pages from the PDF.")
```
### OUTPUT :
<img width="609" height="239" alt="image" src="https://github.com/user-attachments/assets/f2cb6090-88c2-4d3e-b62b-4fec269f792f" />

### RESULT :
Thus, a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain is executed successfully.
