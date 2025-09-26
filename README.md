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
import os, sys, datetime, panel as pn, param
from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import Chroma, DocArrayInMemorySearch
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

pn.extension()
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ['OPENAI_API_KEY']

llm_name = "gpt-3.5-turbo" if datetime.datetime.now().date() >= datetime.date(2023, 9, 2) else "gpt-3.5-turbo-0301"
llm = ChatOpenAI(model_name=llm_name, temperature=0)

def load_db(file, chain_type, k):
    docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(PyPDFLoader(file).load())
    db = DocArrayInMemorySearch.from_documents(docs, OpenAIEmbeddings())
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0),
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True
    )

class cbfs(param.Parameterized):
    chat_history = param.List([])
    answer = param.String("")
    db_query = param.String("")
    db_response = param.List([])

    def __init__(self, **params):
        super().__init__(**params)
        self.panels = []
        self.loaded_file = "docs/cs229_lectures/gen_ai_paper.pdf"
        self.qa = load_db(self.loaded_file, "stuff", 4)

    def call_load_db(self, count):
        if count and file_input.value:
            file_input.save("temp.pdf")
            self.loaded_file = file_input.filename
            self.qa = load_db("temp.pdf", "stuff", 4)
        self.clr_history()
        return pn.pane.Markdown(f"Loaded File: {self.loaded_file}")

    def convchain(self, query):
        if not query: return pn.WidgetBox(pn.Row('User:', pn.pane.Markdown("")), scroll=True)
        result = self.qa({"question": query, "chat_history": self.chat_history})
        self.chat_history.append((query, result["answer"]))
        self.db_query, self.db_response, self.answer = result["generated_question"], result["source_documents"], result["answer"]
        self.panels.extend([pn.Row('User:', pn.pane.Markdown(query, width=600)),
                            pn.Row('ChatBot:', pn.pane.Markdown(self.answer, width=600, style={'background-color':'#F6F6F6'}))])
        inp.value = ''
        return pn.WidgetBox(*self.panels, scroll=True)

    @param.depends('db_query')
    def get_lquest(self):
        return pn.Column(pn.Row(pn.pane.Markdown("DB query:", styles={'background-color':'#F6F6F6'})), 
                         pn.pane.Str(self.db_query or "no DB accesses so far"))

    @param.depends('db_response')
    def get_sources(self):
        if not self.db_response: return
        return pn.WidgetBox(*[pn.Row(pn.pane.Str(doc)) for doc in self.db_response], width=600, scroll=True)

    @param.depends('convchain', 'clr_history')
    def get_chats(self):
        if not self.chat_history: return pn.pane.Str("No History Yet")
        return pn.WidgetBox(*[pn.Row(pn.pane.Str(x)) for x in self.chat_history], width=600, scroll=True)

    def clr_history(self, *_):
        self.chat_history, self.panels = [], []

cb = cbfs()
file_input = pn.widgets.FileInput(accept='.pdf')
button_load = pn.widgets.Button(name="Load DB", button_type='primary')
button_clearhistory = pn.widgets.Button(name="Clear History", button_type='warning')
button_clearhistory.on_click(cb.clr_history)
inp = pn.widgets.TextInput(placeholder='Enter text here…')
bound_button_load = pn.bind(cb.call_load_db, button_load.param.clicks)
conversation = pn.bind(cb.convchain, inp)

tab1 = pn.Column(inp, pn.layout.Divider(), pn.panel(conversation, loading_indicator=True, height=300))
tab2 = pn.Column(pn.panel(cb.get_lquest), pn.layout.Divider(), pn.panel(cb.get_sources))
tab3 = pn.Column(pn.panel(cb.get_chats), pn.layout.Divider())
tab4 = pn.Column(pn.Row(file_input, button_load, bound_button_load),
                  pn.Row(button_clearhistory, pn.pane.Markdown("Clears chat history.")))

dashboard = pn.Column(pn.pane.Markdown('# ChatWithYourData_Bot'),
                      pn.Tabs(('Conversation', tab1), ('Database', tab2), ('Chat History', tab3), ('Configure', tab4)))
dashboard
```
### OUTPUT :
<img width="928" height="559" alt="image" src="https://github.com/user-attachments/assets/2a39b2da-2fb3-4991-baf5-da896671f417" />
<img width="923" height="559" alt="image" src="https://github.com/user-attachments/assets/8d9fb66a-3bac-4830-adc0-ba3b0e7e93b2" />
<img width="790" height="375" alt="image" src="https://github.com/user-attachments/assets/c0e44046-b18f-41fc-9990-973bce9cd14f" />
<img width="787" height="236" alt="image" src="https://github.com/user-attachments/assets/2812a195-8e16-4588-ac2e-d63f7f5057fb" />

### RESULT :
Thus, a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain is executed successfully.
