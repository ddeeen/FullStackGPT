from uuid import UUID
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredFileLoader
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler
import openai
import os

# page tab 제목, 파비콘 추가
st.set_page_config(
    page_title="DocumentGPT",
    page_icon="🧊"
)

@st.cache_data(show_spinner="Embedding file...")
def embed_file(file, api_key):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(file_path, "wb") as f:
        f.write(file_content)
    loader = UnstructuredFileLoader(file_path)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    cache_file_dir = f"./.cache/embeddings/{file.name}"
    folder_path = os.path.dirname(cache_file_dir)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    cache_dir = LocalFileStore(cache_file_dir)
    cache_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstores = FAISS.from_documents(docs, cache_embeddings)
    retriever = vectorstores.as_retriever()
    return retriever

def save_message(message, role):
    st.session_state["messages"].append({"message":message, "role":role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

def check_api_key(api_key):
    try:
        openai.api_key = api_key
        openai.Model.list()
        return True
    except Exception:
        st.error("Wrong API Key")
        return False

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kargs):
        self.message_box = st.empty()
    
    def on_llm_end(self, *args, **kargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kargs):
        self.message += token
        self.message_box.markdown(self.message)

template = ChatPromptTemplate.from_messages([
    ("system", """
Answer the question using ONLY the following context.
If you don't know the answer just say you don't know.
DON't make anything up.
	
Context:{context}
"""),
    ("human", "{question}")
])

st.title("DocumentGPT")

st.markdown("""
Welcome!
            
Use this chatbot to ask questions to an AI about your files!
            
1. Input your OpenAI API key.
            
2. Upload your files on the sidebar.
""")

with st.sidebar:
    st.write("Github repo:\nhttps://github.com/ddeeen/FullStackGPT")

api_key = None

with st.sidebar:
    with st.form("api_key"):
        api_key = st.text_input(label="Enter your OpenAI API key")
        submit = st.form_submit_button("Submit")
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4o-mini",
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
    api_key=api_key
)

if api_key:
    if check_api_key(api_key) and file:
        retriever = embed_file(file, api_key)
        send_message("I'm ready! Ask away!", "ai", save=False)
        paint_history()
        message = st.chat_input("Ask anything about your file...")

        if message:
            send_message(message, "human")
            chain = {
                "context":retriever | RunnableLambda(format_docs), 
                "question":RunnablePassthrough()
            } | template | llm
            with st.chat_message("ai"):
                chain.invoke(message)
    else:
        st.session_state["messages"] = []