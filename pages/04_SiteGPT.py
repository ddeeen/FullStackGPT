from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
import openai
import os

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kargs):
        self.message_box = st.empty()
    
    def on_llm_end(self, *args, **kargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kargs):
        self.message += token
        self.message_box.markdown(self.message)

def parse_page(soup):
    header = soup.find("header")
    if header:
        header.decompose()
    footer = soup.find("footer")
    if footer:
        footer.decompose()
    for nav in soup.find_all("nav"):
        if nav:
            nav.decompose()
    related_products = soup.find("div", class_="items-start")
    if related_products:
        related_products.decompose()
    more_resources = soup.find("div", class_="astro-zntqmydn")
    if more_resources:
        more_resources.decompose()
    for footer_link in soup.find_all("a", class_="mx-2"):
        footer_link.decompose()
    return str(soup.get_text()).replace("\n", " ")
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

# function========================================================================
@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        filter_urls=[ #입력으로 받을 수 있는 list는 2 종류. 1) data를 load 하고 싶은 url들을 담은 list. 2) 정규식(regular expression)을 사용하는 것.
            r"^(.*\/ai-gateway\/).*",
            r"^(.*\/vectorize\/).*",
            r"^(.*\/workers-ai\/).*",
        ],
        parsing_function=parse_page, 
    )
    loader.requests_per_second = 1 # 1초당 몇 번 요청할 것인가.
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()

answers_template = ChatPromptTemplate.from_template("""
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
""")

# chain에서 받아온 input의 각 문서마다 질문을 적용해서 답변+점수의 list를 반환해주는 함수
def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_template | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"context": doc.page_content,
                    "question":question}).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            } for doc in docs
        ],
    }

choose_template = ChatPromptTemplate.from_messages([
    ("system","""
    Use ONLY the following pre-existing answers to answer the user's question.

    Use the answers that have the highest score (more helpful) and favor the most recent ones.

    Cite sources and return the sources of the answers as they are, do not change them.

    Answers: {answers}
    """,
    ),
    ("human", "{question}"),
])

def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_template | llm
    # 2
    condensed = "".join(f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n" for answer in answers)
    return choose_chain.invoke({"question":question, "answers": condensed})

def check_api_key(api_key):
    try:
        openai.api_key = api_key
        openai.Model.list()
        os.environ["OPENAI_API_KEY"] = api_key
        return True
    except Exception:
        st.error("Wrong API Key")
        return False
# ==================================================================================

if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.set_page_config(
    page_title="SiteGPT Home",
    page_icon="🤩"
)
st.markdown("""
# SiteGPT for Cloudflare
Ask questions about the content of a website.
            
Start by writing the URL of the website on the sidebar.
""")
with st.sidebar:
    st.write("Github repo:\nhttps://github.com/ddeeen/FullStackGPT")
    with st.form("api_key"):
        api_key = st.text_input(label="Enter your OpenAI API key")
        submit = st.form_submit_button("Submit")

with st.sidebar:
    url = st.text_input("Write down a Cloudflare sitemap.xml", placeholder="ex) https://developers.cloudflare.com/sitemap-0.xml")

if check_api_key(api_key) and url:
    if ".xml" not in url or "cloudflare" not in url:
        with st.sidebar:
            st.error("Please write down a cloudflare Sitemap URL")
    else:
        llm = ChatOpenAI(
            temperature=0.1,
            model="gpt-4o-mini",
            callbacks=[
                ChatCallbackHandler()
            ]
        )
        retriever = load_website(url)
        send_message("I'm ready! Ask away!!", "ai", save=False)
        paint_history()
        message = st.chat_input("Ask a question to the website!")
        if message:
            send_message(message, "human")
            chain = {"docs": retriever, "question":RunnablePassthrough()} | RunnableLambda(get_answers) | RunnableLambda(choose_answer)
            with st.chat_message("ai"):
                chain.invoke(message).content