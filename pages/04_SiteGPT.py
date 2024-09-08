import asyncio
from langchain.document_loaders import AsyncChromiumLoader, SitemapLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate
import streamlit as st

asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose() # decompose() 함수를 사용하면 해당 tag의 요소만 제거할 수 있음.
    if footer:
        footer.decompose()
    return str(soup.get_text()).replace("\n", "").replace("  ", "")

# function========================================================================
@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
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


# ==================================================================================

st.set_page_config(
    page_title="SiteGPT Home",
    page_icon="🤩"
)
st.markdown("""
# SiteGPT
Ask questions about the content of a website.
            
Start by writing the URL of the website on the sidebar.
""")

with st.sidebar:
    url = st.text_input("Write down a  URL", placeholder="ex) example.com")

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4o-mini",
)



if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL")
    else:
        # 2
        retriever = load_website(url)
        query = st.text_input("Ask a question to the website.")
        if query:
            chain = {"docs": retriever, "question":RunnablePassthrough()} | RunnableLambda(get_answers) | RunnableLambda(choose_answer)
            result = chain.invoke(query)
            st.write(result.replace("$", "\$"))