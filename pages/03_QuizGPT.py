import streamlit as st
import os
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers import WikipediaRetriever
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import openai
import json

if "disabled" not in st.session_state:
    st.session_state["disabled"] = False

# 함수==================================================
@st.cache_data(show_spinner="Embedding file...")
def split_file(file):
    file_content = file.read()
    folder_path = "./.cache/quiz_files/"
    file_path = f"{folder_path}{file.name}"
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
    return docs

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

@st.cache_data(show_spinner="Searching Wikipedia...")
def search_wiki(topic):
    retriever = WikipediaRetriever(
        top_k_result=1, # language 설정 가능
    )
    docs = retriever.invoke(topic)
    return docs

def check_api_key(api_key):
    try:
        openai.api_key = api_key
        openai.Model.list()
        os.environ["OPENAI_API_KEY"] = api_key
        return True
    except Exception:
        st.error("Wrong API Key")
        return False

@st.cache_data(show_spinner="Making quiz...")
def invoke_quiz_chain(_docs, topic, level):
    chain = quiz_template | llm
    message = chain.invoke({"context":_docs, "level":level})
    message = message.additional_kwargs["function_call"]["arguments"]
    return json.loads(message)

def disabled_button(question_num, correct_num):
    if question_num == correct_num:
        st.session_state["disabled"] = True
    else:
        st.session_state["disabled"] = False
# 함수==================================================

# page tab 제목, 파비콘 추가
st.set_page_config(
    page_title="QuizGPT",
    page_icon="💋"
)

st.title("QuizGPT")

quiz_function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and asnwers and return a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type":"object",
                    "properties": {
                        "question": {
                            "type": "string"
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string"
                                    },
                                    "correct": {
                                        "type": "boolean"
                                    },
                                },
                                "required":["answer", "correct"]
                            },
                        },
                    },
                    "required":["question", "answers"]
                },
            },
        },
        "required":["questions"]
    }
}

quiz_template = ChatPromptTemplate.from_messages([
    ("system", """
Context-based Questions:
All questions must be created based on the provided context.
The goal is to test how well the user understands the content of the context.
**Make 10 questions
     
Difficulty Levels:
For easy mode: Generate ten multiple-choice questions with four answer choices for each question.
For hard mode: Generate ten multiple-choice questions with five answer choices for each question.
     
Answer Key:
Randomize the position of the correct answer across all questions (i.e., the correct answer should not always be in the same position).
Ensure that the correct answer can appear in any of the options. 
**Mix up the order of answers** and make sure the correct one is not always the first option.
"""),
    ("human", """
     Quiz level: {level}
     Context: {context}
"""),
])

with st.sidebar:
    docs = None
    topic = None

    st.write("Github repo:\nhttps://github.com/ddeeen/FullStackGPT")

    with st.form("api_key"):
        api_key = st.text_input(label="Enter your OpenAI API key")
        submit = st.form_submit_button("Submit")

    level = st.radio(
        "Quiz Level", options=["Easy", "Hard"]
    )

    choice = st.selectbox(
        "Choosde what you want to use.",
        ("File", "Wikipedia Article"), 
    )
    if choice == "File":
        file = st.file_uploader("Upload your file(.pdf .docx or .txt)", type=[".pdf", ".txt", ".docx"])
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic and check_api_key(api_key):
            docs = search_wiki(topic)


if not docs:
    st.markdown(
        """
Welcome to QuizGPT.
I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.

Get started by uploading a file or searching on Wikipedia in the sidebar.
"""
    )
else:
    if check_api_key(api_key):
        llm = ChatOpenAI(
            temperature=0.1,
            model="gpt-4o-mini",
        ).bind(
            function_call = {"name":"create_quiz",},
            functions=[quiz_function,]
        )
        questions = invoke_quiz_chain(docs, topic if topic else file.name, level)
        st.markdown("## Choose the correct answer!")
        with st.form("quiz_form"):
            question_number = 0
            correct_number = 0
            for question in questions["questions"]:
                question_number += 1
                value = st.radio(
                    question["question"],
                    options=[answer["answer"] for answer in question["answers"]],
                    index=None
                )
                if {"answer":value, "correct":True} in question["answers"]:
                    st.success("Correct!")
                    correct_number += 1
                elif value:
                    st.error("Wrong")
            disabled_button(question_number, correct_number)
            st.form_submit_button("Submit", disabled=st.session_state["disabled"], 
                on_click=disabled_button, args=(question_number, correct_number,)
            )
        if st.session_state["disabled"]:
            st.balloons()