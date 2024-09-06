import streamlit as st
import os
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers import WikipediaRetriever
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda
from langchain.schema import BaseOutputParser
import json

class JsonOutputParser(BaseOutputParser):

    def parse(self, text):
        text = text.replace("```json", "").replace("```", "")
        return json.loads(text)
        # quiz = json.load(text)
        # with st.form(key="quiz"):
        #     for question in quiz["questions"]:
        #         st.write(question["question"])
        #         for answer in question["answers"]:
        #             st.checkbox(answer["answer"], value=answer["correct"])
        #     st.form_submit_button("Submit")

output_parser = JsonOutputParser()

# 함수==================================================
@st.cache_data(show_spinner="Embedding file...")
def split_file(file):
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

@st.cache_data(show_spinner="Making quiz...")
def invoke_quiz_chain(_docs, topic):
    chain = {"context":question_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)
# 함수==================================================

# page tab 제목, 파비콘 추가
st.set_page_config(
    page_title="QuizGPT",
    page_icon="💋"
)

st.title("QuizGPT")

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4o-mini",
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler(),
    ]
)

question_template = ChatPromptTemplate.from_messages([
    ("system", """
     You are a helpful assistant that is role playing as a teacher.

     Based ONLY on the following context make 10 questions to test the user's knowledge about the text.

     Each question should have 4 answers, three of them must be incorrect and one should be correct.

     Use (o) to signal the correct answer.

     Question examples:

     Question: What is the color of the ocean?
     Answers: Red|Yellow|Green|Blue(o)

     Question: What is the capital of Georgia?
     Answers: Baku|Tbilisi(o)|Manila|Beirut

     Question: When was Avatar released?
     Answers: 2007|2001|2009(o)|1998

     Question: Who was Julius Caesar?
     Answers: A Roman Emperor(o)|Painter|Actor|Model

     Your turn!

     Context: {context}
""")
])

question_chain = {"context":format_docs} | question_template | llm

formatting_template = ChatPromptTemplate.from_messages([
    ("system", """
     You are a powerful formatting algorithm.
     
     You format exam quesitons into JSON format.
     Answers with (o) are the correct ones.

     Example Input:

     Question: What is the color of the ocean?
     Answers: Red|Yellow|Green|Blue(o)

     Question: What is the captial of Georgia?
     Answers: Baku|Tbilisi(o)|Manila|Beirut

     Question:When was Avatar released?
     Answers 2007|2001|2009(o)|2024

     Question: Who was Julius Caesar?
     Answers: A Roman Emperor(o)|Painter|Actor|Model


     Example Output:
     
     ```json
     {{ "questions": [
        {{
            "question": "What is the color of the ocean?",
            "answers": [
                {{
                    "answer": "Red",
                    "correct": false
                }},
                {{
                    "answer": "Yellow",
                    "correct": false
                }},
                {{
                    "answer": "Green",
                    "correct": false
                }},
                {{
                    "answer": "Blue",
                    "correct": true
                }},
            ]
        }},
        {{
            "question": "What is the captial of Georgia",
            "answers": [
                {{
                    "answer": "Baku",
                    "correct": false
                }},
                {{
                    "answer": "Tbilisi",
                    "correct": true
                }},
                {{
                    "answer": "Manila",
                    "correct": false
                }},
                {{
                    "answer": "Beirut",
                    "correct": false
                }},
            ]
        }},
        {{
            "question": "When was Avatar released",
            "answers": [
                {{
                    "answer": "2007",
                    "correct": false
                }},
                {{
                    "answer": "2001",
                    "correct": false
                }},
                {{
                    "answer": "2009",
                    "correct": true
                }},
                {{
                    "answer": "2024",
                    "correct": false
                }},
            ]
        }},
        {{
            "question": "Who was Julius Caesar?",
            "answers": [
                {{
                    "answer": "A Roman Emperor",
                    "correct": true
                }},
                {{
                    "answer": "Painter",
                    "correct": false
                }},
                {{
                    "answer": "Actor",
                    "correct": false
                }},
                {{
                    "answer": "Model",
                    "correct": false
                }},
            ]
        }},
     ]}}
     ```
     Your turn!

     Context: {context}
""")
])

formatting_chain = formatting_template | llm

with st.sidebar:
    docs = None
    topic = None
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
        if topic:
            #1
            # retriever = WikipediaRetriever(
            #     top_k_result=1, # language 설정 가능
            # )
            # with st.status("Searching wikipedia..."):
            #     docs = retriever.invoke(topic)
            #2
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
    start = st.button("Generate Quiz")
    if start:
        #1
        # questions_response = question_chain.invoke(docs)
        # format_response = formatting_chain.invoke({"context" : questions_response.content})
        #2
        # chain = {"context":question_chain} | formatting_chain | output_parser
        # response = chain.invoke(docs)
        #3
        response = invoke_quiz_chain(docs, topic if topic else file.name)
        st.write(response)