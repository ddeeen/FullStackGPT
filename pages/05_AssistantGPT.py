from openai import OpenAI, AssistantEventHandler
from langchain.retrievers import WikipediaRetriever
from langchain.tools import DuckDuckGoSearchResults
from langchain.document_loaders import WebBaseLoader
import streamlit as st
import json
import os
import time

client = OpenAI()

def check_api_key(api_key):
    try:
        client.api_key = api_key
        client.models.list()
        os.environ["OPENAI_API_KEY"] = api_key
        return True
    except Exception:
        st.error("Wrong API Key")
        return False

st.cache_data(show_spinner="Searching wikipedia...")
def search_wiki(inputs):
    topic = inputs["term"]
    retriever = WikipediaRetriever(
        top_k_result=4,
    )
    docs = retriever.invoke(topic)
    result = "\n".join([doc.page_content for doc in docs]).replace("'", "\'").replace('"', '\"')
    return "\n".join([doc.page_content for doc in docs]).replace("'", "\'").replace('"', '\"')

st.cache_data(show_spinner="Searching duckduckgo...")
def search_duckduckgo(inputs):
    query = inputs["query"]
    search = DuckDuckGoSearchResults()
    str = search.invoke(query)
    find_s_string = "link: https://"
    find_e_string = "]"
    del_string = "link: "
    s_index = str.find(find_s_string)
    e_index = 0
    urls = []
    while s_index != -1:
        s_index += e_index
        e_index = str[s_index:].find(find_e_string) + s_index
        urls.append(str[s_index + len(del_string):e_index])
        s_index = str[e_index:].find(find_s_string)
    result = json.dumps({"urls":"\n".join(urls)})
    return json.dumps({"urls":"\n".join(urls)})

st.cache_data(show_spinner="Scrapping...")
def web_scrapping(inputs):
    urls = inputs["urls"]
    urls_list = urls.split("|")
    if len(urls_list) >= 2:
        loader = WebBaseLoader(urls_list[:2])
    else:
        loader = WebBaseLoader(urls_list)
    docs = loader.load()
    result = "\n\n".join([doc.page_content.replace("\n", " ").replace("  ", " ") for doc in docs]).replace("'","\'").replace('"', '\"')
    return "\n\n".join([doc.page_content.replace("\n", " ").replace("  ", " ") for doc in docs]).replace("'","\'").replace('"', '\"')

st.cache_data(show_spinner="Saving content...")
def save_content_to_txt(inputs):
    content = inputs["content"]
    filename = inputs["filename"]
    st.session_state["filename"] = filename
    folder_dir = "./.cache/agent"
    os.makedirs(folder_dir, exist_ok=True)
    with open(f"{folder_dir}/{filename}.txt", "w", encoding="utf-8") as file:
        file.write(content)
    return inputs["content"]

def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )

def get_messages(thread_id):
    messages = client.beta.threads.messages.list(
        thread_id=thread_id,
    )
    messages = list(messages)
    messages.reverse()
    for message in messages:
        print(f"{message.role}:{message.content[0].text.value}")

def run_message(thread_id, assistant_id):
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )
    st.session_state["run_id"] = run.id

def add_message(thread_id, assistant_id, content):
    try:
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=content
        )
        run_message(thread_id, assistant_id)
    except:
        try:
            create_thread()
            thread_id = st.session_state["thread_id"]
            client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=content
            )
            run_message(thread_id, assistant_id)
        except:
            st.error("Error. Plesae Refresh.")

def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id=action.id
        function = action.function
        try:
            output = functions_map[function.name](json.loads(function.arguments))
            outputs.append({
                "output":output,
                "tool_call_id": action_id,
            })
        except:
            outputs.append({
                "output":"Error",
                "tool_call_id": action_id,
            })
    return outputs

def submit_tool_outputs(run_id, thread_id):
    output = get_tool_outputs(run_id, thread_id)
    client.beta.threads.runs.submit_tool_outputs(
        run_id=run_id,
        thread_id=thread_id,
        tool_outputs=output,
    )
    return output

def create_thread():
    thread = client.beta.threads.create()
    st.session_state["thread_id"] = thread.id

def save_message(message, role):
    st.session_state["messages"].append({"message":message, "role":role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        if "message" in message:
            send_message(message["message"], message["role"], save=False)
        else:
            with st.chat_message("ai"):
                with open(f"./.cache/agent/{message['filename']}.txt", "r", encoding="utf-8") as file:
                    st.download_button(
                        label=f"File download >> {message['filename']}.txt", 
                        data=file,
                        file_name=f"{message['filename']}.txt",
                    )

functions_map = {
    "search_wiki":search_wiki,
    "search_duckduckgo":search_duckduckgo,
    "web_scrapping":web_scrapping,
    "save_content_to_txt":save_content_to_txt,
}

functions = [
    {
        "type":"function",
        "function":{
            "name":"search_wiki",
            "description":"Receives a single term, searches Wikipedia, and returns information about that term.",
            "parameters":{
                "type":"object",
                "properties":{
                    "term":{
                        "type":"string",
                        "description":"The term you will search for on Wikipedia."
                    },
                },
                "required":["term"],
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"search_duckduckgo",
            "description":"It receives a query, searches the input content using DuckDuckGo, and returns list of the website address(url)",
            "parameters":{
                "type":"object",
                "properties":{
                    "query":{
                        "type":"string",
                        "description":"The query you will search for. Example query: Research about the XZ backdoor",
                    },
                },
                "required":["query"],
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"web_scrapping",
            "description":"""This function accesses the specified URL and extracts information from them.
            This function receives URL list, scrapes the information from those URL, and returns web information.""",
            "parameters":{
                "type":"object",
                "properties":{
                    "urls":{
                        "type":"string",
                        "description":"url list to extracts web information. Example url list: 'https://en.wikipedia.org/wiki/URL|https://naver.com|https://daum.net'",
                    },
                },
                "required":["urls"],
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"save_content_to_txt",
            "description":"This tool receives content and a filename and saves them as a text file.",
            "parameters":{
                "type":"object",
                "properties":{
                    "content":{
                        "type":"string",
                        "description":"Content to be saved as a text file.",
                    },
                    "filename":{
                        "type":"string",
                        "description":"Filename to save the content. A single word that contains the content.",
                    },
                },
                "required":["content", "filename"],
            }
        }
    }
]

if "assistant_id" not in st.session_state:
    st.session_state["assistant_id"] = None

if "bef_api_key" not in st.session_state:
    st.session_state["bef_api_key"] = ""

if "model" not in st.session_state:
    st.session_state["model"] = ""

if "run_id" not in st.session_state:
    st.session_state["run_id"] = ""

if "filename" not in st.session_state:
    st.session_state["filename"] = ""

if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.set_page_config(
    page_title="AssistantGPT Home",
    page_icon="😎"
)

st.title("AssistantGPT")

st.markdown("""
Ask questions about a term.
""")

with st.sidebar:
    st.write("Github repo:\nhttps://github.com/ddeeen/FullStackGPT")
    with st.form("api_key"):
        api_key = st.text_input(label="Enter your OpenAI API key")
        submit = st.form_submit_button("Submit")
    model = None
    model = st.selectbox("Model", options=["", "gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"])
    if not model:
        st.error("Choose a model")

if check_api_key(api_key) and model:
    if "thread_id" not in st.session_state:
        create_thread()
    if not st.session_state["assistant_id"] or st.session_state["bef_api_key"] != api_key or st.session_state["model"] != model:
        st.session_state["bef_api_key"] = api_key
        st.session_state["model"] = model
        st.write("Assistant Create")
        assistant = client.beta.assistants.create(
            name="Research Assistant",
            instructions="""
        You are a research expert.
        When you get a research request, decide between Wikipedia and DuckDuckGo.
        Perform the search using the better option.
        If one source lacks information, gather more from the other.
        With DuckDuckGo, obtain the website's URL.
        Use the URL to extract information from the site.
        Write all collected content.
        And save all collected content as a text file.
        Name the file with the key term.
        If the content contains single quotes, please add an escape character '\' before them.
        """,
            model=model,
            tools=functions,
        )
        st.session_state["assistant_id"] = assistant.id


    send_message("What would you like to research?", "ai", save=False)
    paint_history()
    question = st.chat_input("Ask anything. ex)Research about the XZ backdoor")
    if question:
        send_message(question, "human")
        add_message(st.session_state["thread_id"], st.session_state["assistant_id"], question)
        while get_run(st.session_state["run_id"], st.session_state["thread_id"]).status != "completed" and get_run(st.session_state["run_id"], st.session_state["thread_id"]).status != "expired":
            if get_run(st.session_state["run_id"], st.session_state["thread_id"]).status == "requires_action":
                output = submit_tool_outputs(st.session_state["run_id"], st.session_state["thread_id"])
            if get_run(st.session_state["run_id"], st.session_state["thread_id"]).status == "in_progress":
                while get_run(st.session_state["run_id"], st.session_state["thread_id"]).status == "in_progress":
                    time.sleep(1)
        if get_run(st.session_state["run_id"], st.session_state["thread_id"]).status == "completed":
            send_message(output[0]["output"], "ai")
            st.session_state["messages"].append({"filename":st.session_state['filename']})
            with st.chat_message("ai"):
                with open(f"./.cache/agent/{st.session_state['filename']}.txt", "r", encoding="utf-8") as file:
                    st.download_button(
                        label=f"File download >> {st.session_state['filename']}.txt", 
                        data=file,
                        file_name=f"{st.session_state['filename']}.txt",
                    )
        elif get_run(st.session_state["run_id"], st.session_state["thread_id"]).status == "expired":
            send_message("Error: Timeout. Input new question", "ai", save=False)