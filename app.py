import streamlit as st

# page tab 제목, 파비콘 추가
st.set_page_config(
    page_title="FullstackGPT Home",
    page_icon="💋"
)

if "success" not in st.session_state:
    st.session_state["success"] = False
if "api_ok" not in st.session_state:
    st.session_state.api_ok = False

st.title("FullstackGPT Home")

with st.sidebar:
    st.write("Github repo:\nhttps://github.com/ddeeen/FullStackGPT")

st.markdown(
    """
# Hello!

Welcome to my FullstackGPT Portfolio!

Here are the apps I made:

- [x] [DocumentGPT](/DocumentGPT)
"""
)