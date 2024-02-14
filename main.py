import os
import streamlit as st
import pickle
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from helper_funcs import get_qa_chain, get_embeddings, get_model


def add_source():
    st.session_state['input_keys'].append(len(st.session_state.input_keys) + 1)
    st.session_state['calc_vectorstore'] = True


def is_embedding_changed():
    st.session_state['calc_vectorstore'] = True


def disable_query():
    st.session_state["disabled_query"] = True


if "disabled_query" not in st.session_state:
    st.session_state["disabled_query"] = False

if 'input_keys' not in st.session_state:
    st.session_state['input_keys'] = [1]

if 'calc_vectorstore' not in st.session_state:
    st.session_state['calc_vectorstore'] = True

st.title("ChatBot Research Tool:")
st.sidebar.title("Article URLs")

urls = []
for input_key in st.session_state.input_keys:
    input_value = st.sidebar.text_input(f"URL {input_key}", key=input_key)
    urls.append(input_value)

st.sidebar.button("Add new source", on_click=add_source)
model_radio = st.sidebar.radio("Choose model", ("OpenAI", "Google PaLM"))
embeddings_radio = st.sidebar.radio("Choose embeddings", ("OpenAI", "HuggingFace"), on_change=is_embedding_changed)
file_path = "faiss_db.pkl"
main_placeholder = st.empty()

process_url_clicked = st.sidebar.button("Process")
if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Loading data...")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Splitting text...")
    docs = text_splitter.split_documents(data)
    # create embeddings and save it to FAISS index
    embeddings = get_embeddings(embeddings_radio)
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Building embeddings vector database")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

    st.session_state.calc_vectorstore = False

if st.session_state.calc_vectorstore:
    disable_query()
    query = main_placeholder.text_input("Question: ", disabled=st.session_state.disabled_query)
else:
    query = main_placeholder.text_input("Question: ")
    if query:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
                llm = get_model(model_radio)
                chain = get_qa_chain(vectorstore, llm)
                result = chain({"question": query}, return_only_outputs=True)
                st.header("Answer")
                st.write(result["answer"])
                sources = result.get("sources", "")
                if sources:
                    st.subheader("Sources:")
                    sources_list = sources.split("\n")  # Split the sources by newline
                    for source in sources_list:
                        st.write(source)
