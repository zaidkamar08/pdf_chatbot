import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
import tempfile

load_dotenv()

st.set_page_config(page_title="PDF Chatbot", page_icon="")
st.title(" Chat with your PDF")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and st.session_state.vectorstore is None:
    with st.spinner("Reading your PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(uploaded_file.read())
            tmp_path = f.name
        docs = PyMuPDFLoader(tmp_path).load()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        ).split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
    st.success(f"Ready! {len(chunks)} chunks loaded.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if question := st.chat_input("Ask something about your PDF..."):
    if st.session_state.vectorstore is None:
        st.warning("Please upload a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                docs = st.session_state.vectorstore.similarity_search(question, k=3)
                context = "\n\n".join([d.page_content for d in docs])
                pages = list(set([str(d.metadata.get('page','?')) for d in docs]))
                llm = ChatGroq(model="llama-3.1-8b-instant")
                response = llm.invoke([
                    SystemMessage(content=f"Answer using only this context. If unsure say 'I don't know'.\n\n{context}"),
                    HumanMessage(content=question)
                ])
                answer = f"{response.content}\n\n *Pages: {', '.join(pages)}*"
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
